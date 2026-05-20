import argparse
import sys
import random
import os
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from enum import Enum
import torch.nn as nn
import math
from timm.data.mixup import one_hot
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

# 设置随机种子，确保程序在运行时的随机行为是可以重复的。
def seed_torch(seed=2):
    print('seed=',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False #禁用 CuDNN 的自动优化模式。
    torch.backends.cudnn.deterministic=True #启用 CuDNN 的确定性模式。


# 将pytorch张量(tensor)表示的图像转换为Numpy数组表示的图像
def To_img(tensor_image):
    numpy_image = tensor_image.mul(255).cpu().numpy() #将像素值从[0,1]缩放到[0,255]
    integer_image = numpy_image.astype(np.uint8) #数据类型转为uint8
    hwc_image = np.transpose(integer_image, (1, 2, 0)) #调整维度顺序，从(C,H,W)转换为(H,W,C)
    return hwc_image

class Logger(object):
    def __init__(self,log_name):
        self.terminal = sys.stdout
        self.log_name=log_name
        self.log = open(self.log_name, "a")

    def write(self, message):
        if message.strip() != "":
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            message = f"[{timestamp}] {message}"
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __del__(self):
        self.log.close()    

def generate_flow_mask(f1,f2, tau_m1, tau_m2, tau_d):
    """
    生成混合掩码，基于幅度显著性和方向一致性。

    参数:
        f1 (np.ndarray): 原始光流，形状为 (H, W, 2)，形状为 (B, 2, H, W)，其中 f1[:, 0, :, :] 是 u 分量，f1[:, 1, :, :] 是 v 分量。
        f2 (np.ndarray): 扰动光流，形状为 (H, W, 2)，形状为 (B, 2, H, W)，其中 f2[:, 0, :, :] 是 u 分量，f2[:, 1, :, :] 是 v 分量。
        tau_m1 (float): 原始光流的幅度显著性阈值。
        tau_m2 (float): 扰动光流的幅度显著性阈值。
        tau_d (float): 方向一致性阈值（弧度）。
        lam_alpha (float): Beta 分布的超参数，用于控制扰动系数 λ 的分布。

    返回:
        M_p (torch.Tensor): 扰动掩码，形状为 (B, 1, H, W)，值为 0 或 1。
        lam_list (torch.Tensor): 扰动系数 λ，形状为 (B,)。
    """

    B,C,H,W=f1.shape
    # 计算光流幅度
    magnitude_f1 = torch.norm(f1, dim=1)  # ||f1|| = sqrt(u1^2 + v1^2), 形状为 (B, H, W)
    magnitude_f2 = torch.norm(f2, dim=1)  # ||f2|| = sqrt(u2^2 + v2^2), 形状为 (B, H, W)

    # 幅度显著性掩码
    R1 = (magnitude_f1 >= tau_m1)  # 原始光流的显著区域
    R2 = (magnitude_f2 >= tau_m2)  # 扰动光流的显著区域
    M_m = torch.logical_or(R1, R2)    # 显著区域的并集

    # 计算光流方向
    theta_f1 = torch.atan2(f1[:, 1, :, :], f1[:, 0, :, :])  # θ1 = arctan(v1 / u1), 形状为 (B, H, W)
    theta_f2 = torch.atan2(f2[:, 1, :, :], f2[:, 0, :, :])  # θ2 = arctan(v2 / u2), 形状为 (B, H, W)

    # 方向一致性掩码
    theta_diff = torch.abs(theta_f1 - theta_f2)  # |θ1 - θ2|
    theta_diff = torch.minimum(theta_diff, 2 * np.pi - theta_diff)  # 处理角度环绕问题
    M_d = (theta_diff <= tau_d)  # 方向一致性掩码

    # 综合掩码
    M_p = M_m * M_d      


    # 扩展掩码维度
    M_p = M_p.unsqueeze(1).float()  # 形状为 (B, 1, H, W)    

    return M_p


def calculate_thresholds(f1, f2, magnitude_percentile, direction_percentile):
    """
    计算幅度显著性阈值和方向一致性阈值。

    参数:
        f1 (torch.Tensor): 原始光流，形状为 (B, 3, H, W)，其中 f1[:, 0, :, :] 是 u 分量，f1[:, 1, :, :] 是 v 分量。
        f2 (torch.Tensor): 扰动光流，形状为 (B, 3, H, W)，其中 f2[:, 0, :, :] 是 u 分量，f2[:, 1, :, :] 是 v 分量。
        magnitude_percentile (float): 幅度显著性百分位数，用于计算幅度显著性阈值。例如，0.8 表示选择幅度值的前 80% 作为显著区域。
        direction_percentile (float): 方向一致性百分位数，用于计算方向一致性阈值。例如，0.9 表示选择方向差异的前 90% 作为一致性区域。

    返回:
        tau_m1 (float): 原始光流的幅度显著性阈值。
        tau_m2 (float): 扰动光流的幅度显著性阈值。
        tau_d (float): 方向一致性阈值（弧度）。
    """
    # 计算光流幅度
    magnitude_f1 = torch.norm(f1, dim=1)  # ||f1|| = sqrt(u1^2 + v1^2), 形状为 (B, H, W)
    magnitude_f2 = torch.norm(f2, dim=1)  # ||f2|| = sqrt(u2^2 + v2^2), 形状为 (B, H, W)

    # 计算幅度显著性阈值
    tau_m1 = torch.quantile(magnitude_f1, magnitude_percentile)  # 原始光流的幅度显著性阈值
    tau_m2 = torch.quantile(magnitude_f2, magnitude_percentile)  # 扰动光流的幅度显著性阈值

    # 计算光流方向
    theta_f1 = torch.atan2(f1[:, 1, :, :], f1[:, 0, :, :])  # θ1 = arctan(v1 / u1), 形状为 (B, H, W)
    theta_f2 = torch.atan2(f2[:, 1, :, :], f2[:, 0, :, :])  # θ2 = arctan(v2 / u2), 形状为 (B, H, W)

    # 计算方向差异
    theta_diff = torch.abs(theta_f1 - theta_f2)  # |θ1 - θ2|, 形状为 (B, H, W)
    theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)  # 处理角度环绕问题

    # 计算方向一致性阈值
    tau_d = torch.quantile(theta_diff, direction_percentile)  # 方向一致性阈值

    return tau_m1.item(), tau_m2.item(), tau_d.item()

from sklearn.feature_extraction.image import extract_patches_2d
from skimage.filters import threshold_otsu

def calculate_thresholds_otsu(f1, f2):
    """
    基于 Otsu 方法计算幅度显著性阈值和方向一致性阈值。

    参数:
        f1 (torch.Tensor): 原始光流，形状为 (B, 2, H, W)。
        f2 (torch.Tensor): 扰动光流，形状为 (B, 2, H, W)。

    返回:
        tau_m1 (float): 原始光流的幅度显著性阈值。
        tau_m2 (float): 扰动光流的幅度显著性阈值。
        tau_d (float): 方向一致性阈值（弧度）。
    """
    

    # 计算光流幅度
    magnitude_f1 = torch.norm(f1, dim=1).cpu().numpy()  # ||f1|| = sqrt(u1^2 + v1^2)
    magnitude_f2 = torch.norm(f2, dim=1).cpu().numpy()  # ||f2|| = sqrt(u2^2 + v2^2)

    # 使用 Otsu 方法计算幅度显著性阈值
    tau_m1 = threshold_otsu(magnitude_f1)  # 原始光流的幅度显著性阈值
    tau_m2 = threshold_otsu(magnitude_f2)  # 扰动光流的幅度显著性阈值

    # 计算光流方向
    theta_f1 = torch.atan2(f1[:, 1, :, :], f1[:, 0, :, :]).cpu().numpy()  # θ1 = arctan(v1 / u1)
    theta_f2 = torch.atan2(f2[:, 1, :, :], f2[:, 0, :, :]).cpu().numpy()  # θ2 = arctan(v2 / u2)

    # 计算方向差异
    theta_diff = np.abs(theta_f1 - theta_f2)  # |θ1 - θ2|
    theta_diff = np.minimum(theta_diff, 2 * np.pi - theta_diff)  # 处理角度环绕问题

    # 使用 Otsu 方法计算方向一致性阈值
    tau_d = threshold_otsu(theta_diff)  # 方向一致性阈值

    return tau_m1, tau_m2, tau_d

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    return Variable(y_onehot.cuda(), requires_grad=False)

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''
    

class mAP:
    """mean average precision: 1/|Right| * sum( P@k )
    """
    def __init__(self):
        self.type = 0

    def eval_scalar(self,pred_s, true_s):
        if pred_s.shape[1] > 1 or true_s.shape[1] > 1:
            print('Inputs need to be a torch scalar!')

    def eval_vector(self,pred_mat, true_mat, bins = None ):
        """mean average precision with input of vectored labels:
            pred_mat: N*C matrix
            true_mat: N*C matrix
        """
        if not (torch.is_tensor(pred_mat) and torch.is_tensor(true_mat)):
            print('Inputs need to be a torch tensor!')

        num_classes = pred_mat.shape[1]
        num_samples = pred_mat.shape[0]
        if bins is None:
            K = num_samples
        else:
            K = bins
        pred_sorted, idx_mat = torch.sort(pred_mat,dim=0, descending=True)
        precisions = torch.zeros(num_classes)
        for i in range(num_classes):
            idx = idx_mat[:,i]
            # x = true_mat[idx,i]
            x = torch.index_select(true_mat[:,i],0,idx)
            y = torch.cumsum(x,dim=0)
            num = torch.FloatTensor(range(num_samples))+1
            y /= num
            precisions[i] = torch.mean(y[:K])

        map = torch.mean(precisions)

        return map

    def eval_matrix(self,pred_mat, true_mat, bins = None ):
        """mean average precision with input of vectored labels:
            pred_mat: N*C_1 matrix
            true_mat: N*C_2 matrix
        """
        if not (torch.is_tensor(pred_mat) and torch.is_tensor(true_mat)):
            print('Inputs need to be a torch tensor!')

        num_bins = pred_mat.shape[1]
        num_samples = pred_mat.shape[0]
        num_classes = true_mat.shape[1]
        if bins is None:
            K = num_samples
        else:
            K = bins

        # calculating similarity matrix
        pred_s = pred_mat.mm(pred_mat.t())
        pred_s = torch.div(pred_s,torch.diag(pred_s))
        true_s = true_mat.mm(true_mat.t())
        idx_rm = [i for i, v in enumerate(torch.diag(true_s)) if v == 0]
        # np.savetxt(os.path.join('data', 'tmp.csv'), torch.diag(true_s).numpy(), fmt="%d")
        true_s = torch.div(true_s, torch.diag(true_s))
        pred_sorted, idx_mat = torch.sort(pred_s,dim=0, descending=True)

        precisions = torch.zeros(num_samples)
        for i in set(range(num_samples))-set(idx_rm):
            idx = idx_mat[:,i]
            x = torch.index_select(true_s[:,i],0,idx)
            y = torch.cumsum(x,dim=0)
            num = torch.FloatTensor(range(num_samples))+1
            y /= num
            precisions[i] = torch.mean(y[:K])

        map = torch.mean(precisions)

        return map

class accuracy:
    """accuracy:
    """
    def __init__(self):
        self.type = 0

    def eval(self,pred_v, true_v):
        # calulate the weighted accuracy or unbalanced accuracy
        idx_a = [i for i, value in enumerate(pred_v) if pred_v[i] == true_v[i]]
        acc_weighted = float(len(idx_a))/float(len(pred_v))
        # calculate the unweighted accuracy or balanced accuracy
        labels = torch.unique(true_v)
        acc = torch.zeros(len(labels))
        for i in range(len(labels)):
            idx_c = [j for j in range(len(true_v)) if true_v[j] == labels[i]]
            acc[i] = torch.sum(pred_v[idx_c] == true_v[idx_c]).double()/float(len(idx_c))
        acc_unweighted = torch.mean(acc)
        return acc_weighted, acc_unweighted

class f1score:
    """f1score: weighted and unweighted
    """

    def __init__(self):
        self.type = 0

    def eval(self, pred_v, true_v):
        # calulate the weighted f1 score
        f1 = metrics.f1_score(true_v.float(), pred_v.float(), average='micro')
        f1_weighted = metrics.f1_score(true_v.float(), pred_v.float(), average='macro')
        return f1, f1_weighted