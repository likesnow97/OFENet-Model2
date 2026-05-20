import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from preprocess import *

def get_mean_std(all_data):
    # all_data: N x C x H x W 的 ndarray
    all_data = np.concatenate(all_data, axis=0)  # 合并所有样本 (N, C, H, W)
    # 对样本数量 N、图片的高 H 和 宽 W 这三个维度求平均,不计算 C 维度上的均值，而是 分别对每个通道单独计算 mean 和 std
    mean = np.mean(all_data, axis=(0, 2, 3))  # 通道维度
    std = np.std(all_data, axis=(0, 2, 3))

    return mean, std

# ========================= #
# 主流程
# ========================= #

raf_path = '/nfs/users/yanghuiru/AMER/dataset/CASMEII'  
df = pd.read_excel(os.path.join(raf_path, 'CASME2-coding-20140508.xlsx'), usecols=[0, 1, 3, 4, 5, 7, 8])
df['Subject'] = df['Subject'].apply(str)

# 统计三类和五类
for num_classes in [3, 5]:
    print(f"\n====== {num_classes} 类处理 ======")

    dataset = df.copy()

    Subject = dataset.iloc[:, 0].values
    File_names = dataset.iloc[:, 1].values
    Onset_num = dataset.iloc[:, 2].values
    Apex_num = dataset.iloc[:, 3].values
    Offset_num = dataset.iloc[:, 4].values
    Label_all = dataset.iloc[:, 6].values

    apex_images = []
    flow_features = []

    for (f, sub, onset, apex, offset, label_all) in tqdm(zip(File_names, Subject, Onset_num, Apex_num, Offset_num, Label_all),
                                                         total=len(Label_all)):
        # 类别筛选
        if num_classes == 3:
            valid_classes = ['happiness', 'repression', 'disgust', 'surprise', 'fear', 'sadness']
        else:  # 5类
            valid_classes = ['happiness', 'repression', 'disgust', 'surprise', 'others']

        if label_all not in valid_classes:
            continue

        # 文件名路径拼接
        on0_j = f'reg_img{onset}.jpg'
        apex0_j = f'reg_img{apex}.jpg'

        path_on0 = os.path.join(raf_path, 'Cropped', f'sub{int(sub):02d}', f, on0_j)
        path_apex0 = os.path.join(raf_path, 'Cropped', f'sub{int(sub):02d}', f, apex0_j)

        if not os.path.exists(path_on0) or not os.path.exists(path_apex0):
            print(f'路径缺失: {path_on0} 或 {path_apex0}')
            continue

        # 读取和resize（保持统一）
        image_on0 = cv2.imread(path_on0)
        image_apex0 = cv2.imread(path_apex0)

        if image_on0 is None or image_apex0 is None:
            print(f'图片读取失败: {path_on0} 或 {path_apex0}')
            continue

        image_on0 = cv2.resize(image_on0, (224, 224))
        image_apex0 = cv2.resize(image_apex0, (224, 224))

        # # 保存apex图像做统计,调整维度顺序为CHW
        apex_tensor = image_apex0.transpose(2, 0, 1) / 255.0
        apex_images.append(apex_tensor[np.newaxis, :]) 

        # 计算光流
        flow_feature = calc_os_flow(image_on0, image_apex0)
        if flow_feature.ndim == 3 and flow_feature.shape[-1] == 3:  # 如果是HWC格式
            flow_feature = flow_feature.transpose(2, 0, 1)  # 转为CHW
        flow_feature /= 255.0
        flow_features.append(flow_feature[np.newaxis, :]) 

    # 计算mean/std
    apex_mean, apex_std = get_mean_std(apex_images)
    flow_mean, flow_std = get_mean_std(flow_features)

    print(f'\n分类 {num_classes} Apex 图像 Mean: {apex_mean}, Std: {apex_std}')
    print(f'分类 {num_classes} 光流特征 Mean: {flow_mean}, Std: {flow_std}')


    '''
====== 3 类处理 ======
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [01:09<00:00,  3.66it/s]

分类 3 Apex 图像 Mean: [0.30275449 0.36524517 0.50842999], Std: [0.1190318  0.13523947 0.18652917]
分类 3 光流特征 Mean: [0.4744762  0.5149655  0.02466276], Std: [0.20347348 0.23907797 0.045638  ]

====== 5 类处理 ======
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [01:48<00:00,  2.34it/s]

分类 5 Apex 图像 Mean: [0.2955144  0.35766289 0.50003973], Std: [0.12269197 0.13605498 0.18462566]
分类 5 光流特征 Mean: [0.48156697 0.48047075 0.02433573], Std: [0.19904253 0.2394165  0.04526653]
    '''


    
