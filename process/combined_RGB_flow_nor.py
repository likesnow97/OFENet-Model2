import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from preprocess import calc_os_flow

def get_mean_std(all_data):
    # all_data: N x C x H x W 的 ndarray
    all_data = np.concatenate(all_data, axis=0)  # 合并所有样本 (N, C, H, W)
    # 对样本数量 N、图片的高 H 和 宽 W 这三个维度求平均
    mean = np.mean(all_data, axis=(0, 2, 3))  # 通道维度
    std = np.std(all_data, axis=(0, 2, 3))

    return mean, std

def process_combined_dataset(raf_path, num_classes=3):
    """
    处理组合数据集（CASMEII + SAMM + SMIC）
    """
    print(f"\n====== 组合数据集 {num_classes} 类处理 ======")
    
    # 读取CSV文件
    csv_path = os.path.join(raf_path, 'combined_3class_with_frames.csv')
    df = pd.read_csv(csv_path)
    df.columns = ['Dataset', 'Subject', 'Filename', 'Label', 'OnsetFrame', 'ApexFrame', 'OffsetFrame']
    
    # 类别映射
    if num_classes == 3:
        # 三类：0-negative, 1-positive, 2-surprise
        valid_labels = [0, 1, 2]
    else:
        # 五类或其他分类
        print("暂不支持5类处理，仅支持3类")
        return
    
    # 筛选有效类别
    dataset = df[df['Label'].isin(valid_labels)].copy()
    
    Database_name = dataset['Dataset'].values
    Label_all = dataset['Label'].values
    Subject = dataset['Subject'].values
    File_names = dataset['Filename'].values
    Onset_num = dataset['OnsetFrame'].values
    Apex_num = dataset['ApexFrame'].values
    Offset_num = dataset['OffsetFrame'].values
    
    apex_images = []
    flow_features = []
    
    # 统计各类别数量
    count_dict = {0: 0, 1: 0, 2: 0}
    
    for (f, sub, db_name, onset, apex, offset, label_all) in tqdm(
        zip(File_names, Subject, Database_name, Onset_num, Apex_num, Offset_num, Label_all),
        total=len(Label_all),
        desc="处理组合数据集"
    ):
        # 统计类别数量
        count_dict[label_all] += 1
        
        # 根据数据集类型构建路径
        on0 = str(onset)
        apex0 = str(apex)
        sub_str = str(sub)
        
        # CASMEII
        if db_name == 'casme2':
            on0_j = 'reg_img' + on0 + '.jpg'
            apex0_j = 'reg_img' + apex0 + '.jpg'
            path_on = os.path.join(raf_path, 'CASMEII', 'Cropped', sub_str, f, on0_j)
            path_apex = os.path.join(raf_path, 'CASMEII', 'Cropped', sub_str, f, apex0_j)
        
        # SAMM
        elif db_name == "samm":
            sub_str = sub_str.zfill(3)
            on0_j = f"{sub_str}_{on0}.jpg"
            apex0_j = f"{sub_str}_{apex}.jpg"
            path_on = os.path.join(raf_path, 'SAMM', 'Cropped', sub_str, f, on0_j)
            path_apex = os.path.join(raf_path, 'SAMM', 'Cropped', sub_str, f, apex0_j)
        
        # SMIC
        elif db_name == 'smic':
            on0 = on0.zfill(6)
            apex0 = apex0.zfill(6)
            on0_j = 'reg_image' + on0 + '.bmp'
            apex0_j = 'reg_image' + apex0 + '.bmp'
            sub_str = 's' + str(int(sub_str[1:])) if isinstance(sub_str, str) and sub_str.startswith('s') else 's' + sub_str
            f_name = sub_str + f[3:] if isinstance(f, str) and len(f) > 3 else f
            
            # 确定文件标签
            if 'ne' in f_name.lower():
                file_label = 'negative'
            elif 'po' in f_name.lower():
                file_label = 'positive'
            else:
                file_label = 'surprise'
            
            path_on = os.path.join(raf_path, 'SMIC', 'HS', sub_str, 'micro', file_label, f_name, on0_j)
            path_apex = os.path.join(raf_path, 'SMIC', 'HS', sub_str, 'micro', file_label, f_name, apex0_j)
        
        else:
            print(f"未知数据集: {db_name}")
            continue
        
        # 检查文件是否存在
        if not os.path.exists(path_on) or not os.path.exists(path_apex):
            # print(f'路径缺失: {path_on} 或 {path_apex}')
            continue
        
        # 读取图像
        try:
            image_on0 = cv2.imread(path_on)
            image_apex0 = cv2.imread(path_apex)
            
            if image_on0 is None or image_apex0 is None:
                # print(f'图片读取失败: {path_on} 或 {path_apex}')
                continue
            
            # 统一resize到224x224
            # image_on0 = cv2.resize(image_on0, (28, 28))
            # image_apex0 = cv2.resize(image_apex0, (28, 28))
            image_on0 = cv2.resize(image_on0, (224, 224))
            image_apex0 = cv2.resize(image_apex0, (224, 224))
            
            # 保存apex图像做统计，调整维度顺序为CHW
            apex_tensor = image_apex0.transpose(2, 0, 1) / 255.0
            apex_images.append(apex_tensor[np.newaxis, :])
            
            # # 计算光流
            # flow_feature = calc_os_flow(image_on0, image_apex0)
            # if flow_feature.ndim == 3 and flow_feature.shape[-1] == 3:  # 如果是HWC格式
            #     flow_feature = flow_feature.transpose(2, 0, 1)  # 转为CHW
            # flow_feature /= 255.0
            # flow_features.append(flow_feature[np.newaxis, :])
            
        except Exception as e:
            print(f"处理文件失败 {path_apex}: {e}")
            continue
    
    print(f"\n类别统计: negative={count_dict[0]}, positive={count_dict[1]}, surprise={count_dict[2]}")
    
    # 计算mean/std
    if apex_images:
        apex_mean, apex_std = get_mean_std(apex_images)
        print(f'Apex 图像 Mean: {apex_mean}, Std: {apex_std}')
    else:
        print("警告: 没有找到有效的Apex图像")
    
    # if flow_features:
    #     flow_mean, flow_std = get_mean_std(flow_features)
    #     print(f'光流特征 Mean: {flow_mean}, Std: {flow_std}')
    # else:
    #     print("警告: 没有找到有效的光流特征")
    
    # return apex_mean, apex_std, flow_mean, flow_std
    return apex_mean, apex_std

def main():
    # 设置数据集路径
    raf_path = '/nfs/users/yanghuiru/AMER/dataset'  # 修改为您的组合数据集根目录
    
    # 处理组合数据集
    print("\n=== 处理组合数据集 (CASMEII + SAMM + SMIC) ===")
    
    # 检查文件是否存在
    csv_path = os.path.join(raf_path, 'combined_3class_with_frames.csv')
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 处理3类数据集
    process_combined_dataset(raf_path, num_classes=3)

if __name__ == "__main__":
    main()

# 224*224 光流堆叠：os v u
# 类别统计: negative=250, positive=109, surprise=83
# Apex 图像 Mean: [0.39540449 0.38597813 0.47191015], Std: [0.18725356 0.1817951  0.19961848]
# 光流特征 Mean: [0.01992355 0.51963115 0.49643812], Std: [0.04085233 0.22144313 0.20650046]

# 28*28 光流堆叠：u v os
# 类别统计: negative=250, positive=109, surprise=83
# Apex 图像 Mean: [0.39558027 0.38615964 0.47214476], Std: [0.18718252 0.18171958 0.19955507]
# 光流特征 Mean: [0.50430346 0.47149783 0.06291154], Std: [0.24467677 0.26548716 0.10847765]

# 224*224
# Apex 图像 Mean: [0.39540449 0.38597813 0.47191015], Std: [0.18725356 0.1817951  0.19961848]
