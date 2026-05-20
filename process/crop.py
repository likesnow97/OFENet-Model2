import os
import cv2
import pandas as pd
from mtcnn import MTCNN
import re

def rename_samm_files(root_dir):
    """
    重命名SAMM数据集文件，将_后面的数字串左边的0去掉
    例如：006_05562.jpg → 006_5562.jpg
    现在可以匹配 XXX_X_X 和 XXX_X_XX 形式的目录（如013_1_10）
    """
    # 编译正则表达式匹配模式
    file_pattern = re.compile(r'^(\d{3})_0+(\d+)\.jpg$')
    dir_pattern = re.compile(r'^\d{3}_\d_\d+$')  
    
    # 遍历根目录下的所有子文件夹
    for subdir, _, files in os.walk(root_dir):
        # 检查是否匹配SAMM/orig/XXX/XXX_X_XX/的结构
        dir_parts = subdir.split(os.sep)
        if len(dir_parts) >= 4 and dir_pattern.match(dir_parts[-1]):
            for filename in files:
                # 只处理.jpg文件
                if filename.lower().endswith('.jpg'):
                    match = file_pattern.match(filename)
                    if match:
                        # 获取前缀和后缀数字
                        prefix = match.group(1)
                        suffix = match.group(2)
                        
                        # 构建新文件名
                        new_filename = f"{prefix}_{suffix}.jpg"
                        old_path = os.path.join(subdir, filename)
                        new_path = os.path.join(subdir, new_filename)
                        
                        # 重命名文件
                        os.rename(old_path, new_path)
                        print(f"Renamed: {os.path.join(subdir, filename)} → {new_filename}")


# dataset_path = "dataset/SAMM/orig" 
# rename_samm_files(dataset_path)
# print("文件重命名完成！")

# 根目录
root_dir = 'dataset/SAMM'
orig_dir = os.path.join(root_dir, 'orig')
cropped_dir = os.path.join(root_dir, 'Cropped')

# 读取Excel文件
excel_path = 'dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'  # 改为实际路径
df = pd.read_excel(excel_path)

detector = MTCNN()

def crop_face(img):
    # 使用MTCNN检测人脸
    results = detector.detect_faces(img)
    if len(results) == 0:
        return None  # 没检测到人脸
    # 默认取第一个人脸
    box = results[0]['box']  # [x, y, width, height]
    x, y, w, h = box
    # 确保坐标非负且在图像范围内
    x, y = max(0, x), max(0, y)
    cropped = img[y:y+h, x:x+w]
    return cropped

def process_row(row):
    subject = str(row['Subject']).zfill(3)  # 比如006
    filename = row['Filename']  # 例如006_1_2
    onset = row['Onset Frame']
    apex = row['Apex Frame']

    # print(subject,filename,onset,apex)

    # 处理onset和apex帧对应的图片
    for frame_num in [onset, apex]:

        img_name = f"{subject}_{frame_num}.jpg"
        img_path = os.path.join(orig_dir, subject, filename, img_name)

        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            continue

        # 裁剪人脸
        cropped_face = crop_face(img)
        if cropped_face is None:
            print(f"未检测到人脸: {img_path}")
            continue

        # 构建保存路径
        save_dir = os.path.join(cropped_dir, subject, filename)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, img_name)

        # 保存裁剪图片
        cv2.imwrite(save_path, cropped_face)
        print(f"保存裁剪图片: {save_path}")



# # 遍历每一行处理
for idx, row in df.iterrows():
    process_row(row)

# img = cv2.imread('/nfs/users/yanghuiru/AMER/dataset/SAMM/orig/032/032_3_1/032_4930.jpg')
# cropped_face = crop_face(img)
# cv2.imwrite('/nfs/users/yanghuiru/AMER/dataset/SAMM/Cropped/032/032_3_1/032_4930.jpg', cropped_face)