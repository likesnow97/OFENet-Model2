# precompute_flows_with_table.py
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
import sys

def calc_os_flow(img1, img2):
    """TV-L1光流计算函数"""
    
    def tvl1_ofcalc(img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.optflow.DualTVL1OpticalFlow_create()
        of = flow.calc(img1, img2, None)
        return of
    
    def minmax_norm(x):
        x_flat = x.reshape(-1)
        x_max = np.max(x)
        x_min = np.min(x)
        if x_max == x_min:
            x_flat *= 0
        else:
            x_flat = (x_flat - x_min) / (x_max - x_min)
        return x_flat.reshape(x.shape)
    
    flow = tvl1_ofcalc(img1, img2)
    u_flow = minmax_norm(flow[:, :, 0]) * 255
    v_flow = minmax_norm(flow[:, :, 1]) * 255
    
    ux, uy = np.gradient(flow[:, :, 0])
    vx, vy = np.gradient(flow[:, :, 1])
    
    os_flow = np.sqrt(ux ** 2 + vy ** 2 + 0.25 * (uy + vx) ** 2)
    os_flow = minmax_norm(os_flow) * 255
    
    return np.concatenate((u_flow.reshape(*u_flow.shape, 1), 
                          v_flow.reshape(*v_flow.shape, 1), 
                          os_flow.reshape(*os_flow.shape, 1)), axis=2)

def construct_image_path(row, raf_path):
    """
    
    Args:
        row: DataFrame的一行数据
        raf_path: 数据集根路径
    
    Returns:
        path_on: onset图像路径
        path_apex: apex图像路径
    """
    subject = int(row['Subject'])
    filename = row['Filename']
    onset = str(row['OnsetFrame'])
    apex = str(row['ApexFrame'])
    offset = str(row['OffsetFrame'])
    
    # 构造文件名
    on0_j = 'reg_img' + onset + '.jpg'
    apex0_j = 'reg_img' + apex + '.jpg'
    
    # 构造路径
    path_on = os.path.join(raf_path, 'CASMEII', 'Cropped', 
                            'sub%02d' % int(subject), filename, on0_j)
    path_apex = os.path.join(raf_path, 'CASMEII', 'Cropped', 
                            'sub%02d' % int(subject), filename, apex0_j)
        
    
    return path_on, path_apex

def generate_flow_filename(row):
    """生成光流文件名"""

    subject = row['Subject']
    filename = row['Filename']
    onset = row['OnsetFrame']
    apex = row['ApexFrame']
    
    # 清理文件名中的特殊字符
    clean_filename = str(filename).replace('/', '_').replace('\\', '_')
    
    # 生成唯一文件名
    flow_filename = f"flow_casme2_{subject}_{clean_filename}_{onset}_{apex}.npy"
    
    return flow_filename

def precompute_flows(csv_path, raf_path, output_dir, overwrite=False):
    """
    主预计算函数
    
    Args:
        csv_path: 原始CSV文件路径
        raf_path: 数据集根路径
        output_dir: 输出目录
        overwrite: 是否覆盖已存在的光流文件
    """
    
    # 1. 读取数据表
    print(f"读取数据表: {csv_path}")
    file_ext = os.path.splitext(csv_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(csv_path)
    elif file_ext == '.xlsx' or file_ext == '.xls':
        # 读取Excel文件
        try:
            # 尝试读取所有工作表，或者指定工作表
            df = pd.read_excel(csv_path)
        except Exception as e:
            print(f"读取Excel文件出错: {e}")
            # 尝试指定工作表名称
            try:
                # 获取工作表名称
                xl = pd.ExcelFile(csv_path)
                sheet_names = xl.sheet_names
                print(f"可用工作表: {sheet_names}")
                
                # 尝试第一个工作表
                df = pd.read_excel(csv_path, sheet_name=sheet_names[0])
                print(f"使用工作表: {sheet_names[0]}")
            except Exception as e2:
                print(f"无法读取Excel文件: {e2}")
                return None
    else:
        print(f"不支持的文件格式: {file_ext}，支持.csv和.xlsx/.xls")
        return None
    
    print(f"总样本数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 添加图像路径列
    print("构造图像路径...")
    image_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="构造路径"):
        try:
            path_on, path_apex = construct_image_path(row, raf_path)
            image_paths.append({
                'path_on': path_on,
                'path_apex': path_apex
            })
        except Exception as e:
            print(f"第{idx}行构造路径失败: {e}")
            image_paths.append({
                'path_on': '',
                'path_apex': ''
            })
    
    # 添加到DataFrame
    df['OnsetPath'] = [p['path_on'] for p in image_paths]
    df['ApexPath'] = [p['path_apex'] for p in image_paths]
    
    # 4. 计算并保存光流
    print("\n计算光流...")
    flow_paths = []
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算光流"):
        path_on = row['OnsetPath']
        path_apex = row['ApexPath']
        
        # 检查路径是否有效
        if not path_on or not path_apex:
            print(f"第{idx}行: 图像路径为空")
            flow_paths.append('')
            fail_count += 1
            continue
        
        # 生成光流文件名和路径
        flow_filename = generate_flow_filename(row)
        flow_path = os.path.join(output_dir, flow_filename)
        
        # 检查是否已存在
        if os.path.exists(flow_path) and not overwrite:
            # print(f"第{idx}行: 光流文件已存在 {flow_path}")
            flow_paths.append(flow_path)
            success_count += 1
            continue
        
        try:
            # 读取图像
            img1 = cv2.imread(path_on)
            img2 = cv2.imread(path_apex)
            
            if img1 is None:
                print(f"第{idx}行: 无法读取图像 {path_on}")
                flow_paths.append('')
                fail_count += 1
                continue
            
            if img2 is None:
                print(f"第{idx}行: 无法读取图像 {path_apex}")
                flow_paths.append('')
                fail_count += 1
                continue
            
            # 调整图像大小到28×28
            img1 = cv2.resize(img1, (224, 224))
            img2 = cv2.resize(img2, (224, 224))
            
            # 计算光流
            flow = calc_os_flow(img1, img2)
            
            # 保存光流文件
            np.save(flow_path, flow)
            
            # 验证保存的文件
            if os.path.exists(flow_path):
                flow_paths.append(flow_path)
                success_count += 1
            else:
                print(f"第{idx}行: 保存光流文件失败 {flow_path}")
                flow_paths.append('')
                fail_count += 1
                
        except Exception as e:
            print(f"第{idx}行: 计算光流出错 {e}")
            flow_paths.append('')
            fail_count += 1
        
        # 每50个样本清理一次内存
        if idx % 50 == 0:
            import gc
            gc.collect()
    
    # 5. 添加光流路径列
    df['FlowPath'] = flow_paths
    
    # 6. 保存新的数据表
    output_csv = os.path.join(output_dir, 'dataset_with_flows.csv')
    df.to_csv(output_csv, index=False)
    
    # 7. 生成统计报告
    print(f"\n{'='*60}")
    print("预计算完成！")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")
    print(f"成功计算: {success_count}")
    print(f"失败: {fail_count}")
    print(f"光流文件目录: {output_dir}")
    print(f"新数据表: {output_csv}")
    
    # 打印失败样本（如果有）
    if fail_count > 0:
        print(f"\n失败样本:")
        failed_df = df[df['FlowPath'] == '']
        for idx, row in failed_df.head(10).iterrows():  # 只显示前10个
            print(f"  第{idx}行: {row['Dataset']}/{row['Subject']}/{row['Filename']}")
    
    return output_csv

def verify_precomputation(csv_path, sample_check=5):
    """验证预计算结果"""
    print(f"\n验证预计算结果...")
    df = pd.read_csv(csv_path)
    
    # 检查列
    required_columns = ['OnsetPath', 'ApexPath', 'FlowPath']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺失列 {missing_columns}")
        return False
    
    # 随机检查几个样本
    import random
    valid_indices = df[df['FlowPath'] != ''].index.tolist()
    
    if len(valid_indices) == 0:
        print("错误: 没有有效的光流路径")
        return False
    
    test_indices = random.sample(valid_indices, min(sample_check, len(valid_indices)))
    
    print(f"随机检查 {len(test_indices)} 个样本:")
    
    for idx in test_indices:
        row = df.iloc[idx]
        flow_path = row['FlowPath']
        
        try:
            flow = np.load(flow_path)
            print(f"  样本{idx}: 光流文件 {flow_path}")
            print(f"    形状: {flow.shape}, 数据类型: {flow.dtype}")
            print(f"    值范围: [{flow.min():.2f}, {flow.max():.2f}]")
            
            # 检查图像文件是否存在
            if os.path.exists(row['OnsetPath']):
                print(f"    Onset图像: ✓")
            else:
                print(f"    Onset图像: ✗ 不存在")
                
            if os.path.exists(row['ApexPath']):
                print(f"    Apex图像: ✓")
            else:
                print(f"    Apex图像: ✗ 不存在")
                
        except Exception as e:
            print(f"  样本{idx}: 加载失败 {e}")
    
    return True

if __name__ == '__main__':
    
    # 1. 原始CSV文件路径
    CSV_PATH = '/nfs/users/yanghuiru/AMER/dataset/CASMEII/CASME2-coding-20140508.xlsx'
    
    # 2. 数据集根路径
    RAF_PATH = '/nfs/users/yanghuiru/AMER/dataset'
    
    # 3. 输出目录
    OUTPUT_DIR = '/nfs/users/yanghuiru/AMER/dataset/CASMEII/precomputed_flows_tvl1'
    
    # 4. 是否覆盖已存在的光流文件
    OVERWRITE = True   # True: 重新计算所有, False: 跳过已存在的
    
    # # 5. 是否验证结果
    # VERIFY = True
    
    # ==================== 开始执行 ====================
    
    print(f"{'='*60}")
    print("光流预计算脚本")
    print(f"{'='*60}")
    print(f"CSV文件: {CSV_PATH}")
    print(f"数据集根路径: {RAF_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"覆盖模式: {'是' if OVERWRITE else '否'}")
    print(f"{'='*60}")
    
    # 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"错误: CSV文件不存在 {CSV_PATH}")
        sys.exit(1)
    
    if not os.path.exists(RAF_PATH):
        print(f"错误: 数据集根路径不存在 {RAF_PATH}")
        sys.exit(1)
    
    # 执行预计算
    try:
        new_csv = precompute_flows(CSV_PATH, RAF_PATH, OUTPUT_DIR, OVERWRITE)
        
        # # 验证结果
        # if VERIFY:
        #     verify_precomputation(new_csv)
            
        # print(f"\n{'='*60}")
        # print("所有任务完成！")
        # print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()