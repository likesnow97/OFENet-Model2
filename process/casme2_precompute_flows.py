# precompute_dual_flows_with_table.py
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
import sys

def calc_farneback_flow(img1, img2):
    """Farneback方法计算光流和光学应变"""
    
    def farneback_ofcalc(img1, img2):
        """Farneback光流计算"""
        # 转换为灰度图
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Farneback光流参数
        flow = cv2.calcOpticalFlowFarneback(
            prev=img1_gray,
            next=img2_gray,
            flow=None,
            pyr_scale=0.5,      # 金字塔缩放比例
            levels=3,           # 金字塔层数
            winsize=15,         # 窗口大小
            iterations=3,       # 迭代次数
            poly_n=5,           # 像素邻域大小
            poly_sigma=1.2,     # 高斯标准差
            flags=0             # 可选标志
        )
        return flow
    
    def minmax_norm(x):
        """最小-最大归一化"""
        x_flat = x.reshape(-1)
        x_max = np.max(x)
        x_min = np.min(x)
        if x_max == x_min:
            x_flat *= 0
        else:
            x_flat = (x_flat - x_min) / (x_max - x_min)
        return x_flat.reshape(x.shape)
    
    # 计算Farneback光流
    flow_uv = farneback_ofcalc(img1, img2)  # shape: (H, W, 2)
    
    # 分离u, v分量
    flow_u = flow_uv[..., 0]
    flow_v = flow_uv[..., 1]
    
    # 归一化
    u_flow = minmax_norm(flow_u) * 255
    v_flow = minmax_norm(flow_v) * 255
    
    # 使用Sobel算子计算梯度
    flow_u_float64 = flow_u.astype(np.float64)
    flow_v_float64 = flow_v.astype(np.float64)
    
    # 计算一阶梯度
    u_x = cv2.Sobel(flow_u_float64, cv2.CV_64F, 1, 0, ksize=3)
    u_y = cv2.Sobel(flow_u_float64, cv2.CV_64F, 0, 1, ksize=3)
    v_x = cv2.Sobel(flow_v_float64, cv2.CV_64F, 1, 0, ksize=3)
    v_y = cv2.Sobel(flow_v_float64, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算光学应变（Strain）
    strain = np.sqrt(u_x ** 2 + v_y ** 2 + 0.5 * (u_y + v_x) ** 2)
    
    # 归一化应变
    os_flow = minmax_norm(strain) * 255
    
    # 拼接为3通道输出
    return np.concatenate((
        u_flow.reshape(*u_flow.shape, 1), 
        v_flow.reshape(*v_flow.shape, 1), 
        os_flow.reshape(*os_flow.shape, 1)
    ), axis=2)


def calc_tvl1_flow(img1, img2):
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
    构造图像路径
    
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


def generate_flow_filename(row, flow_type='farneback'):
    """生成光流文件名"""
    subject = row['Subject']
    filename = row['Filename']
    onset = row['OnsetFrame']
    apex = row['ApexFrame']
    
    # 清理文件名中的特殊字符
    clean_filename = str(filename).replace('/', '_').replace('\\', '_')
    
    # 根据光流类型生成文件名
    if flow_type == 'farneback':
        flow_filename = f"flow_farneback_{subject}_{clean_filename}_{onset}_{apex}.npy"
    elif flow_type == 'tvl1':
        flow_filename = f"flow_tvl1_{subject}_{clean_filename}_{onset}_{apex}.npy"
    else:
        flow_filename = f"flow_{subject}_{clean_filename}_{onset}_{apex}.npy"
    
    return flow_filename


def precompute_dual_flows(csv_path, raf_path, output_dir, overwrite=False):
    """
    主预计算函数 - 同时计算两种光流
    
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
        try:
            df = pd.read_excel(csv_path)
        except Exception as e:
            print(f"读取Excel文件出错: {e}")
            try:
                xl = pd.ExcelFile(csv_path)
                sheet_names = xl.sheet_names
                print(f"可用工作表: {sheet_names}")
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
    
    # 创建子目录分别存放两种光流
    farneback_dir = os.path.join(output_dir, 'farneback_flows')
    tvl1_dir = os.path.join(output_dir, 'tvl1_flows')
    os.makedirs(farneback_dir, exist_ok=True)
    os.makedirs(tvl1_dir, exist_ok=True)
    
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
    
    # 4. 计算并保存两种光流
    print("\n计算两种光流...")
    farneback_paths = []
    tvl1_paths = []
    success_farneback = 0
    success_tvl1 = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="计算光流"):
        path_on = row['OnsetPath']
        path_apex = row['ApexPath']
        
        # 检查路径是否有效
        if not path_on or not path_apex:
            print(f"第{idx}行: 图像路径为空")
            farneback_paths.append('')
            tvl1_paths.append('')
            fail_count += 1
            continue
        
        # 生成两种光流文件名和路径
        farneback_filename = generate_flow_filename(row, flow_type='farneback')
        tvl1_filename = generate_flow_filename(row, flow_type='tvl1')
        
        farneback_path = os.path.join(farneback_dir, farneback_filename)
        tvl1_path = os.path.join(tvl1_dir, tvl1_filename)
        
        # 初始化路径
        current_farneback_path = ''
        current_tvl1_path = ''
        
        try:
            # 读取图像
            img1 = cv2.imread(path_on)
            img2 = cv2.imread(path_apex)
            
            if img1 is None:
                print(f"第{idx}行: 无法读取图像 {path_on}")
                farneback_paths.append('')
                tvl1_paths.append('')
                fail_count += 1
                continue
            
            if img2 is None:
                print(f"第{idx}行: 无法读取图像 {path_apex}")
                farneback_paths.append('')
                tvl1_paths.append('')
                fail_count += 1
                continue
            
            # 调整图像大小
            img1 = cv2.resize(img1, (224, 224))
            img2 = cv2.resize(img2, (224, 224))
            
            # ========== 计算Farneback光流 ==========
            need_farneback = overwrite or not os.path.exists(farneback_path)
            if need_farneback:
                farneback_flow = calc_farneback_flow(img1, img2)
                np.save(farneback_path, farneback_flow)
            
            if os.path.exists(farneback_path):
                current_farneback_path = farneback_path
                success_farneback += 1
            else:
                print(f"第{idx}行: 保存Farneback光流失败 {farneback_path}")
            
            # ========== 计算TV-L1光流 ==========
            need_tvl1 = overwrite or not os.path.exists(tvl1_path)
            if need_tvl1:
                tvl1_flow = calc_tvl1_flow(img1, img2)
                np.save(tvl1_path, tvl1_flow)
            
            if os.path.exists(tvl1_path):
                current_tvl1_path = tvl1_path
                success_tvl1 += 1
            else:
                print(f"第{idx}行: 保存TV-L1光流失败 {tvl1_path}")
            
            # 添加到列表
            farneback_paths.append(current_farneback_path)
            tvl1_paths.append(current_tvl1_path)
            
            # 如果两个都失败，则增加失败计数
            if not current_farneback_path and not current_tvl1_path:
                fail_count += 1
                
        except Exception as e:
            print(f"第{idx}行: 计算光流出错 {e}")
            farneback_paths.append('')
            tvl1_paths.append('')
            fail_count += 1
        
        # 每50个样本清理一次内存
        if idx % 50 == 0:
            import gc
            gc.collect()
    
    # 5. 添加光流路径列
    df['FlowPath_Farneback'] = farneback_paths
    df['FlowPath_TVL1'] = tvl1_paths
    
    # 6. 保存新的数据表
    output_csv = os.path.join(output_dir, 'dataset_with_dual_flows.csv')
    df.to_csv(output_csv, index=False)
    
    # 7. 生成统计报告
    print(f"\n{'='*60}")
    print("双光流预计算完成！")
    print(f"{'='*60}")
    print(f"总样本数: {len(df)}")
    print(f"成功计算Farneback光流: {success_farneback}")
    print(f"成功计算TV-L1光流: {success_tvl1}")
    print(f"完全失败的样本: {fail_count}")
    print(f"光流文件目录: {output_dir}")
    print(f"  - Farneback光流: {farneback_dir}")
    print(f"  - TV-L1光流: {tvl1_dir}")
    print(f"新数据表: {output_csv}")
    
    # 分析结果
    print(f"\n结果分析:")
    valid_farneback = df[df['FlowPath_Farneback'] != '']
    valid_tvl1 = df[df['FlowPath_TVL1'] != '']
    valid_both = df[(df['FlowPath_Farneback'] != '') & (df['FlowPath_TVL1'] != '')]
    
    print(f"有效的Farneback光流: {len(valid_farneback)} ({len(valid_farneback)/len(df)*100:.1f}%)")
    print(f"有效的TV-L1光流: {len(valid_tvl1)} ({len(valid_tvl1)/len(df)*100:.1f}%)")
    print(f"两种光流都有效的: {len(valid_both)} ({len(valid_both)/len(df)*100:.1f}%)")
    
    # 打印失败样本（如果有）
    failed_both = df[(df['FlowPath_Farneback'] == '') & (df['FlowPath_TVL1'] == '')]
    if len(failed_both) > 0:
        print(f"\n两种光流都失败的样本 ({len(failed_both)}个):")
        for idx, row in failed_both.head(10).iterrows():
            print(f"  第{idx}行: Subject={row['Subject']}, File={row['Filename']}")
    
    return output_csv


def verify_dual_flows(csv_path, sample_check=3):
    """验证双光流预计算结果"""
    print(f"\n验证双光流预计算结果...")
    df = pd.read_csv(csv_path)
    
    # 检查列
    required_columns = ['OnsetPath', 'ApexPath', 'FlowPath_Farneback', 'FlowPath_TVL1']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺失列 {missing_columns}")
        return False
    
    # 随机检查几个样本
    import random
    # 选择两种光流都有的样本
    valid_indices = df[(df['FlowPath_Farneback'] != '') & (df['FlowPath_TVL1'] != '')].index.tolist()
    
    if len(valid_indices) == 0:
        print("错误: 没有同时有两种光流的样本")
        return False
    
    test_indices = random.sample(valid_indices, min(sample_check, len(valid_indices)))
    
    print(f"随机检查 {len(test_indices)} 个样本:")
    
    for idx in test_indices:
        row = df.iloc[idx]
        
        print(f"\n样本{idx}: {row['Subject']}/{row['Filename']}")
        print(f"  Onset图像: {row['OnsetPath']}")
        print(f"  Apex图像: {row['ApexPath']}")
        
        # 检查Farneback光流
        try:
            farneback_flow = np.load(row['FlowPath_Farneback'])
            print(f"  Farneback光流: ✓")
            print(f"    形状: {farneback_flow.shape}, 数据类型: {farneback_flow.dtype}")
            print(f"    值范围: [{farneback_flow.min():.2f}, {farneback_flow.max():.2f}]")
        except Exception as e:
            print(f"  Farneback光流: ✗ 加载失败 {e}")
        
        # 检查TV-L1光流
        try:
            tvl1_flow = np.load(row['FlowPath_TVL1'])
            print(f"  TV-L1光流: ✓")
            print(f"    形状: {tvl1_flow.shape}, 数据类型: {tvl1_flow.dtype}")
            print(f"    值范围: [{tvl1_flow.min():.2f}, {tvl1_flow.max():.2f}]")
        except Exception as e:
            print(f"  TV-L1光流: ✗ 加载失败 {e}")
    
    return True


def create_merged_dataset_from_existing(farneback_csv, tvl1_csv, output_csv):
    """
    从现有的两个单独CSV创建合并的CSV
    适用于已经分别计算好两种光流的情况
    """
    print(f"从现有CSV创建合并数据集...")
    print(f"Farneback CSV: {farneback_csv}")
    print(f"TV-L1 CSV: {tvl1_csv}")
    
    # 读取两个CSV
    df_farneback = pd.read_csv(farneback_csv)
    df_tvl1 = pd.read_csv(tvl1_csv)
    
    print(f"Farneback数据: {len(df_farneback)} 行")
    print(f"TV-L1数据: {len(df_tvl1)} 行")
    
    # 确保都有需要的列
    if 'FlowPath' not in df_farneback.columns:
        print("错误: Farneback CSV缺少'FlowPath'列")
        return None
    
    if 'FlowPath' not in df_tvl1.columns:
        print("错误: TV-L1 CSV缺少'FlowPath'列")
        return None
    
    # 重命名FlowPath列
    df_farneback = df_farneback.rename(columns={'FlowPath': 'FlowPath_Farneback'})
    df_tvl1 = df_tvl1.rename(columns={'FlowPath': 'FlowPath_TVL1'})
    
    # 确定合并键 - 假设两个CSV有相同的结构和顺序
    merge_keys = ['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame']
    
    # 检查哪些列是共同的
    common_columns = list(set(df_farneback.columns) & set(df_tvl1.columns))
    print(f"共同列: {common_columns}")
    
    # 尝试合并
    df_merged = pd.merge(
        df_farneback,
        df_tvl1[['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'FlowPath_TVL1']],
        on=['Subject', 'Filename', 'OnsetFrame', 'ApexFrame'],
        how='inner'
    )
    
    print(f"合并后数据: {len(df_merged)} 行")
    
    # 保存合并后的CSV
    df_merged.to_csv(output_csv, index=False)
    print(f"合并CSV已保存到: {output_csv}")
    
    return output_csv


if __name__ == '__main__':
    
    # ==================== 配置参数 ====================
    # 1. 原始CSV文件路径
    CSV_PATH = '/nfs/users/yanghuiru/AMER/dataset/CASMEII/CASME2-coding-20140508.xlsx'
    
    # 2. 数据集根路径
    RAF_PATH = '/nfs/users/yanghuiru/AMER/dataset'
    
    # 3. 输出目录
    OUTPUT_DIR = '/nfs/users/yanghuiru/AMER/dataset/CASMEII/precomputed_dual_flows'
    
    # 4. 是否覆盖已存在的光流文件
    OVERWRITE = True   # True: 重新计算所有, False: 跳过已存在的
    
    # 5. 是否验证结果
    VERIFY = True
    
    # ==================== 开始执行 ====================
    
    print(f"{'='*60}")
    print("双光流预计算脚本")
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
    
    print("\n开始计算两种光流...")
    new_csv = precompute_dual_flows(CSV_PATH, RAF_PATH, OUTPUT_DIR, OVERWRITE)
    
    # 验证结果
    if VERIFY and new_csv:
        verify_dual_flows(new_csv)
    
    print(f"\n{'='*60}")
    print("所有任务完成！")
    print(f"{'='*60}")
        
