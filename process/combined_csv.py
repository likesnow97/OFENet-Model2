import pandas as pd

# 1. 读取 combined 表
combined_df = pd.read_csv('combined_3class_gt.csv',header=None, names=['Dataset','Subject','Filename','Class'])

# 2. 读取三个参考表
casme2_df = pd.read_excel('CASMEII/CASME2-coding-20140508.xlsx',usecols=[0, 1, 3, 4, 5])
smic_df = pd.read_csv('SMIC/smic_coding.csv',usecols=[0, 1, 3, 4, 5])
samm_df = pd.read_excel('SAMM/SAMM_Micro_FACS_Codes_v2.xlsx',usecols=[0, 1, 3, 4, 5])

# 3. 标准化列名
casme2_df.columns = samm_df.columns = smic_df.columns = ['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame']
casme2_df['Subject'] = casme2_df['Subject'].apply(lambda x: f"sub{int(x):02d}")
samm_df['Subject'] = samm_df['Subject'].apply(lambda x: str(int(x)))  # 006 -> 6
smic_df['Subject'] = smic_df['Subject'].apply(lambda x: f"s{int(x[1:]):02d}")  # s1 -> s01
smic_df['Filename'] = smic_df['Filename'].apply(
    lambda x: x.replace(x.split('_')[0], f"s{int(x.split('_')[0][1:]):02d}")
)

# 4. 初始化新列
combined_df['OnsetFrame'] = None
combined_df['ApexFrame'] = None
combined_df['OffsetFrame'] = None

# 5. 遍历合并表，填充帧信息
for idx, row in combined_df.iterrows():
    dataset = row['Dataset']
    subject = str(row['Subject']).strip()
    filename = str(row['Filename']).strip()
    
    if dataset == 'casme2':
        match = casme2_df[(casme2_df['Subject'].astype(str).str.strip() == subject) &
                          (casme2_df['Filename'].astype(str).str.strip() == filename)]
    elif dataset == 'smic':
        match = smic_df[(smic_df['Subject'].astype(str).str.strip() == subject) &
                        (smic_df['Filename'].astype(str).str.strip() == filename)]
    elif dataset == 'samm':
        match = samm_df[(samm_df['Subject'].astype(str).str.strip() == subject) &
                        (samm_df['Filename'].astype(str).str.strip() == filename)]
    else:
        match = pd.DataFrame()

    if not match.empty:
        combined_df.at[idx, 'OnsetFrame'] = match.iloc[0]['OnsetFrame']
        combined_df.at[idx, 'ApexFrame'] = match.iloc[0]['ApexFrame']
        combined_df.at[idx, 'OffsetFrame'] = match.iloc[0]['OffsetFrame']

# 6. 保存新文件
combined_df.to_csv('combined_3class_with_frames.csv', index=False)
print("已保存含帧信息的新表格：combined_3class_with_frames.csv")
