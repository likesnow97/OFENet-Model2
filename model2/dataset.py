import torch
from torchvision import transforms
import os
from os.path import join
import glob
import cv2
import numpy as np
import pandas as pd
from math import floor

from PIL import Image
import torch.utils.data as data
from process.preprocess import *
from pathlib import Path
try:
    from process.facial_matrix import PrecomputedMatrixLoader
except ModuleNotFoundError as e:
    if e.name != "dlib":
        raise
    PrecomputedMatrixLoader = None



class RafDataSet(data.Dataset):
    def __init__(self,raf_path, phase, num_loso, transform_flow=None,transform_apex=None,transform_aug=None,num_classes=5,db='smic',combined=False):
        self.phase = phase
        self.raf_path = raf_path
        self.transform_flow = transform_flow
        self.transform_apex=transform_apex
        self.transform_aug=transform_aug
        self.db = db

        self.db_name=[]
        self.file_paths_onset = []
        self.file_paths_apex = []
        self.file_paths_offset=[]
        self.file_names = []
        self.label= []
        self.sub = []
        # self.label_au=[]
        self.label_db=[]
        self.onset_paths=[]
        self.apex_paths=[]
        self.farneback_flow_paths=[]
        self.tvl1_flow_paths=[]


        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
       
        if combined:
            df = pd.read_csv(os.path.join(self.raf_path, '3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv'))
            # df.columns = ['Dataset','Subject','Filename','Label','OnsetFrame','ApexFrame','OffsetFrame','OnsetPath','ApexPath','FlowPath_Farneback','FlowPath_TVL1']
            df.columns=['Dataset','Subject','Filename','Label','OnsetFrame','ApexFrame','OffsetFrame','OnsetPath','ApexPath','FlowPath']

            # Engineering compatibility fix:
            # single-dataset SAMM CSV may be read with integer Subject values,
            # while the training entry passes LOSO keys as strings.
            # This preserves the original LOSO semantics while avoiding dtype mismatch.
            df['Subject'] = df['Subject'].astype(str)
            num_loso = str(num_loso)

            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
            else:
                dataset = df.loc[df['Subject'] == num_loso]      

            Label_all = dataset['Label'].values
            Onset_path = dataset['OnsetPath'].values
            Apex_path = dataset['ApexPath'].values
            Flow_path_tvl1=dataset['FlowPath'].values          



            for (label_all,onset_path,apex_path,flow_path_tvl1) in zip(Label_all,Onset_path,Apex_path,Flow_path_tvl1):
                #three classes
                if num_classes == 3:

                    self.onset_paths.append(onset_path)
                    self.apex_paths.append(apex_path)
                    self.tvl1_flow_paths.append(flow_path_tvl1)                
                    

                    # (‘0‘ for negative, ‘1‘ for positive, ‘2‘ for surprise).
                    if label_all == 0:
                        self.label.append(0)
                        a=a+1
                    elif label_all == 1:
                        self.label.append(1)
                        b=b+1
                    else:
                        self.label.append(2)
                        c=c+1

                elif num_classes == 5:
                    # five classes, consistent with the original single-dataset 5-class mapping:
                    # happiness -> 0, surprise -> 1, disgust -> 2, repression -> 3, others -> 4.
                    label_key = str(label_all).strip().lower()
                    label_map = {
                        '0': 0, 'happiness': 0,
                        '1': 1, 'surprise': 1,
                        '2': 2, 'disgust': 2,
                        '3': 3, 'repression': 3,
                        '4': 4, 'others': 4,
                    }

                    if label_key in label_map:
                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.tvl1_flow_paths.append(flow_path_tvl1)
                        self.label.append(label_map[label_key])

                        if label_map[label_key] == 0:
                            a = a + 1
                        elif label_map[label_key] == 1:
                            b = b + 1
                        elif label_map[label_key] == 2:
                            c = c + 1
                        elif label_map[label_key] == 3:
                            d = d + 1
                        elif label_map[label_key] == 4:
                            e = e + 1

                else:
                    print('wrong')

        
        
        elif self.db == 'casme2':
            df = pd.read_csv(os.path.join(self.raf_path, 'CASMEII','precomputed_flows_tvl1/dataset_with_flows.csv'), usecols=[0, 1, 3, 4, 5, 7, 8, 9, 10, 11])
            df.columns = ['Subject', 'Filename', 'OnsetFrame', 'ApexFrame', 'OffsetFrame', 'Label_AU', 'Label','OnsetPath','ApexPath', 'FlowPath']
            df['Subject'] = df['Subject'].apply(str)
            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
            else:
                dataset = df.loc[df['Subject'] == num_loso]

            Label_all = dataset['Label'].values
            # Onset_num = dataset['OnsetFrame'].values
            # Apex_num = dataset['ApexFrame'].values
            # Offset_num = dataset['OffsetFrame'].values

            Subject = dataset['Subject'].values
            File_names = dataset['Filename'].values
            Onset_path = dataset['OnsetPath'].values
            Apex_path = dataset['ApexPath'].values
            # Flow_path = dataset['FlowPath_Farneback'].values
            # Flow_path_farneback = dataset['FlowPath_Farneback'].values  # Farneback光流路径
            Flow_path_tvl1 = dataset['FlowPath'].values 




            # for (f, sub, onset, apex, offset, label_all) in zip(File_names,Subject, Onset_num, Apex_num,Offset_num, Label_all):
            for (f, sub, label_all,onset_path,apex_path,flow_path_tvl1) in zip(File_names,Subject, Label_all,Onset_path,Apex_path,Flow_path_tvl1):
            #     #three classes
                if num_classes == 3:
                    if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'fear' or label_all == 'sadness' :#or label_all=='others':

                        # self.file_paths_onset.append(onset)
                        # self.file_paths_offset.append(offset)
                        # self.file_paths_apex.append(apex)
                        # self.sub.append(sub)
                        # self.file_names.append(f)

                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.tvl1_flow_paths.append(flow_path_tvl1)

                        
                        if label_all == 'happiness':
                            self.label.append(0)
                            a=a+1
                        elif label_all == 'surprise':
                            self.label.append(1)
                            b=b+1
                        else:
                            self.label.append(2)
                            c=c+1

                elif num_classes==5:
                #five classes
                    if label_all == 'happiness' or label_all == 'repression' or label_all == 'disgust' or label_all == 'surprise' or label_all == 'others':

                        # self.file_paths_onset.append(onset)
                        # self.file_paths_offset.append(offset)
                        # self.file_paths_apex.append(apex)
                        # self.sub.append(sub)
                        # self.file_names.append(f)

                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.tvl1_flow_paths.append(flow_path_tvl1)


                        if label_all == 'happiness':
                            self.label.append(0)

                            a = a + 1
                        elif label_all == 'repression':
                            self.label.append(1)

                            b = b + 1
                        elif label_all == 'disgust':
                            self.label.append(2)

                            c = c + 1
                        elif label_all == 'surprise':
                            self.label.append(3)

                            d = d + 1
                        else:
                            self.label.append(4)
                            e = e + 1

                else:
                    print('wrong')


        elif self.db == 'samm':
            
            df = pd.read_csv(os.path.join(self.raf_path, "SAMM","precomputed_flows/dataset_with_flows.csv"), usecols=[0, 1, 6,7,8,9])
            df.columns = ['Subject', 'Filename', 'Label','OnsetPath','ApexPath','FlowPath']

            df['Subject'] = df['Subject'].astype(str)
            num_loso = str(num_loso)

            if self.phase == 'train':
                dataset = df.loc[df['Subject'] != num_loso]
                # print('train_dataset:',len(dataset))
            else:
                dataset = df.loc[df['Subject'] == num_loso] 
                # print('test_dataset:',len(dataset))   


            Label_all = dataset['Label'].values
            Subject = dataset['Subject'].values
            File_names = dataset['Filename'].values
            Onset_path = dataset['OnsetPath'].values
            Apex_path = dataset['ApexPath'].values
            # Flow_path = dataset['FlowPath_Farneback'].values
            # Flow_path_farneback = dataset['FlowPath_Farneback'].values  # Farneback光流路径
            Flow_path_tvl1 = dataset['FlowPath'].values 

            for (f, sub, onset_path, apex_path, flow_path_tvl1, label_all) in zip(File_names,Subject, Onset_path, Apex_path, Flow_path_tvl1, Label_all):
                # Anger,Sadness,Surprise,Fear,Other,Happiness,Disgust,Contempt               
                #three classes
                if num_classes == 3:
                    if label_all == 'Happiness' or label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Surprise':
                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.tvl1_flow_paths.append(flow_path_tvl1)
                        
                        if label_all == 'Happiness':
                            self.label.append(0)
                            a=a+1
                        elif label_all == 'Surprise':
                            self.label.append(1)
                            b=b+1
                        else:
                            self.label.append(2)
                            c=c+1

                elif num_classes==5:
                #five classes
                    if label_all == 'Happiness' or label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Surprise' or label_all == 'Other':

                        self.onset_paths.append(onset_path)
                        self.apex_paths.append(apex_path)
                        self.tvl1_flow_paths.append(flow_path_tvl1)


                        if label_all == 'Happiness':
                            self.label.append(0)

                            a = a + 1
                        elif label_all == 'Anger':
                            self.label.append(1)

                            b = b + 1
                        elif label_all == 'Contempt':
                            self.label.append(2)

                            c = c + 1
                        elif label_all == 'Surprise':
                            self.label.append(3)

                            d = d + 1
                        else:
                            self.label.append(4)
                            e = e + 1
        
        else:
            print('wrong') 


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        onset_path=self.onset_paths[idx]
        apex_path=self.apex_paths[idx]
        tvl1_flow_path=self.tvl1_flow_paths[idx]


        image_onset_bgr = cv2.imread(onset_path)
        image_apex_bgr = cv2.imread(apex_path) #BGR


        if image_apex_bgr is None:
            raise ValueError(f"无法读取图像: {apex_path}")
            
        # 检查图像尺寸
        if image_apex_bgr.size == 0:
            raise ValueError(f"空图像: {apex_path}")
        
        image_onset_bgr=cv2.resize(image_onset_bgr,(224,224))
        image_apex_bgr=cv2.resize(image_apex_bgr,(224,224))

        tvl1_flow=np.load(tvl1_flow_path)
        flow = tvl1_flow.transpose(2, 0, 1)
        flow = torch.from_numpy(flow).float()

        label = self.label[idx]

        if self.phase!='test':  
            # flow_nor = self.transform_flow(flow)
            flow_nor=flow
            image_apex_nor = self.transform_apex(image_apex_bgr)
            image_onset_nor = self.transform_apex(image_onset_bgr)
            if self.transform_aug is not None:
                ALL = torch.cat((flow_nor, image_onset_nor, image_apex_nor), dim=0)
                ALL = self.transform_aug(ALL)
                flow_nor = ALL[0:3, :, :]
                image_onset_nor = ALL[3:6, :, :]
                image_apex_nor = ALL[6:9, :, :]
            # print(f"\n最终返回数据:")
            # print(f"flow_nor形状: {flow_nor.shape}")
            # print(f"image_onset_nor形状: {image_onset_nor.shape}")
            # print(f"image_apex_nor形状: {image_apex_nor.shape}")
            # print(f"标签: {label}")
            # print(f"返回类型将是: (Tensor, Tensor, Tensor, int)")

            return flow_nor,image_onset_nor,image_apex_nor,label
        else:
            # flow_nor = self.transform_flow(flow)
            flow_nor=flow
            image_apex_nor = self.transform_apex(image_apex_bgr)
            image_onset_nor = self.transform_apex(image_onset_bgr)
            return flow_nor,image_onset_nor,image_apex_nor,label      
        


        