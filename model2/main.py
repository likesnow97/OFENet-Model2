import argparse
import sys
import torch
import torch.utils.data as data
from timm.loss import *
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import torchvision.models as models
import os
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics2


torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from utils import *
from model2.dataset import *
from model2.model import *
from model2.model_new import *
from model2.improved_resnet_v2 import ImprovedResNetV2
from model2.improved_resnet_v2_mod import ImprovedResNetV2Mod
from model2.improved_resnet_v3 import ImprovedResNetV3
from model2.improved_resnet_v4 import ImprovedResNetV4
from model2.improved_resnet_v5 import ImprovedResNetV5
from model2.improved_resnet_v6 import ImprovedResNetV6
from model2.improved_resnet_v7 import ImprovedResNetV7
from model2.improved_resnet_v8 import ImprovedResNetV8
from model2.improved_resnet_v9 import ImprovedResNetV9
from model2.improved_resnet_v10 import ImprovedResNetV10    
from model2.improved_resnet_v11 import ImprovedResNetV11
from model2.improved_resnet_v12 import ImprovedResNetV12
from model2.improved_resnet_v13 import ImprovedResNetV13
from model2.improved_resnet_v14 import ImprovedResNetV14
from model2.improved_resnet_v15 import ImprovedResNetV15
from model2.improved_resnet_v16 import ImprovedResNetV16
from model2.improved_resnet_v17 import ImprovedResNetV17
from model2.improved_resnet_v18 import ImprovedResNetV18
from model2.improved_resnet_v19 import ImprovedResNetV19
from model2.improved_resnet_v20 import ImprovedResNetV20
from model2.improved_resnet_v21 import ImprovedResNetV21
from model2.ResNet import MicroExpressionResNet
from model2.improved_resnet_v22 import ImprovedResNetV22

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/nfs/users/yanghuiru/AMER/dataset', help='Raf-DB dataset path.')
    parser.add_argument('--model_name', default='improved_resnet_v22', help='the model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0008, help='Initial learning rate for sgd.')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs.')
    parser.add_argument('--loss_function', default='crossentropy', help='the loss functions')
    parser.add_argument('--optimizer_name', default='adamW', help='the optimizer')
    parser.add_argument('--num_classes',type=int, default=3, help='Classes') 
    parser.add_argument('--db',type=str, default='3DB', help='Dataset')    #casme2,samm,3DB
    parser.add_argument('--combined', default=True, dest='combined', action='store_true',
                    help='if the label of dataset is the combined 3 classes. ')
    

    parser.add_argument('--is_of', default=True)
    parser.add_argument('--is_motion', default=False)

    # v2 v3 v7 v8 v9 v11
    parser.add_argument('--use_attention_propagation', default=True)    

    # improved_resnet
    parser.add_argument('--use_bidirectional', default=False)
    parser.add_argument('--use_attention_residual', default=False)    
    parser.add_argument('--use_attention_refinement', default=False)
    parser.add_argument('--attention_loss_weight', default=0.0)    

    # improved_resnet_v2
    parser.add_argument('--use_au_attention', default=False)

    parser.add_argument('--use_motion_guidance', default=False)    

    # improved_resnet_v3 v4 v5 v6 v7 v8 v9 v11 v12 v14 v15 v16 v17 v18 v19 region_reweight
    parser.add_argument('--use_region_reweight', default=False)

    # improved_resnet_v4 v16
    parser.add_argument('--use_rma', default=True)
    # # improved_resnet_v4 v5 v15 v18 v20
    # parser.add_argument('--use_rce', default=True)

    # improved_resnet_v5
    parser.add_argument('--use_mg', default=True)

    # improved_resnet_v6
    parser.add_argument('--use_rdi', default=True)

    # improved_resnet_v7
    parser.add_argument('--use_diff', default=False)
    parser.add_argument('--use_flow', default=True)

    # improved_resnet_v8
    parser.add_argument('--use_attention_branch', default=True)
    parser.add_argument('--use_region_branch', default=True)  

    # v9
    parser.add_argument('--use_apex_branch', default=True)  
    parser.add_argument('--use_flow_branch',default=True)

    # V10
    parser.add_argument('--use_mca',default=True)
    # # v10 v20
    # parser.add_argument('--use_rfr',default=True)

    # v11
    parser.add_argument('--use_motion_channel_attention', default=True)

    # v12 v14 v15 v16 v17
    parser.add_argument('--use_fcm', default=True)

    # v13
    parser.add_argument('--fusion_alpha',default=0.5)

    # v14 v15 v16 v17
    parser.add_argument('--use_mme',default=True)
    # v17
    parser.add_argument('--use_rim', default=True)
    # v19
    parser.add_argument('--use_rice',default=True)

    # # v20
    # parser.add_argument('--use_ril',default=True)
    # # v10 v20
    # parser.add_argument('--use_rfr',default=True)
    # # improved_resnet_v4 v5 v15 v18 v20
    # parser.add_argument('--use_rce', default=True)
    # parser.add_argument('--num_regions',default=8)
    # parser.add_argument('--ril_dropout',default=0.2)
    # parser.add_argument('--fusion_type',default="sum")
    # parser.add_argument('--aggregation_type',default="sum")
    # parser.add_argument('--rce_reduction',default=32)
    # parser.add_argument('--band_width',default=5)

    # v22
    parser.add_argument('--use_rfr',default=True)
    parser.add_argument('--use_adaf', default=True)
    parser.add_argument('--num_regions',default=8)
    parser.add_argument('--ril_dropout',default=0.2)
    parser.add_argument('--aggregation_type',default="sum")
    parser.add_argument('--rce_reduction',default=32)
    parser.add_argument('--band_width',default=7)

    
    parser.add_argument('--image_size', default=224, type=int,help='input image size (default: 224)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('--patchup_prob', type=float, default=.7, help='PatchUp probability')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')



    return parser.parse_args()

def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)

def main():

    args = parse_args()
    raf_path=args.raf_path
    batch_size=args.batch_size
    optimizer_name=args.optimizer_name
    lr=args.lr
    epochs=args.epochs
    model_name=args.model_name
    db=args.db
    combined=args.combined    
    loss_function=args.loss_function
    num_classes=args.num_classes
    workers=args.workers

    image_size=args.image_size


    is_of=args.is_of
    is_motion=args.is_motion
    use_attention_propagation=args.use_attention_propagation
    use_bidirectional=args.use_bidirectional
    use_attention_residual=args.use_attention_residual
    use_attention_refinement=args.use_attention_refinement
    attention_loss_weight=args.attention_loss_weight  



    if model_name=='resnet':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"
    elif model_name=='resca':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"
    elif model_name=='cross_resnet':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"
    elif model_name=='ResNet':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}"
    elif model_name=='improved_resnet':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"
    elif model_name=='improved_resnet_v2':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_motion_guidance}_{args.use_au_attention}_{args.use_attention_propagation}"
    elif model_name=='improved_resnet_v2_mod':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_motion_guidance}_{args.use_region_reweight}_{args.use_attention_propagation}"
    elif model_name=='improved_resnet_v3':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_attention_propagation}"
    elif model_name=='improved_resnet_v4':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_rma}_{args.use_rce}"
    elif model_name=='improved_resnet_v5':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_mg}_{args.use_rce}"
    elif model_name=='improved_resnet_v6':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_rdi}"
    elif model_name=='improved_resnet_v7':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_flow}_{args.use_region_reweight}_{args.use_diff}_{args.use_attention_propagation}"
    elif model_name=='improved_resnet_v8':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_branch}_{args.use_attention_branch}"
    elif model_name=='improved_resnet_v9':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_apex_branch}_{args.use_flow_branch}"
    elif model_name=='improved_resnet_v10':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_mca}_{args.use_rfr}"
    elif model_name=='improved_resnet_v11':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_attention_propagation}_{args.use_motion_channel_attention}"
    elif model_name=='improved_resnet_v12':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_fcm}"
    elif model_name=='improved_resnet_v13':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.fusion_alpha}"
    elif model_name=='improved_resnet_v14':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}"
    elif model_name=='improved_resnet_v15':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rce}"
    elif model_name=='improved_resnet_v16':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rma}"
    elif model_name=='improved_resnet_v17':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rim}"
    elif model_name=='improved_resnet_v18':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_rce}"
    elif model_name=='improved_resnet_v19':
        log_name= f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_region_reweight}_{args.use_rice}"
    elif model_name=='improved_resnet_v20' or model_name=='improved_resnet_v21':
        log_name = f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_ril}_{args.use_rfr}_{args.use_rce}_numregion{args.num_regions}_dropout{args.ril_dropout}_fusiontype{args.fusion_type}_aggre{args.aggregation_type}_reduction{args.rce_reduction}_bandwidth{args.band_width}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    elif model_name=='improved_resnet_v22':
        log_name=f"{model_name}_lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_cls{num_classes}_{args.use_rfr}_{args.use_adaf}_numregion{args.num_regions}_dropout{args.ril_dropout}_aggre{args.aggregation_type}_reduction{args.rce_reduction}_bandwidth{args.band_width}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"


    log_filename = f"logs/class{num_classes}/{model_name}/{db}/{image_size}/{log_name}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout

    start_time = time.time()
    data_loading_time = 0


    ### data augmentation for training set only
    data_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),
    ])

    if db=='casme2':
        if num_classes==5:
            LOSO = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3',
                '19', '1','18','10','20', '21', '22', '15', '6', '25', '7']
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            # 通道0 (OS Flow): 光流应变（Optical Strain）通道1 (V Flow): 垂直方向光流 通道2 (U Flow): 水平方向光流
            transforms.Normalize(mean=[0.482,0.480,0.024],
                                std=[0.199,0.239,0.045]),
        ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                # 图像为BGR格式，与OpenCV默认读取一致
                transforms.Normalize(mean=[0.296,0.358,0.500],
                                    std=[0.123,0.136,0.185]),
            ])
        elif num_classes==3:
            LOSO = ['17', '26', '16', '9', '5', '24', '2', '13', '4', '23', '11', '12', '8', '14', '3', '19', '1', '10',
                '20', '21', '22', '15', '6', '25', '7']
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.474,0.515,0.025],   
                                std=[0.203,0.239,0.046]),
            ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.303,0.365,0.508],   
                                    std=[0.119,0.135,0.187]),
            ])

    elif db=='samm':
        LOSO=[6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,30,31,32,33,34,35,36,37]
        
        if num_classes==5:
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.018,0.507,0.496],
                                std=[0.032, 0.230,0.243]),
        ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.504,0.504,0.504],
                                    std=[0.209, 0.209,0.209]),
            ])
        elif num_classes==3:
            flow_normal = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.018, 0.505,0.491],   
                                std=[0.033,0.232,0.238]),
            ])
            apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.501,0.501,0.501],   
                                    std=[0.210,0.210,0.210]),
            ])
    elif db=='smic':
        LOSO=['s1','s2','s3','s4','s5','s6','s8','s9','s11','s12','s13','s14','s15','s18','s19','s20']
    else:
        LOSO=['sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09','sub11','sub12','sub13','sub14','sub15','sub16','sub17','sub19','sub20','sub21','sub22','sub23','sub24','sub25','sub26',
              's01','s02','s03','s04','s05','s06','s08','s09','s11','s12','s13','s14','s15','s18','s19','s20',
              '6','7','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','26','28','30','31','32','33','34','35','36','37']
        # flow_normal = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.020,0.520,0.496],   
        #                         std=[0.041,0.221,0.207]),
        #     ])
        # apex_normal = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.395,0.386,0.472],   
        #                             std=[0.187,0.182,0.200]),
        #     ])
        flow_normal = transforms.Compose([
            transforms.ToTensor(), ##3个通道对应 flow_x, flow_y, strain 
            transforms.Normalize(mean=[0.504,0.471,0.062],   
                                std=[0.245,0.265,0.108]),
            ])
        
        apex_normal = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.395,0.386,0.472],   
                                    std=[0.187,0.182,0.200]),
            ])

    preds_db = {}
    preds_db['casme2'] = torch.tensor([])
    preds_db['smic'] = torch.tensor([])
    preds_db['samm'] = torch.tensor([])
    preds_db['all'] = torch.tensor([])
    labels_db = {}
    labels_db['casme2'] = torch.tensor([])
    labels_db['smic'] = torch.tensor([])
    labels_db['samm'] = torch.tensor([])
    labels_db['all'] = torch.tensor([])
    allRST = []

    all_accuracy_dict = {}
    total_gt = []
    total_pred = []
    best_total_pred = []

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(num_classes)
    pos_label_ALL = torch.zeros(num_classes)
    TP_ALL = torch.zeros(num_classes)
    acc_list = []
    epoch_list = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for subj in LOSO:

        # 构建路径字符串
        if model_name=='resnet':
            params_model_path = f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"
        elif model_name=='resca':
            params_model_path = f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"       
        elif model_name=='cross_resnet':
            params_model_path = f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"    
        elif model_name=='ResNet':
            params_model_path = f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}"
        elif model_name=='improved_resnet':
            params_model_path = f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.is_of}_{args.is_motion}_{args.use_attention_propagation}_{args.use_bidirectional}_{args.use_attention_residual}_{args.use_attention_refinement}_{args.attention_loss_weight}"   
        elif model_name=='improved_resnet_v2':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_motion_guidance}_{args.use_au_attention}_{args.use_attention_propagation}"
        elif model_name=='improved_resnet_v2_mod':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_motion_guidance}_{args.use_region_reweight}_{args.use_attention_propagation}"
        elif model_name=='improved_resnet_v3':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_attention_propagation}"
        elif model_name=='improved_resnet_v4':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_rma}_{args.use_rce}"
        elif model_name=='improved_resnet_v5':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_mg}_{args.use_rce}"
        elif model_name=='improved_resnet_v6':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_rdi}"
        elif model_name=='improved_resnet_v7':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_flow}_{args.use_region_reweight}_{args.use_diff}_{args.use_attention_propagation}"
        elif model_name=='improved_resnet_v8':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_branch}_{args.use_attention_branch}"
        elif model_name=='improved_resnet_v9':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_apex_branch}_{args.use_flow_branch}"
        elif model_name=='improved_resnet_v10':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_mca}_{args.use_rfr}"
        elif model_name=='improved_resnet_v11':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_attention_propagation}_{args.use_motion_channel_attention}"
        elif model_name=='improved_resnet_v12':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_fcm}"
        elif model_name=='improved_resnet_v13':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.fusion_alpha}"
        elif model_name=='improved_resnet_v14':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}"
        elif model_name=='improved_resnet_v15':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rce}"
        elif model_name=='improved_resnet_v16':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rma}"
        elif model_name=='improved_resnet_v17':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_mme}_{args.use_fcm}_{args.use_region_reweight}_{args.use_rim}"
        elif model_name=='improved_resnet_v18':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_rce}"
        elif model_name=='improved_resnet_v19':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_region_reweight}_{args.use_rice}"
        elif model_name=='improved_resnet_v20' or model_name=='improved_resnet_v21':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_ril}_{args.use_rfr}_{args.use_rce}_numregion{args.num_regions}_dropout{args.ril_dropout}_fusiontype{args.fusion_type}_aggre{args.aggregation_type}_reduction{args.rce_reduction}_bandwidth{args.band_width}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
        elif model_name=='improved_resnet_v22':
            params_model_path= f"lr={lr}_epoch{epochs}_batch{batch_size}_{optimizer_name}_{loss_function}_{args.use_rfr}_{args.use_adaf}_{args.num_regions}_{args.ril_dropout}_{args.aggregation_type}_{args.rce_reduction}_{args.band_width}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"        
        
        weight_path = Path(
            f"class{num_classes}"
        ) / model_name /db/str(image_size)/ params_model_path / f"{subj}.pth"
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        data_load_start = time.time()
        train_dataset = RafDataSet(raf_path, phase='train', num_loso=subj,  transform_flow=flow_normal,transform_apex=apex_normal,transform_aug=data_aug,num_classes=num_classes,db=db,combined=combined)
        val_dataset = RafDataSet(raf_path, phase='test', num_loso=subj,  transform_flow=flow_normal,transform_apex=apex_normal,num_classes=num_classes,db=db,combined=combined)
        data_loading_time += time.time() - data_load_start
        print(f'数据加载时间: {time.time() - data_load_start:.2f}秒')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=workers,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_epoch = 0
        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(num_classes)
        max_pos_label = torch.zeros(num_classes)
        max_TP = torch.zeros(num_classes)

        if model_name=='resnet':
            # net_all = create_resnet_model(num_classes=num_classes, is_of=args.is_of, is_motion=args.is_motion)
            net_all = create_improved_resnet(num_classes=num_classes, is_of=args.is_of, is_motion=args.is_motion,use_attention_propagation=args.use_attention_propagation,use_bidirectional=args.use_bidirectional,use_attention_residual=args.use_attention_residual,use_attention_refinement=args.use_attention_refinement,attention_loss_weight=args.attention_loss_weight)
        elif model_name=='resca':
            net_all=create_model(num_classes=num_classes, is_of=args.is_of, is_motion=args.is_motion)
        elif model_name=='cross_resnet':
            net_all=create_new_model(num_classes=num_classes, is_of=args.is_of, is_motion=args.is_motion)
        elif model_name=='ResNet':
            net_all = MicroExpressionResNet(num_classes=num_classes, pretrained=True)
        elif model_name=='improved_resnet':
            net_all = create_improved_resnet(
        num_classes=num_classes,
        is_of=args.is_of,
        is_motion=args.is_motion,
        use_attention_propagation=args.use_attention_propagation,
        use_bidirectional=args.use_bidirectional,
        use_attention_residual=args.use_attention_residual,
        use_attention_refinement=args.use_attention_refinement,
        attention_loss_weight=args.attention_loss_weight
    )
        elif model_name=='improved_resnet_v2':
            net_all = ImprovedResNetV2(
                num_classes=num_classes,
                use_motion_guidance=args.use_motion_guidance,
                use_au_attention=args.use_au_attention,
                use_attention_propagation=args.use_attention_propagation)     
        elif model_name=='improved_resnet_v2_mod':
            net_all = ImprovedResNetV2Mod(
                num_classes=num_classes,
                use_motion_guidance=args.use_motion_guidance,
                use_region_reweight=args.use_region_reweight,
                use_attention_propagation=args.use_attention_propagation)     
        elif model_name=='improved_resnet_v3':
            net_all = ImprovedResNetV3(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_attention_propagation=args.use_attention_propagation)   
        elif model_name=='improved_resnet_v4':
            net_all = ImprovedResNetV4(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_rma=args.use_rma,
                use_rce=args.use_rce)
        elif model_name=='improved_resnet_v5':
            net_all = ImprovedResNetV5(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_mg=args.use_mg,
                use_rce=args.use_rce)
        elif model_name=='improved_resnet_v6':
            net_all = ImprovedResNetV6(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_rdi=args.use_rdi)
        elif model_name=='improved_resnet_v7':
            net_all = ImprovedResNetV7(
                num_classes=num_classes,
                use_flow=args.use_flow,
                use_diff=args.use_diff,
                use_region_reweight=args.use_region_reweight,
                use_attention_propagation=args.use_attention_propagation)
            
        elif model_name=='improved_resnet_v8':
            net_all = ImprovedResNetV8(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_attention_branch=args.use_attention_branch,
                use_region_branch=args.use_region_branch,
                use_attention_propagation=args.use_attention_propagation
                )
            
        elif model_name=='improved_resnet_v9':
            net_all = ImprovedResNetV9(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_apex_branch=args.use_apex_branch,
                use_flow_branch=args.use_flow_branch,
                use_attention_propagation=args.use_attention_propagation
                )
        elif model_name=='improved_resnet_v10':
            net_all = ImprovedResNetV10(
                num_classes=num_classes,
                use_mca=args.use_mca,
                use_rfr=args.use_rfr,
                )
            
        elif model_name=='improved_resnet_v11':
            net_all = ImprovedResNetV11(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_attention_propagation=args.use_attention_propagation,
                use_motion_channel_attention=args.use_motion_channel_attention)
            
        elif model_name=='improved_resnet_v12':
            net_all = ImprovedResNetV12(
                num_classes=num_classes,
                use_fcm=args.use_fcm,
                use_region_reweight=args.use_region_reweight)
        
        elif model_name=='improved_resnet_v13':
            net_all = ImprovedResNetV13(
                num_classes=num_classes,
                fusion_alpha=args.fusion_alpha)
        elif model_name=='improved_resnet_v14':
            net_all = ImprovedResNetV14(
                num_classes=num_classes,
                use_mme=args.use_mme,
                use_fcm=args.use_fcm,
                use_region_reweight=args.use_region_reweight)
        elif model_name=='improved_resnet_v15':
            net_all = ImprovedResNetV15(
                num_classes=num_classes,
                use_mme=args.use_mme,
                use_fcm=args.use_fcm,
                use_region_reweight=args.use_region_reweight,
                use_rce=args.use_rce)
        
        elif model_name=='improved_resnet_v16':
            net_all = ImprovedResNetV16(
                num_classes=num_classes,
                use_mme=args.use_mme,
                use_fcm=args.use_fcm,
                use_region_reweight=args.use_region_reweight,
                use_rma=args.use_rma)
        elif model_name=='improved_resnet_v17':
            net_all = ImprovedResNetV17(
                num_classes=num_classes,
                use_mme=args.use_mme,
                use_fcm=args.use_fcm,
                use_region_reweight=args.use_region_reweight,
                use_rim=args.use_rim)  
        elif model_name=='improved_resnet_v18':
            net_all = ImprovedResNetV18(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_rce=args.use_rce) 
        elif model_name=='improved_resnet_v19':  
            net_all = ImprovedResNetV19(
                num_classes=num_classes,
                use_region_reweight=args.use_region_reweight,
                use_rice=args.use_rice)     
        elif model_name=='improved_resnet_v20':
            net_all = ImprovedResNetV20(
                num_classes=num_classes,
                use_ril=args.use_ril,
                use_rfr=args.use_rfr,
                use_rce=args.use_rce,

                num_regions=args.num_regions,       # 区域数量
                ril_dropout=args.ril_dropout,       # RIL dropout
                fusion_type=args.fusion_type,      # RIL only 时 spatial fusion
                aggregation_type=args.aggregation_type,  # RFR 区域聚合方式
                rce_reduction=args.rce_reduction,     # RCE reduction ratio
                band_width=args.band_width,     # RCE band width
            )
        elif model_name=='improved_resnet_v21':
            net_all = ImprovedResNetV21(
                num_classes=num_classes,
                use_ril=args.use_ril,
                use_rfr=args.use_rfr,
                use_rce=args.use_rce,

                num_regions=args.num_regions,       # 区域数量
                ril_dropout=args.ril_dropout,       # RIL dropout
                fusion_type=args.fusion_type,      # RIL only 时 spatial fusion
                aggregation_type=args.aggregation_type,  # RFR 区域聚合方式
                rce_reduction=args.rce_reduction,     # RCE reduction ratio
                band_width=args.band_width
            )

        elif model_name=='improved_resnet_v22':
            net_all = ImprovedResNetV22(
                num_classes=num_classes,
                use_rfr=args.use_rfr,
                use_adaf=args.use_adaf,

                num_regions=args.num_regions,       # 区域数量
                ril_dropout=args.ril_dropout,       # RIL dropout
                aggregation_type=args.aggregation_type,  # RFR 区域聚合方式
                rce_reduction=args.rce_reduction,     # RCE reduction ratio
                band_width=args.band_width
            )
        
        else:
            net_all=None
        
        params_all = net_all.parameters()

        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params_all, lr=lr,weight_decay=0.0001)
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params_all, lr=lr)
        elif optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params_all, lr=lr, weight_decay=0.7)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.987)

        if loss_function == 'crossentropy':
            criterion = torch.nn.CrossEntropyLoss()
            # 若wetghtedCE loss
            # class_weight = torch.FloatTensor([0.2857, 0.1329, 0.1695]).cuda(args.gpuid)  # half-oversampling  [1.0, 2.2936, 3.0120]和[0.2857, 0.1329,0.1695]
            # criterion = nn.CrossEntropyLoss(weight=class_weight).cuda(args.gpuid)
        # elif loss_function == 'focal':
        #     criterion = LossFunctions.FocalLoss(class_num=classes, device=device)
        # elif loss_function == 'balanced':
        #     criterion = LossFunctions.BalancedLoss(class_num=classes, device=device)
        # elif loss_function == 'cosine':
        #     criterion = LossFunctions.CosineLoss()
            
        net_all=net_all.to(device)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []
        best_each_subject_labels = []

        num_samples = len(val_loader.dataset)
        predVec = torch.zeros(num_samples)
        labelVec = torch.zeros(num_samples)
        start_idx = 0
        end_idx = 0     


        for epoch in range(1,epochs+1):
            # print('\tEpoch {}/{}'.format(epoch, epochs))
            running_loss = 0.0
            running_corrects = 0  #num_train_correct
            num_train_examples = 0
            iter_cnt = 0
            
            net_all.train()

            for i, (flow, onset, apex,label) in enumerate(train_loader):
                flow = flow.cuda()
                onset=onset.cuda()
                apex = apex.cuda()
                class_labels = label.cuda()

                iter_cnt+=1

                if model_name=='improved_resnet_v2' or model_name=='improved_resnet_v3' or model_name=='improved_resnet_v4' or model_name=='improved_resnet_v5' or model_name=='improved_resnet_v6' or model_name=='improved_resnet_v8' or model_name=='improved_resnet_v10' or model_name=='improved_resnet_v11' or model_name=='improved_resnet_v12' or model_name=='improved_resnet_v13' or model_name=='improved_resnet_v14' or model_name=='improved_resnet_v15' or model_name=='improved_resnet_v16' or model_name=='improved_resnet_v17' or model_name=='improved_resnet_v18' or model_name=='improved_resnet_v19' or model_name=='improved_resnet_v20' or model_name=='ResNet' or model_name=='improved_resnet_v21' or model_name=='improved_resnet_v22':
                    output_class= net_all(flow)
                elif model_name=='improved_resnet_v9':
                    output_class=net_all(flow=flow,apex=apex)
                else:
                    output_class= net_all(onset=onset, apex=apex, flow=flow)


                loss_class = criterion(output_class, class_labels)


                if model_name=='improved_resnet':
                    attn_loss = net_all.get_attention_loss()
                    total_loss =loss_class+attn_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item() * flow.size(0)
                    _, predicted = torch.max(output_class, 1)
                    running_corrects += torch.sum(predicted == class_labels)
                else:
                    optimizer.zero_grad()
                    loss_class.backward()
                    optimizer.step()

                    running_loss += loss_class.item() * flow.size(0)
                    _, predicted = torch.max(output_class, 1)
                    running_corrects += torch.sum(predicted == class_labels)                    


            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / float(len(train_loader.dataset))
            print('[Epoch %d] Training accuracy: %.4f. Training Loss: %.3f' % (epoch, epoch_acc, epoch_loss))

            # batch_loss = running_loss / iter_cnt
            # print('\t{}Epoch Loss: {:.4f} Acc: {:.4f}'.format('Train', epoch_loss, epoch_acc))

            net_all.eval()
            all_predictions = []  # 当前epoch的所有预测
            all_labels = []       # 当前epoch的所有标签

            with torch.no_grad():  # 禁用梯度计算
                val_loss=0.0
                num_val_correct = 0
                num_val_examples = 0

                pos_label = torch.zeros(num_classes)
                pos_pred = torch.zeros(num_classes)
                TP = torch.zeros(num_classes)

                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0


                for i, (flow, onset, apex,class_labels) in enumerate(val_loader):
                    onset=onset.cuda()
                    class_labels=class_labels.cuda()
                    apex=apex.cuda()
                    flow=flow.cuda()

                    if model_name=='improved_resnet_v2' or model_name=='improved_resnet_v3' or model_name=='improved_resnet_v4' or model_name=='improved_resnet_v5' or model_name=='improved_resnet_v6' or model_name=='improved_resnet_v8' or model_name=='improved_resnet_v10' or model_name=='improved_resnet_v11' or model_name=='improved_resnet_v12' or model_name=='improved_resnet_v13' or model_name=='improved_resnet_v14' or model_name=='improved_resnet_v15' or model_name=='improved_resnet_v16' or model_name=='improved_resnet_v17' or model_name=='improved_resnet_v18' or model_name=='improved_resnet_v19' or model_name=='improved_resnet_v20' or model_name=='ResNet' or model_name=='improved_resnet_v21' or model_name=='improved_resnet_v22':
                        output_class= net_all(flow)
                    elif model_name=='improved_resnet_v9':
                        output_class=net_all(flow=flow,apex=apex)
                    else:
                        output_class= net_all(onset=onset, apex=apex, flow=flow)


                    loss_class = criterion(output_class, class_labels)
                    val_loss+=loss_class.data.item()*flow.size(0)

                    running_loss += loss_class
                    iter_cnt += 1

                    _, predicts = torch.max(output_class, 1)

                    # 关键：收集所有batch的数据
                    all_predictions.extend(predicts.cpu().tolist())
                    all_labels.extend(class_labels.cpu().tolist())

                    num_val_correct += (torch.max(output_class, 1)[1] == class_labels).sum().item()
                    num_val_examples += class_labels.size(0)  

                    correct_num = torch.eq(predicts,class_labels)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += output_class.size(0)

                    for cls in range(num_classes):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in class_labels:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, class_labels):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1

                    count = 0
                    SUM_F1 = 0
                    for index in range(num_classes):
                        if pos_label[index] != 0 or pos_pred[index] != 0:
                            count = count + 1
                            SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])
                    AVG_F1 = SUM_F1 / count

                val_acc = num_val_correct / num_val_examples
                val_loss = val_loss / len(val_loader.dataset) 

                temp_best_each_subject_pred = []
                if best_accuracy_for_each_subject<=val_acc:
                    best_accuracy_for_each_subject=val_acc
                    # temp_best_each_subject_pred.extend(torch.max(output_class, 1)[1].tolist())
                    best_each_subject_pred = all_predictions.copy()
                    best_each_subject_labels = all_labels.copy()

                    # 保存模型参数
                    torch.save(net_all.state_dict(), weight_path)
            
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, val_acc, val_loss))

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                    max_epoch = epoch
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (epoch, acc, running_loss, AVG_F1))
                if val_acc==1.:
                    print('achieve 100%acc, break')
                    break     

                if epoch <= 50:
                    scheduler.step() 



        print("评价1：--------------------------------------------------------------------------------------------------")
        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = best_each_subject_labels
        all_accuracy_dict[subj] = accuracydict

        print('Ground Truth :', best_each_subject_labels)
        print('Evaluation until this subject: ')
        total_pred.extend(best_each_subject_pred)
        total_gt.extend(best_each_subject_labels)
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4)) 

        print("评价2：--------------------------------------------------------------------------------------------------") 
        end_idx = start_idx + len(val_dataset)
        predVec[start_idx:end_idx] = torch.tensor(best_each_subject_pred).to(device)
        labelVec[start_idx:end_idx] = torch.tensor(best_each_subject_labels).to(device)
        start_idx =end_idx
        preds_np = np.array(predVec)
        labels_np = np.array(labelVec)
        allRST.append([preds_np, labels_np])
        acc = torch.sum(predVec == labelVec).double() / len(predVec)
        print('\tSubject {} has the accuracy:{:.4f}\n'.format(subj, acc))

        preds_db['all'] = torch.cat((preds_db['all'], predVec), 0)
        labels_db['all'] = torch.cat((labels_db['all'], labelVec), 0)
        if combined:
            if subj.find('sub') != -1:
                preds_db['casme2'] = torch.cat((preds_db['casme2'], predVec), 0)
                labels_db['casme2'] = torch.cat((labels_db['casme2'], labelVec), 0)
            else:
                if subj.find('s') != -1:
                    preds_db['smic'] = torch.cat((preds_db['smic'], predVec), 0)
                    labels_db['smic'] = torch.cat((labels_db['smic'], labelVec), 0)
                else:
                    preds_db['samm'] = torch.cat((preds_db['samm'], predVec), 0)
                    labels_db['samm'] = torch.cat((labels_db['samm'], labelVec), 0)


        print("评价3：--------------------------------------------------------------------------------------------------")
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        for index in range(num_classes):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        if len(val_dataset) > 0:
            acc_now=max_corr/val_dataset.__len__()
        else:
            acc_now=0

        # 确保 acc_now 是 Python 数字而不是 Tensor
        if isinstance(acc_now, torch.Tensor):
            acc_now_value = acc_now.item()
        else:
            acc_now_value = acc_now
        
        #writer.add_scalar('trans_acc', acc_now, subj)
        print("[subject %s] correct_num:%d sum:%d ACC: %.4f  " % (subj, max_corr, val_dataset.__len__(),acc_now))
        print("[ALL_corr]: %d [ALL_val]: %d [ALL_ACC]:%.4f" % (int(num_sum), int(val_now), num_sum / val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
        print('max_epoch:', str(max_epoch))
        acc_list.append(round(acc_now_value,4))
        epoch_list.append(str(max_epoch))

    print("最终评价1：--------------------------------------------------------------------------------------------------")
    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print(all_accuracy_dict)

    print("最终评价2：--------------------------------------------------------------------------------------------------")
    eval_acc = metrics2.accuracy()
    eval_f1 = metrics2.f1score()
    acc_w, acc_uw = eval_acc.eval(preds_db['all'], labels_db['all'])
    f1_w, f1_uw = eval_f1.eval(preds_db['all'], labels_db['all'])
    print('\nThe dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))

    if combined:
        # casme2
        if preds_db['casme2'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['casme2'], labels_db['casme2'])
            f1_w, f1_uw = eval_f1.eval(preds_db['casme2'], labels_db['casme2'])
            print('\nThe casme2 dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
            
        # smic
        if preds_db['smic'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['smic'], labels_db['smic'])
            f1_w, f1_uw = eval_f1.eval(preds_db['smic'], labels_db['smic'])
            print('\nThe smic dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))
            
        # samm
        if preds_db['samm'].nelement() != 0:
            acc_w, acc_uw = eval_acc.eval(preds_db['samm'], labels_db['samm'])
            f1_w, f1_uw = eval_f1.eval(preds_db['samm'], labels_db['samm'])
            print('\nThe samm dataset has the ACC:{:.4f} and UAR and UF1:{:.4f} and {:.4f}'.format(acc_w, acc_uw, f1_uw))

    print("最终评价3：--------------------------------------------------------------------------------------------------")
    print("Acc_list:", acc_list)
    print('max epoches:', epoch_list)
    print("Total acc: %.4f"% (num_sum * 1.0 / val_now))
        
    # writing parameters into log file
    print("\n" + "="*40)
    print("🏷️ 实验参数配置".center(40))
    print("="*40)

    # 1. 路径参数
    print("\n📂 路径参数:")
    print(f"  RAF数据集路径: {args.raf_path}")
    print(f"  数据库选择: {args.db}")

    # 2. 模型参数
    print("\n🧠 模型参数:")
    print(f"  模型名称: {args.model_name}")
    print(f"  分类数: {args.num_classes}")
    print(f"  Dropout率: {args.drop_rate}")
    print(f"  多数据集: {'是' if args.combined else '否'}")

    # 3. 训练参数
    print("\n⚙️ 训练参数:")
    print(f"  批量大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  优化器: {args.optimizer_name}")
    print(f"  损失函数: {args.loss_function}")
    print("="*40 + "\n")

    end_time = time.time()
    print("start_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print("end_time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))) 

if __name__ == '__main__':
    seed=2
    seed_torch(seed)
    import torch
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
   
         