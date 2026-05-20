import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


############################################################
# Region Feature Reweighting (RFR)
############################################################

class RegionFeatureReweighting(nn.Module):

    def __init__(self, in_channels=512, num_regions=8,aggregation="sum",dropout=0.2):

        super().__init__()

        self.num_regions = num_regions
        self.aggregation = aggregation
        self.dropout=dropout

        self.region_conv = nn.Conv2d(
            in_channels,
            num_regions,
            kernel_size=1
        )

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):

        B, C, H, W = x.shape

        region_logits = self.region_conv(x)

        region_weights = self.softmax(region_logits)

        region_weights = self.dropout(region_weights)

        region_weights = region_weights.unsqueeze(2)

        x_expand = x.unsqueeze(1)

        region_features = region_weights * x_expand

        if self.aggregation == "sum":

            out = region_features.sum(dim=1)

        elif self.aggregation == "mean":

            out = region_features.mean(dim=1)

        elif self.aggregation == "max":

            out, _ = region_features.max(dim=1)

        else:

            raise ValueError("Unknown aggregation type")

        return out, region_weights


############################################################
# Region Channel Excitation (RCE)
############################################################

class AdaptiveDualAttentionFusion(nn.Module):
# class DualPathAttention(nn.Module):
    """并行SE和GLEA，通道级自适应混合，完全保留两者优势"""
    def __init__(self, channels=512, reduction=16, band_width=5):
        super().__init__()
        # SE分支
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        # GLEA分支（保持原样）
        self.local_conv = nn.Conv1d(channels, channels, kernel_size=band_width,
                                     padding=band_width//2, groups=channels, bias=False)
        nn.init.normal_(self.local_conv.weight, 0, 0.1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )
        self.corr_enhance = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        
        # 通道级混合门控（基于输入特征生成每个通道的融合系数）
        self.mix_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()  # 输出0~1，表示GLEA权重，SE权重为1-该值
        )
        
    def glea_attention(self, x):
        B, C, H, W = x.shape
        U = F.adaptive_avg_pool2d(x, 1).view(B, C, 1)
        U_flat = U.squeeze(-1)
        U_g = self.global_fc(U_flat).unsqueeze(-1)
        U_gc = self.sigmoid(U_g)
        U_lc = self.sigmoid(self.local_conv(U))
        correlation = U_gc * U_lc
        corr_enhanced = self.corr_enhance(correlation)
        combined = U_gc + U_lc + corr_enhanced
        att = self.sigmoid(combined).view(B, C, 1, 1)
        return att
    
    def forward(self, x):
        # 计算SE注意力权重
        se_att = self.se(x).view(x.size(0), -1, 1, 1)
        # 计算GLEA注意力权重
        glea_att = self.glea_attention(x)
        # 生成混合系数（每个通道独立）
        gate = self.mix_gate(x).view(x.size(0), -1, 1, 1)  # [B,C,1,1]
        # 融合权重
        final_att = (1 - gate) * se_att + gate * glea_att
        # 应用注意力，并加上残差连接（保持原始信息）
        out = final_att * x + x  # 或 x * final_att 但加残差更稳定
        return out

############################################################
# Improved ResNet
############################################################

class ImprovedResNetV22(nn.Module):

    def __init__(
            self,
            num_classes=3,

            use_rfr=True,
            use_adaf=True,

            num_regions=8,
            ril_dropout=0.2,
            aggregation_type="sum",
            rce_reduction=32,
            band_width=3):

        super().__init__()


        self.use_rfr = use_rfr
        self.use_adaf = use_adaf


        resnet = models.resnet18(pretrained=True)

        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4


        if self.use_rfr:

            self.rfr = RegionFeatureReweighting(
                512,
                num_regions,
                aggregation_type,
                ril_dropout
            )

        if self.use_adaf:

            self.adaf = AdaptiveDualAttentionFusion(
                512,
                reduction=rce_reduction,
                band_width=band_width
            )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

        self.region_weights = None


    def extract_features(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)


        if self.use_rfr:

            x, region_weights = self.rfr(x)

            self.region_weights = region_weights

        if self.use_adaf:

            x = self.adaf(x)

        return x


    ########################################################

    def forward(self, flow):

        x = self.extract_features(flow)

        out = self.classifier(x)

        return out


    ########################################################

    def get_region_weights(self):

        return self.region_weights