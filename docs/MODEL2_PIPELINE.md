# Model2 Pipeline

Model2 is an optical-flow-based micro-expression recognition pipeline.

## Input

Each sample is represented by the TV-L1 optical flow computed between the onset frame and the apex frame.

Although the dataset loader also returns onset and apex RGB images, `ImprovedResNetV22` uses the optical-flow tensor as its network input.

The input tensor shape is:

    [batch_size, 3, 224, 224]

## Backbone

The model uses an ImageNet-pretrained ResNet18 backbone.

The backbone is decomposed into:

    conv1 + bn1 + relu + maxpool
    layer1
    layer2
    layer3
    layer4

The output feature map is then refined by RFR and ADAF.

## RFR

RFR means Region Feature Reweighting.

It predicts soft region weights from the final feature map and reweights spatial regions before classification.

## ADAF

ADAF means Adaptive Dual Attention Fusion.

It combines:

    SE channel attention
    GLEA local-global channel attention

A learned channel-wise gate adaptively fuses the two attention branches.

## Classifier

The classifier contains:

    AdaptiveAvgPool2d
    Linear
    BatchNorm1d
    ReLU
    Dropout
    Linear

The output is class logits.

## Supported Variants

The training script supports the following variants:

    full
    no_pretrain
    no_rfr
    no_adaf
    no_rfr_no_adaf

These variants are used for ablation experiments.
