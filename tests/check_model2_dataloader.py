import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model2.dataset import RafDataSet

raf_path = "/nfs/users/lixinglin/data/microemotion"
num_loso = "sub01"

apex_normal = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

flow_normal = transforms.Compose([
    transforms.ToTensor(),
])

dataset = RafDataSet(
    raf_path=raf_path,
    phase="train",
    num_loso=num_loso,
    transform_flow=flow_normal,
    transform_apex=apex_normal,
    transform_aug=None,
    num_classes=3,
    db="3DB",
    combined=True,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

flow, onset, apex, label = next(iter(loader))

print("len:", len(dataset))
print("flow:", flow.shape, flow.dtype)
print("onset:", onset.shape, onset.dtype)
print("apex:", apex.shape, apex.dtype)
print("label:", label.shape, label.dtype, label)

assert flow.shape == torch.Size([4, 3, 224, 224])
assert onset.shape == torch.Size([4, 3, 224, 224])
assert apex.shape == torch.Size([4, 3, 224, 224])

print("DataLoader smoke test OK")
