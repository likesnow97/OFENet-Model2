import os
import sys

sys.path.insert(0, os.getcwd())

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from model2.dataset import RafDataSet
from model2.improved_resnet_v22 import ImprovedResNetV22

raf_path = "/nfs/users/lixinglin/data/microemotion"
num_loso = "sub01"
device = "cuda" if torch.cuda.is_available() else "cpu"

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

loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

flow, onset, apex, label = next(iter(loader))

model = ImprovedResNetV22(num_classes=3).to(device)
model.eval()

flow = flow.to(device)

with torch.no_grad():
    out = model(flow)

print("device:", device)
print("input flow:", flow.shape)
print("output:", out.shape)
print("output dtype:", out.dtype)

assert out.shape == torch.Size([4, 3])

print("ImprovedResNetV22 real-batch forward OK")
