import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import torch
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

for phase in ["train", "test"]:
    print("\n===", phase, "===")

    dataset = RafDataSet(
        raf_path=raf_path,
        phase=phase,
        num_loso=num_loso,
        transform_flow=flow_normal,
        transform_apex=apex_normal,
        transform_aug=None,
        num_classes=3,
        db="3DB",
        combined=True,
    )

    print("len:", len(dataset))
    assert len(dataset) > 0

    flow, onset, apex, label = dataset[0]

    print("flow:", flow.shape, flow.dtype, float(flow.min()), float(flow.max()))
    print("onset:", onset.shape, onset.dtype, float(onset.min()), float(onset.max()))
    print("apex:", apex.shape, apex.dtype, float(apex.min()), float(apex.max()))
    print("label:", label, type(label))

    assert isinstance(flow, torch.Tensor)
    assert isinstance(onset, torch.Tensor)
    assert isinstance(apex, torch.Tensor)
    assert flow.shape == torch.Size([3, 224, 224])
    assert onset.shape == torch.Size([3, 224, 224])
    assert apex.shape == torch.Size([3, 224, 224])

print("\nRafDataSet smoke test OK")
