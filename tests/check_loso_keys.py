import os
import sys
from pathlib import Path
import pandas as pd
from torchvision import transforms

sys.path.insert(0, os.getcwd())

from model2.dataset import RafDataSet

raf_path = "/nfs/users/lixinglin/data/microemotion"
csv_path = Path(raf_path) / "3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv"
df = pd.read_csv(csv_path)

print("columns:", df.columns.tolist())

subject_col = None
for c in df.columns:
    if "subject" in c.lower() or c.lower() in ["sub", "subj"]:
        subject_col = c
        break

print("subject_col:", subject_col)

if subject_col is None:
    raise RuntimeError("No subject-like column found")

subjects = df[subject_col].dropna().astype(str).unique().tolist()
print("num subjects:", len(subjects))
print("first subjects:", subjects[:30])

apex_normal = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

flow_normal = transforms.Compose([
    transforms.ToTensor(),
])

print("\n=== LOSO check ===")
for s in subjects[:30]:
    try:
        train_set = RafDataSet(
            raf_path=raf_path,
            phase="train",
            num_loso=s,
            transform_flow=flow_normal,
            transform_apex=apex_normal,
            transform_aug=None,
            num_classes=3,
            db="3DB",
            combined=True,
        )
        test_set = RafDataSet(
            raf_path=raf_path,
            phase="test",
            num_loso=s,
            transform_flow=flow_normal,
            transform_apex=apex_normal,
            transform_aug=None,
            num_classes=3,
            db="3DB",
            combined=True,
        )
        print(s, "train:", len(train_set), "test:", len(test_set))
    except Exception as e:
        print(s, "ERROR:", repr(e))
