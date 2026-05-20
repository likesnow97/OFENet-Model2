#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

OLD_ROOT = "/nfs/users/yanghuiru/AMER/dataset"
NEW_ROOT = "/nfs/users/lixinglin/data/microemotion"

CASME2_CSV = Path("/nfs/users/lixinglin/data/microemotion/CASMEII/precomputed_flows_tvl1/dataset_with_flows.csv")
SAMM_CSV = Path("/nfs/users/lixinglin/data/microemotion/SAMM/precomputed_flows/dataset_with_flows.csv")
OUT_CSV = Path("/nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2/runtime_data/mixed5_casme2_samm/3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv")

VALID_LABELS = {"happiness", "surprise", "disgust", "repression", "others"}

def norm_path(x):
    return str(x).replace(OLD_ROOT, NEW_ROOT)

def standardize_casme2(path):
    df = pd.read_csv(path, usecols=[0, 1, 3, 4, 5, 7, 8, 9, 10, 11])
    df.columns = ["Subject", "Filename", "OnsetFrame", "ApexFrame", "OffsetFrame", "Label_AU", "Label", "OnsetPath", "ApexPath", "FlowPath"]

    out = pd.DataFrame()
    out["Dataset"] = "casme2"
    out["Subject"] = df["Subject"].astype(str).apply(lambda x: f"casme2_sub{int(x):02d}" if x.isdigit() else f"casme2_{x}")
    out["Filename"] = df["Filename"].astype(str)
    out["Label"] = df["Label"].astype(str).str.strip().str.lower()
    out["OnsetFrame"] = df["OnsetFrame"]
    out["ApexFrame"] = df["ApexFrame"]
    out["OffsetFrame"] = df["OffsetFrame"]
    out["OnsetPath"] = df["OnsetPath"].map(norm_path)
    out["ApexPath"] = df["ApexPath"].map(norm_path)
    out["FlowPath"] = df["FlowPath"].map(norm_path)
    out = out[out["Label"].isin(VALID_LABELS)].copy()
    return out

def standardize_samm(path):
    # Real SAMM CSV columns:
    # 0 Subject, 1 Filename, 2 OnsetFrame, 3 ApexFrame, 4 OffsetFrame,
    # 5 Label_AU, 6 Estimated Emotion, 7 OnsetPath, 8 ApexPath, 9 FlowPath
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["Dataset"] = "samm"
    out["Subject"] = df["Subject"].astype(str).apply(lambda x: f"samm_{x}")
    out["Filename"] = df["Filename"].astype(str)
    out["Label"] = df["Estimated Emotion"].astype(str).str.strip().str.lower()
    out["OnsetFrame"] = df["OnsetFrame"]
    out["ApexFrame"] = df["ApexFrame"]
    out["OffsetFrame"] = df["OffsetFrame"]
    out["OnsetPath"] = df["OnsetPath"].map(norm_path)
    out["ApexPath"] = df["ApexPath"].map(norm_path)
    out["FlowPath"] = df["FlowPath"].map(norm_path)

    # Keep the same five labels used by the combined 5-class mapping.
    # Note: original SAMM labels include Anger/Contempt/Other; map them before filtering.
    label_map = {
        "happiness": "happiness",
        "surprise": "surprise",
        "disgust": "disgust",
        "anger": "repression",
        "contempt": "repression",
        "other": "others",
        "others": "others",
    }
    out["Label"] = out["Label"].map(label_map)
    out = out[out["Label"].notna()].copy()
    return out


def check_paths(df, name):
    for col in ["OnsetPath", "ApexPath", "FlowPath"]:
        miss_mask = ~df[col].astype(str).map(lambda p: Path(p).exists())
        miss = int(miss_mask.sum())
        print(f"{name} {col}: missing {miss} / {len(df)}")
        if miss:
            print(df.loc[miss_mask, col].head(10).tolist())

def main():
    print("CASME2_CSV:", CASME2_CSV, "exists:", CASME2_CSV.exists())
    print("SAMM_CSV:", SAMM_CSV, "exists:", SAMM_CSV.exists())

    if not CASME2_CSV.exists():
        raise FileNotFoundError(CASME2_CSV)
    if not SAMM_CSV.exists():
        raise FileNotFoundError(SAMM_CSV)

    casme2 = standardize_casme2(CASME2_CSV)
    samm = standardize_samm(SAMM_CSV)

    check_paths(casme2, "casme2")
    check_paths(samm, "samm")

    out = pd.concat([casme2, samm], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    print("samples:", len(out))
    print("dataset distribution:")
    print(out["Dataset"].value_counts())
    print("subject count:", out["Subject"].nunique())
    print("label distribution:")
    print(out["Label"].value_counts().sort_index())
    print("columns:", out.columns.tolist())

if __name__ == "__main__":
    main()
