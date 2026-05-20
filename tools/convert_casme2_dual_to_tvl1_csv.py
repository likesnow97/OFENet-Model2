#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


SRC = Path("/nfs/users/lixinglin/data/microemotion/CASMEII/precomputed_dual_flows/dataset_with_dual_flows.csv")
OUT = Path("/nfs/users/lixinglin/data/microemotion/CASMEII/precomputed_flows_tvl1/dataset_with_flows.csv")

OLD_ROOT = "/nfs/users/yanghuiru/AMER/dataset"
NEW_ROOT = "/nfs/users/lixinglin/data/microemotion"


def find_col(df, candidates=None, contains=None, required=True):
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}

    if candidates:
        for cand in candidates:
            for c in cols:
                if lower[c] == cand.lower():
                    return c

    if contains:
        for c in cols:
            lc = lower[c]
            if all(x.lower() in lc for x in contains):
                return c

    if required:
        raise KeyError(f"Cannot find column candidates={candidates}, contains={contains}. Columns={cols}")
    return None


def norm_path(s):
    return s.astype(str).str.replace(OLD_ROOT, NEW_ROOT, regex=False)


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    df = pd.read_csv(SRC)
    print("source:", SRC)
    print("shape:", df.shape)
    print("columns:")
    for i, c in enumerate(df.columns):
        print(i, repr(c))

    subject_col = find_col(df, candidates=["Subject", "subject", "Sub", "sub"], contains=["subject"], required=False)
    if subject_col is None:
        subject_col = df.columns[0]

    filename_col = find_col(df, candidates=["Filename", "FileName", "file", "File"], contains=["file"], required=False)
    if filename_col is None:
        filename_col = df.columns[1]

    onset_frame_col = find_col(df, candidates=["OnsetFrame", "Onset", "Onset_Frame"], contains=["onset", "frame"], required=False)
    apex_frame_col = find_col(df, candidates=["ApexFrame", "Apex", "Apex_Frame"], contains=["apex", "frame"], required=False)
    offset_frame_col = find_col(df, candidates=["OffsetFrame", "Offset", "Offset_Frame"], contains=["offset", "frame"], required=False)

    label_au_col = find_col(df, candidates=["Label_AU", "AU", "ActionUnit"], contains=["au"], required=False)

    label_col = find_col(df, candidates=["Label", "Emotion", "emotion"], contains=["label"], required=False)
    if label_col is None:
        label_col = find_col(df, contains=["emotion"], required=True)

    onset_path_col = find_col(df, candidates=["OnsetPath", "onset_path"], contains=["onset", "path"], required=False)
    apex_path_col = find_col(df, candidates=["ApexPath", "apex_path"], contains=["apex", "path"], required=False)

    if onset_path_col is None:
        onset_path_col = find_col(df, contains=["onset"], required=True)
    if apex_path_col is None:
        apex_path_col = find_col(df, contains=["apex"], required=True)

    # Prefer TVL1 flow path from dual-flow CSV.
    flow_tvl1_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "flow" in lc and "tvl1" in lc:
            flow_tvl1_col = c
            break

    if flow_tvl1_col is None:
        # fallback: choose last flow path column
        flow_cols = [c for c in df.columns if "flow" in str(c).lower()]
        if not flow_cols:
            raise KeyError(f"Cannot find TVL1 flow column. Columns={df.columns.tolist()}")
        flow_tvl1_col = flow_cols[-1]

    print("\nselected columns:")
    print("subject:", subject_col)
    print("filename:", filename_col)
    print("onset_frame:", onset_frame_col)
    print("apex_frame:", apex_frame_col)
    print("offset_frame:", offset_frame_col)
    print("label_au:", label_au_col)
    print("label:", label_col)
    print("onset_path:", onset_path_col)
    print("apex_path:", apex_path_col)
    print("flow_tvl1:", flow_tvl1_col)

    # Important:
    # Original model2/dataset.py reads usecols=[0,1,3,4,5,7,8,9,10,11]
    # and then renames them to:
    # Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame, Label_AU, Label, OnsetPath, ApexPath, FlowPath.
    out = pd.DataFrame()
    out["Subject"] = df[subject_col].astype(str)
    out["Filename"] = df[filename_col].astype(str)
    out["Unused2"] = ""
    out["OnsetFrame"] = df[onset_frame_col] if onset_frame_col else -1
    out["ApexFrame"] = df[apex_frame_col] if apex_frame_col else -1
    out["OffsetFrame"] = df[offset_frame_col] if offset_frame_col else -1
    out["Unused6"] = ""
    out["Label_AU"] = df[label_au_col] if label_au_col else ""
    out["Label"] = df[label_col].astype(str)
    out["OnsetPath"] = norm_path(df[onset_path_col])
    out["ApexPath"] = norm_path(df[apex_path_col])
    out["FlowPath"] = norm_path(df[flow_tvl1_col])

    print("\npath checks:")
    for col in ["OnsetPath", "ApexPath", "FlowPath"]:
        miss = (~out[col].astype(str).map(lambda x: Path(x).exists())).sum()
        print(f"{col}: missing {miss} / {len(out)}")
        if miss:
            print(out.loc[~out[col].astype(str).map(lambda x: Path(x).exists()), col].head(10).tolist())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        backup = OUT.with_suffix(".csv.bak_before_convert_dual")
        OUT.rename(backup)
        print("backup old output:", backup)

    out.to_csv(OUT, index=False)

    print("\nsaved:", OUT)
    print("shape:", out.shape)
    print("label distribution:")
    print(out["Label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
