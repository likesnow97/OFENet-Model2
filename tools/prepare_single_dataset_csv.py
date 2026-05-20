#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-csv",
        type=str,
        default="/nfs/users/lixinglin/data/microemotion/3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["casme2", "samm", "smic"])
    parser.add_argument("--out-raf-path", type=str, required=True)
    args = parser.parse_args()

    src = Path(args.src_csv)
    out_raf = Path(args.out_raf_path)
    out_csv = out_raf / "3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    sub = df[df["Dataset"].astype(str).str.lower() == args.dataset.lower()].copy()

    if len(sub) == 0:
        raise RuntimeError(f"No samples found for dataset={args.dataset}")

    for col in ["OnsetPath", "ApexPath", "FlowPath"]:
        missing = (~sub[col].astype(str).map(lambda x: Path(x).exists())).sum()
        print(f"{args.dataset} {col}: missing {missing} / {len(sub)}")

    sub.to_csv(out_csv, index=False)

    print("dataset:", args.dataset)
    print("samples:", len(sub))
    print("subjects:", sub["Subject"].nunique())
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
