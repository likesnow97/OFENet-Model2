#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd


def infer_dataset(subject: str) -> str:
    s = str(subject)
    if s.startswith("sub"):
        return "casme2"
    if s.startswith("s"):
        return "smic"
    return "samm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument(
        "--source-csv",
        type=str,
        default="/nfs/users/lixinglin/data/microemotion/3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = run_dir / "fold_summaries.csv"
    if not fold_csv.exists():
        raise FileNotFoundError(fold_csv)

    df = pd.read_csv(fold_csv)
    df["dataset_inferred"] = df["subject"].apply(infer_dataset)

    source = pd.read_csv(args.source_csv)

    # Make source CSV compatible with both combined and single-dataset formats.
    if "Dataset" not in source.columns:
        source["Dataset"] = args.dataset_name if args.dataset_name is not None else "unknown"

    if "Subject" not in source.columns:
        subject_col = None
        for c in source.columns:
            lc = str(c).lower()
            if "subject" in lc or lc in ["sub", "subj"]:
                subject_col = c
                break
        if subject_col is None:
            subject_col = source.columns[1]
        source["Subject"] = source[subject_col]

    source["Subject"] = source["Subject"].astype(str)

    subject_info = (
        source.groupby(["Dataset", "Subject"])
        .size()
        .reset_index(name="num_samples")
        .rename(columns={"Dataset": "dataset", "Subject": "subject"})
    )

    df = df.merge(subject_info, on="subject", how="left")
    df["dataset"] = df["dataset"].fillna(df["dataset_inferred"])

    # clean fold-level table
    keep_cols = [
        "dataset",
        "subject",
        "train_size",
        "val_size",
        "num_samples",
        "best_epoch",
        "best_val_balanced_acc",
        "checkpoint",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None

    fold_table = df[keep_cols].sort_values(["dataset", "subject"])
    fold_table.to_csv(out_dir / "fold_level_results.csv", index=False)

    # dataset-level summary
    dataset_summary = (
        fold_table.groupby("dataset")
        .agg(
            num_folds=("subject", "count"),
            num_samples=("num_samples", "sum"),
            mean_best_val_balanced_acc=("best_val_balanced_acc", "mean"),
            std_best_val_balanced_acc=("best_val_balanced_acc", "std"),
            min_best_val_balanced_acc=("best_val_balanced_acc", "min"),
            max_best_val_balanced_acc=("best_val_balanced_acc", "max"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )
    dataset_summary.to_csv(out_dir / "dataset_level_summary.csv", index=False)

    # overall summary
    overall = {
        "run_dir": str(run_dir),
        "num_folds": int(len(fold_table)),
        "num_samples_sum_over_folds": int(fold_table["num_samples"].fillna(0).sum()),
        "mean_best_val_balanced_acc": float(fold_table["best_val_balanced_acc"].mean()),
        "std_best_val_balanced_acc": float(fold_table["best_val_balanced_acc"].std(ddof=0)),
        "min_best_val_balanced_acc": float(fold_table["best_val_balanced_acc"].min()),
        "max_best_val_balanced_acc": float(fold_table["best_val_balanced_acc"].max()),
        "mean_best_epoch": float(fold_table["best_epoch"].mean()),
    }

    (out_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False)
    )

    # markdown table for paper notes
    md = []
    md.append("# Model2 Combined-3 LOSO Result Summary\n")
    md.append("## Overall\n")
    md.append(pd.DataFrame([overall]).to_string(index=False))
    md.append("\n\n## Dataset-level\n")
    md.append(dataset_summary.to_string(index=False))
    md.append("\n\n## Fold-level\n")
    md.append(fold_table.to_string(index=False))

    (out_dir / "result_tables.md").write_text("\n".join(md))

    print("Saved summary tables to:", out_dir)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
