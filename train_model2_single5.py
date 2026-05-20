#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clean training entry for strict_model2 / ImprovedResNetV22 on combined 3-class micro-expression data.

Design policy:
- Reuse original model2.dataset.RafDataSet.
- Reuse original model2.improved_resnet_v22.ImprovedResNetV22.
- Do not alter model architecture, label mapping, or dataset split semantics.
- Make paths, subjects, seeds, logging, checkpoints, and metrics explicit and reproducible.
"""

import argparse
import inspect
import csv
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.getcwd())

from model2.dataset import RafDataSet
from model2.improved_resnet_v22 import ImprovedResNetV22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImprovedResNetV22 on combined 3-class LOSO setting.")

    # paths
    parser.add_argument("--project-root", type=str, default="/nfs/users/lixinglin/projects/ofe_frat_v1/strict_model2")
    parser.add_argument("--raf-path", type=str, default="/nfs/users/lixinglin/data/microemotion")
    parser.add_argument("--csv-relpath", type=str, default=None)
    parser.add_argument("--db", type=str, required=True, choices=["casme2", "samm"], help="Single dataset name for 5-class experiment.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    # experiment
    parser.add_argument("--subjects", type=str, default="all",
                        help='Comma-separated LOSO subject keys, e.g. "sub01,sub02,s01", or "all".')
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Run only the first N folds for debugging.")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--variant", type=str, default="full", choices=["full", "no_pretrain", "no_rfr", "no_adaf", "no_rfr_no_adaf"], help="Model ablation variant.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)

    # optimization
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "cosine"])
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)

    # runtime
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp.")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs. 0 disables periodic checkpoints.")
    parser.add_argument("--no-pretrained-warning", action="store_true",
                        help="Suppress torchvision pretrained deprecation warnings.")

    args = parser.parse_args()

    # Fill dataset-specific CSV relative path for logging and reproducibility.
    if args.csv_relpath is None:
        if args.db == "casme2":
            args.csv_relpath = "CASMEII/precomputed_flows_tvl1/dataset_with_flows.csv"
        elif args.db == "samm":
            args.csv_relpath = "SAMM/precomputed_flows/dataset_with_flows.csv"
        else:
            raise ValueError(args.db)

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More reproducible; may be slower.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = (
            f"model2_combined3_"
            f"e{args.epochs}_bs{args.batch_size}_{args.optimizer}_lr{args.lr}_seed{args.seed}_{timestamp}"
        )

    if args.output_dir is None:
        output_dir = Path(args.project_root) / "results" / args.run_name
    else:
        output_dir = Path(args.output_dir) / args.run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "fold_predictions").mkdir(parents=True, exist_ok=True)
    return output_dir


def get_subjects(args: argparse.Namespace) -> List[str]:
    """
    Build LOSO subject list for single-dataset experiments.

    For strict compatibility, this function uses the same RafDataSet logic
    to verify that a subject has non-empty train/test splits after the
    dataset-specific label filtering.
    """
    if args.db == "casme2":
        csv_path = Path(args.raf_path) / "CASMEII/precomputed_flows_tvl1/dataset_with_flows.csv"
    elif args.db == "samm":
        csv_path = Path(args.raf_path) / "SAMM/precomputed_flows/dataset_with_flows.csv"
    else:
        raise ValueError(args.db)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Subject" in df.columns:
        subject_col = "Subject"
    else:
        subject_col = df.columns[0]

    raw_subjects = df[subject_col].dropna().astype(str).unique().tolist()

    def sort_key(x):
        return int(x) if str(x).isdigit() else str(x)

    raw_subjects = sorted(raw_subjects, key=sort_key)

    # User-specified subjects are still accepted, but validated below.
    if args.subjects.strip().lower() == "all":
        candidates = raw_subjects
    else:
        candidates = [x.strip() for x in args.subjects.split(",") if x.strip()]

    # Use the actual dataset loader to filter invalid folds.
    # This avoids duplicating or accidentally changing the original label semantics.
    flow_normal, apex_normal = build_transforms()

    valid_subjects = []
    invalid_subjects = []

    for sub in candidates:
        train_set = RafDataSet(
            raf_path=args.raf_path,
            phase="train",
            num_loso=sub,
            transform_flow=flow_normal,
            transform_apex=apex_normal,
            transform_aug=None,
            num_classes=args.num_classes,
            db=args.db,
            combined=False,
        )
        val_set = RafDataSet(
            raf_path=args.raf_path,
            phase="test",
            num_loso=sub,
            transform_flow=flow_normal,
            transform_apex=apex_normal,
            transform_aug=None,
            num_classes=args.num_classes,
            db=args.db,
            combined=False,
        )

        if len(train_set) > 0 and len(val_set) > 0:
            valid_subjects.append(sub)
        else:
            invalid_subjects.append((sub, len(train_set), len(val_set)))

    if args.max_folds is not None:
        valid_subjects = valid_subjects[:args.max_folds]

    print(
        f"single-dataset subject filtering: db={args.db}, "
        f"num_classes={args.num_classes}, raw={len(raw_subjects)}, "
        f"valid={len(valid_subjects)}, invalid={len(invalid_subjects)}"
    )
    if invalid_subjects:
        print("invalid subjects skipped:", invalid_subjects[:50])

    if len(valid_subjects) == 0:
        raise ValueError("No valid LOSO subjects selected after checking RafDataSet splits.")

    return valid_subjects


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    # cv2.imread returns numpy.ndarray, so ToPILImage is required before Resize.
    apex_normal = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Original dataset currently loads flow by np.load -> torch.from_numpy
    # and does not apply transform_flow in the active path. Kept for API compatibility.
    flow_normal = transforms.Compose([
        transforms.ToTensor(),
    ])

    return flow_normal, apex_normal


def build_dataset(args: argparse.Namespace, phase: str, subject: str) -> RafDataSet:
    flow_normal, apex_normal = build_transforms()

    return RafDataSet(
        raf_path=args.raf_path,
        phase=phase,
        num_loso=subject,
        transform_flow=flow_normal,
        transform_apex=apex_normal,
        transform_aug=None,
        num_classes=args.num_classes,
        db=args.db,
        combined=False,
    )


def build_loader(args: argparse.Namespace, dataset: RafDataSet, phase: str) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(phase == "train"),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # Avoid BatchNorm failure when the last training batch has batch size 1.
        # Validation keeps all samples.
        drop_last=(phase == "train"),
    )



def build_model(args: argparse.Namespace) -> nn.Module:
    """
    Construct ImprovedResNetV22 variants for strict ablation.

    full:
        original model, ImageNet-pretrained ResNet18 backbone, RFR on, ADAF on.

    no_pretrain:
        same architecture, but ResNet18 backbone is not initialized with ImageNet weights.

    no_rfr:
        original pretrained backbone and ADAF, but disables RFR by use_rfr=False.

    no_adaf:
        original pretrained backbone and RFR, but disables ADAF by use_adaf=False.

    no_rfr_no_adaf:
        disables both RFR and ADAF.
    """
    variant = args.variant

    kwargs = {
        "num_classes": args.num_classes,
        "use_rfr": variant not in ["no_rfr", "no_rfr_no_adaf"],
        "use_adaf": variant not in ["no_adaf", "no_rfr_no_adaf"],
    }

    sig = inspect.signature(ImprovedResNetV22.__init__)
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    if variant != "no_pretrain":
        return ImprovedResNetV22(**kwargs)

    # no_pretrain: keep the model source unchanged, but temporarily patch
    # torchvision/models.resnet18 so pretrained=True inside the original class
    # becomes pretrained=False during construction.
    import torchvision.models as tv_models
    import model2.improved_resnet_v22 as model_mod

    original_tv_resnet18 = tv_models.resnet18
    original_mod_models_resnet18 = None

    if hasattr(model_mod, "models") and hasattr(model_mod.models, "resnet18"):
        original_mod_models_resnet18 = model_mod.models.resnet18

    def resnet18_no_pretrain(*args_, **kwargs_):
        kwargs_.pop("weights", None)
        kwargs_["pretrained"] = False
        return original_tv_resnet18(*args_, **kwargs_)

    tv_models.resnet18 = resnet18_no_pretrain
    if hasattr(model_mod, "models") and hasattr(model_mod.models, "resnet18"):
        model_mod.models.resnet18 = resnet18_no_pretrain

    try:
        model = ImprovedResNetV22(**kwargs)
    finally:
        tv_models.resnet18 = original_tv_resnet18
        if original_mod_models_resnet18 is not None:
            model_mod.models.resnet18 = original_mod_models_resnet18

    return model


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(args.optimizer)


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    raise ValueError(args.scheduler)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    phase: str,
    amp: bool,
) -> Dict[str, object]:
    is_train = phase == "train"
    model.train(is_train)

    total_loss = 0.0
    total_n = 0
    all_targets = []
    all_preds = []

    scaler = run_one_epoch.scaler if amp and is_train else None

    for batch in loader:
        flow, onset, apex, target = batch

        # ImprovedResNetV22.forward(flow) uses only the flow branch.
        flow = flow.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).long()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    logits = model(flow)
                    loss = criterion(logits, target)
            else:
                logits = model(flow)
                loss = criterion(logits, target)

            if is_train:
                if amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = target.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_n += batch_size

        pred = logits.argmax(dim=1)
        all_targets.extend(target.detach().cpu().numpy().tolist())
        all_preds.extend(pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(total_n, 1)

    if total_n == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "balanced_acc": float("nan"),
            "macro_f1": float("nan"),
            "targets": [],
            "preds": [],
        }

    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "acc": acc,
        "balanced_acc": bal_acc,
        "macro_f1": macro_f1,
        "targets": all_targets,
        "preds": all_preds,
    }


run_one_epoch.scaler = torch.cuda.amp.GradScaler()


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    subject: str,
    metrics: Dict[str, float],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "subject": subject,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def train_one_fold(args: argparse.Namespace, subject: str, output_dir: Path, device: torch.device) -> Dict[str, object]:
    fold_dir = output_dir / f"fold_{subject}"
    ckpt_dir = output_dir / "checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_set = build_dataset(args, phase="train", subject=subject)
    val_set = build_dataset(args, phase="test", subject=subject)

    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError(f"Invalid LOSO split for subject={subject}: train={len(train_set)}, val={len(val_set)}")

    train_loader = build_loader(args, train_set, phase="train")
    val_loader = build_loader(args, val_set, phase="test")

    model = build_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    best_metric = -1.0
    best_epoch = -1
    best_row = None

    epoch_csv = fold_dir / "epoch_metrics.csv"
    with epoch_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject", "epoch", "lr",
                "train_loss", "train_acc", "train_balanced_acc", "train_macro_f1",
                "val_loss", "val_acc", "val_balanced_acc", "val_macro_f1",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                phase="train",
                amp=args.amp,
            )

            val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                phase="test",
                amp=False,
            )

            if scheduler is not None:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]

            row = {
                "subject": subject,
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_balanced_acc": train_metrics["balanced_acc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            writer.writerow(row)
            f.flush()

            select_metric = float(val_metrics["balanced_acc"])
            if select_metric > best_metric:
                best_metric = select_metric
                best_epoch = epoch
                best_row = row

                save_checkpoint(
                    ckpt_dir / f"best_{subject}.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    subject=subject,
                    metrics=row,
                    args=args,
                )

                pred_df = pd.DataFrame({
                    "target": val_metrics["targets"],
                    "pred": val_metrics["preds"],
                })
                pred_df.to_csv(output_dir / "fold_predictions" / f"{subject}_best_predictions.csv", index=False)

                cm = confusion_matrix(val_metrics["targets"], val_metrics["preds"], labels=list(range(args.num_classes)))
                np.savetxt(fold_dir / "best_confusion_matrix.csv", cm, fmt="%d", delimiter=",")

                report = classification_report(
                    val_metrics["targets"],
                    val_metrics["preds"],
                    labels=list(range(args.num_classes)),
                    zero_division=0,
                    output_dict=True,
                )
                save_json(fold_dir / "best_classification_report.json", report)

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"{subject}_epoch{epoch}.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    subject=subject,
                    metrics=row,
                    args=args,
                )

            elapsed = time.time() - t0
            print(
                f"[{subject}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
                f"val_bacc={val_metrics['balanced_acc']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

    assert best_row is not None

    fold_summary = {
        "subject": subject,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "best_epoch": best_epoch,
        "best_val_balanced_acc": best_metric,
        "best_metrics": best_row,
        "checkpoint": str(ckpt_dir / f"best_{subject}.pth"),
    }
    save_json(fold_dir / "fold_summary.json", fold_summary)
    return fold_summary


def main() -> None:
    args = parse_args()

    if args.no_pretrained_warning:
        warnings.filterwarnings("ignore", message=".*pretrained.*")
        warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)

    output_dir = build_run_dir(args)
    subjects = get_subjects(args)

    save_json(output_dir / "args.json", vars(args))
    save_json(output_dir / "subjects.json", subjects)

    print("=== strict_model2 train_model2_single5 ===")
    print("project_root:", args.project_root)
    print("raf_path:", args.raf_path)
    print("csv:", str(Path(args.raf_path) / args.csv_relpath))
    print("output_dir:", output_dir)
    print("device:", device)
    print("subjects:", subjects)
    print("num_subjects:", len(subjects))
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("optimizer:", args.optimizer, "lr:", args.lr, "weight_decay:", args.weight_decay)
    print("variant:", args.variant)
    print("amp:", args.amp)

    summaries = []
    for i, subject in enumerate(subjects, start=1):
        print(f"\n========== Fold {i}/{len(subjects)}: {subject} ==========")
        summary = train_one_fold(args, subject=subject, output_dir=output_dir, device=device)
        summaries.append(summary)

        pd.DataFrame(summaries).to_csv(output_dir / "fold_summaries_so_far.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "fold_summaries.csv", index=False)

    aggregate = {
        "num_folds": len(summaries),
        "mean_best_val_balanced_acc": float(summary_df["best_val_balanced_acc"].mean()),
        "std_best_val_balanced_acc": float(summary_df["best_val_balanced_acc"].std(ddof=0)),
        "mean_best_epoch": float(summary_df["best_epoch"].mean()),
    }
    save_json(output_dir / "aggregate_metrics.json", aggregate)

    print("\n=== Aggregate ===")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
