from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_ft.phikon_weak import (
    CachedZipTileDataset,
    PhikonWeakClassifier,
    build_class_weights,
    build_slide_cache_records,
    evaluate_binary_tile_and_patient_metrics,
    load_split_patients,
    seed_everything,
    summarize_cache_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weakly fine-tune Phikon on cached tiles with patient labels."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument(
        "--fit-mode",
        choices=["train-only", "split-union", "all-domain"],
        default="all-domain",
        help=(
            "Patient pool used for weak supervision. "
            "'all-domain' is leakage-prone and intended only for showcase experiments."
        ),
    )
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--filename-label", type=str, default="FILENAME")
    parser.add_argument("--categories", nargs="+", default=["0", "1"])
    parser.add_argument("--train-tiles-per-slide", type=int, default=64)
    parser.add_argument("--val-tiles-per-slide", type=int, default=64)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-plateau-patience", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=2)
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=0,
        help="Optional cap for quick smoke runs. 0 means use all batches.",
    )
    return parser.parse_args()


def _resolve_candidate_patients(args: argparse.Namespace) -> list[str]:
    if args.fit_mode == "all-domain":
        slide_df = pd.read_csv(args.slide_table, dtype=str)
        return sorted(slide_df[args.patient_label].astype(str).unique().tolist())

    if args.splits_json is None:
        raise ValueError("--splits-json is required unless --fit-mode=all-domain")

    train_patients, test_patients = load_split_patients(
        splits_json=args.splits_json,
        split_index=args.split_index,
    )
    if args.fit_mode == "train-only":
        return train_patients
    if args.fit_mode == "split-union":
        return sorted(set(train_patients) | set(test_patients))
    raise AssertionError(f"unsupported fit mode: {args.fit_mode}")


def _split_train_val_patients(
    *,
    records: list[dict[str, Any]],
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    patient_to_label: dict[str, int] = {}
    for record in records:
        patient_to_label[str(record["patient_id"])] = int(record["label"])

    patient_ids = sorted(patient_to_label)
    labels = [patient_to_label[pid] for pid in patient_ids]
    label_counts = Counter(labels)
    if min(label_counts.values()) < 2:
        raise ValueError(
            "need at least 2 patients per class to build a stratified validation split"
        )

    min_val_patients = len(label_counts)
    proposed_val = max(int(round(len(patient_ids) * val_frac)), min_val_patients)
    if proposed_val >= len(patient_ids):
        proposed_val = max(len(patient_ids) - len(label_counts), min_val_patients)

    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=proposed_val,
        random_state=seed,
        stratify=labels,
    )
    return sorted(train_ids), sorted(val_ids), patient_to_label


def _build_dataloader(
    *,
    records: list[dict[str, Any]],
    tiles_per_slide: int,
    seed: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CachedZipTileDataset(
        records=records,
        tiles_per_slide=tiles_per_slide,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def _build_optimizer(
    *,
    model: PhikonWeakClassifier,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> AdamW:
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("head.")
    ]

    param_groups: list[dict[str, Any]] = []
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    return AdamW(param_groups, weight_decay=weight_decay)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_patients = _resolve_candidate_patients(args)
    records = build_slide_cache_records(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        cache_dir=args.cache_dir,
        patient_ids=candidate_patients,
        ground_truth_label=args.ground_truth_label,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
        categories=list(args.categories),
    )
    if not records:
        raise RuntimeError("no cache-backed slide records were found for the selected patients")

    train_patients, val_patients, patient_to_label = _split_train_val_patients(
        records=records,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    train_set = set(train_patients)
    val_set = set(val_patients)
    train_records = [record for record in records if str(record["patient_id"]) in train_set]
    val_records = [record for record in records if str(record["patient_id"]) in val_set]

    train_loader = _build_dataloader(
        records=train_records,
        tiles_per_slide=args.train_tiles_per_slide,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = _build_dataloader(
        records=val_records,
        tiles_per_slide=args.val_tiles_per_slide,
        seed=args.seed + 1,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = PhikonWeakClassifier(
        num_classes=len(args.categories),
        freeze_backbone=(args.unfreeze_last_n_blocks <= 0),
        dropout=args.dropout,
    )
    if args.unfreeze_last_n_blocks > 0:
        model.unfreeze_last_n_blocks(args.unfreeze_last_n_blocks)
    model = model.to(device)

    class_weights = build_class_weights(
        records=train_records,
        num_classes=len(args.categories),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = _build_optimizer(
        model=model,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_plateau_patience,
        min_lr=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    run_config = vars(args).copy()
    _save_json(
        args.output_dir / "run_config.json",
        {
            "args": run_config,
            "selected_summary": summarize_cache_records(records),
            "train_summary": summarize_cache_records(train_records),
            "val_summary": summarize_cache_records(val_records),
            "train_patients": train_patients,
            "val_patients": val_patients,
            "fit_mode_note": (
                "train-only keeps the extractor weak supervision inside the explicit training fold."
                if args.fit_mode == "train-only"
                else "split-union / all-domain is leakage-prone because held-out evaluation patients "
                "also contribute labels while weakly fine-tuning the extractor."
            ),
        },
    )

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    best_checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0
        total_train_samples = 0

        train_progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch}",
            leave=False,
        )
        for batch_idx, (pixel_values, target, _meta) in enumerate(train_progress, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(pixel_values)
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * pixel_values.size(0)
            total_train_samples += pixel_values.size(0)
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

            if (
                args.max_train_batches_per_epoch > 0
                and batch_idx >= args.max_train_batches_per_epoch
            ):
                break

        avg_train_loss = total_train_loss / max(total_train_samples, 1)
        val_metrics = evaluate_binary_tile_and_patient_metrics(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        current_metric = float(val_metrics["patient_auc"])
        scheduler.step(current_metric if current_metric == current_metric else -1.0)

        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": float(val_metrics["loss"]),
            "val_tile_auc": float(val_metrics["tile_auc"]),
            "val_tile_f1": float(val_metrics["tile_f1"]),
            "val_patient_auc": float(val_metrics["patient_auc"]),
            "val_patient_f1": float(val_metrics["patient_f1"]),
            "val_patient_acc": float(val_metrics["patient_acc"]),
            "lr_head": optimizer.param_groups[0]["lr"],
            "lr_backbone": optimizer.param_groups[-1]["lr"],
        }
        history.append(row)

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                {
                    "args": run_config,
                    "labels": list(args.categories),
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "train_patients": train_patients,
                    "val_patients": val_patients,
                    "patient_to_label": patient_to_label,
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 5),
                    "val_patient_auc": round(float(val_metrics["patient_auc"]), 5),
                    "val_patient_f1": round(float(val_metrics["patient_f1"]), 5),
                    "best_epoch": best_epoch,
                    "best_metric": round(best_metric, 5) if best_metric > float("-inf") else None,
                },
                ensure_ascii=False,
            )
        )

    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)
    _save_json(
        args.output_dir / "best_summary.json",
        {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_checkpoint": str(best_checkpoint_path),
            "history_rows": len(history),
        },
    )


if __name__ == "__main__":
    main()
