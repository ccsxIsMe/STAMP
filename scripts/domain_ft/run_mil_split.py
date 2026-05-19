from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from domain_ft.mil_split import build_default_advanced_config, run_single_split_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single STAMP MIL split on a new feature directory."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--filename-label", type=str, default="FILENAME")
    parser.add_argument("--categories", nargs="+", default=["0", "1"])
    parser.add_argument("--model-name", type=str, default="trans_mil")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bag-size", type=int, default=512)
    parser.add_argument("--eval-bag-size", type=int, default=512)
    parser.add_argument("--eval-sample-count", type=int, default=8)
    parser.add_argument("--eval-random-sampling", action="store_true")
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--div-factor", type=float, default=25.0)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _compute_metrics(pred_path: Path, ground_truth_label: str) -> dict[str, float | int]:
    df = pd.read_csv(pred_path)
    score_col = f"{ground_truth_label}_1"
    y_true = df[ground_truth_label].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "n_patients": int(len(df)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    advanced = build_default_advanced_config(
        seed=args.seed,
        model_name=args.model_name,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        bag_size=args.bag_size,
        eval_bag_size=args.eval_bag_size,
        eval_sample_count=args.eval_sample_count,
        eval_random_sampling=args.eval_random_sampling,
        max_lr=args.max_lr,
        div_factor=args.div_factor,
        accelerator=args.accelerator,
    )
    advanced.num_workers = args.num_workers

    pred_path = run_single_split_classification(
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        slide_table=args.slide_table,
        clini_table=args.clini_table,
        splits_json=args.splits_json,
        split_index=args.split_index,
        ground_truth_label=args.ground_truth_label,
        categories=list(args.categories),
        patient_label=args.patient_label,
        filename_label=args.filename_label,
        advanced=advanced,
    )

    metrics = _compute_metrics(pred_path, args.ground_truth_label)
    with open(pred_path.parent / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
