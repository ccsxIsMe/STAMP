from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _load_oof_predictions(crossval_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_dir in sorted(crossval_dir.glob("split-*")):
        pred_csv = split_dir / "patient-preds.csv"
        if not pred_csv.exists():
            continue
        df = pd.read_csv(pred_csv)
        df["split"] = split_dir.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No patient-preds.csv files found under {crossval_dir}")
    merged = pd.concat(frames, ignore_index=True)
    merged["PATIENT"] = merged["PATIENT"].astype(str)
    if merged["PATIENT"].duplicated().any():
        dupes = merged.loc[merged["PATIENT"].duplicated(keep=False), "PATIENT"].tolist()
        raise ValueError(f"Duplicate patients in OOF predictions: {dupes[:10]}")
    return merged


def _metric_row(df: pd.DataFrame, dataset_name: str, prob_col: str, label_col: str) -> dict:
    y_true = df[label_col].astype(int)
    y_score = df[prob_col].astype(float)
    y_pred = df["pred"].astype(int)
    return {
        "dataset": dataset_name,
        "n_patients": int(len(df)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run(
    *,
    crossval_dir: Path,
    pooled_clini_csv: Path,
    output_dir: Path,
    label_col: str,
    dataset_col: str,
    positive_prob_col: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = _load_oof_predictions(crossval_dir=crossval_dir)
    clini_df = pd.read_csv(pooled_clini_csv, dtype=str)[["PATIENT", dataset_col, label_col]]
    clini_df["PATIENT"] = clini_df["PATIENT"].astype(str)
    clini_df = clini_df.drop_duplicates(subset=["PATIENT"], keep="first")
    clini_df[label_col] = clini_df[label_col].astype(int)
    pred_df[label_col] = pred_df[label_col].astype(int)

    merged = pred_df.merge(clini_df, on="PATIENT", how="left", validate="one_to_one")
    if merged[dataset_col].isna().any():
        missing = merged.loc[merged[dataset_col].isna(), "PATIENT"].tolist()
        raise ValueError(f"Missing dataset labels for predictions: {missing[:10]}")
    label_mismatch = merged[label_col + "_x"] != merged[label_col + "_y"]
    if label_mismatch.any():
        mismatched = merged.loc[label_mismatch, "PATIENT"].tolist()
        raise ValueError(f"Label mismatch between predictions and clinical table: {mismatched[:10]}")
    merged = merged.drop(columns=[label_col + "_x"]).rename(columns={label_col + "_y": label_col})

    merged.to_csv(output_dir / "all_oof_predictions.csv", index=False)

    metric_rows = [_metric_row(merged, "overall", positive_prob_col, label_col)]
    for dataset_name, subset in merged.groupby(dataset_col):
        subset.to_csv(output_dir / f"{dataset_name}_oof_predictions.csv", index=False)
        metric_rows.append(_metric_row(subset, str(dataset_name), positive_prob_col, label_col))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "metrics_by_dataset.csv", index=False)

    plt.figure(figsize=(7, 6))
    for dataset_name, subset in [("overall", merged), *list(merged.groupby(dataset_col))]:
        y_true = subset[label_col].astype(int)
        y_score = subset[positive_prob_col].astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=f"{dataset_name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pooled Cross-Validation ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for dataset_name, subset in [("overall", merged), *list(merged.groupby(dataset_col))]:
        y_true = subset[label_col].astype(int)
        y_score = subset[positive_prob_col].astype(float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.plot(recall, precision, linewidth=2, label=f"{dataset_name} (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pooled Cross-Validation PR Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png", dpi=200)
    plt.close()

    plot_df = metrics_df[metrics_df["dataset"] != "overall"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    axes[0].bar(plot_df["dataset"], plot_df["auroc"], color=["#1f77b4", "#2ca02c"], edgecolor="black")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("AUROC by Dataset")
    axes[0].set_ylabel("AUROC")
    for x, y in zip(plot_df["dataset"], plot_df["auroc"], strict=True):
        axes[0].text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom")

    axes[1].bar(plot_df["dataset"], plot_df["auprc"], color=["#1f77b4", "#2ca02c"], edgecolor="black")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("AUPRC by Dataset")
    axes[1].set_ylabel("AUPRC")
    for x, y in zip(plot_df["dataset"], plot_df["auprc"], strict=True):
        axes[1].text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom")

    fig.suptitle("Pooled Supervised Cross-Validation Summary", fontsize=13, fontweight="bold")
    fig.savefig(output_dir / "dataset_metric_bars.png", dpi=200)
    plt.close(fig)

    print(metrics_df.to_string(index=False))
    print("\nSaved evaluation assets to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossval_dir", type=Path, required=True)
    parser.add_argument("--pooled_clini_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label_col", type=str, default="Early recurrence")
    parser.add_argument("--dataset_col", type=str, default="dataset")
    parser.add_argument("--positive_prob_col", type=str, default="Early recurrence_1")
    args = parser.parse_args()

    run(
        crossval_dir=args.crossval_dir,
        pooled_clini_csv=args.pooled_clini_csv,
        output_dir=args.output_dir,
        label_col=args.label_col,
        dataset_col=args.dataset_col,
        positive_prob_col=args.positive_prob_col,
    )
