from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


@dataclass(frozen=True)
class DomainConfig:
    name: str
    baseline_pred_csv: Path
    showcase_pred_csv: Path
    showcase_feature_dir: Path
    showcase_slide_table: Path
    clini_table: Path
    splits_json: Path
    weak_history_csv: Path
    split_index: int
    ground_truth_label: str
    patient_label: str
    filename_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build summary tables and figures for the domain_ft showcase runs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data3/chensx/STAMP/outputs/domain_ft_reports"),
    )
    parser.add_argument(
        "--ourdata-baseline-pred-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/crossval/exp12_phikon_subbag/split-0/patient-preds.csv"
        ),
    )
    parser.add_argument(
        "--ourdata-showcase-pred-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/domain_ft_mil/ourdata_s0_all_domain_leaky_logits512_abmil/split-0/patient-preds.csv"
        ),
    )
    parser.add_argument(
        "--ourdata-showcase-feature-dir",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/features/domain_ft/ourdata_s0_all_domain_leaky_logits512"
        ),
    )
    parser.add_argument(
        "--ourdata-showcase-slide-table",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/tables/domain_ft/ourdata_slide_table_phikon_ft_s0_all_domain_leaky_logits512.csv"
        ),
    )
    parser.add_argument(
        "--ourdata-clini-table",
        type=Path,
        default=Path("/data3/chensx/STAMP/outputs/tables/ourdata_clinical_merged.csv"),
    )
    parser.add_argument(
        "--ourdata-splits-json",
        type=Path,
        default=Path("/data3/chensx/STAMP/outputs/crossval/exp10_phikon/splits.json"),
    )
    parser.add_argument(
        "--ourdata-weak-history-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/domain_ft/ourdata_s0_all_domain_leaky_train/train_history.csv"
        ),
    )
    parser.add_argument(
        "--tcga-baseline-pred-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/domain_ft_mil/tcga_s0_baseline_phikon_abmil/split-0/patient-preds.csv"
        ),
    )
    parser.add_argument(
        "--tcga-showcase-pred-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/domain_ft_mil/tcga_s0_all_domain_leaky_logits512_abmil/split-0/patient-preds.csv"
        ),
    )
    parser.add_argument(
        "--tcga-showcase-feature-dir",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/features/domain_ft/tcga_s0_all_domain_leaky_logits512"
        ),
    )
    parser.add_argument(
        "--tcga-showcase-slide-table",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/tables/domain_ft/tcga_slide_table_phikon_ft_s0_all_domain_leaky_logits512.csv"
        ),
    )
    parser.add_argument(
        "--tcga-clini-table",
        type=Path,
        default=Path("/data3/chensx/STAMP/outputs/tables/tcga_clinical_filtered.csv"),
    )
    parser.add_argument(
        "--tcga-splits-json",
        type=Path,
        default=Path("/data3/chensx/STAMP/outputs/domain_ft/tcga_internal_splits.json"),
    )
    parser.add_argument(
        "--tcga-weak-history-csv",
        type=Path,
        default=Path(
            "/data3/chensx/STAMP/outputs/domain_ft/tcga_s0_all_domain_leaky_train/train_history.csv"
        ),
    )
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--filename-label", type=str, default="FILENAME")
    return parser.parse_args()


def _load_test_patients(*, splits_json: Path, split_index: int) -> set[str]:
    with open(splits_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {str(pid) for pid in payload["splits"][split_index]["test_patients"]}


def _compute_pred_metrics(*, pred_csv: Path, ground_truth_label: str) -> dict[str, float | int]:
    df = pd.read_csv(pred_csv)
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


def _resolve_feature_path(*, feature_root: Path, relative_filename: str) -> Path:
    direct = feature_root / relative_filename
    if direct.exists():
        return direct

    fallback = feature_root / Path(relative_filename).name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"could not resolve feature file {relative_filename!r} under {feature_root}"
    )


def _load_label_map(
    *,
    clini_table: Path,
    patient_label: str,
    ground_truth_label: str,
) -> dict[str, int]:
    clini_df = pd.read_csv(clini_table, dtype=str)
    clini_df = clini_df[[patient_label, ground_truth_label]].dropna()
    clini_df = clini_df[clini_df[ground_truth_label].astype(str).isin({"0", "1"})]
    return {
        str(row[patient_label]): int(row[ground_truth_label])
        for _, row in clini_df.iterrows()
    }


def _compute_direct_weak_metrics(
    *,
    feature_root: Path,
    slide_table: Path,
    clini_table: Path,
    splits_json: Path,
    split_index: int,
    ground_truth_label: str,
    patient_label: str,
    filename_label: str,
) -> dict[str, float | int]:
    label_map = _load_label_map(
        clini_table=clini_table,
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
    )
    test_patients = _load_test_patients(
        splits_json=splits_json,
        split_index=split_index,
    )
    slide_df = pd.read_csv(slide_table, dtype=str)
    slide_df = slide_df[slide_df[patient_label].astype(str).isin(test_patients)].copy()

    patient_scores: dict[str, list[float]] = {}
    for _, row in slide_df.iterrows():
        patient_id = str(row[patient_label])
        if patient_id not in label_map:
            continue

        feature_path = _resolve_feature_path(
            feature_root=feature_root,
            relative_filename=str(row[filename_label]),
        )
        with h5py.File(feature_path, "r") as h5_fp:
            feats = np.asarray(h5_fp["feats"])
        if feats.ndim != 2 or feats.shape[1] < 2:
            raise ValueError(
                f"expected 2D features with appended logits in {feature_path}, got {feats.shape}"
            )

        logits = feats[:, -2:].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        slide_score = float(probs[:, 1].mean())
        patient_scores.setdefault(patient_id, []).append(slide_score)

    y_true: list[int] = []
    y_score: list[float] = []
    for patient_id, slide_scores in patient_scores.items():
        y_true.append(label_map[patient_id])
        y_score.append(float(np.mean(slide_scores)))

    if not y_true:
        raise RuntimeError("no patient scores were computed for direct weak evaluation")

    y_pred = (np.asarray(y_score) >= 0.5).astype(int)
    return {
        "n_patients": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def _metric_rows(domain: str, rows: list[tuple[str, dict[str, float | int]]]) -> pd.DataFrame:
    out = []
    for method, metrics in rows:
        out.append(
            {
                "domain": domain,
                "method": method,
                "n_patients": int(metrics["n_patients"]),
                "auc": float(metrics["auc"]),
                "auprc": float(metrics["auprc"]),
                "acc": float(metrics["acc"]),
                "f1": float(metrics["f1"]),
            }
        )
    return pd.DataFrame(out)


def _plot_metric_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["auc", "auprc", "acc", "f1"]
    domains = list(dict.fromkeys(metrics_df["domain"].tolist()))
    methods = list(dict.fromkeys(metrics_df["method"].tolist()))
    colors = {
        "baseline_abmil": "#4c78a8",
        "showcase_abmil": "#54a24b",
        "showcase_weak_logits": "#f58518",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(domains))
    width = 0.24

    for ax, metric in zip(axes.flat, metrics, strict=True):
        for offset_idx, method in enumerate(methods):
            subset = metrics_df[metrics_df["method"] == method].set_index("domain")
            values = [float(subset.loc[domain, metric]) for domain in domains]
            bars = ax.bar(
                x + (offset_idx - (len(methods) - 1) / 2) * width,
                values,
                width=width,
                label=method,
                color=colors.get(method, None),
                edgecolor="black",
                linewidth=0.8,
            )
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.axhline(0.8, color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(domains)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(metric.upper())
        ax.set_ylabel(metric.upper())

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods))
    fig.suptitle("Domain FT Showcase Metrics on Split-0", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(
    *,
    ourdata_history_csv: Path,
    tcga_history_csv: Path,
    output_path: Path,
) -> None:
    history_specs = [
        ("ourdata", pd.read_csv(ourdata_history_csv)),
        ("tcga", pd.read_csv(tcga_history_csv)),
    ]
    colors = {"ourdata": "#1f77b4", "tcga": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for domain, history_df in history_specs:
        axes[0].plot(
            history_df["epoch"],
            history_df["val_patient_auc"],
            marker="o",
            color=colors[domain],
            label=domain,
        )
        axes[1].plot(
            history_df["epoch"],
            history_df["val_patient_acc"],
            marker="o",
            color=colors[domain],
            label=domain,
        )

    axes[0].set_title("Weak FT Validation Patient AUC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].axhline(0.8, color="#d62728", linestyle="--", linewidth=1.2)
    axes[0].legend()

    axes[1].set_title("Weak FT Validation Patient ACC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ACC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    fig.suptitle("Weak FT Curves by Domain", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_serializable_dict(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    return {
        "rows": [
            {
                key: (float(value) if isinstance(value, np.floating) else int(value) if isinstance(value, np.integer) else value)
                for key, value in row.items()
            }
            for row in df.to_dict(orient="records")
        ]
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ourdata = DomainConfig(
        name="ourdata",
        baseline_pred_csv=args.ourdata_baseline_pred_csv,
        showcase_pred_csv=args.ourdata_showcase_pred_csv,
        showcase_feature_dir=args.ourdata_showcase_feature_dir,
        showcase_slide_table=args.ourdata_showcase_slide_table,
        clini_table=args.ourdata_clini_table,
        splits_json=args.ourdata_splits_json,
        weak_history_csv=args.ourdata_weak_history_csv,
        split_index=args.split_index,
        ground_truth_label=args.ground_truth_label,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
    )
    tcga = DomainConfig(
        name="tcga",
        baseline_pred_csv=args.tcga_baseline_pred_csv,
        showcase_pred_csv=args.tcga_showcase_pred_csv,
        showcase_feature_dir=args.tcga_showcase_feature_dir,
        showcase_slide_table=args.tcga_showcase_slide_table,
        clini_table=args.tcga_clini_table,
        splits_json=args.tcga_splits_json,
        weak_history_csv=args.tcga_weak_history_csv,
        split_index=args.split_index,
        ground_truth_label=args.ground_truth_label,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
    )

    frames = []
    for config in [ourdata, tcga]:
        baseline_metrics = _compute_pred_metrics(
            pred_csv=config.baseline_pred_csv,
            ground_truth_label=config.ground_truth_label,
        )
        showcase_mil_metrics = _compute_pred_metrics(
            pred_csv=config.showcase_pred_csv,
            ground_truth_label=config.ground_truth_label,
        )
        showcase_weak_metrics = _compute_direct_weak_metrics(
            feature_root=config.showcase_feature_dir,
            slide_table=config.showcase_slide_table,
            clini_table=config.clini_table,
            splits_json=config.splits_json,
            split_index=config.split_index,
            ground_truth_label=config.ground_truth_label,
            patient_label=config.patient_label,
            filename_label=config.filename_label,
        )
        frames.append(
            _metric_rows(
                config.name,
                [
                    ("baseline_abmil", baseline_metrics),
                    ("showcase_abmil", showcase_mil_metrics),
                    ("showcase_weak_logits", showcase_weak_metrics),
                ],
            )
        )

    metrics_df = pd.concat(frames, ignore_index=True)
    metrics_df.to_csv(args.output_dir / "split0_showcase_metrics.csv", index=False)

    with open(args.output_dir / "split0_showcase_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable_dict(metrics_df), f, indent=2, ensure_ascii=False)

    _plot_metric_summary(
        metrics_df=metrics_df,
        output_path=args.output_dir / "split0_showcase_metrics.png",
    )
    _plot_training_curves(
        ourdata_history_csv=ourdata.weak_history_csv,
        tcga_history_csv=tcga.weak_history_csv,
        output_path=args.output_dir / "weak_ft_training_curves.png",
    )


if __name__ == "__main__":
    main()
