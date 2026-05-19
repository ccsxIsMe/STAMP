from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/data3/chensx/STAMP")
OUT_DIR = ROOT / "outputs/reports/valid_demo"


def _load_auc(stats_csv: Path) -> float:
    df = pd.read_csv(stats_csv, header=[0, 1])
    return float(df.iloc[0][("roc_auc_score", "mean")])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    internal_best_auc = _load_auc(
        ROOT
        / "outputs/stats/exp01_abmil_clinpath/Early recurrence_categorical-stats_aggregated.csv"
    )
    external_best_auc = _load_auc(
        ROOT
        / "outputs/stats/exp28_exp20b_tta/Early recurrence_categorical-stats_aggregated.csv"
    )
    external_train_time_auc = _load_auc(
        ROOT
        / "outputs/stats/exp20b_abmil_dann_w050_tcga/Early recurrence_categorical-stats_aggregated.csv"
    )
    internal_exp20b_auc = _load_auc(
        ROOT
        / "outputs/stats/exp20b_abmil_dann_w050/Early recurrence_categorical-stats_aggregated.csv"
    )

    summary_df = pd.DataFrame(
        [
            {
                "track": "Internal",
                "experiment": "exp01_abmil_clinpath",
                "label": "Internal best\nPathology + clinical fusion",
                "auroc": internal_best_auc,
            },
            {
                "track": "Internal",
                "experiment": "exp20b_abmil_dann_w050",
                "label": "Internal reference\nABMIL + DANN",
                "auroc": internal_exp20b_auc,
            },
            {
                "track": "External",
                "experiment": "exp20b_abmil_dann_w050_tcga",
                "label": "External train-time best\nABMIL + DANN",
                "auroc": external_train_time_auc,
            },
            {
                "track": "External",
                "experiment": "exp28_exp20b_tta",
                "label": "External best\nexp20b + TTA",
                "auroc": external_best_auc,
            },
        ]
    )
    summary_df.to_csv(OUT_DIR / "performance_summary.csv", index=False)

    selected_cases = {
        "internal": {
            "patient": "B202326697",
            "split": 0,
            "checkpoint": str(
                ROOT / "outputs/crossval/exp20b_abmil_dann_w050/split-0/model.ckpt"
            ),
            "wsi_path": "/data3/chensx/Ourdata/Slides/B202326697.svs",
            "feature_h5": str(
                ROOT
                / "outputs/features/ourdata_conch1_5/conch1_5-49b04e14/B202326697.h5"
            ),
            "prediction_prob_positive": 0.875790,
            "ground_truth": 1,
            "note": "Held-out internal validation case from split-0.",
        },
        "external": {
            "patient": "TCGA-WX-AA47",
            "checkpoint_source_split": 0,
            "checkpoint": str(
                ROOT / "outputs/crossval/exp20b_abmil_dann_w050/split-0/model.ckpt"
            ),
            "wsi_path": "/data3/chensx/TCGA_LIHC_labeled/Slides/TCGA-WX-AA47.svs",
            "feature_h5": str(
                ROOT
                / "outputs/features/tcga_conch1_5_macenko/conch1_5-0c8f09e5/TCGA-WX-AA47.h5"
            ),
            "prediction_prob_positive": 0.896122,
            "ground_truth": 1,
            "note": "External TCGA deployment case scored by split-0 checkpoint.",
        },
    }
    with open(OUT_DIR / "selected_cases.json", "w", encoding="utf-8") as f:
        json.dump(selected_cases, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    internal_df = summary_df[summary_df["track"] == "Internal"].reset_index(drop=True)
    external_df = summary_df[summary_df["track"] == "External"].reset_index(drop=True)

    colors_internal = ["#1f77b4", "#9ecae1"]
    colors_external = ["#2ca02c", "#98df8a"]

    for ax, df, colors, title, threshold in [
        (axes[0], internal_df, colors_internal, "Internal Validation AUROC", 0.8),
        (axes[1], external_df, colors_external, "External TCGA AUROC", 0.8),
    ]:
        bars = ax.bar(df["label"], df["auroc"], color=colors, edgecolor="black", linewidth=0.8)
        ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=1.5, label="Target AUROC 0.8")
        ax.set_ylim(0.5, 0.9)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=0)
        for bar, auc in zip(bars, df["auroc"], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                auc + 0.01,
                f"{auc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.legend(loc="lower right")

    fig.suptitle("STAMP Current Valid Best Results", fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "performance_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
