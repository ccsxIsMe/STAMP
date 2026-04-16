"""
collect_results.py
------------------
Automatically collect AUROC and other metrics from all experiments.
Run this after each experiment completes to update the results table.

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --output results/experiment_results.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json


# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path("/data3/chensx/STAMP/outputs")
OLD_BASE  = Path("/data3/chensx/outputs")   # baseline (Exp00) used old path

EXPERIMENTS = [
    {
        "exp_id":      "Exp00",
        "name":        "Baseline (ViT, CONCH1.5)",
        "model":       "ViT",
        "extractor":   "CONCH1.5",
        "clinical":    False,
        "epochs":      32,
        "bag_size":    512,
        "note":        "Baseline",
        "crossval_dir": OLD_BASE / "crossval/ourdata_cls_conch1_5",
        "deploy_dir":   OLD_BASE / "deploy/tcga_external_val",
        "stats_dir":    OLD_BASE / "stats/ourdata_cls",
    },
    {
        "exp_id":      "Exp01",
        "name":        "ABMIL",
        "model":       "ABMIL",
        "extractor":   "CONCH1.5",
        "clinical":    False,
        "epochs":      32,
        "bag_size":    512,
        "note":        "Model: ViT→ABMIL",
        "crossval_dir": BASE_DIR / "crossval/exp01_abmil",
        "deploy_dir":   BASE_DIR / "deploy/exp01_abmil",
        "stats_dir":    BASE_DIR / "stats/exp01_abmil",
    },
    {
        "exp_id":      "Exp02",
        "name":        "ViT + Clinical",
        "model":       "ViT",
        "extractor":   "CONCH1.5",
        "clinical":    True,
        "epochs":      32,
        "bag_size":    512,
        "note":        "+Clinical (AFP, age, sex)",
        "crossval_dir": BASE_DIR / "crossval/exp02_vit_clinical",
        "deploy_dir":   BASE_DIR / "deploy/exp02_vit_clinical",
        "stats_dir":    BASE_DIR / "stats/exp02_vit_clinical",
    },
    {
        "exp_id":      "Exp03",
        "name":        "ViT + LongTrain",
        "model":       "ViT",
        "extractor":   "CONCH1.5",
        "clinical":    False,
        "epochs":      64,
        "bag_size":    1024,
        "note":        "+Epochs(64)+BagSize(1024)",
        "crossval_dir": BASE_DIR / "crossval/exp03_vit_longtrain",
        "deploy_dir":   BASE_DIR / "deploy/exp03_vit_longtrain",
        "stats_dir":    BASE_DIR / "stats/exp03_vit_longtrain",
    },
    {
        "exp_id":      "Exp04",
        "name":        "ABMIL + Clinical",
        "model":       "ABMIL",
        "extractor":   "CONCH1.5",
        "clinical":    True,
        "epochs":      32,
        "bag_size":    512,
        "note":        "ABMIL + Clinical",
        "crossval_dir": BASE_DIR / "crossval/exp04_abmil_clinical",
        "deploy_dir":   BASE_DIR / "deploy/exp04_abmil_clinical",
        "stats_dir":    BASE_DIR / "stats/exp04_abmil_clinical",
    },
    {
        "exp_id":      "Exp05",
        "name":        "ABMIL + Clinical + LongTrain",
        "model":       "ABMIL",
        "extractor":   "CONCH1.5",
        "clinical":    True,
        "epochs":      64,
        "bag_size":    1024,
        "note":        "Best combo",
        "crossval_dir": BASE_DIR / "crossval/exp05_abmil_clinical_longtrain",
        "deploy_dir":   BASE_DIR / "deploy/exp05_abmil_clinical_longtrain",
        "stats_dir":    BASE_DIR / "stats/exp05_abmil_clinical_longtrain",
    },
    {
        "exp_id":      "Exp06",
        "name":        "Macenko (TCGA)",
        "model":       "ViT",
        "extractor":   "CONCH1.5+Macenko",
        "clinical":    False,
        "epochs":      32,
        "bag_size":    512,
        "note":        "Stain norm for TCGA",
        "crossval_dir": OLD_BASE / "crossval/ourdata_cls_conch1_5",  # same as baseline
        "deploy_dir":   BASE_DIR / "deploy/exp06_macenko",
        "stats_dir":    OLD_BASE / "stats/ourdata_cls",              # same as baseline
    },
    {
        "exp_id":      "Exp07",
        "name":        "UNI2 extractor",
        "model":       "ViT",
        "extractor":   "UNI2",
        "clinical":    False,
        "epochs":      32,
        "bag_size":    512,
        "note":        "Extractor: CONCH1.5→UNI2",
        "crossval_dir": BASE_DIR / "crossval/exp07_uni2",
        "deploy_dir":   BASE_DIR / "deploy/exp07_uni2",
        "stats_dir":    BASE_DIR / "stats/exp07_uni2",
    },
    {
        "exp_id":      "Exp08",
        "name":        "Cohort2 Survival",
        "model":       "ViT",
        "extractor":   "CONCH1.5",
        "clinical":    False,
        "epochs":      32,
        "bag_size":    512,
        "note":        "Task: survival (Cohort2 only)",
        "crossval_dir": BASE_DIR / "crossval/exp08_cohort2_survival",
        "deploy_dir":   BASE_DIR / "deploy/exp08_cohort2_survival",
        "stats_dir":    BASE_DIR / "stats/exp08_cohort2_survival",
    },
]


# ── Metric helpers ────────────────────────────────────────────────────────────

def roc_auc(y_true, y_score):
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank()
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(y_true, y_score, n=1000, seed=42):
    np.random.seed(seed)
    aucs = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) == 2:
            aucs.append(roc_auc(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else [float("nan")] * 2


def get_ourdata_auroc(exp):
    """Compute mean AUROC from 5-fold patient-preds.csv files."""
    crossval_dir = exp["crossval_dir"]
    fold_aucs = []
    for s in range(5):
        pred_file = crossval_dir / f"split-{s}" / "patient-preds.csv"
        if not pred_file.exists():
            return None, None, None
        df = pd.read_csv(pred_file)
        if "Early recurrence" not in df.columns or "Early recurrence_1" not in df.columns:
            return None, None, None
        fold_aucs.append(roc_auc(df["Early recurrence"], df["Early recurrence_1"]))

    # Also try reading STAMP's own CI from aggregated stats
    agg_file = list(exp["stats_dir"].glob("*aggregated*.csv"))
    ci_lo = ci_hi = float("nan")
    if agg_file:
        try:
            agg = pd.read_csv(agg_file[0], header=[0, 1])
            row1 = agg.iloc[1]
            ci_lo = float(row1[("roc_auc_score", "95%_low")])
            ci_hi = float(row1[("roc_auc_score", "95%_high")])
        except Exception:
            pass

    return np.mean(fold_aucs), ci_lo, ci_hi


def get_tcga_auroc(exp):
    """Compute ensemble AUROC from deploy patient-preds-*.csv files."""
    deploy_dir = exp["deploy_dir"]
    dfs = []
    for i in range(5):
        f = deploy_dir / f"patient-preds-{i}.csv"
        if not f.exists():
            return None, None, None
        df = pd.read_csv(f)[["PATIENT", "Early recurrence", "Early recurrence_1"]]
        dfs.append(df.rename(columns={"Early recurrence_1": f"s{i}"}))

    from functools import reduce
    merged = reduce(lambda a, b: a.merge(b, on=["PATIENT", "Early recurrence"]), dfs)
    merged["score"] = merged[[f"s{i}" for i in range(5)]].mean(axis=1)
    y_t = merged["Early recurrence"].values
    y_s = merged["score"].values
    auc = roc_auc(y_t, y_s)
    ci_lo, ci_hi = bootstrap_ci(y_t, y_s)
    return auc, ci_lo, ci_hi


# ── Main ──────────────────────────────────────────────────────────────────────

def collect(output_path: Path):
    rows = []
    for exp in EXPERIMENTS:
        row = {
            "Exp":       exp["exp_id"],
            "Name":      exp["name"],
            "Model":     exp["model"],
            "Extractor": exp["extractor"],
            "Clinical":  "Yes" if exp["clinical"] else "No",
            "Epochs":    exp["epochs"],
            "BagSize":   exp["bag_size"],
            "Note":      exp["note"],
        }

        our_auc, our_lo, our_hi = get_ourdata_auroc(exp)
        tcga_auc, tcga_lo, tcga_hi = get_tcga_auroc(exp)

        def fmt(v, lo, hi):
            if v is None or np.isnan(v):
                return "—"
            ci = f"({lo:.3f}–{hi:.3f})" if lo and not np.isnan(lo) else ""
            return f"{v:.4f} {ci}".strip()

        row["Ourdata AUROC"] = fmt(our_auc, our_lo, our_hi)
        row["TCGA AUROC"]    = fmt(tcga_auc, tcga_lo, tcga_hi)
        row["ΔAUROC"]        = f"{abs(our_auc - tcga_auc):.4f}" if our_auc and tcga_auc else "—"
        row["Status"] = "Done" if our_auc else "Pending"

        rows.append(row)
        status = "✓" if our_auc else "⏳"
        print(f"  {status} {exp['exp_id']:6s} {exp['name']:35s} "
              f"Ourdata={row['Ourdata AUROC'][:6] if our_auc else '——':>6}  "
              f"TCGA={row['TCGA AUROC'][:6] if tcga_auc else '——':>6}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("/data3/chensx/STAMP/outputs/results/experiment_results.csv"))
    args = parser.parse_args()

    print("=" * 70)
    print("Collecting experiment results...")
    print("=" * 70)
    df = collect(args.output)
    print("\n" + df[["Exp", "Name", "Ourdata AUROC", "TCGA AUROC", "ΔAUROC", "Status"]].to_string(index=False))
