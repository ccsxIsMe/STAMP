"""
clinical_ensemble.py
--------------------
Post-hoc ensemble of WSI prediction scores + clinical features.

For each cross-validation fold:
  - Train logistic regression on clinical features (using same fold split)
  - Average WSI score + clinical LR score
  - Compute ensemble AUROC

Usage:
    python scripts/clinical_ensemble.py \
        --crossval_dir /data3/chensx/STAMP/outputs/crossval/exp02_vit_clinical \
        --clini_csv    /data3/chensx/outputs/tables/ourdata_clinical_merged.csv \
        --output_dir   /data3/chensx/STAMP/outputs/crossval/exp02_vit_clinical_ensemble

Clinical features used: AFP (ng/ml), 年龄, 性别
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


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


def logistic_regression_predict(X_train, y_train, X_val, lr=0.01, n_iter=500):
    """Minimal logistic regression via gradient descent (no sklearn needed)."""
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    X_val = np.array(X_val, dtype=float)

    # Standardize
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std

    # Add bias
    X_train = np.column_stack([X_train, np.ones(len(X_train))])
    X_val   = np.column_stack([X_val,   np.ones(len(X_val))])

    w = np.zeros(X_train.shape[1])
    for _ in range(n_iter):
        pred = sigmoid(X_train @ w)
        grad = X_train.T @ (pred - y_train) / len(y_train)
        w -= lr * grad

    return sigmoid(X_val @ w)


# ── Main ──────────────────────────────────────────────────────────────────────

CLINICAL_CONT = ["AFP (ng/ml)", "年龄"]
CLINICAL_CAT  = ["性别"]


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Strip leading comparison operators (>, <, ≥, ≤ and full-width variants) then cast to float."""
    import re
    def _clean(v):
        if pd.isna(v):
            return float("nan")
        s = str(v).strip()
        s = re.sub(r"^[><≥≤＞＜≧≦＝=\s]+", "", s)
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return series.map(_clean)


def run(crossval_dir: Path, clini_csv: Path, output_dir: Path, alpha: float = 0.5):
    """
    alpha: weight for WSI score (1-alpha for clinical score)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clini_df = pd.read_csv(clini_csv)
    clini_df["PATIENT"] = clini_df["PATIENT"].astype(str)

    # Clean numeric columns that may contain comparison-operator strings (e.g. ＞60500)
    for col in CLINICAL_CONT:
        if col in clini_df.columns:
            clini_df[col] = _parse_numeric(clini_df[col])

    # One-hot encode sex
    sex_map = {"男": 1, "女": 0, "M": 1, "F": 0, "male": 1, "female": 0}
    for col in CLINICAL_CAT:
        if col in clini_df.columns:
            clini_df[col + "_enc"] = clini_df[col].map(sex_map).fillna(0)

    feat_cols = CLINICAL_CONT + [c + "_enc" for c in CLINICAL_CAT
                                  if c + "_enc" in clini_df.columns]
    print(f"Clinical features: {feat_cols}")

    all_preds = []
    fold_aucs_wsi, fold_aucs_clin, fold_aucs_ens = [], [], []

    for split in range(5):
        pred_file = crossval_dir / f"split-{split}" / "patient-preds.csv"
        if not pred_file.exists():
            print(f"  split-{split}: prediction file not found, skipping")
            continue

        wsi_df = pd.read_csv(pred_file)
        wsi_df["PATIENT"] = wsi_df["PATIENT"].astype(str)

        # Merge clinical
        merged = wsi_df.merge(clini_df[["PATIENT"] + feat_cols], on="PATIENT", how="inner")
        if len(merged) < len(wsi_df):
            print(f"  split-{split}: {len(wsi_df)-len(merged)} patients lost in clinical merge")

        y_val = merged["Early recurrence"].values
        wsi_score = merged["Early recurrence_1"].values

        # Determine training patients (all patients NOT in this validation fold)
        all_patients = set(clini_df["PATIENT"].astype(str))
        val_patients = set(merged["PATIENT"].astype(str))
        train_patients = all_patients - val_patients

        train_df = clini_df[clini_df["PATIENT"].isin(train_patients)].dropna(subset=feat_cols)
        X_train = train_df[feat_cols].values
        y_train = train_df["Early recurrence"].values
        X_val   = merged[feat_cols].values

        clin_score = logistic_regression_predict(X_train, y_train, X_val)

        # Ensemble
        ens_score = alpha * wsi_score + (1 - alpha) * clin_score

        auc_wsi  = roc_auc(y_val, wsi_score)
        auc_clin = roc_auc(y_val, clin_score)
        auc_ens  = roc_auc(y_val, ens_score)

        fold_aucs_wsi.append(auc_wsi)
        fold_aucs_clin.append(auc_clin)
        fold_aucs_ens.append(auc_ens)

        print(f"  split-{split}: WSI={auc_wsi:.4f}  Clinical={auc_clin:.4f}  Ensemble={auc_ens:.4f}")

        merged["ens_score"] = ens_score
        merged["clin_score"] = clin_score
        all_preds.append(merged[["PATIENT", "Early recurrence",
                                  "Early recurrence_1", "clin_score", "ens_score"]])

        # Save fold predictions
        fold_out = output_dir / f"split-{split}"
        fold_out.mkdir(exist_ok=True)
        merged["Early recurrence_1_ensemble"] = ens_score
        out_df = merged[["PATIENT", "Early recurrence",
                          "Early recurrence_1", "Early recurrence_1_ensemble"]]
        out_df.to_csv(fold_out / "patient-preds.csv", index=False)

    if not fold_aucs_ens:
        print("No completed folds found.")
        return

    all_df = pd.concat(all_preds)
    ci_lo, ci_hi = bootstrap_ci(all_df["Early recurrence"], all_df["ens_score"])

    print("\n" + "=" * 55)
    print(f"  WSI only:    {np.mean(fold_aucs_wsi):.4f} ± {np.std(fold_aucs_wsi):.4f}")
    print(f"  Clinical LR: {np.mean(fold_aucs_clin):.4f} ± {np.std(fold_aucs_clin):.4f}")
    print(f"  Ensemble:    {np.mean(fold_aucs_ens):.4f} ± {np.std(fold_aucs_ens):.4f}  "
          f"(95%CI {ci_lo:.4f}–{ci_hi:.4f})")

    # Save summary
    summary = {
        "wsi_mean": np.mean(fold_aucs_wsi),
        "clinical_mean": np.mean(fold_aucs_clin),
        "ensemble_mean": np.mean(fold_aucs_ens),
        "ensemble_ci_lo": ci_lo,
        "ensemble_ci_hi": ci_hi,
        "alpha_wsi": alpha,
    }
    with open(output_dir / "ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossval_dir", type=Path, required=True)
    parser.add_argument("--clini_csv",    type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for WSI score (default 0.5 = equal ensemble)")
    args = parser.parse_args()

    run(args.crossval_dir, args.clini_csv, args.output_dir, args.alpha)
