"""
model_ensemble.py
-----------------
Post-hoc probability ensemble of two or more STAMP crossval/deploy outputs.

Usage (crossval):
    python scripts/model_ensemble.py \
        --dirs /data3/chensx/STAMP/outputs/crossval/exp01_abmil \
               /data3/chensx/STAMP/outputs/crossval/exp07_uni2 \
        --output_dir /data3/chensx/STAMP/outputs/crossval/ens_transmil_uni2 \
        --mode crossval

Usage (deploy / TCGA):
    python scripts/model_ensemble.py \
        --dirs /data3/chensx/STAMP/outputs/deploy/exp01_abmil \
               /data3/chensx/STAMP/outputs/deploy/exp07_uni2 \
        --output_dir /data3/chensx/STAMP/outputs/deploy/ens_transmil_uni2 \
        --mode deploy
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


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
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    np.random.seed(seed)
    aucs = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) == 2:
            aucs.append(roc_auc(y_true[idx], y_score[idx]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else [float("nan")] * 2


def ensemble_crossval(dirs, output_dir):
    """Average predictions across folds from multiple crossval directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_preds = []
    fold_aucs_individual = {d.name: [] for d in dirs}
    fold_aucs_ens = []

    for split in range(5):
        dfs = []
        for d in dirs:
            f = d / f"split-{split}" / "patient-preds.csv"
            if not f.exists():
                print(f"  WARNING: {f} not found, skipping split-{split}")
                break
            dfs.append(pd.read_csv(f).set_index("PATIENT"))
        else:
            # Intersect patients present in all models
            common = dfs[0].index
            for df in dfs[1:]:
                common = common.intersection(df.index)
            dfs = [df.loc[common] for df in dfs]

            y = dfs[0]["Early recurrence"].values
            scores = np.stack([df["Early recurrence_1"].values for df in dfs], axis=0)
            ens_score = scores.mean(axis=0)

            for d, df in zip(dirs, dfs):
                fold_aucs_individual[d.name].append(roc_auc(y, df["Early recurrence_1"].values))
            fold_aucs_ens.append(roc_auc(y, ens_score))

            out_df = dfs[0][["Early recurrence"]].copy()
            out_df["Early recurrence_1_ensemble"] = ens_score
            out_df["Early recurrence_1"] = ens_score  # STAMP-compatible column
            out_df = out_df.reset_index()

            split_dir = output_dir / f"split-{split}"
            split_dir.mkdir(exist_ok=True)
            out_df.to_csv(split_dir / "patient-preds.csv", index=False)
            all_preds.append(out_df)

    if not fold_aucs_ens:
        print("No completed folds.")
        return

    print("\n Individual model AUROCs (per fold):")
    for name, aucs in fold_aucs_individual.items():
        print(f"  {name}: {[round(a, 4) for a in aucs]}  mean={np.mean(aucs):.4f}")

    all_df = pd.concat(all_preds, ignore_index=True)
    ci_lo, ci_hi = bootstrap_ci(all_df["Early recurrence"], all_df["Early recurrence_1"])
    print(f"\n  Ensemble: {[round(a, 4) for a in fold_aucs_ens]}  "
          f"mean={np.mean(fold_aucs_ens):.4f}  (95%CI {ci_lo:.4f}–{ci_hi:.4f})")
    print(f"\nSaved to: {output_dir}")


def ensemble_deploy(dirs, output_dir):
    """Average predictions from multiple deploy directories (no folds)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_preds = []
    individual_aucs = {}

    for split in range(5):
        dfs = []
        for d in dirs:
            f = d / f"patient-preds-{split}.csv"
            if not f.exists():
                break
            dfs.append(pd.read_csv(f).set_index("PATIENT"))
        else:
            common = dfs[0].index
            for df in dfs[1:]:
                common = common.intersection(df.index)
            dfs = [df.loc[common] for df in dfs]

            y = dfs[0]["Early recurrence"].values
            scores = np.stack([df["Early recurrence_1"].values for df in dfs], axis=0)
            ens_score = scores.mean(axis=0)

            for d, df in zip(dirs, dfs):
                individual_aucs.setdefault(d.name, []).append(
                    roc_auc(y, df["Early recurrence_1"].values))

            out_df = dfs[0][["Early recurrence"]].copy().reset_index()
            out_df["Early recurrence_1"] = ens_score
            out_df.to_csv(output_dir / f"patient-preds-{split}.csv", index=False)
            all_preds.append(out_df)

    if not all_preds:
        print("No completed folds.")
        return

    print("\n Individual model AUROCs:")
    for name, aucs in individual_aucs.items():
        print(f"  {name}: {[round(a, 4) for a in aucs]}  mean={np.mean(aucs):.4f}")

    all_df = pd.concat(all_preds, ignore_index=True)
    ens_aucs = []
    for split_df in all_preds:
        ens_aucs.append(roc_auc(split_df["Early recurrence"].values,
                                split_df["Early recurrence_1"].values))
    ci_lo, ci_hi = bootstrap_ci(all_df["Early recurrence"], all_df["Early recurrence_1"])
    print(f"\n  Ensemble: {[round(a, 4) for a in ens_aucs]}  "
          f"mean={np.mean(ens_aucs):.4f}  (95%CI {ci_lo:.4f}–{ci_hi:.4f})")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", type=Path, required=True,
                        help="Two or more crossval/deploy directories to ensemble")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["crossval", "deploy"], default="crossval")
    args = parser.parse_args()

    if args.mode == "crossval":
        ensemble_crossval(args.dirs, args.output_dir)
    else:
        ensemble_deploy(args.dirs, args.output_dir)
