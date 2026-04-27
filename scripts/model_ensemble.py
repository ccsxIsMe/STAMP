"""
model_ensemble.py
-----------------
Post-hoc probability ensemble of two or more STAMP crossval/deploy outputs.

Supports:
1. Cross-validation ensembling with optional internal weight search.
2. External/deploy ensembling with fixed weights.
3. STAMP-compatible binary classification export:
   - Early recurrence_0
   - Early recurrence_1

Examples
--------
Search the best 2-model weight on internal crossval:
    python scripts/model_ensemble.py \
        --dirs /data3/chensx/STAMP/outputs/crossval/exp11_abmil_subbag \
               /data3/chensx/STAMP/outputs/crossval/exp12_phikon_subbag \
        --output_dir /data3/chensx/STAMP/outputs/crossval/exp13_conch_phikon_ens \
        --mode crossval \
        --search_weights

Apply the learned weight to TCGA deploy outputs:
    python scripts/model_ensemble.py \
        --dirs /data3/chensx/STAMP/outputs/deploy/exp11_abmil_subbag \
               /data3/chensx/STAMP/outputs/deploy/exp12_phikon_subbag \
        --output_dir /data3/chensx/STAMP/outputs/deploy/exp13_conch_phikon_ens \
        --mode deploy \
        --weights 0.65 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _normalize_weights(weights: list[float]) -> np.ndarray:
    arr = np.array(weights, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("weights must be a non-empty 1D list")
    if np.any(arr < 0):
        raise ValueError("weights must be non-negative")
    if arr.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    return arr / arr.sum()


def _resolve_prediction_file(base_dir: Path, *, mode: str, split_idx: int | None = None) -> Path | None:
    candidates: list[Path] = []
    if mode == "crossval":
        if split_idx is None:
            raise ValueError("split_idx is required for crossval mode")
        candidates.append(base_dir / f"split-{split_idx}" / "patient-preds.csv")
    else:
        if split_idx is not None:
            candidates.append(base_dir / f"patient-preds-{split_idx}.csv")
        candidates.append(base_dir / "patient-preds_95_confidence_interval.csv")
        candidates.append(base_dir / "patient-preds.csv")

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_prediction_df(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    required = {"PATIENT", "Early recurrence", "Early recurrence_1"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{pred_path} is missing columns: {sorted(missing)}")
    return df


def _build_stamp_output(base_df: pd.DataFrame, ens_score: np.ndarray) -> pd.DataFrame:
    out_df = base_df[["PATIENT", "Early recurrence"]].copy()
    out_df["Early recurrence_0"] = 1.0 - ens_score
    out_df["Early recurrence_1"] = ens_score
    return out_df


def _score_weighted_ensemble(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(scores, axis=0, weights=weights)


def _weight_grid_two_models(step: float = 0.05) -> list[np.ndarray]:
    grid = []
    for w0 in np.arange(0.0, 1.0 + 1e-9, step):
        grid.append(np.array([w0, 1.0 - w0], dtype=float))
    return grid


def _search_weights_crossval(
    aligned_scores: list[np.ndarray],
    aligned_labels: list[np.ndarray],
    model_names: list[str],
) -> np.ndarray:
    if len(model_names) != 2:
        raise ValueError("--search_weights currently supports exactly 2 models")

    best_auc = -np.inf
    best_weights = None
    for weights in _weight_grid_two_models(step=0.05):
        y_all = np.concatenate(aligned_labels, axis=0)
        score_all = np.concatenate(
            [_score_weighted_ensemble(split_scores, weights) for split_scores in aligned_scores],
            axis=0,
        )
        auc = roc_auc(y_all, score_all)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    assert best_weights is not None
    print(
        f"Best internal OOF weights: "
        f"{model_names[0]}={best_weights[0]:.2f}, {model_names[1]}={best_weights[1]:.2f} "
        f"(AUROC={best_auc:.4f})"
    )
    return best_weights


def ensemble_crossval(dirs: list[Path], output_dir: Path, weights: np.ndarray | None, search_weights: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [d.name for d in dirs]

    split_scores_all: list[np.ndarray] = []
    split_labels_all: list[np.ndarray] = []
    split_base_dfs: list[pd.DataFrame] = []
    fold_aucs_individual = {d.name: [] for d in dirs}

    for split_idx in range(5):
        dfs = []
        for d in dirs:
            pred_path = _resolve_prediction_file(d, mode="crossval", split_idx=split_idx)
            if pred_path is None:
                print(f"WARNING: missing split-{split_idx} prediction in {d}, skipping this split")
                break
            dfs.append(_load_prediction_df(pred_path).set_index("PATIENT"))
        else:
            common = dfs[0].index
            for df in dfs[1:]:
                common = common.intersection(df.index)
            dfs = [df.loc[common].sort_index() for df in dfs]

            y = dfs[0]["Early recurrence"].astype(int).to_numpy()
            scores = np.stack([df["Early recurrence_1"].astype(float).to_numpy() for df in dfs], axis=0)

            for d, df in zip(dirs, dfs):
                fold_aucs_individual[d.name].append(roc_auc(y, df["Early recurrence_1"].values))

            split_scores_all.append(scores)
            split_labels_all.append(y)
            split_base_dfs.append(dfs[0].reset_index()[["PATIENT", "Early recurrence"]])

    if not split_scores_all:
        raise RuntimeError("No completed cross-validation splits found")

    if search_weights:
        weights = _search_weights_crossval(
            aligned_scores=split_scores_all,
            aligned_labels=split_labels_all,
            model_names=model_names,
        )
    elif weights is None:
        weights = np.ones(len(dirs), dtype=float) / len(dirs)

    fold_aucs_ens = []
    all_preds = []
    for split_idx, (base_df, scores, y) in enumerate(zip(split_base_dfs, split_scores_all, split_labels_all)):
        ens_score = _score_weighted_ensemble(scores, weights)
        fold_aucs_ens.append(roc_auc(y, ens_score))

        out_df = _build_stamp_output(base_df, ens_score)
        split_dir = output_dir / f"split-{split_idx}"
        split_dir.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(split_dir / "patient-preds.csv", index=False)
        all_preds.append(out_df)

    print("\nIndividual model AUROCs (per fold):")
    for name, aucs in fold_aucs_individual.items():
        print(f"  {name}: {[round(a, 4) for a in aucs]}  mean={np.mean(aucs):.4f}")

    all_df = pd.concat(all_preds, ignore_index=True)
    ci_lo, ci_hi = bootstrap_ci(all_df["Early recurrence"], all_df["Early recurrence_1"])
    print(
        f"\nEnsemble weights: "
        + ", ".join(f"{name}={weight:.2f}" for name, weight in zip(model_names, weights))
    )
    print(
        f"Ensemble fold AUROCs: {[round(a, 4) for a in fold_aucs_ens]}  "
        f"mean={np.mean(fold_aucs_ens):.4f}  (95% CI {ci_lo:.4f} to {ci_hi:.4f})"
    )

    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "crossval",
                "model_dirs": [str(d) for d in dirs],
                "model_names": model_names,
                "weights": weights.tolist(),
                "fold_aurocs": [float(x) for x in fold_aucs_ens],
                "mean_auroc": float(np.mean(fold_aucs_ens)),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved to: {output_dir}")


def ensemble_deploy(dirs: list[Path], output_dir: Path, weights: np.ndarray | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [d.name for d in dirs]
    if weights is None:
        weights = np.ones(len(dirs), dtype=float) / len(dirs)

    dfs = []
    for d in dirs:
        pred_path = _resolve_prediction_file(d, mode="deploy", split_idx=None)
        if pred_path is None:
            raise FileNotFoundError(f"No deploy prediction file found in {d}")
        dfs.append(_load_prediction_df(pred_path).set_index("PATIENT"))

    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    dfs = [df.loc[common].sort_index() for df in dfs]

    y = dfs[0]["Early recurrence"].astype(int).to_numpy()
    scores = np.stack([df["Early recurrence_1"].astype(float).to_numpy() for df in dfs], axis=0)
    ens_score = _score_weighted_ensemble(scores, weights)

    for name, df in zip(model_names, dfs):
        print(f"{name} AUROC: {roc_auc(y, df['Early recurrence_1'].values):.4f}")

    auc_ens = roc_auc(y, ens_score)
    ci_lo, ci_hi = bootstrap_ci(y, ens_score)

    out_df = _build_stamp_output(dfs[0].reset_index()[["PATIENT", "Early recurrence"]], ens_score)
    out_df.to_csv(output_dir / "patient-preds.csv", index=False)

    print(
        f"\nEnsemble weights: "
        + ", ".join(f"{name}={weight:.2f}" for name, weight in zip(model_names, weights))
    )
    print(f"Ensemble deploy AUROC: {auc_ens:.4f}")
    print(f"Ensemble 95% CI: {ci_lo:.4f} to {ci_hi:.4f}")

    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "deploy",
                "model_dirs": [str(d) for d in dirs],
                "model_names": model_names,
                "weights": weights.tolist(),
                "deploy_auroc": float(auc_ens),
                "ci_low": float(ci_lo),
                "ci_high": float(ci_hi),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", type=Path, required=True, help="Two or more crossval/deploy directories to ensemble")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["crossval", "deploy"], default="crossval")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional ensemble weights, one per directory")
    parser.add_argument("--search_weights", action="store_true", help="Search the best 2-model weight on internal crossval OOF predictions")
    args = parser.parse_args()

    if args.weights is not None and len(args.weights) != len(args.dirs):
        raise ValueError("--weights must have the same length as --dirs")
    if args.search_weights and args.mode != "crossval":
        raise ValueError("--search_weights is only supported in crossval mode")
    if args.search_weights and args.weights is not None:
        raise ValueError("Use either --weights or --search_weights, not both")

    weights = _normalize_weights(args.weights) if args.weights is not None else None

    if args.mode == "crossval":
        ensemble_crossval(
            dirs=args.dirs,
            output_dir=args.output_dir,
            weights=weights,
            search_weights=args.search_weights,
        )
    else:
        ensemble_deploy(dirs=args.dirs, output_dir=args.output_dir, weights=weights)
