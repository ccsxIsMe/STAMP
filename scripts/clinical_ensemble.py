"""
clinical_ensemble.py
--------------------
Post-hoc ensemble of WSI prediction scores + clinical features.

For each cross-validation fold:
  - Train a clinical logistic-regression model on the training patients
  - Predict clinical risk on the validation fold
  - Blend WSI and clinical scores with either a fixed alpha or an alpha search
  - Save blended out-of-fold predictions for downstream statistics

Example:
    python scripts/clinical_ensemble.py \
        --crossval_dir /data3/chensx/STAMP/outputs/crossval/exp01_abmil \
        --clini_csv    /data3/chensx/STAMP/outputs/tables/ourdata_clinical_merged.csv \
        --output_dir   /data3/chensx/STAMP/outputs/crossval/exp01_abmil_clinpath \
        --preset ourdata_pathology \
        --search_alpha
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRESETS: dict[str, dict[str, list[str]]] = {
    "basic": {
        "cont": ["AFP (ng/ml)", "年龄"],
        "cat": ["性别"],
    },
    "ourdata_pathology": {
        "cont": [
            "AFP (ng/ml)",
            "年龄",
            "肿瘤大小（影像学）(mm)",
            "手术切缘宽度（mm）",
        ],
        "cat": [
            "性别",
            "微血管侵犯分级",
            "组织学分级（分化）",
            "卫星结节",
            "门静脉癌栓",
            "瘤内坏死",
            "瘤内出血",
            "CNLC",
            "BCLC",
            "分化",
            "cohort",
        ],
    },
    "shared_basic": {
        "cont": ["age_at_initial_pathologic_diagnosis"],
        "cat": ["gender", "Histological_grade", "Ajcc_pathologic_tumor_stage"],
    },
}


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


def _parse_numeric(series: pd.Series) -> pd.Series:
    def _clean(v):
        if pd.isna(v):
            return float("nan")
        s = str(v).strip()
        s = re.sub(r"^[><>=≤≥\s]+", "", s)
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return series.map(_clean)


def _load_splits(crossval_dir: Path) -> list[dict]:
    with open(crossval_dir / "splits.json", "r", encoding="utf-8") as f:
        return json.load(f)["splits"]


def _resolve_feature_columns(
    clini_df: pd.DataFrame,
    preset: str | None,
    cont_cols: list[str],
    cat_cols: list[str],
) -> tuple[list[str], list[str]]:
    preset_cols = PRESETS.get(preset, {"cont": [], "cat": []}) if preset else {"cont": [], "cat": []}
    cont = list(dict.fromkeys([*preset_cols["cont"], *cont_cols]))
    cat = list(dict.fromkeys([*preset_cols["cat"], *cat_cols]))

    cont = [c for c in cont if c in clini_df.columns]
    cat = [c for c in cat if c in clini_df.columns]
    if not cont and not cat:
        raise ValueError("No valid clinical feature columns were found in the clinical table.")
    return cont, cat


def _fit_predict_clinical(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cont_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, cont_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    X_train = train_df[cont_cols + cat_cols].copy()
    X_val = val_df[cont_cols + cat_cols].copy()
    y_train = train_df["Early recurrence"].astype(int).to_numpy()

    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def _alpha_grid() -> list[float]:
    return [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]


def run(
    crossval_dir: Path,
    clini_csv: Path,
    output_dir: Path,
    alpha: float = 0.5,
    search_alpha: bool = False,
    preset: str | None = None,
    cont_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    clini_df = pd.read_csv(clini_csv)
    clini_df["PATIENT"] = clini_df["PATIENT"].astype(str)

    cont_cols = cont_cols or []
    cat_cols = cat_cols or []
    cont_cols, cat_cols = _resolve_feature_columns(clini_df, preset, cont_cols, cat_cols)

    for col in cont_cols:
        clini_df[col] = _parse_numeric(clini_df[col])
    for col in cat_cols:
        clini_df[col] = clini_df[col].astype(str).replace({"nan": np.nan, "None": np.nan})

    print(f"Continuous features : {cont_cols}")
    print(f"Categorical features: {cat_cols}")

    splits = _load_splits(crossval_dir)
    fold_outputs = []
    fold_aucs_wsi, fold_aucs_clin = [], []
    alpha_scores: dict[float, list[float]] = {a: [] for a in _alpha_grid()} if search_alpha else {alpha: []}

    for split_idx, split in enumerate(splits):
        pred_file = crossval_dir / f"split-{split_idx}" / "patient-preds.csv"
        if not pred_file.exists():
            print(f"split-{split_idx}: prediction file not found, skipping")
            continue

        wsi_df = pd.read_csv(pred_file)
        wsi_df["PATIENT"] = wsi_df["PATIENT"].astype(str)

        merged = wsi_df.merge(
            clini_df[["PATIENT", "Early recurrence", *cont_cols, *cat_cols]],
            on=["PATIENT", "Early recurrence"],
            how="inner",
        )
        if len(merged) < len(wsi_df):
            print(f"split-{split_idx}: {len(wsi_df) - len(merged)} patients lost in clinical merge")

        train_patients = set(split["train_patients"])
        train_df = clini_df[clini_df["PATIENT"].isin(train_patients)].copy()
        train_df = train_df.dropna(subset=["Early recurrence"])

        clin_score = _fit_predict_clinical(
            train_df=train_df,
            val_df=merged,
            cont_cols=cont_cols,
            cat_cols=cat_cols,
        )

        y_val = merged["Early recurrence"].astype(int).to_numpy()
        wsi_score = merged["Early recurrence_1"].to_numpy()

        auc_wsi = roc_auc(y_val, wsi_score)
        auc_clin = roc_auc(y_val, clin_score)
        fold_aucs_wsi.append(auc_wsi)
        fold_aucs_clin.append(auc_clin)

        for a in alpha_scores:
            ens_score = a * wsi_score + (1 - a) * clin_score
            alpha_scores[a].append(roc_auc(y_val, ens_score))

        fold_outputs.append(
            pd.DataFrame(
                {
                    "PATIENT": merged["PATIENT"],
                    "Early recurrence": y_val,
                    "Early recurrence_1": wsi_score,
                    "clin_score": clin_score,
                }
            )
        )
        print(f"split-{split_idx}: WSI={auc_wsi:.4f} Clinical={auc_clin:.4f}")

    if not fold_outputs:
        print("No completed folds found.")
        return

    if search_alpha:
        best_alpha = max(alpha_scores, key=lambda a: np.nanmean(alpha_scores[a]))
    else:
        best_alpha = alpha

    all_df = pd.concat(fold_outputs, ignore_index=True)
    all_df["Early recurrence_1_ensemble"] = (
        best_alpha * all_df["Early recurrence_1"] + (1 - best_alpha) * all_df["clin_score"]
    )

    for split_idx, fold_df in enumerate(fold_outputs):
        fold_out = output_dir / f"split-{split_idx}"
        fold_out.mkdir(parents=True, exist_ok=True)
        out_df = fold_df.copy()
        out_df["Early recurrence_1_ensemble"] = (
            best_alpha * out_df["Early recurrence_1"] + (1 - best_alpha) * out_df["clin_score"]
        )
        out_df[
            ["PATIENT", "Early recurrence", "Early recurrence_1", "Early recurrence_1_ensemble"]
        ].to_csv(fold_out / "patient-preds.csv", index=False)

    ci_lo, ci_hi = bootstrap_ci(
        all_df["Early recurrence"].to_numpy(),
        all_df["Early recurrence_1_ensemble"].to_numpy(),
    )
    ensemble_auc = roc_auc(
        all_df["Early recurrence"].to_numpy(),
        all_df["Early recurrence_1_ensemble"].to_numpy(),
    )

    print("\n" + "=" * 60)
    print(f"WSI only mean fold AUROC   : {np.mean(fold_aucs_wsi):.4f}")
    print(f"Clinical mean fold AUROC   : {np.mean(fold_aucs_clin):.4f}")
    print(f"Ensemble OOF AUROC         : {ensemble_auc:.4f}")
    print(f"Ensemble 95% CI            : {ci_lo:.4f} to {ci_hi:.4f}")
    print(f"Best alpha (WSI weight)    : {best_alpha:.2f}")

    summary = {
        "preset": preset,
        "continuous_features": cont_cols,
        "categorical_features": cat_cols,
        "wsi_fold_mean": float(np.mean(fold_aucs_wsi)),
        "clinical_fold_mean": float(np.mean(fold_aucs_clin)),
        "ensemble_oof_auc": float(ensemble_auc),
        "ensemble_ci_lo": float(ci_lo),
        "ensemble_ci_hi": float(ci_hi),
        "alpha_wsi": float(best_alpha),
        "search_alpha": search_alpha,
        "alpha_grid_scores": {str(a): float(np.nanmean(v)) for a, v in alpha_scores.items()},
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crossval_dir", type=Path, required=True)
    parser.add_argument("--clini_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed WSI weight when alpha search is disabled.")
    parser.add_argument("--search_alpha", action="store_true", help="Search the best WSI/clinical blending weight on out-of-fold predictions.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="basic")
    parser.add_argument("--cont_col", action="append", default=[], help="Additional continuous clinical feature column. Can be passed multiple times.")
    parser.add_argument("--cat_col", action="append", default=[], help="Additional categorical clinical feature column. Can be passed multiple times.")
    args = parser.parse_args()

    run(
        crossval_dir=args.crossval_dir,
        clini_csv=args.clini_csv,
        output_dir=args.output_dir,
        alpha=args.alpha,
        search_alpha=args.search_alpha,
        preset=args.preset,
        cont_cols=args.cont_col,
        cat_cols=args.cat_col,
    )
