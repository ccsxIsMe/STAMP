"""
clinical_transfer_ensemble.py
-----------------------------
Train a clinical risk model on a source cohort and apply it to a target cohort,
then blend the resulting clinical score with an existing WSI prediction CSV.

This is intended for external validation where we must not tune on target labels.

Example:
    python scripts/clinical_transfer_ensemble.py \
        --source_clini_csv /data3/chensx/STAMP/outputs/tables/ourdata_clinical_merged.csv \
        --target_clini_csv /data3/chensx/STAMP/outputs/tables/tcga_clinical_filtered.csv \
        --pred_csv /data3/chensx/STAMP/outputs/deploy/exp01_abmil/patient-preds_95_confidence_interval.csv \
        --output_dir /data3/chensx/STAMP/outputs/deploy/exp01_abmil_clinpath_tcga \
        --preset tcga_shared \
        --alpha 0.15
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


PRESETS = {
    "tcga_shared": {
        "continuous": [
            {
                "name": "age",
                "source": ["年龄", "骞撮緞"],
                "target": ["age_at_initial_pathologic_diagnosis"],
            },
        ],
        "categorical": [
            {
                "name": "sex",
                "source": ["性别", "鎬у埆"],
                "target": ["gender"],
            },
            {
                "name": "grade",
                "source": [
                    "组织学分级（分化）",
                    "缁勭粐瀛﹀垎绾э紙鍒嗗寲锛?",
                ],
                "target": ["Histological_grade", "histological_grade"],
            },
        ],
    }
}

PATIENT_COL_ALIASES = ["PATIENT", "patient", "bcr_patient_barcode", "case_id"]
LABEL_COL_ALIASES = ["Early recurrence", "early_recurrence"]
WSI_SCORE_COL_ALIASES = [
    "Early recurrence_1",
    "early_recurrence_1",
    "prob_1",
    "prediction_1",
]


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
        s = re.sub(r"^[><>=鈮も墺\s]+", "", s)
        try:
            return float(s)
        except ValueError:
            return float("nan")

    return series.map(_clean)


def _normalize_gender(series: pd.Series) -> pd.Series:
    def _map(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in {"male", "m", "1", "鐢?", "男"}:
            return "MALE"
        if s in {"female", "f", "0", "濂?", "女"}:
            return "FEMALE"
        return np.nan

    return series.map(_map)


def _normalize_grade(series: pd.Series) -> pd.Series:
    def _map(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().upper()
        s_lower = s.lower()
        if "楂?" in s or "高" in s or "well" in s_lower or s in {"G1", "1"}:
            return "G1"
        if "涓?" in s or "中" in s or "moder" in s_lower or s in {"G2", "2"}:
            return "G2"
        if "浣?" in s or "低" in s or "poor" in s_lower or s in {"G3", "G4", "3", "4"}:
            return "G3"
        return np.nan

    return series.map(_map)


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip()).lower()


def _find_column(df: pd.DataFrame, candidates: list[str], *, required: bool = False, field_name: str = "") -> str | None:
    normalized_to_actual = {_normalize_column_name(col): col for col in df.columns}
    for candidate in candidates:
        actual = normalized_to_actual.get(_normalize_column_name(candidate))
        if actual is not None:
            return actual
    if required:
        raise KeyError(
            f"Could not find column for {field_name or 'field'}. "
            f"Tried aliases: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def _resolve_feature_pairs(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    preset_name: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    preset = PRESETS[preset_name]
    source_cont_cols, source_cat_cols = [], []
    target_cont_cols, target_cat_cols = [], []

    for feature in preset["continuous"]:
        source_col = _find_column(source_df, feature["source"])
        target_col = _find_column(target_df, feature["target"])
        if source_col and target_col:
            source_cont_cols.append(source_col)
            target_cont_cols.append(target_col)
        else:
            print(f"Skip continuous feature '{feature['name']}' because it is not present in both cohorts")

    for feature in preset["categorical"]:
        source_col = _find_column(source_df, feature["source"])
        target_col = _find_column(target_df, feature["target"])
        if source_col and target_col:
            source_cat_cols.append(source_col)
            target_cat_cols.append(target_col)
        else:
            print(f"Skip categorical feature '{feature['name']}' because it is not present in both cohorts")

    if not (source_cont_cols or source_cat_cols):
        raise ValueError(f"No shared clinical features could be resolved for preset '{preset_name}'")

    return source_cont_cols, source_cat_cols, target_cont_cols, target_cat_cols


def _prepare_df(df: pd.DataFrame, cont_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    for col in cont_cols:
        work[col] = _parse_numeric(work[col])
    for col in cat_cols:
        col_key = _normalize_column_name(col)
        if "鎬у埆" in col or "性别" in col or col_key == "gender":
            work[col] = _normalize_gender(work[col])
        elif "鍒嗙骇" in col or "鍒嗗寲" in col or "grade" in col_key:
            work[col] = _normalize_grade(work[col])
        else:
            work[col] = work[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    return work


def _fit_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_cont_cols: list[str],
    train_cat_cols: list[str],
    eval_cont_cols: list[str],
    eval_cat_cols: list[str],
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
            ("num", numeric_transformer, train_cont_cols),
            ("cat", categorical_transformer, train_cat_cols),
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

    X_train = train_df[train_cont_cols + train_cat_cols].copy()
    y_train = train_df["Early recurrence"].astype(int).to_numpy()
    X_eval = eval_df[eval_cont_cols + eval_cat_cols].copy()
    X_eval.columns = X_train.columns

    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def _resolve_patient_and_label_columns(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> tuple[str, str, str, str, str | None]:
    source_patient_col = _find_column(
        source_df,
        PATIENT_COL_ALIASES,
        required=True,
        field_name="source patient id",
    )
    target_patient_col = _find_column(
        target_df,
        PATIENT_COL_ALIASES,
        required=True,
        field_name="target patient id",
    )
    pred_patient_col = _find_column(
        pred_df,
        PATIENT_COL_ALIASES,
        required=True,
        field_name="prediction patient id",
    )
    target_label_col = _find_column(
        target_df,
        LABEL_COL_ALIASES,
        required=True,
        field_name="target label",
    )
    pred_label_col = _find_column(pred_df, LABEL_COL_ALIASES)

    return (
        source_patient_col,
        target_patient_col,
        pred_patient_col,
        target_label_col,
        pred_label_col,
    )


def _resolve_wsi_score_column(pred_df: pd.DataFrame) -> str:
    direct_match = _find_column(pred_df, WSI_SCORE_COL_ALIASES)
    if direct_match is not None:
        return direct_match

    for col in pred_df.columns:
        col_key = _normalize_column_name(col)
        if col_key.endswith("_1") and "earlyrecurrence" in col_key:
            return col

    raise KeyError(
        "Could not find the WSI positive-class score column in prediction CSV. "
        f"Tried aliases: {WSI_SCORE_COL_ALIASES}. Available columns: {list(pred_df.columns)}"
    )


def run(
    source_clini_csv: Path,
    target_clini_csv: Path,
    pred_csv: Path,
    output_dir: Path,
    preset: str,
    alpha: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(source_clini_csv)
    target_df = pd.read_csv(target_clini_csv)
    pred_df = pd.read_csv(pred_csv)

    (
        source_patient_col,
        target_patient_col,
        pred_patient_col,
        target_label_col,
        pred_label_col,
    ) = _resolve_patient_and_label_columns(source_df, target_df, pred_df)

    source_df = source_df.rename(columns={source_patient_col: "PATIENT"})
    target_df = target_df.rename(columns={target_patient_col: "PATIENT", target_label_col: "Early recurrence"})
    pred_rename_map = {pred_patient_col: "PATIENT"}
    if pred_label_col is not None:
        pred_rename_map[pred_label_col] = "Early recurrence"
    pred_df = pred_df.rename(columns=pred_rename_map)

    source_df["PATIENT"] = source_df["PATIENT"].astype(str)
    target_df["PATIENT"] = target_df["PATIENT"].astype(str)
    pred_df["PATIENT"] = pred_df["PATIENT"].astype(str)

    source_cont_cols, source_cat_cols, target_cont_cols, target_cat_cols = _resolve_feature_pairs(
        source_df=source_df,
        target_df=target_df,
        preset_name=preset,
    )
    source_df = _prepare_df(source_df, source_cont_cols, source_cat_cols)
    target_df = _prepare_df(target_df, target_cont_cols, target_cat_cols)

    print(f"Source continuous : {source_cont_cols}")
    print(f"Source categorical: {source_cat_cols}")
    print(f"Target continuous : {target_cont_cols}")
    print(f"Target categorical: {target_cat_cols}")

    if "Early recurrence" in pred_df.columns:
        pred_df["Early recurrence"] = pred_df["Early recurrence"].astype(str)
    target_df["Early recurrence"] = target_df["Early recurrence"].astype(str)

    merged = pred_df.merge(
        target_df[["PATIENT", "Early recurrence", *target_cont_cols, *target_cat_cols]],
        on=["PATIENT"],
        how="inner",
        suffixes=("", "_target"),
    )
    if len(merged) < len(pred_df):
        print(f"{len(pred_df) - len(merged)} patients lost in target clinical merge")

    if "Early recurrence_target" in merged.columns:
        mismatch_count = (merged["Early recurrence"] != merged["Early recurrence_target"]).sum()
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} patient labels differ between prediction CSV and target clinical CSV")
        merged["Early recurrence"] = merged["Early recurrence_target"]
        merged = merged.drop(columns=["Early recurrence_target"])

    train_df = source_df.dropna(subset=["Early recurrence"]).copy()
    clin_score = _fit_predict(
        train_df=train_df,
        eval_df=merged,
        train_cont_cols=source_cont_cols,
        train_cat_cols=source_cat_cols,
        eval_cont_cols=target_cont_cols,
        eval_cat_cols=target_cat_cols,
    )

    wsi_score_col = _resolve_wsi_score_column(pred_df)
    wsi_score = merged[wsi_score_col].astype(float).to_numpy()
    ens_score = alpha * wsi_score + (1 - alpha) * clin_score

    out_df = merged[["PATIENT", "Early recurrence"]].copy()
    out_df["Early recurrence_0"] = 1.0 - ens_score
    out_df["Early recurrence_1"] = ens_score
    out_df["Early recurrence_1_wsi"] = wsi_score
    out_df["Early recurrence_1_clinical"] = clin_score
    out_df.to_csv(output_dir / "patient-preds.csv", index=False)

    y_true = merged["Early recurrence"].astype(int).to_numpy()
    auc_wsi = roc_auc(y_true, wsi_score)
    auc_clin = roc_auc(y_true, clin_score)
    auc_ens = roc_auc(y_true, ens_score)
    ci_lo, ci_hi = bootstrap_ci(y_true, ens_score)

    print("\n" + "=" * 60)
    print(f"WSI AUROC         : {auc_wsi:.4f}")
    print(f"Clinical AUROC    : {auc_clin:.4f}")
    print(f"Ensemble AUROC    : {auc_ens:.4f}")
    print(f"Ensemble 95% CI   : {ci_lo:.4f} to {ci_hi:.4f}")
    print(f"Fixed alpha (WSI) : {alpha:.2f}")

    summary = {
        "preset": preset,
        "alpha_wsi": alpha,
        "source_continuous_features": source_cont_cols,
        "source_categorical_features": source_cat_cols,
        "target_continuous_features": target_cont_cols,
        "target_categorical_features": target_cat_cols,
        "wsi_auc": float(auc_wsi),
        "clinical_auc": float(auc_clin),
        "ensemble_auc": float(auc_ens),
        "ensemble_ci_lo": float(ci_lo),
        "ensemble_ci_hi": float(ci_hi),
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_clini_csv", type=Path, required=True)
    parser.add_argument("--target_clini_csv", type=Path, required=True)
    parser.add_argument("--pred_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tcga_shared")
    parser.add_argument("--alpha", type=float, default=0.15)
    args = parser.parse_args()

    run(
        source_clini_csv=args.source_clini_csv,
        target_clini_csv=args.target_clini_csv,
        pred_csv=args.pred_csv,
        output_dir=args.output_dir,
        preset=args.preset,
        alpha=args.alpha,
    )
