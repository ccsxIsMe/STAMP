"""
clinical_transfer_stack.py
--------------------------
Train a source-domain stacker on source OOF WSI predictions plus shared
clinical variables, then apply it to an external target cohort.

This is intended for external validation without any target-label tuning.

Example:
    python scripts/clinical_transfer_stack.py \
        --source_crossval_dir /data3/chensx/STAMP/outputs/crossval/exp01_abmil \
        --source_clini_csv /data3/chensx/STAMP/outputs/tables/ourdata_clinical_merged.csv \
        --target_clini_csv /data3/chensx/STAMP/outputs/tables/tcga_clinical_filtered.csv \
        --target_pred_csv /data3/chensx/STAMP/outputs/deploy/exp01_abmil/patient-preds_95_confidence_interval.csv \
        --output_dir /data3/chensx/STAMP/outputs/deploy/exp01_abmil_clinstack_tcga \
        --preset tcga_shared
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


def _resolve_source_feature_pairs(source_df: pd.DataFrame, target_df: pd.DataFrame, preset_name: str) -> list[dict[str, str]]:
    preset = PRESETS[preset_name]
    pairs: list[dict[str, str]] = []

    for feature in preset["continuous"]:
        source_col = _find_column(source_df, feature["source"])
        target_col = _find_column(target_df, feature["target"])
        if source_col and target_col:
            pairs.append({"type": "continuous", "name": feature["name"], "source_col": source_col, "target_col": target_col})
        else:
            print(f"Skip continuous feature '{feature['name']}' because it is not present in both cohorts")

    for feature in preset["categorical"]:
        source_col = _find_column(source_df, feature["source"])
        target_col = _find_column(target_df, feature["target"])
        if source_col and target_col:
            pairs.append({"type": "categorical", "name": feature["name"], "source_col": source_col, "target_col": target_col})
        else:
            print(f"Skip categorical feature '{feature['name']}' because it is not present in both cohorts")

    if not pairs:
        raise ValueError(f"No shared clinical features could be resolved for preset '{preset_name}'")
    return pairs


def _prepare_feature_frame(df: pd.DataFrame, feature_pairs: list[dict[str, str]], *, cohort: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for feature in feature_pairs:
        source_key = "source_col" if cohort == "source" else "target_col"
        raw = df[feature[source_key]]
        if feature["type"] == "continuous":
            out[feature["name"]] = _parse_numeric(raw)
        elif feature["name"] == "sex":
            out[feature["name"]] = _normalize_gender(raw)
        elif feature["name"] == "grade":
            out[feature["name"]] = _normalize_grade(raw)
        else:
            out[feature["name"]] = raw.astype(str).replace({"nan": np.nan, "None": np.nan})
    return out


def _load_and_standardize_pred_csv(pred_csv: Path) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv)
    patient_col = _find_column(pred_df, PATIENT_COL_ALIASES, required=True, field_name="prediction patient id")
    label_col = _find_column(pred_df, LABEL_COL_ALIASES)
    score_col = _resolve_wsi_score_column(pred_df)

    rename_map = {patient_col: "PATIENT", score_col: "wsi_score"}
    if label_col is not None:
        rename_map[label_col] = "Early recurrence"
    pred_df = pred_df.rename(columns=rename_map)
    pred_df["PATIENT"] = pred_df["PATIENT"].astype(str)
    return pred_df


def _load_source_oof_preds(crossval_dir: Path) -> pd.DataFrame:
    pred_files = sorted(crossval_dir.glob("split-*/patient-preds.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No split prediction files found under {crossval_dir}")

    preds = []
    for pred_file in pred_files:
        pred_df = _load_and_standardize_pred_csv(pred_file)
        pred_df["source_split"] = pred_file.parent.name
        preds.append(pred_df[["PATIENT", "Early recurrence", "wsi_score", "source_split"]])

    source_pred_df = pd.concat(preds, ignore_index=True)
    source_pred_df = source_pred_df.drop_duplicates(subset=["PATIENT"], keep="first")
    return source_pred_df


def _build_transfer_model(X_train: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
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
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
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
    model.fit(X_train, X_train.pop("__label__"))
    return model


def run(
    source_crossval_dir: Path,
    source_clini_csv: Path,
    target_clini_csv: Path,
    target_pred_csv: Path,
    output_dir: Path,
    preset: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    source_clini_df = pd.read_csv(source_clini_csv)
    target_clini_df = pd.read_csv(target_clini_csv)
    target_pred_df = _load_and_standardize_pred_csv(target_pred_csv)
    source_pred_df = _load_source_oof_preds(source_crossval_dir)

    source_patient_col = _find_column(source_clini_df, PATIENT_COL_ALIASES, required=True, field_name="source patient id")
    target_patient_col = _find_column(target_clini_df, PATIENT_COL_ALIASES, required=True, field_name="target patient id")
    source_label_col = _find_column(source_clini_df, LABEL_COL_ALIASES, required=True, field_name="source label")
    target_label_col = _find_column(target_clini_df, LABEL_COL_ALIASES, required=True, field_name="target label")

    source_clini_df = source_clini_df.rename(columns={source_patient_col: "PATIENT", source_label_col: "Early recurrence"})
    target_clini_df = target_clini_df.rename(columns={target_patient_col: "PATIENT", target_label_col: "Early recurrence"})
    source_clini_df["PATIENT"] = source_clini_df["PATIENT"].astype(str)
    target_clini_df["PATIENT"] = target_clini_df["PATIENT"].astype(str)

    feature_pairs = _resolve_source_feature_pairs(source_clini_df, target_clini_df, preset)
    print("Resolved shared features:")
    for feature in feature_pairs:
        print(f"  {feature['name']}: source='{feature['source_col']}' target='{feature['target_col']}'")

    source_merged = source_pred_df.merge(
        source_clini_df[["PATIENT", "Early recurrence", *[f["source_col"] for f in feature_pairs]]],
        on=["PATIENT", "Early recurrence"],
        how="inner",
    )
    target_merged = target_pred_df.merge(
        target_clini_df[["PATIENT", "Early recurrence", *[f["target_col"] for f in feature_pairs]]],
        on=["PATIENT"],
        how="inner",
        suffixes=("", "_target"),
    )

    if len(source_merged) < len(source_pred_df):
        print(f"{len(source_pred_df) - len(source_merged)} source patients lost in clinical merge")
    if len(target_merged) < len(target_pred_df):
        print(f"{len(target_pred_df) - len(target_merged)} target patients lost in clinical merge")

    if "Early recurrence_target" in target_merged.columns:
        mismatch_count = (target_merged["Early recurrence"] != target_merged["Early recurrence_target"]).sum()
        if mismatch_count > 0:
            print(f"Warning: {mismatch_count} target labels differ between prediction CSV and target clinical CSV")
        target_merged["Early recurrence"] = target_merged["Early recurrence_target"]
        target_merged = target_merged.drop(columns=["Early recurrence_target"])

    X_source = _prepare_feature_frame(source_merged, feature_pairs, cohort="source")
    X_target = _prepare_feature_frame(target_merged, feature_pairs, cohort="target")
    X_source["wsi_score"] = source_merged["wsi_score"].astype(float).to_numpy()
    X_target["wsi_score"] = target_merged["wsi_score"].astype(float).to_numpy()

    numeric_cols = ["wsi_score"] + [f["name"] for f in feature_pairs if f["type"] == "continuous"]
    categorical_cols = [f["name"] for f in feature_pairs if f["type"] == "categorical"]

    train_df = X_source.copy()
    train_df["__label__"] = source_merged["Early recurrence"].astype(int).to_numpy()
    model = _build_transfer_model(train_df, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    stack_score = model.predict_proba(X_target)[:, 1]

    out_df = target_merged[["PATIENT", "Early recurrence"]].copy()
    out_df["Early recurrence_0"] = 1.0 - stack_score
    out_df["Early recurrence_1"] = stack_score
    out_df["Early recurrence_1_wsi"] = target_merged["wsi_score"].astype(float).to_numpy()
    out_df.to_csv(output_dir / "patient-preds.csv", index=False)

    y_true = target_merged["Early recurrence"].astype(int).to_numpy()
    auc_wsi = roc_auc(y_true, out_df["Early recurrence_1_wsi"].to_numpy())
    auc_stack = roc_auc(y_true, stack_score)
    ci_lo, ci_hi = bootstrap_ci(y_true, stack_score)

    print("\n" + "=" * 60)
    print(f"Target WSI AUROC    : {auc_wsi:.4f}")
    print(f"Target stack AUROC  : {auc_stack:.4f}")
    print(f"Target stack 95% CI : {ci_lo:.4f} to {ci_hi:.4f}")

    summary = {
        "preset": preset,
        "source_crossval_dir": str(source_crossval_dir),
        "target_pred_csv": str(target_pred_csv),
        "shared_features": feature_pairs,
        "wsi_auc": float(auc_wsi),
        "stack_auc": float(auc_stack),
        "stack_ci_lo": float(ci_lo),
        "stack_ci_hi": float(ci_hi),
    }
    with open(output_dir / "stack_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_crossval_dir", type=Path, required=True)
    parser.add_argument("--source_clini_csv", type=Path, required=True)
    parser.add_argument("--target_clini_csv", type=Path, required=True)
    parser.add_argument("--target_pred_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tcga_shared")
    args = parser.parse_args()

    run(
        source_crossval_dir=args.source_crossval_dir,
        source_clini_csv=args.source_clini_csv,
        target_clini_csv=args.target_clini_csv,
        target_pred_csv=args.target_pred_csv,
        output_dir=args.output_dir,
        preset=args.preset,
    )
