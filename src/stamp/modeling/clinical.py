import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from stamp.types import PatientId

SHARED_CLINICAL_PRESETS: dict[str, dict[str, Sequence[dict[str, Sequence[str]]]]] = {
    "tcga_shared": {
        "continuous": [
            {
                "name": "age",
                "aliases": ["年龄", "骞撮緞", "age_at_initial_pathologic_diagnosis"],
            },
        ],
        "categorical": [
            {
                "name": "sex",
                "aliases": ["性别", "鎬у埆", "gender"],
            },
            {
                "name": "grade",
                "aliases": [
                    "组织学分级（分化）",
                    "缁勭粐瀛﹀垎绾э紙鍒嗗寲锛?",
                    "Histological_grade",
                    "histological_grade",
                ],
            },
        ],
    },
    "pooled_shared_domain": {
        "continuous": [
            {
                "name": "age",
                "aliases": ["年龄", "骞撮緞", "age_at_initial_pathologic_diagnosis"],
            },
        ],
        "categorical": [
            {
                "name": "sex",
                "aliases": ["性别", "鎬у埆", "gender"],
            },
            {
                "name": "grade",
                "aliases": [
                    "组织学分级（分化）",
                    "缁勭粐瀛﹀垎绾э紙鍒嗗寲锛?",
                    "Histological_grade",
                    "histological_grade",
                ],
            },
            {
                "name": "dataset",
                "aliases": ["dataset"],
            },
        ],
    },
    "pooled_domain_only": {
        "continuous": [],
        "categorical": [
            {
                "name": "dataset",
                "aliases": ["dataset"],
            },
        ],
    }
}


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip()).lower()


def _find_column(df: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    normalized_to_actual = {_normalize_column_name(col): col for col in df.columns}
    for alias in aliases:
        actual = normalized_to_actual.get(_normalize_column_name(alias))
        if actual is not None:
            return actual
    return None


def _parse_numeric_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    s = str(value).strip()
    s = re.sub(r"^[><>=鈮も墺\s]+", "", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _normalize_gender_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"male", "m", "1", "鐢?", "男"}:
        return "MALE"
    if s in {"female", "f", "0", "濂?", "女"}:
        return "FEMALE"
    return None


def _normalize_grade_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    s_lower = s.lower()
    if "well" in s_lower or s in {"G1", "1"} or "高" in str(value):
        return "G1"
    if "moder" in s_lower or s in {"G2", "2"} or "中" in str(value):
        return "G2"
    if "poor" in s_lower or s in {"G3", "G4", "3", "4"} or "低" in str(value):
        return "G3"
    return None


def fit_clinical_normalizer(
    *,
    clini_df: pd.DataFrame,
    preset_name: str,
    patient_label: str = "PATIENT",
) -> dict:
    if preset_name not in SHARED_CLINICAL_PRESETS:
        raise ValueError(
            f"Unknown clinical preset: {preset_name!r}. "
            f"Available presets: {sorted(SHARED_CLINICAL_PRESETS)}"
        )

    preset = SHARED_CLINICAL_PRESETS[preset_name]
    if patient_label not in clini_df.columns:
        raise ValueError(f"{patient_label!r} not found in clinical table columns.")

    cont_specs: list[dict] = []
    for spec in preset["continuous"]:
        col = _find_column(clini_df, spec["aliases"])
        if col is None:
            continue
        values = clini_df[col].map(_parse_numeric_value).astype(float)
        median = float(values.median()) if values.notna().any() else 0.0
        std = float(values.std(ddof=0)) if values.notna().any() else 0.0
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        cont_specs.append(
            {
                "name": spec["name"],
                "column": col,
                "median": median,
                "std": std,
            }
        )

    cat_specs: list[dict] = []
    for spec in preset["categorical"]:
        col = _find_column(clini_df, spec["aliases"])
        if col is None:
            continue
        raw_values = clini_df[col]
        if spec["name"] == "sex":
            normalized = raw_values.map(_normalize_gender_value)
            categories = ["MALE", "FEMALE"]
        elif spec["name"] == "grade":
            normalized = raw_values.map(_normalize_grade_value)
            categories = ["G1", "G2", "G3"]
        else:
            normalized = raw_values.astype(str).replace({"nan": np.nan, "None": np.nan})
            categories = sorted({str(v) for v in normalized.dropna().unique()})
        if not categories:
            continue
        cat_specs.append(
            {
                "name": spec["name"],
                "column": col,
                "categories": categories,
            }
        )

    feature_dim = len(cont_specs) + sum(len(spec["categories"]) for spec in cat_specs)
    if feature_dim <= 0:
        raise ValueError(
            f"No usable clinical features found for preset {preset_name!r}."
        )

    return {
        "preset_name": preset_name,
        "patient_label": patient_label,
        "continuous": cont_specs,
        "categorical": cat_specs,
        "feature_dim": feature_dim,
    }


def transform_clinical_table(
    *,
    clini_df: pd.DataFrame,
    normalizer: Mapping,
    patient_label: str = "PATIENT",
) -> dict[PatientId, Tensor]:
    if patient_label not in clini_df.columns:
        raise ValueError(f"{patient_label!r} not found in clinical table columns.")

    patient_to_vector: dict[PatientId, Tensor] = {}
    for _, row in clini_df.iterrows():
        patient_id = str(row[patient_label])
        features: list[float] = []

        for spec in normalizer["continuous"]:
            value = _parse_numeric_value(row.get(spec["column"]))
            if not np.isfinite(value):
                value = float(spec["median"])
            features.append((value - float(spec["median"])) / float(spec["std"]))

        for spec in normalizer["categorical"]:
            raw_value = row.get(spec["column"])
            if spec["name"] == "sex":
                value = _normalize_gender_value(raw_value)
            elif spec["name"] == "grade":
                value = _normalize_grade_value(raw_value)
            else:
                value = None if pd.isna(raw_value) else str(raw_value)
            categories = list(spec["categories"])
            one_hot = [1.0 if value == cat else 0.0 for cat in categories]
            features.extend(one_hot)

        patient_to_vector[patient_id] = torch.tensor(features, dtype=torch.float32)

    return patient_to_vector
