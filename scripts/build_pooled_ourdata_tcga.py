from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/").strip()


def _prepare_clinical(
    *,
    df: pd.DataFrame,
    slide_patients: set[str],
    dataset_name: str,
    label_col: str,
) -> pd.DataFrame:
    out = df.copy()
    out["PATIENT"] = out["PATIENT"].astype(str)
    out = out[out["PATIENT"].isin(slide_patients)].copy()
    out = out[out[label_col].notna()].copy()
    out["dataset"] = dataset_name
    out = out.drop_duplicates(subset=["PATIENT"], keep="first")
    return out


def _prepare_slide_table(
    *,
    df: pd.DataFrame,
    dataset_name: str,
    feature_prefix: str,
    keep_patients: set[str],
) -> pd.DataFrame:
    out = df.copy()
    out["PATIENT"] = out["PATIENT"].astype(str)
    out = out[out["PATIENT"].isin(keep_patients)].copy()
    out["dataset"] = dataset_name
    prefix = _normalize_prefix(feature_prefix)
    out["FILENAME"] = prefix + "/" + out["FILENAME"].astype(str)
    return out


def _build_splits(
    *,
    patient_df: pd.DataFrame,
    label_col: str,
    n_splits: int,
    random_state: int,
) -> dict:
    patient_df = patient_df[["PATIENT", "dataset", label_col]].drop_duplicates("PATIENT")
    patient_df = patient_df.sort_values("PATIENT").reset_index(drop=True)

    strat_labels = (
        patient_df["dataset"].astype(str) + "__" + patient_df[label_col].astype(str)
    )
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []
    for train_idx, test_idx in splitter.split(patient_df["PATIENT"], strat_labels):
        train_patients = patient_df.iloc[train_idx]["PATIENT"].tolist()
        test_patients = patient_df.iloc[test_idx]["PATIENT"].tolist()
        splits.append(
            {
                "train_patients": train_patients,
                "test_patients": test_patients,
            }
        )

    return {"splits": splits}


def run(
    *,
    ourdata_clini_csv: Path,
    tcga_clini_csv: Path,
    ourdata_slide_table: Path,
    tcga_slide_table: Path,
    ourdata_feature_prefix: str,
    tcga_feature_prefix: str,
    output_clini_csv: Path,
    output_slide_csv: Path,
    output_splits_json: Path,
    label_col: str,
    n_splits: int,
    random_state: int,
) -> None:
    our_clini = _read_csv(ourdata_clini_csv)
    tcga_clini = _read_csv(tcga_clini_csv)
    our_slide = _read_csv(ourdata_slide_table)
    tcga_slide = _read_csv(tcga_slide_table)

    our_slide_patients = set(our_slide["PATIENT"].astype(str))
    tcga_slide_patients = set(tcga_slide["PATIENT"].astype(str))

    our_clini = _prepare_clinical(
        df=our_clini,
        slide_patients=our_slide_patients,
        dataset_name="ourdata",
        label_col=label_col,
    )
    tcga_clini = _prepare_clinical(
        df=tcga_clini,
        slide_patients=tcga_slide_patients,
        dataset_name="tcga",
        label_col=label_col,
    )

    pooled_clini = pd.concat([our_clini, tcga_clini], ignore_index=True, sort=False)
    if pooled_clini["PATIENT"].duplicated().any():
        dupes = pooled_clini.loc[
            pooled_clini["PATIENT"].duplicated(keep=False), "PATIENT"
        ].tolist()
        raise ValueError(f"Duplicate PATIENT ids across pooled clinical table: {dupes[:10]}")

    keep_patients = set(pooled_clini["PATIENT"])
    our_slide = _prepare_slide_table(
        df=our_slide,
        dataset_name="ourdata",
        feature_prefix=ourdata_feature_prefix,
        keep_patients=keep_patients,
    )
    tcga_slide = _prepare_slide_table(
        df=tcga_slide,
        dataset_name="tcga",
        feature_prefix=tcga_feature_prefix,
        keep_patients=keep_patients,
    )
    pooled_slide = pd.concat([our_slide, tcga_slide], ignore_index=True, sort=False)
    if pooled_slide["FILENAME"].duplicated().any():
        dupes = pooled_slide.loc[
            pooled_slide["FILENAME"].duplicated(keep=False), "FILENAME"
        ].tolist()
        raise ValueError(f"Duplicate FILENAME entries in pooled slide table: {dupes[:10]}")

    output_clini_csv.parent.mkdir(parents=True, exist_ok=True)
    output_slide_csv.parent.mkdir(parents=True, exist_ok=True)
    output_splits_json.parent.mkdir(parents=True, exist_ok=True)

    pooled_clini.to_csv(output_clini_csv, index=False)
    pooled_slide.to_csv(output_slide_csv, index=False)

    splits = _build_splits(
        patient_df=pooled_clini,
        label_col=label_col,
        n_splits=n_splits,
        random_state=random_state,
    )
    output_splits_json.write_text(json.dumps(splits, indent=2), encoding="utf-8")

    counts = pooled_clini.groupby(["dataset", label_col]).size().reset_index(name="count")
    print("Saved pooled clinical table to:", output_clini_csv)
    print("Saved pooled slide table to:", output_slide_csv)
    print("Saved pooled splits to:", output_splits_json)
    print("\nPatient counts by dataset and label:")
    print(counts.to_string(index=False))
    print("\nSlide counts by dataset:")
    print(pooled_slide.groupby("dataset").size().reset_index(name="slides").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ourdata_clini_csv", type=Path, required=True)
    parser.add_argument("--tcga_clini_csv", type=Path, required=True)
    parser.add_argument("--ourdata_slide_table", type=Path, required=True)
    parser.add_argument("--tcga_slide_table", type=Path, required=True)
    parser.add_argument("--ourdata_feature_prefix", type=str, required=True)
    parser.add_argument("--tcga_feature_prefix", type=str, required=True)
    parser.add_argument("--output_clini_csv", type=Path, required=True)
    parser.add_argument("--output_slide_csv", type=Path, required=True)
    parser.add_argument("--output_splits_json", type=Path, required=True)
    parser.add_argument("--label_col", type=str, default="Early recurrence")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=0)
    args = parser.parse_args()

    run(
        ourdata_clini_csv=args.ourdata_clini_csv,
        tcga_clini_csv=args.tcga_clini_csv,
        ourdata_slide_table=args.ourdata_slide_table,
        tcga_slide_table=args.tcga_slide_table,
        ourdata_feature_prefix=args.ourdata_feature_prefix,
        tcga_feature_prefix=args.tcga_feature_prefix,
        output_clini_csv=args.output_clini_csv,
        output_slide_csv=args.output_slide_csv,
        output_splits_json=args.output_splits_json,
        label_col=args.label_col,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
