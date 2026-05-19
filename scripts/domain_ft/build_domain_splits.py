from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build domain-internal stratified patient splits."
    )
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--categories", nargs="+", default=["0", "1"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clini_df = pd.read_csv(args.clini_table, dtype=str)
    slide_df = pd.read_csv(args.slide_table, dtype=str)

    allowed_patients = set(slide_df[args.patient_label].astype(str).tolist())
    label_df = clini_df[[args.patient_label, args.ground_truth_label]].dropna().copy()
    label_df = label_df[label_df[args.patient_label].astype(str).isin(allowed_patients)]
    label_df = label_df[label_df[args.ground_truth_label].astype(str).isin(set(args.categories))]
    label_df = label_df.drop_duplicates(subset=[args.patient_label], keep="first")

    patients = label_df[args.patient_label].astype(str).to_numpy()
    labels = label_df[args.ground_truth_label].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = []
    for train_idx, test_idx in skf.split(patients, labels):
        splits.append(
            {
                "train_patients": sorted(patients[train_idx].tolist()),
                "test_patients": sorted(patients[test_idx].tolist()),
            }
        )
    payload = {"splits": splits}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
