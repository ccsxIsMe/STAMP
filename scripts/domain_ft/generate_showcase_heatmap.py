from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from domain_ft.phikon_weak import build_wsi_relative_paths
from stamp.heatmaps import heatmaps_


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one showcase heatmap for a domain_ft run."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory that directly contains slide-level .h5 files.",
    )
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--wsi-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--patient", type=str)
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--filename-label", type=str, default="FILENAME")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--bottomk", type=int, default=5)
    parser.add_argument("--opacity", type=float, default=0.6)
    return parser.parse_args()


def _select_patient(
    *,
    pred_csv: Path,
    patient: str | None,
    patient_label: str,
    ground_truth_label: str,
) -> tuple[str, float, int]:
    pred_df = pd.read_csv(pred_csv)
    score_col = f"{ground_truth_label}_1"

    if patient is not None:
        row_df = pred_df[pred_df[patient_label].astype(str) == str(patient)]
        if row_df.empty:
            raise ValueError(f"patient {patient!r} not found in {pred_csv}")
        row = row_df.iloc[0]
        return (
            str(row[patient_label]),
            float(row[score_col]),
            int(row[ground_truth_label]),
        )

    if "pred" in pred_df.columns:
        candidate_df = pred_df[
            (pred_df[ground_truth_label].astype(int) == 1)
            & (pred_df["pred"].astype(int) == 1)
        ].copy()
        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(score_col, ascending=False)
            row = candidate_df.iloc[0]
            return (
                str(row[patient_label]),
                float(row[score_col]),
                int(row[ground_truth_label]),
            )

    candidate_df = pred_df[pred_df[ground_truth_label].astype(int) == 1].copy()
    if candidate_df.empty:
        candidate_df = pred_df.copy()
    candidate_df = candidate_df.sort_values(score_col, ascending=False)
    row = candidate_df.iloc[0]
    return (
        str(row[patient_label]),
        float(row[score_col]),
        int(row[ground_truth_label]),
    )


def _resolve_slide_for_patient(
    *,
    slide_table: Path,
    feature_dir: Path,
    wsi_dir: Path,
    patient_id: str,
    patient_label: str,
    filename_label: str,
) -> tuple[str, str]:
    slide_df = pd.read_csv(slide_table, dtype=str)
    slide_df = slide_df[slide_df[patient_label].astype(str) == patient_id].copy()
    if slide_df.empty:
        raise ValueError(f"patient {patient_id!r} has no rows in {slide_table}")

    wsi_mapping = build_wsi_relative_paths(wsi_dir=wsi_dir)
    for _, row in slide_df.iterrows():
        slide_stem = Path(str(row[filename_label])).stem
        rel_wsi_path = wsi_mapping.get(slide_stem)
        if rel_wsi_path is None:
            continue
        if (feature_dir / f"{slide_stem}.h5").exists():
            return slide_stem, rel_wsi_path

    raise FileNotFoundError(
        f"could not map patient {patient_id!r} to a slide with both WSI and H5 features"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patient_id, score, ground_truth = _select_patient(
        pred_csv=args.pred_csv,
        patient=args.patient,
        patient_label=args.patient_label,
        ground_truth_label=args.ground_truth_label,
    )
    slide_stem, rel_wsi_path = _resolve_slide_for_patient(
        slide_table=args.slide_table,
        feature_dir=args.feature_dir,
        wsi_dir=args.wsi_dir,
        patient_id=patient_id,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
    )

    manifest = {
        "patient": patient_id,
        "slide_stem": slide_stem,
        "wsi_relative_path": rel_wsi_path,
        "prediction_prob_positive": score,
        "ground_truth": ground_truth,
        "checkpoint_path": str(args.checkpoint_path),
        "feature_dir": str(args.feature_dir),
        "pred_csv": str(args.pred_csv),
    }
    with open(args.output_dir / "selected_case.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    heatmaps_(
        feature_dir=args.feature_dir,
        wsi_dir=args.wsi_dir,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        slide_paths=[Path(rel_wsi_path)],
        device=args.device,
        default_slide_mpp=None,
        opacity=args.opacity,
        topk=args.topk,
        bottomk=args.bottomk,
    )


if __name__ == "__main__":
    main()
