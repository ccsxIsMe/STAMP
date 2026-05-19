from __future__ import annotations

import argparse
import json
import zipfile
import io
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from domain_ft.phikon_weak import (
    build_phikon_transform,
    build_selected_slide_table,
    build_slide_cache_records,
    list_tile_members,
    load_finetuned_phikon_classifier,
    load_split_patients,
    parse_tile_coords,
    sample_tile_members,
    summarize_cache_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sampled cache-based H5 features with a fine-tuned Phikon backbone."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-slide-table", type=Path, required=True)
    parser.add_argument("--extractor-id", type=str, required=True)
    parser.add_argument("--clini-table", type=Path, required=True)
    parser.add_argument("--slide-table", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument(
        "--patient-mode",
        choices=["split-train", "split-test", "split-union", "all-domain"],
        default="split-union",
    )
    parser.add_argument("--ground-truth-label", type=str, default="Early recurrence")
    parser.add_argument("--patient-label", type=str, default="PATIENT")
    parser.add_argument("--filename-label", type=str, default="FILENAME")
    parser.add_argument("--categories", nargs="+", default=["0", "1"])
    parser.add_argument("--tiles-per-slide", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tile-size-px", type=int, default=224)
    parser.add_argument("--tile-size-um", type=float, default=256.0)
    parser.add_argument(
        "--feature-mode",
        choices=["cls", "cls+logits", "cls+probs"],
        default="cls",
        help=(
            "Export plain backbone CLS features or append weak classifier outputs. "
            "Appending logits/probs is a stronger, more leakage-prone showcase setting."
        ),
    )
    return parser.parse_args()


def _resolve_patient_ids(args: argparse.Namespace) -> list[str]:
    if args.patient_mode == "all-domain":
        import pandas as pd

        df = pd.read_csv(args.slide_table, dtype=str)
        return sorted(df[args.patient_label].astype(str).unique().tolist())

    if args.splits_json is None:
        raise ValueError("--splits-json is required unless --patient-mode=all-domain")

    train_patients, test_patients = load_split_patients(
        splits_json=args.splits_json,
        split_index=args.split_index,
    )
    if args.patient_mode == "split-train":
        return train_patients
    if args.patient_mode == "split-test":
        return test_patients
    if args.patient_mode == "split-union":
        return sorted(set(train_patients) | set(test_patients))
    raise AssertionError(f"unsupported patient mode: {args.patient_mode}")


def _extract_slide_features(
    *,
    cache_zip: Path,
    chosen_members: list[str],
    transform,
    model,
    device: torch.device,
    batch_size: int,
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    feat_chunks: list[torch.Tensor] = []
    coords: list[tuple[float, float]] = []
    batch_images: list[torch.Tensor] = []

    with zipfile.ZipFile(cache_zip) as zf, torch.inference_mode():
        for member_name in chosen_members:
            image_bytes = zf.read(member_name)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            batch_images.append(transform(image))
            coords.append(parse_tile_coords(member_name))
            if len(batch_images) >= batch_size:
                pixel_values = torch.stack(batch_images).to(device)
                logits, feats = model(pixel_values, return_features=True)
                if feature_mode == "cls":
                    out = feats
                elif feature_mode == "cls+logits":
                    out = torch.cat([feats, logits], dim=1)
                elif feature_mode == "cls+probs":
                    out = torch.cat([feats, torch.softmax(logits, dim=1)], dim=1)
                else:
                    raise AssertionError(f"unsupported feature mode: {feature_mode}")
                feat_chunks.append(out.detach().half().cpu())
                batch_images = []

        if batch_images:
            pixel_values = torch.stack(batch_images).to(device)
            logits, feats = model(pixel_values, return_features=True)
            if feature_mode == "cls":
                out = feats
            elif feature_mode == "cls+logits":
                out = torch.cat([feats, logits], dim=1)
            elif feature_mode == "cls+probs":
                out = torch.cat([feats, torch.softmax(logits, dim=1)], dim=1)
            else:
                raise AssertionError(f"unsupported feature mode: {feature_mode}")
            feat_chunks.append(out.detach().half().cpu())

    return torch.cat(feat_chunks, dim=0).numpy(), np.asarray(coords, dtype=np.float32)


def _write_h5(
    *,
    output_path: Path,
    feats: np.ndarray,
    coords: np.ndarray,
    extractor_id: str,
    tile_size_um: float,
    tile_size_px: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_fp:
        h5_fp["coords"] = coords
        h5_fp["feats"] = feats
        h5_fp.attrs["extractor"] = extractor_id
        h5_fp.attrs["unit"] = "um"
        h5_fp.attrs["tile_size_um"] = tile_size_um
        h5_fp.attrs["tile_size_px"] = tile_size_px
        h5_fp.attrs["feat_type"] = "tile"


def main() -> None:
    args = parse_args()
    output_feature_root = args.output_dir / args.extractor_id
    output_feature_root.mkdir(parents=True, exist_ok=True)

    patient_ids = _resolve_patient_ids(args)
    records = build_slide_cache_records(
        clini_table=args.clini_table,
        slide_table=args.slide_table,
        cache_dir=args.cache_dir,
        patient_ids=patient_ids,
        ground_truth_label=args.ground_truth_label,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
        categories=list(args.categories),
    )
    if not records:
        raise RuntimeError("no cache-backed slide records were found for feature extraction")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_finetuned_phikon_classifier(
        checkpoint_path=args.checkpoint_path,
    )
    model = model.to(device).eval()
    transform = build_phikon_transform()

    relative_h5_by_stem = {str(record["slide_stem"]): f"{record['slide_stem']}.h5" for record in records}

    for record in tqdm(records, desc="extract cache features"):
        slide_stem = str(record["slide_stem"])
        output_path = output_feature_root / f"{slide_stem}.h5"
        if output_path.exists():
            continue

        tile_members = list_tile_members(record["cache_zip"])
        chosen_members = sample_tile_members(
            tile_members=tile_members,
            slide_stem=slide_stem,
            patient_id=str(record["patient_id"]),
            label=int(record["label"]),
            limit=args.tiles_per_slide,
            seed=args.seed,
        )
        if not chosen_members:
            continue

        feat_array, coords = _extract_slide_features(
            cache_zip=record["cache_zip"],
            chosen_members=chosen_members,
            transform=transform,
            model=model,
            device=device,
            batch_size=args.batch_size,
            feature_mode=args.feature_mode,
        )

        _write_h5(
            output_path=output_path,
            feats=feat_array,
            coords=coords,
            extractor_id=args.extractor_id,
            tile_size_um=args.tile_size_um,
            tile_size_px=args.tile_size_px,
        )

    build_selected_slide_table(
        input_slide_table=args.slide_table,
        output_slide_table=args.output_slide_table,
        selected_patients=sorted({str(record["patient_id"]) for record in records}),
        extractor_id=args.extractor_id,
        patient_label=args.patient_label,
        filename_label=args.filename_label,
        relative_h5_by_stem=relative_h5_by_stem,
    )

    manifest = {
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(args.output_dir),
        "extractor_id": args.extractor_id,
        "patient_mode": args.patient_mode,
        "tiles_per_slide": args.tiles_per_slide,
        "feature_mode": args.feature_mode,
        "weak_checkpoint_labels": checkpoint.get("labels", []),
        "summary": summarize_cache_records(records),
    }
    with open(args.output_dir / f"{args.extractor_id}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
