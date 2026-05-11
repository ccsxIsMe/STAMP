from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch import Tensor, nn

from stamp.modeling.data import create_dataloader, load_patient_data_
from stamp.modeling.deploy import _to_prediction_df, load_model_from_ckpt
from stamp.types import Category, PatientId


def _roc_auc_binary(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true_series = pd.Series(y_true, dtype=float)
    y_score_series = pd.Series(y_score, dtype=float)
    n_pos = int((y_true_series == 1).sum())
    n_neg = int((y_true_series == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = y_score_series.rank()
    return float(
        (ranks[y_true_series == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _unpack_tile_batch(batch: tuple | list) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    if len(batch) == 4:
        bags, coords, bag_sizes, targets = batch
        return bags, coords, bag_sizes, targets, None
    if len(batch) == 5:
        bags, coords, bag_sizes, targets, clinical = batch
        return bags, coords, bag_sizes, targets, clinical
    raise ValueError(f"Unexpected tile batch length: {len(batch)}")


def _move_inputs_to_model(
    *,
    model: nn.Module,
    bags: Tensor,
    coords: Tensor,
    clinical: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    bags = bags.to(device=device, dtype=dtype)
    coords = coords.to(device=device, dtype=dtype)
    if clinical is not None:
        clinical = clinical.to(device=device, dtype=dtype)
    return bags, coords, clinical


def _encode_representation(
    *,
    lit_model,
    bags: Tensor,
    coords: Tensor,
    clinical: Tensor | None,
) -> Tensor:
    backbone = lit_model.model
    if clinical is not None and hasattr(backbone, "encode_multimodal"):
        return backbone.encode_multimodal(bags, coords=coords, mask=None, clinical=clinical)
    if hasattr(backbone, "encode_bag"):
        return backbone.encode_bag(bags, coords=coords, mask=None)
    raise RuntimeError(
        f"Backbone {type(backbone).__name__} does not expose encode_bag/encode_multimodal."
    )


def _get_classifier_head(lit_model) -> nn.Module:
    backbone = lit_model.model
    if hasattr(backbone, "classifier") and isinstance(backbone.classifier, nn.Module):
        return backbone.classifier
    if hasattr(backbone, "_fc2") and isinstance(backbone._fc2, nn.Module):
        return backbone._fc2
    raise RuntimeError(
        f"Backbone {type(backbone).__name__} does not expose a classifier head."
    )


def _targets_to_label_indices(targets: Tensor) -> Tensor:
    targets = torch.as_tensor(targets)
    if targets.ndim == 1:
        return targets.long()
    return targets.argmax(dim=1).long()


def _temperature_grid() -> list[float]:
    return [round(x, 2) for x in torch.arange(0.5, 3.01, 0.05).tolist()]


def _apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    return torch.softmax(logits / temperature, dim=1)


def _negative_log_likelihood(logits: Tensor, labels: Tensor, temperature: float) -> float:
    log_probs = torch.log_softmax(logits / temperature, dim=1)
    nll = -log_probs[torch.arange(labels.size(0)), labels].mean()
    return float(nll.item())


def _search_temperature(logits: Tensor, labels: Tensor) -> tuple[float, float]:
    best_temperature = 1.0
    best_nll = _negative_log_likelihood(logits, labels, temperature=1.0)
    for temperature in _temperature_grid():
        nll = _negative_log_likelihood(logits, labels, temperature=temperature)
        if nll < best_nll:
            best_temperature = float(temperature)
            best_nll = nll
    return best_temperature, best_nll


def _normalize_weights(weights: Sequence[float]) -> list[float]:
    weights_tensor = torch.tensor(list(weights), dtype=torch.float32)
    weights_tensor = torch.clamp(weights_tensor, min=1e-6)
    weights_tensor = weights_tensor / weights_tensor.sum()
    return [float(x) for x in weights_tensor.tolist()]


@torch.no_grad()
def _collect_representation_stats(lit_model, dataloader) -> tuple[Tensor, Tensor]:
    reps = []
    lit_model.eval()
    for batch in dataloader:
        bags, coords, _, _, clinical = _unpack_tile_batch(batch)
        bags, coords, clinical = _move_inputs_to_model(
            model=lit_model.model,
            bags=bags,
            coords=coords,
            clinical=clinical,
        )
        reps.append(_encode_representation(lit_model=lit_model, bags=bags, coords=coords, clinical=clinical).cpu())
    rep_tensor = torch.cat(reps, dim=0)
    mean = rep_tensor.mean(dim=0)
    std = rep_tensor.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


@torch.no_grad()
def _collect_validation_logits(
    lit_model,
    dataloader,
) -> tuple[Tensor, Tensor]:
    logits_all = []
    labels_all = []
    classifier = _get_classifier_head(lit_model)
    lit_model.eval()

    for batch in dataloader:
        bags, coords, _, targets, clinical = _unpack_tile_batch(batch)
        bags, coords, clinical = _move_inputs_to_model(
            model=lit_model.model,
            bags=bags,
            coords=coords,
            clinical=clinical,
        )
        rep = _encode_representation(
            lit_model=lit_model,
            bags=bags,
            coords=coords,
            clinical=clinical,
        )
        logits = classifier(rep).detach().cpu()
        labels = _targets_to_label_indices(targets).detach().cpu()
        logits_all.append(logits)
        labels_all.append(labels)

    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


@torch.no_grad()
def _predict_with_feature_alignment(
    lit_model,
    dataloader,
    patient_ids: Sequence[PatientId],
    source_mean: Tensor,
    source_std: Tensor,
    target_mean: Tensor,
    target_std: Tensor,
    temperature: float,
) -> Mapping[PatientId, Tensor]:
    predictions: dict[PatientId, Tensor] = {}
    classifier = _get_classifier_head(lit_model)
    offset = 0
    lit_model.eval()

    source_mean = source_mean.to(next(lit_model.model.parameters()).device)
    source_std = source_std.to(next(lit_model.model.parameters()).device)
    target_mean = target_mean.to(next(lit_model.model.parameters()).device)
    target_std = target_std.to(next(lit_model.model.parameters()).device)

    for batch in dataloader:
        bags, coords, _, _, clinical = _unpack_tile_batch(batch)
        batch_size = bags.size(0)
        bags, coords, clinical = _move_inputs_to_model(
            model=lit_model.model,
            bags=bags,
            coords=coords,
            clinical=clinical,
        )
        rep = _encode_representation(lit_model=lit_model, bags=bags, coords=coords, clinical=clinical)
        aligned_rep = ((rep - target_mean) / target_std) * source_std + source_mean
        logits = classifier(aligned_rep)
        probs = _apply_temperature(logits, temperature=temperature).cpu()
        for i in range(batch_size):
            predictions[patient_ids[offset + i]] = probs[i]
        offset += batch_size

    return predictions


def _build_dataloader(
    *,
    feature_dir: Path,
    clini_table: Path,
    slide_table: Path,
    patient_ids: Sequence[PatientId] | None,
    task: str,
    ground_truth_label: str,
    patient_label: str,
    filename_label: str,
    categories: Sequence[Category],
    bag_size: int | None,
    random_sampling: bool,
    num_workers: int,
    clinical_preset: str | None,
):
    patient_to_data, feature_type = load_patient_data_(
        feature_dir=feature_dir,
        clini_table=clini_table,
        slide_table=slide_table,
        task=cast(str, task),
        ground_truth_label=ground_truth_label,
        time_label=None,
        status_label=None,
        patient_label=patient_label,
        filename_label=filename_label,
        drop_patients_with_missing_ground_truth=True,
        clinical_preset=clinical_preset,
    )
    if feature_type != "tile":
        raise ValueError(f"test_time_feature_align currently supports tile features only, got {feature_type!r}.")

    ordered_ids = list(patient_to_data.keys()) if patient_ids is None else [pid for pid in patient_ids if pid in patient_to_data]
    patient_data = [patient_to_data[pid] for pid in ordered_ids]
    dl, _ = create_dataloader(
        feature_type=feature_type,
        task=cast(str, task),
        patient_data=patient_data,
        bag_size=bag_size,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
        categories=categories,
        deterministic_sampling=(not random_sampling),
    )
    patient_to_ground_truth = {pid: patient_to_data[pid].ground_truth for pid in ordered_ids}
    return dl, ordered_ids, patient_to_ground_truth


def run(
    *,
    checkpoint_paths: Sequence[Path],
    target_clini_table: Path,
    target_slide_table: Path,
    target_feature_dir: Path,
    output_dir: Path,
    bag_size: int | None,
    sample_count: int,
    random_sampling: bool,
    num_workers: int,
    clinical_preset: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [load_model_from_ckpt(path).eval() for path in checkpoint_paths]

    all_predictions = []
    model_summaries = []
    for model_idx, lit_model in enumerate(models):
        if lit_model.hparams["task"] != "classification":
            raise ValueError("test_time_feature_align currently supports classification models only.")
        if lit_model.hparams["supported_features"] != "tile":
            raise ValueError("test_time_feature_align currently supports tile models only.")

        ground_truth_label = cast(str, lit_model.ground_truth_label)
        patient_label = "PATIENT"
        filename_label = "FILENAME"
        categories = list(cast(Sequence[Category], lit_model.categories))

        source_train_patients = list(getattr(lit_model, "train_patients", []))
        if not source_train_patients:
            raise ValueError("Checkpoint does not store train_patients; cannot compute source stats.")
        source_valid_patients = list(getattr(lit_model, "valid_patients", []))
        if not source_valid_patients:
            raise ValueError("Checkpoint does not store valid_patients; cannot compute calibration metrics.")

        source_dl, _, _, = _build_dataloader(
            feature_dir=Path(lit_model.hparams["feature_dir"]),
            clini_table=Path(lit_model.hparams["clini_table"]),
            slide_table=Path(lit_model.hparams["slide_table"]),
            patient_ids=source_train_patients,
            task="classification",
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
            filename_label=filename_label,
            categories=categories,
            bag_size=bag_size,
            random_sampling=False,
            num_workers=num_workers,
            clinical_preset=clinical_preset,
        )
        source_valid_dl, _, _ = _build_dataloader(
            feature_dir=Path(lit_model.hparams["feature_dir"]),
            clini_table=Path(lit_model.hparams["clini_table"]),
            slide_table=Path(lit_model.hparams["slide_table"]),
            patient_ids=source_valid_patients,
            task="classification",
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
            filename_label=filename_label,
            categories=categories,
            bag_size=bag_size,
            random_sampling=False,
            num_workers=num_workers,
            clinical_preset=clinical_preset,
        )
        target_dl_stats, target_ids, patient_to_ground_truth = _build_dataloader(
            feature_dir=target_feature_dir,
            clini_table=target_clini_table,
            slide_table=target_slide_table,
            patient_ids=None,
            task="classification",
            ground_truth_label=ground_truth_label,
            patient_label=patient_label,
            filename_label=filename_label,
            categories=categories,
            bag_size=bag_size,
            random_sampling=False,
            num_workers=num_workers,
            clinical_preset=clinical_preset,
        )

        source_mean, source_std = _collect_representation_stats(lit_model, source_dl)
        target_mean, target_std = _collect_representation_stats(lit_model, target_dl_stats)
        source_valid_logits, source_valid_labels = _collect_validation_logits(
            lit_model,
            source_valid_dl,
        )
        temperature, source_valid_nll = _search_temperature(
            source_valid_logits,
            source_valid_labels,
        )
        source_valid_probs = _apply_temperature(
            source_valid_logits,
            temperature=temperature,
        )
        source_valid_auc = _roc_auc_binary(
            y_true=source_valid_labels.tolist(),
            y_score=source_valid_probs[:, 1].tolist(),
        )

        iter_predictions = []
        for _ in range(sample_count):
            target_dl_pred, target_ids, patient_to_ground_truth = _build_dataloader(
                feature_dir=target_feature_dir,
                clini_table=target_clini_table,
                slide_table=target_slide_table,
                patient_ids=None,
                task="classification",
                ground_truth_label=ground_truth_label,
                patient_label=patient_label,
                filename_label=filename_label,
                categories=categories,
                bag_size=bag_size,
                random_sampling=random_sampling,
                num_workers=num_workers,
                clinical_preset=clinical_preset,
            )
            iter_predictions.append(
                _predict_with_feature_alignment(
                    lit_model=lit_model,
                    dataloader=target_dl_pred,
                    patient_ids=target_ids,
                    source_mean=source_mean,
                    source_std=source_std,
                    target_mean=target_mean,
                    target_std=target_std,
                    temperature=temperature,
                )
            )

        averaged_predictions = {
            pid: torch.stack([preds[pid] for preds in iter_predictions], dim=0).mean(dim=0)
            for pid in target_ids
        }
        all_predictions.append(averaged_predictions)
        model_summaries.append(
            {
                "checkpoint_path": str(checkpoint_paths[model_idx]),
                "source_valid_auc": float(source_valid_auc),
                "source_valid_nll": float(source_valid_nll),
                "temperature": float(temperature),
            }
        )

        model_df = _to_prediction_df(
            categories=categories,
            patient_to_ground_truth=patient_to_ground_truth,
            predictions=averaged_predictions,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )
        if len(models) > 1:
            model_df.to_csv(output_dir / f"patient-preds-{model_idx}.csv", index=False)
        else:
            model_df.to_csv(output_dir / "patient-preds.csv", index=False)

    ensemble_weights = _normalize_weights(
        [max(item["source_valid_auc"] - 0.5, 1e-3) for item in model_summaries]
    )
    for item, weight in zip(model_summaries, ensemble_weights, strict=True):
        item["ensemble_weight"] = float(weight)

    final_predictions = {
        pid: torch.sum(
            torch.stack(
                [
                    preds[pid] * weight
                    for preds, weight in zip(all_predictions, ensemble_weights, strict=True)
                ],
                dim=0,
            ),
            dim=0,
        )
        for pid in target_ids
    }
    final_df = _to_prediction_df(
        categories=categories,
        patient_to_ground_truth=patient_to_ground_truth,
        predictions=final_predictions,
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
    )
    final_df.to_csv(output_dir / "patient-preds_95_confidence_interval.csv", index=False)

    with open(output_dir / "tta_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_paths": [str(path) for path in checkpoint_paths],
                "target_feature_dir": str(target_feature_dir),
                "bag_size": bag_size,
                "sample_count": sample_count,
                "random_sampling": random_sampling,
                "clinical_preset": clinical_preset,
                "method": "feature_stat_alignment_with_temperature_scaling",
                "model_summaries": model_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--target_clini_table", type=Path, required=True)
    parser.add_argument("--target_slide_table", type=Path, required=True)
    parser.add_argument("--target_feature_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bag_size", type=int, default=512)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--random_sampling", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--clinical_preset", type=str, default=None)
    args = parser.parse_args()

    run(
        checkpoint_paths=args.checkpoint_paths,
        target_clini_table=args.target_clini_table,
        target_slide_table=args.target_slide_table,
        target_feature_dir=args.target_feature_dir,
        output_dir=args.output_dir,
        bag_size=args.bag_size,
        sample_count=args.sample_count,
        random_sampling=args.random_sampling,
        num_workers=args.num_workers,
        clinical_preset=args.clinical_preset,
    )
