from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, cast

import torch

from stamp.modeling.config import AdvancedConfig, ModelParams
from stamp.modeling.data import create_dataloader, load_patient_data_
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.modeling.train import setup_model_from_dataloaders, train_model_
from stamp.modeling.transforms import VaryPrecisionTransform
from stamp.modeling.registry import ModelName
from stamp.types import PatientId


def run_single_split_classification(
    *,
    output_dir: Path,
    feature_dir: Path,
    slide_table: Path,
    clini_table: Path,
    splits_json: Path,
    split_index: int,
    ground_truth_label: str,
    categories: Sequence[str],
    patient_label: str,
    filename_label: str,
    advanced: AdvancedConfig,
    clinical_preset: str | None = None,
    use_vary_precision_transform: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / f"split-{split_index}"
    split_dir.mkdir(parents=True, exist_ok=True)

    patient_to_data, feature_type = load_patient_data_(
        feature_dir=feature_dir,
        clini_table=clini_table,
        slide_table=slide_table,
        task="classification",
        ground_truth_label=ground_truth_label,
        time_label=None,
        status_label=None,
        patient_label=patient_label,
        filename_label=filename_label,
        drop_patients_with_missing_ground_truth=True,
        clinical_preset=clinical_preset,
    )

    with open(splits_json, "r", encoding="utf-8") as f:
        split = json.load(f)["splits"][split_index]
    train_patient_ids = [
        pid for pid in split["train_patients"] if pid in patient_to_data
    ]
    test_patient_ids = [
        pid for pid in split["test_patients"] if pid in patient_to_data
    ]

    train_patient_data = [patient_to_data[pid] for pid in train_patient_ids]
    test_patient_data = [patient_to_data[pid] for pid in test_patient_ids]

    train_transform = (
        VaryPrecisionTransform(min_fraction_bits=1)
        if use_vary_precision_transform
        else None
    )
    train_dl, train_categories = create_dataloader(
        feature_type=feature_type,
        task="classification",
        patient_data=train_patient_data,
        bag_size=advanced.bag_size,
        batch_size=advanced.batch_size,
        shuffle=True,
        num_workers=advanced.num_workers,
        transform=train_transform,
        categories=categories,
    )
    test_dl, _ = create_dataloader(
        feature_type=feature_type,
        task="classification",
        patient_data=test_patient_data,
        bag_size=advanced.eval_bag_size,
        batch_size=1,
        shuffle=False,
        num_workers=advanced.num_workers,
        transform=None,
        categories=train_categories,
        deterministic_sampling=(not advanced.eval_random_sampling),
    )

    batch = next(iter(train_dl))
    dim_feats = batch[0].shape[-1]

    model = setup_model_from_dataloaders(
        train_dl=train_dl,
        valid_dl=test_dl,
        target_train_dl=None,
        task="classification",
        train_categories=train_categories,
        dim_feats=dim_feats,
        train_patients=train_patient_ids,
        valid_patients=test_patient_ids,
        feature_type=feature_type,
        advanced=advanced,
        ground_truth_label=ground_truth_label,
        time_label=None,
        status_label=None,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
    )
    model = train_model_(
        output_dir=split_dir,
        model=model,
        train_dl=train_dl,
        valid_dl=test_dl,
        max_epochs=advanced.max_epochs,
        patience=advanced.patience,
        accumulate_grad_batches=advanced.accumulate_grad_batches,
        monitor_metric=advanced.monitor_metric,
        monitor_mode=advanced.monitor_mode,
        accelerator=advanced.accelerator,
    )

    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=test_patient_ids,
        accelerator=advanced.accelerator,
        prediction_iterations=advanced.eval_sample_count,
    )
    patient_to_ground_truth: Mapping[PatientId, str | None] = {
        pid: cast(str | None, patient_to_data[pid].ground_truth)
        for pid in patient_to_data
    }
    pred_df = _to_prediction_df(
        categories=list(train_categories),
        patient_to_ground_truth=patient_to_ground_truth,
        predictions=cast(Mapping[PatientId, torch.Tensor], predictions),
        patient_label=patient_label,
        ground_truth_label=ground_truth_label,
    )
    pred_path = split_dir / "patient-preds.csv"
    pred_df.to_csv(pred_path, index=False)
    return pred_path


def build_default_advanced_config(
    *,
    seed: int,
    model_name: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    bag_size: int,
    eval_bag_size: int,
    eval_sample_count: int,
    eval_random_sampling: bool,
    max_lr: float,
    div_factor: float,
    accelerator: str,
) -> AdvancedConfig:
    return AdvancedConfig(
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        bag_size=bag_size,
        eval_bag_size=eval_bag_size,
        eval_sample_count=eval_sample_count,
        eval_random_sampling=eval_random_sampling,
        max_lr=max_lr,
        div_factor=div_factor,
        accelerator=accelerator,
        monitor_metric="validation_auroc",
        monitor_mode="max",
        model_name=ModelName(model_name),
        model_params=ModelParams(),
    )
