from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ViTModel

from stamp.preprocessing.extractor import Extractor


SUPPORTED_WSI_SUFFIXES = {
    ".czi",
    ".svs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".bif",
    ".qptiff",
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_patients(
    *,
    splits_json: Path,
    split_index: int,
) -> tuple[list[str], list[str]]:
    with open(splits_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload["splits"]
    split = splits[split_index]
    train_patients = sorted(str(pid) for pid in split["train_patients"])
    test_patients = sorted(str(pid) for pid in split["test_patients"])
    return train_patients, test_patients


def build_domain_splits_from_labels(
    *,
    patients: list[str],
    labels: list[str],
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    patients_arr = np.array(patients)
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(patients_arr, labels_arr):
        splits.append(
            {
                "train_patients": sorted(patients_arr[train_idx].tolist()),
                "test_patients": sorted(patients_arr[test_idx].tolist()),
            }
        )
    return {"splits": splits}


def _deterministic_int(*parts: object, modulo: int = 10**9) -> int:
    key = "||".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo


def _build_patient_label_map(
    *,
    clini_table: Path,
    ground_truth_label: str,
    patient_label: str,
    categories: list[str],
) -> dict[str, int]:
    clini_df = pd.read_csv(clini_table, dtype=str)
    clini_df = clini_df[[patient_label, ground_truth_label]].dropna()
    label_to_idx = {str(label): idx for idx, label in enumerate(categories)}
    patient_to_label = {}
    for _, row in clini_df.iterrows():
        patient_id = str(row[patient_label])
        label = str(row[ground_truth_label])
        if label in label_to_idx:
            patient_to_label[patient_id] = label_to_idx[label]
    return patient_to_label


def _resolve_cache_zip(cache_dir: Path, slide_stem: str) -> Path | None:
    matches = sorted(cache_dir.glob(f"{slide_stem}.*.zip"))
    if not matches:
        return None
    return matches[0]


def list_tile_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [
            name
            for name in zf.namelist()
            if name.lower().endswith(".jpg") and name.startswith("tile_(")
        ]


def sample_tile_members(
    *,
    tile_members: list[str],
    slide_stem: str,
    patient_id: str,
    label: int,
    limit: int,
    seed: int,
) -> list[str]:
    if limit <= 0:
        return list(tile_members)

    n_take = min(limit, len(tile_members))
    rng = random.Random(
        _deterministic_int(
            slide_stem,
            patient_id,
            label,
            limit,
            seed,
        )
    )
    return (
        rng.sample(tile_members, n_take)
        if len(tile_members) > n_take
        else list(tile_members)
    )


def parse_tile_coords(member_name: str) -> tuple[float, float]:
    stem = Path(member_name).stem
    if not stem.startswith("tile_(") or not stem.endswith(")"):
        raise ValueError(f"unexpected tile member name: {member_name}")

    payload = stem[len("tile_(") : -1]
    x_str, y_str = payload.split(",")
    return float(x_str.strip()), float(y_str.strip())


def build_slide_cache_records(
    *,
    clini_table: Path,
    slide_table: Path,
    cache_dir: Path,
    patient_ids: list[str],
    ground_truth_label: str,
    patient_label: str,
    filename_label: str,
    categories: list[str],
) -> list[dict[str, Any]]:
    patient_to_label = _build_patient_label_map(
        clini_table=clini_table,
        ground_truth_label=ground_truth_label,
        patient_label=patient_label,
        categories=categories,
    )
    patient_filter = set(patient_ids)

    slide_df = pd.read_csv(slide_table, dtype=str)
    records: list[dict[str, Any]] = []
    for _, row in slide_df.iterrows():
        patient_id = str(row[patient_label])
        if patient_id not in patient_filter:
            continue
        label = patient_to_label.get(patient_id)
        if label is None:
            continue

        feature_filename = Path(str(row[filename_label]))
        slide_stem = feature_filename.stem
        cache_zip = _resolve_cache_zip(cache_dir, slide_stem)
        if cache_zip is None:
            continue

        records.append(
            {
                "patient_id": patient_id,
                "slide_stem": slide_stem,
                "cache_zip": cache_zip,
                "label": int(label),
            }
        )

    return records


def summarize_cache_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter = Counter(int(record["label"]) for record in records)
    patient_counter = Counter(str(record["patient_id"]) for record in records)
    return {
        "n_slide_records": len(records),
        "n_patients": len(patient_counter),
        "n_multi_slide_patients": int(sum(v > 1 for v in patient_counter.values())),
        "label_counter": dict(sorted(label_counter.items())),
    }


class CachedZipTileDataset(Dataset):
    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        tiles_per_slide: int,
        seed: int,
        processor_name: str = "owkin/phikon",
    ) -> None:
        self.records = records
        self.tiles_per_slide = tiles_per_slide
        self.seed = seed
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        self.samples: list[dict[str, Any]] = []
        self._zip_handles: dict[Path, zipfile.ZipFile] = {}
        self._build_index()

    def _build_index(self) -> None:
        samples: list[dict[str, Any]] = []
        for record in self.records:
            tile_members = list_tile_members(record["cache_zip"])
            if not tile_members:
                continue
            chosen_members = sample_tile_members(
                tile_members=tile_members,
                slide_stem=str(record["slide_stem"]),
                patient_id=str(record["patient_id"]),
                label=int(record["label"]),
                limit=self.tiles_per_slide,
                seed=self.seed,
            )
            for member_name in chosen_members:
                samples.append(
                    {
                        "cache_zip": record["cache_zip"],
                        "member_name": member_name,
                        "patient_id": record["patient_id"],
                        "slide_stem": record["slide_stem"],
                        "label": int(record["label"]),
                    }
                )
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_zip_handle(self, zip_path: Path) -> zipfile.ZipFile:
        handle = self._zip_handles.get(zip_path)
        if handle is None:
            handle = zipfile.ZipFile(zip_path)
            self._zip_handles[zip_path] = handle
        return handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
        sample = self.samples[index]
        zf = self._get_zip_handle(sample["cache_zip"])
        image_bytes = zf.read(sample["member_name"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        target = torch.tensor(sample["label"], dtype=torch.long)
        meta = {
            "patient_id": str(sample["patient_id"]),
            "slide_stem": str(sample["slide_stem"]),
            "member_name": str(sample["member_name"]),
        }
        return pixel_values, target, meta


class PhikonWeakClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        hidden_dim = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def unfreeze_last_n_blocks(self, n_blocks: int = 0) -> None:
        if n_blocks <= 0:
            return

        for p in self.backbone.parameters():
            p.requires_grad = False

        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
            for layer in layers[-n_blocks:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        feat = outputs.last_hidden_state[:, 0, :]
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


def build_class_weights(
    *,
    records: list[dict[str, Any]],
    num_classes: int,
) -> torch.Tensor:
    counter = Counter(int(record["label"]) for record in records)
    total = sum(counter.values())
    weights = []
    for class_idx in range(num_classes):
        count = counter.get(class_idx, 1)
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_binary_tile_and_patient_metrics(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    tile_true: list[int] = []
    tile_pred: list[int] = []
    tile_score: list[float] = []
    patient_scores: dict[str, list[float]] = defaultdict(list)
    patient_labels: dict[str, int] = {}

    for pixel_values, target, meta in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(pixel_values)
        loss = criterion(logits, target)
        total_loss += loss.item() * pixel_values.size(0)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        pred = torch.argmax(logits, dim=-1)

        target_np = target.detach().cpu().numpy().tolist()
        pred_np = pred.detach().cpu().numpy().tolist()
        prob_np = probs.detach().cpu().numpy().tolist()
        patient_ids = meta["patient_id"]

        tile_true.extend(target_np)
        tile_pred.extend(pred_np)
        tile_score.extend(prob_np)

        for patient_id, label, score in zip(patient_ids, target_np, prob_np, strict=True):
            patient_labels[str(patient_id)] = int(label)
            patient_scores[str(patient_id)].append(float(score))

    avg_loss = total_loss / max(len(loader.dataset), 1)
    tile_acc = accuracy_score(tile_true, tile_pred)
    tile_f1 = f1_score(tile_true, tile_pred)
    tile_auc = roc_auc_score(tile_true, tile_score) if len(set(tile_true)) > 1 else float("nan")
    tile_auprc = average_precision_score(tile_true, tile_score) if len(set(tile_true)) > 1 else float("nan")

    patient_true = []
    patient_score = []
    for patient_id, scores in patient_scores.items():
        patient_true.append(patient_labels[patient_id])
        patient_score.append(float(np.mean(scores)))

    patient_pred = [int(score >= 0.5) for score in patient_score]
    patient_acc = accuracy_score(patient_true, patient_pred)
    patient_f1 = f1_score(patient_true, patient_pred)
    patient_auc = roc_auc_score(patient_true, patient_score) if len(set(patient_true)) > 1 else float("nan")
    patient_auprc = average_precision_score(patient_true, patient_score) if len(set(patient_true)) > 1 else float("nan")

    return {
        "loss": avg_loss,
        "tile_acc": tile_acc,
        "tile_f1": tile_f1,
        "tile_auc": tile_auc,
        "tile_auprc": tile_auprc,
        "patient_acc": patient_acc,
        "patient_f1": patient_f1,
        "patient_auc": patient_auc,
        "patient_auprc": patient_auprc,
        "n_patient_eval": len(patient_true),
    }


class _PhikonFeatureWrapper(torch.nn.Module):
    def __init__(self, backbone: ViTModel) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]


def build_finetuned_phikon_extractor(
    *,
    checkpoint_path: Path,
    extractor_id: str,
) -> Extractor:
    model, _ = load_finetuned_phikon_classifier(checkpoint_path=checkpoint_path)
    transform = build_phikon_transform()

    return Extractor(
        model=_PhikonFeatureWrapper(model.backbone),
        transform=transform,
        identifier=extractor_id,
    )


def load_finetuned_phikon_classifier(
    *,
    checkpoint_path: Path,
) -> tuple[PhikonWeakClassifier, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt["args"]
    labels = ckpt["labels"]

    model = PhikonWeakClassifier(
        num_classes=len(labels),
        freeze_backbone=False,
        dropout=float(args.get("dropout", 0.3)),
    )
    if int(args.get("unfreeze_last_n_blocks", 0)) > 0:
        model.unfreeze_last_n_blocks(int(args["unfreeze_last_n_blocks"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_phikon_transform(
    processor_name: str = "owkin/phikon",
):
    processor = AutoImageProcessor.from_pretrained(processor_name)
    from torchvision import transforms

    mean = processor.image_mean
    std = processor.image_std
    size = processor.size.get("shortest_edge", 224)
    transform = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def build_selected_slide_table(
    *,
    input_slide_table: Path,
    output_slide_table: Path,
    selected_patients: list[str],
    extractor_id: str,
    patient_label: str,
    filename_label: str,
    relative_h5_by_stem: dict[str, str] | None = None,
) -> None:
    df = pd.read_csv(input_slide_table, dtype=str)
    selected_set = set(selected_patients)
    df = df[df[patient_label].astype(str).isin(selected_set)].copy()

    def _map_filename(value: str) -> str:
        slide_stem = Path(str(value)).stem
        relative_h5 = (
            relative_h5_by_stem[slide_stem]
            if relative_h5_by_stem is not None
            else f"{slide_stem}.h5"
        )
        return f"{extractor_id}/{relative_h5}"

    df[filename_label] = df[filename_label].map(_map_filename)
    output_slide_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_slide_table, index=False)


def build_wsi_relative_paths(
    *,
    wsi_dir: Path,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(wsi_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_SUFFIXES:
            mapping[path.stem] = str(path.relative_to(wsi_dir))
    return mapping
