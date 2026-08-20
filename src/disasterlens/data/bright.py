from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Callable, Sequence

import numpy as np

from .augmentations import SynchronizedGeometry
from .schemas import BRIGHT_V1, DisasterSample, LabelSchema


class BrightDataset:
    """BRIGHT BDA dataset for pre-event optical plus post-event SAR.

    Files are resolved from a manifest created by :func:`build_bright_manifest`.
    """

    def __init__(self, samples: Sequence[DisasterSample], *, schema: LabelSchema = BRIGHT_V1,
                 transform: SynchronizedGeometry | None = None, normalization: dict[str, Any] | None = None) -> None:
        if not samples:
            raise ValueError("BrightDataset requires at least one sample")
        self.samples, self.schema, self.transform = list(samples), schema, transform
        self.normalization = normalization or {}

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read(path: Path, *, channels: int | None = None) -> np.ndarray:
        try:
            import rasterio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rasterio is required to load BRIGHT") from exc
        with rasterio.open(path) as dataset:
            array = dataset.read()
        if channels is not None:
            if array.shape[0] < channels:
                raise ValueError(f"{path} has {array.shape[0]} bands; expected at least {channels}")
            array = array[:channels]
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _normalize(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        mode = config.get("mode", "none")
        if mode == "scale_uint":
            max_value = np.iinfo(np.dtype(np.uint16)).max if image.max(initial=0) > 255 else 255.0
            image = image / max_value
        elif mode == "zscore":
            pass
        elif mode != "none":
            raise ValueError(f"Unknown normalization mode {mode!r}")
        mean = np.asarray(config.get("mean", [0.0]), dtype=np.float32)[:, None, None]
        std = np.asarray(config.get("std", [1.0]), dtype=np.float32)[:, None, None]
        if mean.shape[0] not in (1, image.shape[0]) or std.shape[0] not in (1, image.shape[0]):
            raise ValueError("Normalization statistics do not match image bands")
        if np.any(std <= 0):
            raise ValueError("Normalization standard deviations must be positive")
        return (image - mean) / std

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required to load BRIGHT") from exc
        sample = self.samples[index]
        if not sample.pre_optical or not sample.post_sar or not sample.label:
            raise ValueError(f"M1 BRIGHT samples require pre_optical, post_sar, and label: {sample.tile_id}")
        images = {
            "pre_optical": self._read(sample.pre_optical, channels=3),
            "post_optical": None,
            "pre_sar": None,
            "post_sar": self._read(sample.post_sar, channels=1),
        }
        mask = self._read(sample.label, channels=1)[0].astype(np.int64)
        self.schema.validate(mask)
        dimensions = {image.shape[-2:] for image in images.values() if image is not None} | {mask.shape}
        if len(dimensions) != 1:
            raise ValueError(f"Modalities/mask have mismatched shape for {sample.tile_id}: {dimensions}")
        if self.transform:
            images, mask = self.transform(images, mask, index=index)
        tensor_images = {name: None if image is None else torch.from_numpy(self._normalize(image, self.normalization.get(name, {}))) for name, image in images.items()}
        return {"images": tensor_images, "mask": torch.from_numpy(mask).long(), "event_id": sample.event_id,
                "tile_id": sample.tile_id, "geo": {"crs": sample.crs, "bounds": sample.bounds}}


def collate_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensors while retaining optional geo metadata as a list."""
    import torch

    images: dict[str, Any] = {}
    for modality in ("pre_optical", "post_sar", "pre_sar", "post_optical"):
        values = [item["images"][modality] for item in batch]
        images[modality] = None if all(value is None for value in values) else torch.stack(values)
    return {"images": images, "mask": torch.stack([item["mask"] for item in batch]),
            "event_id": [item["event_id"] for item in batch], "tile_id": [item["tile_id"] for item in batch],
            "geo": [item["geo"] for item in batch]}


def class_weights_from_training_samples(
    samples: Sequence[DisasterSample],
    schema: LabelSchema = BRIGHT_V1,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Inverse-frequency class weights computed from training masks only."""
    counts = np.zeros(len(schema.classes), dtype=np.float64)
    class_ids = sorted(schema.classes)
    total = len(samples)
    for completed, sample in enumerate(samples, start=1):
        if sample.label is None:
            raise ValueError(f"Sample has no label: {sample.tile_id}")
        mask = BrightDataset._read(sample.label, channels=1)[0]
        schema.validate(mask)
        for index, class_id in enumerate(class_ids):
            counts[index] += np.count_nonzero(mask == class_id)
        if progress and (completed == 1 or completed == total or completed % 100 == 0):
            progress(completed, total)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute class weights: a training class is absent ({counts.tolist()})")
    weights = counts.sum() / (len(counts) * counts)
    return (weights / weights.mean()).astype(np.float32)


def select_tiny_overfit_samples(
    samples: Sequence[DisasterSample],
    *,
    count: int,
    crop_size: int,
    cache_path: str | Path | None = None,
    schema: LabelSchema = BRIGHT_V1,
) -> tuple[list[DisasterSample], dict[str, Any]]:
    """Select real train tiles whose deterministic crops cover damage classes.

    This is only for the tiny-overfit pipeline gate.  It never changes a proper
    train/validation/test split and it never introduces non-BRIGHT data.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(samples):
        raise ValueError(f"Requested {count} tiny-overfit tiles from only {len(samples)} training samples")
    if crop_size < 1:
        raise ValueError("crop_size must be positive")
    sample_by_id = {sample.tile_id: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("Tiny-overfit candidates must have unique tile IDs")
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        selected_ids = saved.get("tile_ids", [])
        if (
            saved.get("version") == 1
            and saved.get("crop_size") == crop_size
            and saved.get("count") == count
            and len(selected_ids) == count
            and all(tile_id in sample_by_id for tile_id in selected_ids)
        ):
            print(f"[tiny-overfit] restored label-coverage selection from {cache}", flush=True)
            return [sample_by_id[tile_id] for tile_id in selected_ids], saved

    class_ids = sorted(schema.classes)
    crop_counts: list[np.ndarray] = []
    total = len(samples)
    print(
        f"[tiny-overfit] scanning centre-crop labels for coverage-balanced selection ({total:,} official training tiles)",
        flush=True,
    )
    for index, sample in enumerate(samples, start=1):
        if sample.label is None:
            raise ValueError(f"Sample has no label: {sample.tile_id}")
        mask = BrightDataset._read(sample.label, channels=1)[0].astype(np.int64, copy=False)
        schema.validate(mask)
        height, width = mask.shape
        top, left = max(0, (height - crop_size) // 2), max(0, (width - crop_size) // 2)
        crop = mask[top:top + crop_size, left:left + crop_size]
        crop_counts.append(np.asarray([np.count_nonzero(crop == class_id) for class_id in class_ids], dtype=np.int64))
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[tiny-overfit] scanned {index:,}/{total:,} labels", flush=True)

    counts = np.stack(crop_counts)
    selected_indices: list[int] = []
    aggregate = np.zeros(len(class_ids), dtype=np.int64)
    available = set(range(total))
    for _ in range(count):
        chosen = max(
            available,
            key=lambda index: (
                int(np.min(aggregate[1:] + counts[index, 1:])),
                int(np.sum(np.log1p(aggregate[1:] + counts[index, 1:]) * 1_000_000)),
                int(np.sum(counts[index, 1:])),
                samples[index].tile_id,
            ),
        )
        selected_indices.append(chosen)
        aggregate += counts[chosen]
        available.remove(chosen)
    selected = [samples[index] for index in selected_indices]
    result: dict[str, Any] = {
        "version": 1,
        "count": count,
        "crop_size": crop_size,
        "tile_ids": [sample.tile_id for sample in selected],
        "class_ids": class_ids,
        "aggregate_centre_crop_pixels": {str(class_id): int(aggregate[index]) for index, class_id in enumerate(class_ids)},
        "selection": "greedy_maximise_minimum_damage_class_centre_crop_pixels",
    }
    if np.any(aggregate[1:] == 0):
        raise ValueError(
            "Cannot form a meaningful tiny-overfit set: at least one damage class is absent from all selected real centre crops "
            f"({result['aggregate_centre_crop_pixels']})."
        )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[tiny-overfit] saved label-coverage selection to {cache}", flush=True)
    print(f"[tiny-overfit] selected {count} official tiles with centre-crop pixels {result['aggregate_centre_crop_pixels']}", flush=True)
    return selected, result


def normalization_from_stats(config: dict[str, Any], stats_path: str | Path) -> dict[str, Any]:
    """Attach real audit mean/std values to the configured z-score transforms."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing BRIGHT normalization statistics: {path}. Run scripts/inspect_bright.py on the official data first.")
    stats = json.loads(path.read_text(encoding="utf-8"))
    resolved = {name: dict(values) for name, values in config.items()}
    for modality in ("pre_optical", "post_sar"):
        if modality not in stats:
            raise ValueError(f"Normalization audit does not contain {modality}")
        resolved.setdefault(modality, {}).update({"mean": stats[modality]["mean"], "std": stats[modality]["std"]})
    return resolved
