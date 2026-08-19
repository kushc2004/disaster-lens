from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

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
