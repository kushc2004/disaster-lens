from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryPlan:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    quarter_turns: int = 0
    top: int | None = None
    left: int | None = None
    crop_size: int | None = None


class SynchronizedGeometry:
    """Applies one sampled geometry plan to every image and the label mask."""

    def __init__(self, *, seed: int = 42, crop_size: int | None = None) -> None:
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive")
        self.seed = seed
        self.crop_size = crop_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic augmentation stream for a new training epoch."""
        self.epoch = epoch

    def plan_for(self, index: int, *, height: int, width: int) -> GeometryPlan:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        top = int(rng.integers(height - self.crop_size + 1)) if self.crop_size and height >= self.crop_size else 0 if self.crop_size else None
        left = int(rng.integers(width - self.crop_size + 1)) if self.crop_size and width >= self.crop_size else 0 if self.crop_size else None
        return GeometryPlan(
            bool(rng.integers(2)),
            bool(rng.integers(2)),
            int(rng.integers(4)),
            top,
            left,
            self.crop_size,
        )

    @staticmethod
    def apply(array: np.ndarray, plan: GeometryPlan, *, pad_mode: str = "edge") -> np.ndarray:
        """Apply a plan and return an exact crop_size square when requested.

        Image channels are edge-padded for undersized genuine tiles.  Label masks
        use background (0) padding, supplied by ``__call__`` below.
        """
        result = array
        if plan.top is not None and plan.left is not None and plan.crop_size is not None:
            result = result[..., plan.top:plan.top + plan.crop_size, plan.left:plan.left + plan.crop_size]
            pad_height = plan.crop_size - result.shape[-2]
            pad_width = plan.crop_size - result.shape[-1]
            if pad_height or pad_width:
                padding = [(0, 0)] * (result.ndim - 2) + [(0, pad_height), (0, pad_width)]
                if pad_mode == "constant":
                    result = np.pad(result, padding, mode=pad_mode, constant_values=0)
                else:
                    result = np.pad(result, padding, mode=pad_mode)
        if plan.horizontal_flip:
            result = np.flip(result, axis=-1)
        if plan.vertical_flip:
            result = np.flip(result, axis=-2)
        if plan.quarter_turns:
            result = np.rot90(result, plan.quarter_turns, axes=(-2, -1))
        return np.ascontiguousarray(result)

    def __call__(self, images: dict[str, np.ndarray | None], mask: np.ndarray, *, index: int) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
        plan = self.plan_for(index, height=mask.shape[-2], width=mask.shape[-1])
        return (
            {name: None if image is None else self.apply(image, plan) for name, image in images.items()},
            self.apply(mask, plan, pad_mode="constant"),
        )
