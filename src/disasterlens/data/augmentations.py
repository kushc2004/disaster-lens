from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometryPlan:
    horizontal_flip: bool = False
    vertical_flip: bool = False
    quarter_turns: int = 0


class SynchronizedGeometry:
    """Applies one sampled geometry plan to every image and the label mask."""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = seed

    def plan_for(self, index: int) -> GeometryPlan:
        rng = np.random.default_rng(self.seed + index)
        return GeometryPlan(bool(rng.integers(2)), bool(rng.integers(2)), int(rng.integers(4)))

    @staticmethod
    def apply(array: np.ndarray, plan: GeometryPlan) -> np.ndarray:
        result = array
        if plan.horizontal_flip:
            result = np.flip(result, axis=-1)
        if plan.vertical_flip:
            result = np.flip(result, axis=-2)
        if plan.quarter_turns:
            result = np.rot90(result, plan.quarter_turns, axes=(-2, -1))
        return np.ascontiguousarray(result)

    def __call__(self, images: dict[str, np.ndarray | None], mask: np.ndarray, *, index: int) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
        plan = self.plan_for(index)
        return {name: None if image is None else self.apply(image, plan) for name, image in images.items()}, self.apply(mask, plan)

