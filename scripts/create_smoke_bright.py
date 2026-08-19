#!/usr/bin/env python3
"""Create a tiny georeferenced BRIGHT-shaped fixture outside raw data."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, image: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=image.shape[1], width=image.shape[2], count=image.shape[0], dtype=image.dtype,
                       crs="EPSG:32632", transform=from_origin(500000, 4100000, 10, 10)) as target:
        target.write(image)


def main() -> None:
    destination = ROOT / "data/samples/bright_smoke"
    if destination.exists():
        shutil.rmtree(destination)
    rng = np.random.default_rng(42)
    for event in ("bata-explosion", "surat-flood", "maui-wildfire"):
        tile = f"{event}_00000000"
        _write(destination / "pre-event" / f"{tile}_pre_disaster.tif", rng.integers(0, 255, (3, 32, 32), dtype=np.uint8))
        _write(destination / "post-event" / f"{tile}_post_disaster.tif", rng.normal(size=(1, 32, 32)).astype(np.float32))
        mask = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint8), (32, 8))[None, :, :]
        _write(destination / "target" / f"{tile}_building_damage.tif", mask)
    print(f"Created smoke BRIGHT fixture at {destination}")


if __name__ == "__main__":
    main()
