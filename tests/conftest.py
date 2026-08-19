from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def write_tif(path: Path, image: np.ndarray) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=image.shape[1], width=image.shape[2], count=image.shape[0], dtype=image.dtype,
                       crs="EPSG:32632", transform=from_origin(500000, 4100000, 10, 10)) as target:
        target.write(image)


@pytest.fixture
def bright_root(tmp_path: Path) -> Path:
    for event, ordinal in (("bata-explosion", 0), ("surat-flood", 0), ("maui-wildfire", 0)):
        tile = f"{event}_{ordinal:08d}"
        write_tif(tmp_path / "pre-event" / f"{tile}_pre_disaster.tif", np.full((3, 12, 10), ordinal + 10, dtype=np.uint8))
        write_tif(tmp_path / "post-event" / f"{tile}_post_disaster.tif", np.full((1, 12, 10), ordinal + 1, dtype=np.float32))
        mask = np.tile(np.array([[0, 1, 2, 3, 0]], dtype=np.uint8), (12, 2))[None, :, :]
        write_tif(tmp_path / "target" / f"{tile}_building_damage.tif", mask)
    return tmp_path
