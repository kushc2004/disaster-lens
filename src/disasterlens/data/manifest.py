from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .schemas import DisasterSample

PRE_SUFFIX = "_pre_disaster.tif"
POST_SUFFIX = "_post_disaster.tif"
LABEL_SUFFIX = "_building_damage.tif"
REQUIRED_DIRS = {"pre-event", "post-event", "target"}


class BrightLayoutError(ValueError):
    pass


def _tile_ids(directory: Path, suffix: str) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        tile_id = path.name.removesuffix(suffix)
        if not tile_id:
            raise BrightLayoutError(f"Invalid empty tile ID: {path}")
        if tile_id in matches:
            raise BrightLayoutError(f"Duplicate tile ID {tile_id!r} in {directory}")
        matches[tile_id] = path
    return matches


def _event_and_type(tile_id: str) -> tuple[str, str]:
    if "_" not in tile_id:
        raise BrightLayoutError(f"Cannot derive event ID from tile ID {tile_id!r}")
    event_id, ordinal = tile_id.rsplit("_", 1)
    if not ordinal.isdigit():
        raise BrightLayoutError(f"Expected numeric tile ordinal in {tile_id!r}")
    return event_id, event_id.rsplit("-", 1)[-1]


def _georeference(path: Path) -> tuple[str | None, tuple[float, float, float, float] | None, int, int, int]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - installation error is actionable
        raise RuntimeError("rasterio is required for BRIGHT manifests") from exc
    with rasterio.open(path) as dataset:
        crs = dataset.crs.to_string() if dataset.crs else None
        bounds = tuple(float(v) for v in dataset.bounds) if dataset.crs else None
        return crs, bounds, dataset.width, dataset.height, dataset.count


def build_bright_manifest(
    root: Path, *, progress: Callable[[int, int], None] | None = None
) -> list[DisasterSample]:
    """Discover official BRIGHT BDA files without changing raw data."""
    root = Path(root).expanduser().resolve()
    missing_dirs = REQUIRED_DIRS.difference(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else REQUIRED_DIRS
    if missing_dirs:
        raise BrightLayoutError(f"BRIGHT root {root} is missing directories: {sorted(missing_dirs)}")

    pre = _tile_ids(root / "pre-event", PRE_SUFFIX)
    post = _tile_ids(root / "post-event", POST_SUFFIX)
    labels = _tile_ids(root / "target", LABEL_SUFFIX)
    if not pre:
        raise BrightLayoutError(f"No {PRE_SUFFIX} files found in {root / 'pre-event'}")
    expected = set(pre)
    mismatches = {"post-event": sorted(expected.symmetric_difference(post)), "target": sorted(expected.symmetric_difference(labels))}
    if any(mismatches.values()):
        detail = "; ".join(f"{name}={ids[:5]}" for name, ids in mismatches.items() if ids)
        raise BrightLayoutError(f"Modalities do not align by tile ID: {detail}")

    samples: list[DisasterSample] = []
    total = len(expected)
    for index, tile_id in enumerate(sorted(expected), start=1):
        event_id, disaster_type = _event_and_type(tile_id)
        crs, bounds, width, height, bands = _georeference(pre[tile_id])
        if crs is None:
            raise BrightLayoutError(f"Missing CRS in critical pre-event raster: {pre[tile_id]}")
        samples.append(DisasterSample(
            event_id=event_id, disaster_type=disaster_type, tile_id=tile_id,
            pre_optical=pre[tile_id], post_sar=post[tile_id], label=labels[tile_id], crs=crs, bounds=bounds,
            metadata={"source_layout": "official_bda", "pre_width": width, "pre_height": height, "pre_bands": bands},
        ))
        if progress and (index == 1 or index % 100 == 0 or index == total):
            progress(index, total)
    return samples


def write_manifest(samples: Iterable[DisasterSample], path: Path, *, dataset_root: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_record(root=dataset_root), sort_keys=True) + "\n")
    return path


def load_manifest(path: Path, *, dataset_root: Path | None = None) -> list[DisasterSample]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return [DisasterSample.from_record(json.loads(line), root=dataset_root) for line in handle if line.strip()]
