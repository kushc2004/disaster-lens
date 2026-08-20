#!/usr/bin/env python3
"""Acquire and validate a real WorldPop population-count raster for one event.

Supply either source=<local GeoTIFF> or url=<official download URL>.  The
script never creates fallback population values.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]


def parse(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(source: str | None, url: str | None, target: Path) -> str:
    if bool(source) == bool(url):
        raise ValueError("Supply exactly one of source=<real local GeoTIFF> or url=<official URL>")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    if source:
        local = Path(source).expanduser()
        local = local if local.is_absolute() else ROOT / local
        if not local.is_file():
            raise FileNotFoundError(local)
        shutil.copy2(local, temporary)
        provenance = str(local.resolve())
    else:
        assert url is not None
        request = urllib.request.Request(url, headers={"User-Agent": "DisasterLens/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            if getattr(response, "status", 200) != 200:
                raise RuntimeError(f"Population download returned HTTP {response.status}")
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        provenance = url
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError("Population acquisition produced an empty file")
    temporary.replace(target)
    return provenance


def validate(path: Path) -> dict[str, object]:
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError("WorldPop raster is missing CRS metadata")
        if source.count != 1 or source.width < 1 or source.height < 1:
            raise ValueError("WorldPop input must be a non-empty single-band raster")
        step_y = max(1, source.height // 512)
        step_x = max(1, source.width // 512)
        sample = source.read(1, out_shape=(max(1, source.height // step_y), max(1, source.width // step_x)))
        valid = np.isfinite(sample)
        if source.nodata is not None:
            valid &= sample != source.nodata
        values = sample[valid]
        if not len(values):
            raise ValueError("WorldPop raster has no finite non-nodata values")
        if np.nanmin(values) < 0:
            raise ValueError("Population-count raster contains negative valid values")
        return {
            "crs": str(source.crs),
            "bounds": list(source.bounds),
            "resolution": list(source.res),
            "width": source.width,
            "height": source.height,
            "dtype": source.dtypes[0],
            "nodata": source.nodata,
            "sample_valid_pixels": int(len(values)),
            "sample_population_sum": float(values.sum()),
        }


def main() -> None:
    options = parse(sys.argv[1:])
    event_id = options.get("event_id")
    year = options.get("year") or options.get("population_year")
    source_name = options.get("source_name") or options.get("population_source")
    version = options.get("version") or options.get("population_version")
    license_name = options.get("license") or options.get("population_license")
    download_date = options.get("download_date") or options.get("population_download_date")
    if not all((event_id, year, source_name, version, license_name, download_date)):
        raise ValueError(
            "event_id=, year=, source_name=, version=, license=, and download_date= "
            "are required population provenance"
        )
    output = Path(options.get("output", f"data/external/worldpop/{event_id}/population.tif"))
    output = output if output.is_absolute() else ROOT / output
    provenance = acquire(options.get("source"), options.get("url"), output)
    validation = validate(output)
    metadata = {
        "event_id": event_id,
        "dataset": source_name,
        "population_year": year,
        "dataset_version": version,
        "license": license_name,
        "download_date": download_date,
        "source": provenance,
        "downloaded_or_copied_at": datetime.now(UTC).isoformat(),
        "sha256": sha256(output),
        "raster": validation,
        "semantic_contract": "gridded population-count raster; aggregated by summation",
    }
    metadata_path = output.parent / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[population] validated real population raster: {output}", flush=True)


if __name__ == "__main__":
    main()
