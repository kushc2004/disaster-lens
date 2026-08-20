#!/usr/bin/env python3
"""Run georeferenced event inference from a trained M3/M4 checkpoint.

The script reads only official BRIGHT optical/SAR rasters. Labels are neither
opened nor required during inference.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import torch
import yaml
from rasterio.features import shapes
from rasterio.merge import merge
from scipy import ndimage
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml  # noqa: E402
from disasterlens.data import BrightDataset, load_manifest, normalization_from_stats  # noqa: E402
from disasterlens.models import DamageFusionFormer, PseudoSiameseUNet  # noqa: E402


def options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def path_value(raw: str | None, name: str, *, required: bool = True) -> Path | None:
    if not raw:
        if required:
            raise ValueError(f"{name}= is required")
        return None
    path = Path(raw)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def temperature(path: Path | None) -> float:
    if path is None:
        return 1.0
    value = float(json.loads(path.read_text(encoding="utf-8"))["temperature"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid positive temperature in {path}: {value}")
    return value


def build_model(kind: str, fusion_mode: str) -> torch.nn.Module:
    if kind == "m3":
        config = load_yaml(ROOT / "configs/model/siamese_baseline.yaml")
        return PseudoSiameseUNet(
            num_classes=int(config["num_classes"]),
            base_channels=int(config["base_channels"]),
        )
    if kind != "m4":
        raise ValueError("model_kind must be m3 or m4")
    config = load_yaml(ROOT / "configs/model/damage_fusion_former.yaml")
    return DamageFusionFormer(
        base_channels=int(config["encoder"]["base_channels"]),
        heads=int(config["fusion"]["heads"]),
        dropout=float(config["fusion"]["dropout"]),
        decoder_channels=int(config["decoder"]["channels"]),
        ablation=fusion_mode,
    )


def probabilities(
    output: torch.Tensor | dict[str, torch.Tensor], kind: str, fitted_temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "m4":
        assert isinstance(output, dict)
        localization = torch.softmax(output["localization"], dim=1)
        conditional = torch.softmax(output["damage"] / fitted_temperature, dim=1)
        background = localization[:, :1]
        building = localization[:, 1:2]
    else:
        assert isinstance(output, torch.Tensor)
        raw = torch.softmax(output, dim=1)
        background = raw[:, :1]
        building = 1.0 - background
        conditional = torch.softmax(output[:, 1:] / fitted_temperature, dim=1)
    semantic = torch.cat((background, building * conditional), dim=1)
    return (
        semantic[0].detach().float().cpu().numpy(),
        conditional[0].detach().float().cpu().numpy(),
    )


def write_tile(
    target: Path,
    values: np.ndarray,
    profile: dict[str, Any],
    *,
    dtype: str,
    nodata: int | float | None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output_profile = {
        **profile,
        "driver": "GTiff",
        "count": int(values.shape[0]) if values.ndim == 3 else 1,
        "dtype": dtype,
        "compress": "deflate",
        "tiled": True,
        "nodata": nodata,
    }
    with rasterio.open(target, "w", **output_profile) as destination:
        cast = values.astype(dtype, copy=False)
        if values.ndim == 2:
            destination.write(cast, 1)
        elif values.ndim == 3:
            destination.write(cast)
        else:
            raise ValueError(f"Expected a 2D or band-first 3D raster, found {values.shape}")


def mosaic(paths: list[Path], target: Path, *, dtype: str, nodata: int | float | None) -> None:
    if not paths:
        raise ValueError(f"No tiles available for {target.name}")
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        crs_values = {str(source.crs) for source in sources}
        if len(crs_values) != 1 or None in {source.crs for source in sources}:
            raise ValueError(f"All event tiles must share one CRS, found {sorted(crs_values)}")
        values, transform = merge(sources, nodata=nodata)
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=values.shape[1],
            width=values.shape[2],
            transform=transform,
            count=values.shape[0],
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            tiled=True,
        )
    with rasterio.open(target, "w", **profile) as destination:
        destination.write(values.astype(dtype, copy=False))


def component_rows(
    mask: np.ndarray,
    conditional: np.ndarray,
    *,
    transform: Any,
    event_id: str,
    tile_id: str,
    fitted_temperature: float,
) -> tuple[list[dict[str, Any]], list[Any]]:
    components, count = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    polygon_by_component: dict[int, list[Any]] = {}
    for geometry, component in shapes(components.astype(np.int32), mask=components > 0, transform=transform):
        polygon_by_component.setdefault(int(component), []).append(shape(geometry))
    rows: list[dict[str, Any]] = []
    geometries: list[Any] = []
    labels = ("intact", "damaged", "destroyed")
    from shapely.ops import unary_union

    for component in range(1, count + 1):
        selected = components == component
        if not selected.any():
            continue
        probability = conditional[:, selected].mean(axis=1)
        probability = probability / max(float(probability.sum()), 1e-12)
        entropy = -float(np.sum(probability * np.log(np.clip(probability, 1e-12, 1)))) / np.log(3)
        rows.append(
            {
                "event_id": event_id,
                "tile_id": tile_id,
                "building_id": f"{tile_id}_building_{component:06d}",
                "pixel_count": int(selected.sum()),
                "p_intact": float(probability[0]),
                "p_damaged": float(probability[1]),
                "p_destroyed": float(probability[2]),
                "predicted_class": labels[int(probability.argmax())],
                "expected_severity": float(probability @ np.asarray([0.0, 1.0, 2.0])),
                "predictive_entropy": entropy,
                "temperature": fitted_temperature,
                "aggregation": "mean_conditional_damage_probability_over_predicted_component",
            }
        )
        geometries.append(unary_union(polygon_by_component.get(component, [])))
    return rows, geometries


def main() -> None:
    args = options(sys.argv[1:])
    checkpoint = path_value(args.get("checkpoint"), "checkpoint")
    assert checkpoint is not None
    event_id = args.get("event_id")
    if not event_id:
        raise ValueError("event_id=<audited BRIGHT event> is required")
    split_path = path_value(args.get("split_path"), "split_path", required=False)
    temperature_path = path_value(args.get("temperature_path"), "temperature_path", required=False)
    kind = args.get("model_kind", "m4")
    fusion_mode = args.get("fusion_mode", "full")
    output = Path(args.get("output_dir", f"outputs/predictions/{event_id}"))
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    data = load_yaml(ROOT / "configs/data/bright.yaml")
    samples = load_manifest(ROOT / data["manifest_path"], dataset_root=Path(data["root"]))
    selected = [sample for sample in samples if sample.event_id == event_id]
    if split_path is not None:
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(split.get("test", []))
        selected = [sample for sample in selected if sample.tile_id in test_ids]
        events_in_test = {sample.event_id for sample in samples if sample.tile_id in test_ids}
        if events_in_test != {event_id}:
            raise ValueError(
                f"Inference split test partition must contain only {event_id!r}; found {sorted(events_in_test)}"
            )
    if not selected:
        raise ValueError(f"No official BRIGHT inference tiles found for event {event_id!r}")

    normalization = normalization_from_stats(data["normalization"], ROOT / data["normalization_stats_path"])
    fitted_temperature = temperature(temperature_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(kind, fusion_mode)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()

    tile_dir = output / "tiles"
    optical_tiles: list[Path] = []
    sar_tiles: list[Path] = []
    semantic_tiles: list[Path] = []
    probability_tiles: list[Path] = []
    uncertainty_tiles: list[Path] = []
    building_rows: list[dict[str, Any]] = []
    geometries: list[Any] = []
    shared_crs: str | None = None
    with torch.inference_mode():
        for index, sample in enumerate(selected, start=1):
            with rasterio.open(sample.pre_optical) as optical_source, rasterio.open(sample.post_sar) as sar_source:
                if optical_source.crs is None or sar_source.crs != optical_source.crs:
                    raise ValueError(f"Optical/SAR CRS mismatch for {sample.tile_id}")
                if sar_source.transform != optical_source.transform:
                    raise ValueError(f"Optical/SAR transform mismatch for {sample.tile_id}")
                if optical_source.count < 3 or sar_source.count < 1:
                    raise ValueError(
                        f"Expected at least 3 optical and 1 SAR bands for {sample.tile_id}; "
                        f"found {optical_source.count} and {sar_source.count}"
                    )
                optical_raw = optical_source.read(indexes=(1, 2, 3)).astype(np.float32)
                sar_raw = sar_source.read(indexes=(1,)).astype(np.float32)
                if optical_raw.shape[-2:] != sar_raw.shape[-2:]:
                    raise ValueError(f"Optical/SAR dimensions differ for {sample.tile_id}")
                profile, transform, crs = optical_source.profile.copy(), optical_source.transform, str(optical_source.crs)
            if shared_crs is not None and crs != shared_crs:
                raise ValueError(f"Event spans multiple CRSs ({shared_crs}, {crs}); per-CRS inference is required")
            shared_crs = crs
            optical = BrightDataset._normalize(optical_raw.copy(), normalization["pre_optical"])
            sar = BrightDataset._normalize(sar_raw.copy(), normalization["post_sar"])
            prediction = model(
                torch.from_numpy(optical).unsqueeze(0).to(device),
                torch.from_numpy(sar).unsqueeze(0).to(device),
            )
            semantic_probability, conditional_probability = probabilities(
                prediction, kind, fitted_temperature
            )
            semantic_mask = semantic_probability.argmax(axis=0).astype(np.uint8)
            uncertainty = -np.sum(
                semantic_probability * np.log(np.clip(semantic_probability, 1e-12, 1.0)),
                axis=0,
            ).astype(np.float32) / np.log(4.0)
            optical_path = tile_dir / f"{sample.tile_id}_pre_event_optical.tif"
            sar_path = tile_dir / f"{sample.tile_id}_post_event_sar.tif"
            semantic_path = tile_dir / f"{sample.tile_id}_semantic_mask.tif"
            probability_path = tile_dir / f"{sample.tile_id}_damage_probabilities.tif"
            uncertainty_path = tile_dir / f"{sample.tile_id}_uncertainty.tif"
            write_tile(optical_path, optical_raw, profile, dtype="float32", nodata=None)
            write_tile(sar_path, sar_raw, profile, dtype="float32", nodata=None)
            write_tile(semantic_path, semantic_mask, profile, dtype="uint8", nodata=255)
            write_tile(probability_path, conditional_probability, profile, dtype="float32", nodata=None)
            write_tile(uncertainty_path, uncertainty, profile, dtype="float32", nodata=None)
            rows, polygons = component_rows(
                semantic_mask,
                conditional_probability,
                transform=transform,
                event_id=event_id,
                tile_id=sample.tile_id,
                fitted_temperature=fitted_temperature,
            )
            building_rows.extend(rows)
            geometries.extend(polygons)
            optical_tiles.append(optical_path)
            sar_tiles.append(sar_path)
            semantic_tiles.append(semantic_path)
            probability_tiles.append(probability_path)
            uncertainty_tiles.append(uncertainty_path)
            print(f"[inference] tile {index:,}/{len(selected):,}: {sample.tile_id} ({len(rows):,} predicted buildings)", flush=True)

    mosaic(optical_tiles, output / "pre_event_optical.tif", dtype="float32", nodata=None)
    mosaic(sar_tiles, output / "post_event_sar.tif", dtype="float32", nodata=None)
    mosaic(semantic_tiles, output / "semantic_mask.tif", dtype="uint8", nodata=255)
    mosaic(probability_tiles, output / "damage_probabilities.tif", dtype="float32", nodata=None)
    mosaic(uncertainty_tiles, output / "uncertainty.tif", dtype="float32", nodata=None)
    if not building_rows:
        raise RuntimeError("Model predicted no building components; refusing to create an empty downstream artifact")
    buildings = gpd.GeoDataFrame(building_rows, geometry=geometries, crs=shared_crs)
    buildings.to_parquet(output / "building_predictions.parquet", index=False)
    buildings.to_file(output / "buildings.geojson", driver="GeoJSON")
    metadata = {
        "event_id": event_id,
        "model_kind": kind,
        "fusion_mode": fusion_mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "temperature": fitted_temperature,
        "temperature_source": str(temperature_path) if temperature_path else "uncalibrated_default_1.0",
        "split_path": str(split_path) if split_path else None,
        "tile_count": len(selected),
        "building_count": len(buildings),
        "crs": shared_crs,
        "label_usage": "none; labels are not read during inference",
        "building_definition": "8-connected components of the predicted non-background semantic mask",
        "damage_probabilities": "conditional probabilities over intact, damaged, destroyed",
        "uncertainty": "normalized four-class semantic predictive entropy",
        "artifacts": {
            "pre_event_optical": "pre_event_optical.tif",
            "post_event_sar": "post_event_sar.tif",
            "semantic_mask": "semantic_mask.tif",
            "damage_probabilities": "damage_probabilities.tif",
            "uncertainty": "uncertainty.tif",
            "building_predictions": "building_predictions.parquet",
            "buildings_geojson": "buildings.geojson",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[inference] complete georeferenced event outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
