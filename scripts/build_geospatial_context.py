#!/usr/bin/env python3
"""Build one event's feature table from real WorldPop and OSM files."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.context import build_context_features  # noqa: E402


def parse(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def required_path(options: dict[str, str], name: str) -> Path:
    raw = options.get(name)
    if not raw:
        raise ValueError(f"{name}=<real local source> is required; no substitute data will be created")
    path = Path(raw)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    options = parse(sys.argv[1:])
    building_path = required_path(options, "buildings")
    worldpop_path = required_path(options, "worldpop")
    roads_path = required_path(options, "roads")
    facilities_path = required_path(options, "facilities")
    event_id = options.get("event_id")
    if not event_id:
        raise ValueError("event_id=<audited BRIGHT event> is required")
    provenance_fields = (
        "population_year",
        "population_source",
        "population_version",
        "population_license",
        "population_download_date",
    )
    missing_provenance = [name for name in provenance_fields if not options.get(name)]
    if missing_provenance:
        raise ValueError(
            "Required population provenance is missing: " + ", ".join(missing_provenance)
        )
    output = Path(options.get("output_dir", f"outputs/geospatial/{event_id}"))
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_parquet(building_path)
    if "partition" in buildings:
        buildings = buildings[buildings["partition"] == "test"]
    buildings = buildings[buildings["event_id"].astype(str) == event_id]
    if buildings.empty:
        raise ValueError(f"No test building predictions for event {event_id!r}")
    roads = gpd.read_file(roads_path)
    facilities = gpd.read_file(facilities_path)
    config = yaml.safe_load((ROOT / "configs/priority.yaml").read_text(encoding="utf-8"))
    config["grid_size_m"] = float(options.get("grid_size_m", config["grid_size_m"]))
    features, risk_roads, metadata = build_context_features(
        buildings,
        worldpop_path=worldpop_path,
        roads=roads,
        facilities=facilities,
        config=config,
    )
    features["event_id"] = event_id
    features.to_parquet(output / "features.parquet", index=False)
    features.to_file(output / "features.geojson", driver="GeoJSON")
    risk_roads.to_parquet(output / "roads_with_estimated_risk.parquet", index=False)
    metadata.update(
        {
            "event_id": event_id,
            "created_at": datetime.now(UTC).isoformat(),
            "population_source": options["population_source"],
            "population_year": options["population_year"],
            "population_version": options["population_version"],
            "population_license": options["population_license"],
            "population_download_date": options["population_download_date"],
            "inputs": {
                name: {"path": str(path), "sha256": digest(path)}
                for name, path in {
                    "building_predictions": building_path,
                    "worldpop": worldpop_path,
                    "roads": roads_path,
                    "facilities": facilities_path,
                }.items()
            },
        }
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    figures = ROOT / "outputs/figures" / event_id
    figures.mkdir(parents=True, exist_ok=True)
    axis = features.plot(
        column="population_damage_exposure",
        cmap="magma",
        legend=True,
        figsize=(9, 7),
        legend_kwds={"label": "Population exposure estimate (relative damage-weighted count)"},
        missing_kwds={"color": "lightgrey", "label": "No estimate"},
    )
    axis.set_axis_off()
    axis.set_title(f"Population exposure estimate — {event_id}")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "population_exposure_map.png", dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
    axis = risk_roads.plot(
        column="estimated_risk",
        cmap="inferno",
        legend=True,
        figsize=(9, 7),
        linewidth=1.5,
        legend_kwds={"label": "Estimated accessibility risk (0–1)"},
    )
    axis.set_axis_off()
    axis.set_title(f"Road-network accessibility risk estimate — {event_id}")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "road_risk_map.png", dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
    print(f"[context] complete real-data feature table: {output}", flush=True)


if __name__ == "__main__":
    main()
