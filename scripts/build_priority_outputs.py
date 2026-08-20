#!/usr/bin/env python3
"""Create transparent relative priority outputs from saved M5/M6 artifacts."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.prioritize import priority_analysis  # noqa: E402


def parse(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def existing(options: dict[str, str], name: str) -> Path:
    if not options.get(name):
        raise ValueError(f"{name}=<saved real artifact> is required")
    path = Path(options[name])
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
    event_id = options.get("event_id")
    if not event_id:
        raise ValueError("event_id= is required")
    features_path, buildings_path = existing(options, "features"), existing(options, "buildings")
    output = Path(options.get("output_dir", f"outputs/priority/{event_id}"))
    output = output if output.is_absolute() else ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    cells = gpd.read_parquet(features_path)
    buildings = gpd.read_parquet(buildings_path)
    cells = cells[cells["event_id"].astype(str) == event_id]
    if "partition" in buildings:
        buildings = buildings[buildings["partition"] == "test"]
    buildings = buildings[buildings["event_id"].astype(str) == event_id]
    config = yaml.safe_load((ROOT / "configs/priority.yaml").read_text(encoding="utf-8"))
    if "simulations" in options:
        config["monte_carlo"]["simulations"] = int(options["simulations"])
    if "sensitivity_samples" in options:
        config["sensitivity"]["samples"] = int(options["sensitivity_samples"])
    priority, sensitivity, draws = priority_analysis(cells, buildings, config)
    priority.to_parquet(output / "priority.parquet", index=False)
    priority.to_file(output / "priority.geojson", driver="GeoJSON")
    sensitivity.to_csv(output / "weight_sensitivity.csv", index=False)
    np.savez_compressed(
        output / "monte_carlo_draws.npz",
        priority_scores=draws["priority_scores"],
        ranks=draws["ranks"],
    )
    checkpoint = options.get("checkpoint")
    checkpoint_path = None
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "event_id": event_id,
                "effective_weights": draws["weights"],
                "configured_assumptions": config,
                "monte_carlo_simulations": config["monte_carlo"]["simulations"],
                "sensitivity_samples": config["sensitivity"]["samples"],
                "uncertainty_interval": [
                    config["uncertainty_quantiles"]["lower"],
                    config["uncertainty_quantiles"]["upper"],
                ],
                "inputs": {
                    "context_features": {"path": str(features_path), "sha256": digest(features_path)},
                    "building_predictions": {"path": str(buildings_path), "sha256": digest(buildings_path)},
                },
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "checkpoint_sha256": digest(checkpoint_path) if checkpoint_path else None,
                "priority_semantics": "relative event-level decision-support ranking",
                "policy_warning": "Operational weights require domain-expert validation.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    figures = ROOT / "outputs/figures"
    event_figures = figures / event_id
    reports = ROOT / "outputs/reports"
    figures.mkdir(parents=True, exist_ok=True)
    event_figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    axis = priority.plot(
        column="priority_score",
        cmap="magma",
        legend=True,
        figsize=(9, 7),
        edgecolor="white",
        linewidth=0.25,
        legend_kwds={"label": "Relative relief-priority score (0–1)"},
    )
    axis.set_axis_off()
    axis.set_title(f"Relative relief priority — {event_id}")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "priority_map.png", dpi=180, bbox_inches="tight")
    axis.figure.savefig(event_figures / "priority_map.png", dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
    axis = priority.plot(
        column="prob_top_10_percent", cmap="viridis", legend=True, figsize=(9, 7),
        legend_kwds={"label": "Probability of top-decile relative priority"},
    )
    axis.set_axis_off()
    axis.set_title(f"Rank stability — {event_id}")
    axis.figure.tight_layout()
    axis.figure.savefig(figures / "rank_stability_map.png", dpi=180, bbox_inches="tight")
    axis.figure.savefig(event_figures / "rank_stability_map.png", dpi=180, bbox_inches="tight")
    plt.close(axis.figure)
    report = [
        "# Priority-weight sensitivity",
        "",
        "Priority weights are configurable decision-support assumptions. Operational weights require domain-expert validation.",
        "",
        f"- Event: `{event_id}`",
        f"- Weight samples: `{len(sensitivity):,}`",
        f"- Median Spearman rank correlation: `{sensitivity['spearman_rank_correlation'].median():.6f}`",
        f"- Median top-decile overlap: `{sensitivity['top_decile_overlap'].median():.6f}`",
        "",
        "Bands are relative within this event and are not universal emergency classifications.",
    ]
    (reports / "priority_sensitivity.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[priority] complete uncertainty-aware outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
