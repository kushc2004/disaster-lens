#!/usr/bin/env python3
"""Artifact-only DisasterLens decision-support application.

The application never loads a model or starts training.  It reads the saved
M5--M7 event artifacts and lets an analyst inspect estimates and reweight the
relative priority score without changing the underlying predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _existing(path: Path) -> Path | None:
    return path if path.is_file() and path.stat().st_size else None


@st.cache_data(show_spinner=False)
def read_geoparquet(path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_raster(path: str) -> tuple[np.ndarray, str]:
    with rasterio.open(path) as source:
        return source.read(), str(source.crs)


def events() -> list[str]:
    base = OUTPUTS / "priority"
    if not base.is_dir():
        return []
    return sorted(
        directory.name
        for directory in base.iterdir()
        if directory.is_dir() and _existing(directory / "priority.parquet")
    )


def normalized_weights(raw: dict[str, float], hazard_available: bool) -> dict[str, float]:
    active = {
        key: float(value)
        for key, value in raw.items()
        if key != "hazard" or hazard_available
    }
    total = sum(active.values())
    if total <= 0:
        raise ValueError("At least one active priority weight must be positive")
    return {key: value / total for key, value in active.items()}


def reweighted_priority(
    cells: gpd.GeoDataFrame,
    raw_weights: dict[str, float],
    hazard_enabled: bool,
    bands: dict[str, float],
) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    result = cells.copy()
    hazard_available = (
        hazard_enabled
        and "hazard_score" in result
        and bool(result["hazard_score"].notna().any())
    )
    weights = normalized_weights(raw_weights, hazard_available)
    columns = {
        "damage": "damage_score",
        "population": "population_score",
        "accessibility": "accessibility_score",
        "hazard": "hazard_score",
    }
    score = np.zeros(len(result), dtype=float)
    for component, weight in weights.items():
        if columns[component] not in result:
            raise ValueError(f"Saved priority artifact lacks {columns[component]!r}")
        values = result[columns[component]].fillna(0).to_numpy(float)
        score += weight * np.clip(values, 0, 1)
    result["display_priority_score"] = score
    quantiles = [
        float(bands["moderate_quantile"]),
        float(bands["high_quantile"]),
        float(bands["critical_quantile"]),
    ]
    if not 0 <= quantiles[0] < quantiles[1] < quantiles[2] <= 1:
        raise ValueError("Saved priority band quantiles must be strictly increasing in [0, 1]")
    moderate, high, critical = np.quantile(score, quantiles)
    result["display_priority_band"] = np.select(
        [score >= critical, score >= high, score >= moderate],
        ["CRITICAL", "HIGH", "MODERATE"],
        default="LOW",
    )
    return result, weights


def raster_panel(axis: Any, path: Path, title: str, *, cmap: str | None = None) -> None:
    values, _ = read_raster(str(path))
    if values.shape[0] >= 3 and cmap is None:
        rgb = np.moveaxis(values[:3].astype(float), 0, -1)
        lower, upper = np.nanpercentile(rgb, [2, 98])
        image = np.clip((rgb - lower) / max(upper - lower, 1e-9), 0, 1)
        axis.imshow(image)
    else:
        image = axis.imshow(values[0], cmap=cmap or "gray")
        axis.figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    axis.set_title(title)
    axis.set_axis_off()


def map_panel(
    cells: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    roads: gpd.GeoDataFrame | None,
    *,
    show_buildings: bool,
    show_roads: bool,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 8))
    cells.plot(
        ax=axis,
        column="display_priority_score",
        cmap="magma",
        legend=True,
        edgecolor="white",
        linewidth=0.25,
        legend_kwds={"label": "Relative relief priority"},
    )
    if show_roads and roads is not None and not roads.empty:
        roads.to_crs(cells.crs).plot(
            ax=axis,
            column="estimated_risk",
            cmap="Reds",
            linewidth=1.0,
            alpha=0.75,
        )
    if show_buildings and buildings is not None and not buildings.empty:
        buildings.to_crs(cells.crs).plot(
            ax=axis,
            column="expected_severity",
            cmap="YlOrRd",
            markersize=4,
            alpha=0.75,
        )
    axis.set_title("Relative relief priority and selected estimated layers")
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def main() -> None:
    st.set_page_config(page_title="DisasterLens", layout="wide")
    st.title("DisasterLens")
    st.caption("Multimodal damage estimates and relative relief-priority decision support")
    available_events = events()
    if not available_events:
        st.error(
            "No completed event artifacts were found under outputs/priority/<event>. "
            "Run M5-M7 before launching the application."
        )
        st.stop()

    st.sidebar.header("Event controls")
    st.sidebar.text_input("Dataset", value="Official BRIGHT", disabled=True)
    event_id = st.sidebar.selectbox("Event", available_events)
    prediction_dir = OUTPUTS / "predictions" / event_id
    priority_dir = OUTPUTS / "priority" / event_id
    context_dir = OUTPUTS / "geospatial" / event_id
    metadata_path = priority_dir / "metadata.json"
    metadata = read_json(str(metadata_path)) if _existing(metadata_path) else {}
    checkpoint = metadata.get("checkpoint") or "saved M4 event-held-out checkpoint"
    st.sidebar.text_input("Model checkpoint", value=str(checkpoint), disabled=True)

    st.sidebar.subheader("Map layers")
    show_buildings = st.sidebar.checkbox("Estimated damage (buildings)", value=True)
    show_roads = st.sidebar.checkbox("Estimated accessibility risk (roads)", value=True)
    show_rasters = st.sidebar.checkbox("Input and prediction rasters", value=True)

    cells = read_geoparquet(str(priority_dir / "priority.parquet"))
    building_path = _existing(prediction_dir / "building_predictions.parquet")
    buildings = read_geoparquet(str(building_path)) if building_path else None
    road_path = _existing(context_dir / "roads_with_estimated_risk.parquet")
    roads = read_geoparquet(str(road_path)) if road_path else None

    hazard_available = "hazard_score" in cells and bool(cells["hazard_score"].notna().any())
    hazard_enabled = st.sidebar.toggle(
        "Hazard feature",
        value=hazard_available,
        disabled=not hazard_available,
        help="Hazard is omitted and remaining weights are renormalized when unavailable.",
    )
    st.sidebar.subheader("Priority weights")
    defaults = metadata.get("effective_weights")
    bands = metadata.get("configured_assumptions", {}).get("bands")
    if not isinstance(defaults, dict) or not isinstance(bands, dict):
        st.error("Priority metadata is incomplete: effective_weights and configured band assumptions are required.")
        st.stop()
    raw_weights = {
        name: st.sidebar.slider(name.title(), 0.0, 1.0, float(defaults.get(name, 0.0)), 0.01)
        for name in ("damage", "population", "accessibility", "hazard")
    }
    try:
        display_cells, weights = reweighted_priority(
            cells, raw_weights, hazard_enabled, bands
        )
    except (KeyError, TypeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    st.sidebar.caption(
        "Effective normalized weights: "
        + ", ".join(f"{name}={value:.2f}" for name, value in weights.items())
    )
    st.warning(
        "Priority bands are relative within this event. Weights are policy assumptions "
        "that require domain-expert validation before operational use."
    )

    st.pyplot(
        map_panel(
            display_cells,
            buildings,
            roads,
            show_buildings=show_buildings,
            show_roads=show_roads,
        ),
        clear_figure=True,
    )

    if show_rasters:
        raster_specs = [
            (prediction_dir / "pre_event_optical.tif", "Pre-event optical", None),
            (prediction_dir / "post_event_sar.tif", "Post-event SAR", "gray"),
            (prediction_dir / "semantic_mask.tif", "Estimated damage", "viridis"),
            (prediction_dir / "uncertainty.tif", "Model uncertainty", "magma"),
        ]
        present = [(path, title, cmap) for path, title, cmap in raster_specs if _existing(path)]
        if present:
            figure, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.5))
            axes = np.atleast_1d(axes)
            for axis, (path, title, cmap) in zip(axes, present, strict=True):
                raster_panel(axis, path, title, cmap=cmap)
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)

    left, middle, right = st.columns(3)
    with left:
        st.subheader("Grid-cell inspector")
        cell_id = st.selectbox("Cell", display_cells["cell_id"].astype(str).tolist())
        cell = display_cells.loc[display_cells["cell_id"].astype(str) == cell_id].iloc[0]
        cell_fields = {
            "Population exposure estimate": "population_total",
            "Expected damaged buildings": "expected_damaged_buildings",
            "Expected destroyed buildings": "expected_destroyed_buildings",
            "Estimated accessibility penalty": "accessibility_penalty",
            "Hazard feature": "hazard_score",
            "Relative priority": "display_priority_score",
            "Priority interval (5%-95%)": ("priority_p05", "priority_p95"),
            "Probability of top-decile priority": "prob_top_10_percent",
        }
        for label, column in cell_fields.items():
            if isinstance(column, tuple):
                if all(name in cell for name in column):
                    st.metric(label, f"{cell[column[0]]:.3f} - {cell[column[1]]:.3f}")
            elif column in cell and pd.notna(cell[column]):
                st.metric(label, f"{float(cell[column]):,.3f}")

    with middle:
        st.subheader("Building inspector")
        if buildings is None or buildings.empty:
            st.info("No saved building predictions for this event.")
        else:
            building_id = st.selectbox("Building", buildings["building_id"].astype(str).tolist())
            building = buildings.loc[buildings["building_id"].astype(str) == building_id].iloc[0]
            st.write(f"Estimated damage: **{building['predicted_class']}**")
            for label, column in (
                ("P(intact)", "p_intact"),
                ("P(damaged)", "p_damaged"),
                ("P(destroyed)", "p_destroyed"),
                ("Expected severity", "expected_severity"),
                ("Model uncertainty (entropy)", "predictive_entropy"),
            ):
                st.metric(label, f"{float(building[column]):.3f}")

    with right:
        st.subheader("Road inspector")
        if roads is None or roads.empty:
            st.info("No saved OSM road-risk artifact for this event.")
        else:
            road_id = st.selectbox("Road segment", roads["road_id"].astype(str).tolist())
            road = roads.loc[roads["road_id"].astype(str) == road_id].iloc[0]
            st.metric("Estimated accessibility risk", f"{float(road['estimated_risk']):.3f}")
            st.caption(f"Basis: {road.get('risk_basis', 'not recorded')}")

    st.caption(
        "All displayed values come from saved artifacts. The app does not train a model, "
        "alter damage probabilities, or claim real-time road closure status."
    )


if __name__ == "__main__":
    main()
