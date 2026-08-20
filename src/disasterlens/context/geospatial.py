"""WorldPop exposure and OSM-derived estimated accessibility features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as raster_mask
from shapely.geometry import LineString, MultiLineString, box, mapping


def robust_scale(values: np.ndarray, lower: float = 0.02, upper: float = 0.98) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    low, high = np.quantile(array, [lower, upper])
    if high <= low:
        return np.zeros_like(array)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _validate_config(config: dict[str, Any]) -> None:
    grid_size = float(config["grid_size_m"])
    lower = float(config["normalization"]["lower_quantile"])
    upper = float(config["normalization"]["upper_quantile"])
    road = config["road_risk"]
    if grid_size <= 0:
        raise ValueError("Decision-grid size must be positive")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Normalization quantiles must satisfy 0 <= lower < upper <= 1")
    if float(config["damage"]["destroyed_multiplier"]) < 0:
        raise ValueError("Destroyed-building multiplier must be non-negative")
    if float(road["buffer_fraction_of_grid"]) <= 0:
        raise ValueError("Road-risk buffer fraction must be positive")
    if float(road["minimum_buffer_m"]) <= 0:
        raise ValueError("Road-risk minimum buffer must be positive")
    if float(road["cost_multiplier"]) < 0:
        raise ValueError("Road-risk cost multiplier must be non-negative")


def _grid(bounds: tuple[float, float, float, float], crs: Any, size: float) -> gpd.GeoDataFrame:
    minimum_x, minimum_y, maximum_x, maximum_y = bounds
    start_x, start_y = np.floor(minimum_x / size) * size, np.floor(minimum_y / size) * size
    columns = np.arange(start_x, maximum_x + size, size)
    rows = np.arange(start_y, maximum_y + size, size)
    geometries = [box(x, y, x + size, y + size) for x in columns[:-1] for y in rows[:-1]]
    return gpd.GeoDataFrame(
        {"cell_id": [f"cell_{index:06d}" for index in range(len(geometries))]},
        geometry=geometries,
        crs=crs,
    )


def _population_totals(grid: gpd.GeoDataFrame, worldpop_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(worldpop_path) as source:
        if source.crs is None:
            raise ValueError(f"WorldPop raster has no CRS: {worldpop_path}")
        projected = grid.to_crs(source.crs)
        totals: list[float] = []
        for geometry in projected.geometry:
            try:
                values, _ = raster_mask(source, [mapping(geometry)], crop=True, filled=False)
            except ValueError:
                totals.append(0.0)
                continue
            band = values[0]
            valid = band.compressed() if np.ma.isMaskedArray(band) else band.ravel()
            valid = valid[np.isfinite(valid)]
            totals.append(float(np.clip(valid, 0, None).sum()))
        metadata = {
            "crs": str(source.crs),
            "native_resolution": list(source.res),
            "width": source.width,
            "height": source.height,
            "aggregation": "sum_of_native_population-count_pixels_intersecting_cell",
        }
    return np.asarray(totals, dtype=np.float64), metadata


def _iter_lines(geometry: Any):
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms


def _road_graph(
    roads: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    risk_buffer_m: float,
    risk_cost_multiplier: float,
    normalization_lower: float,
    normalization_upper: float,
) -> tuple[nx.Graph, gpd.GeoDataFrame]:
    graph = nx.Graph()
    expected_destroyed = buildings["p_destroyed"].to_numpy(dtype=float)
    raw_risk: list[float] = []
    road_lines: list[LineString] = []
    for geometry in roads.geometry:
        for line in _iter_lines(geometry):
            if line.is_empty or line.length <= 0:
                continue
            candidates = buildings.sindex.query(line.buffer(risk_buffer_m), predicate="intersects")
            density = float(expected_destroyed[np.asarray(candidates, dtype=int)].sum()) / max(
                line.length * 2 * risk_buffer_m / 1_000_000, 1e-6
            )
            road_lines.append(line)
            raw_risk.append(density)
    if not road_lines:
        raise ValueError("OSM roads contain no usable LineString geometry")
    normalized = robust_scale(
        np.asarray(raw_risk), lower=normalization_lower, upper=normalization_upper
    )
    for edge_id, (line, risk) in enumerate(zip(road_lines, normalized, strict=True)):
        coordinates = list(line.coords)
        for start, end in zip(coordinates[:-1], coordinates[1:], strict=True):
            node_a = (round(start[0], 3), round(start[1], 3))
            node_b = (round(end[0], 3), round(end[1], 3))
            length = float(LineString((start, end)).length)
            if length <= 0:
                continue
            normal_cost = length
            risk_cost = length * (1.0 + risk_cost_multiplier * float(risk))
            if graph.has_edge(node_a, node_b) and graph[node_a][node_b]["normal_cost"] <= normal_cost:
                continue
            graph.add_edge(
                node_a,
                node_b,
                normal_cost=normal_cost,
                risk_cost=risk_cost,
                estimated_risk=float(risk),
                edge_id=edge_id,
            )
    if graph.number_of_edges() == 0:
        raise ValueError("No connected road edges could be constructed")
    risk_roads = gpd.GeoDataFrame(
        {
            "road_id": [f"road_{index:07d}" for index in range(len(road_lines))],
            "estimated_risk": normalized,
            "risk_basis": "nearby_expected_destroyed_building_density",
        },
        geometry=road_lines,
        crs=roads.crs,
    )
    return graph, risk_roads


def _nearest_nodes(graph: nx.Graph, coordinates: np.ndarray) -> list[tuple[float, float]]:
    nodes = list(graph.nodes)
    values = np.asarray(nodes, dtype=np.float64)
    if not len(values):
        raise ValueError("Road graph is empty")
    return [nodes[int(np.square(values - point).sum(axis=1).argmin())] for point in coordinates]


def _accessibility(
    grid: gpd.GeoDataFrame,
    graph: nx.Graph,
    facilities: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    facility_points = facilities.geometry.representative_point()
    facility_nodes = set(_nearest_nodes(graph, np.c_[facility_points.x, facility_points.y]))
    if not facility_nodes:
        raise ValueError("No facilities could be linked to the road graph")
    normal = nx.multi_source_dijkstra_path_length(graph, facility_nodes, weight="normal_cost")
    adjusted = nx.multi_source_dijkstra_path_length(graph, facility_nodes, weight="risk_cost")
    centroids = grid.geometry.centroid
    cell_nodes = _nearest_nodes(graph, np.c_[centroids.x, centroids.y])
    normal_cost, adjusted_cost, penalties, statuses = [], [], [], []
    for node in cell_nodes:
        base, risk = normal.get(node), adjusted.get(node)
        if base is None or risk is None or base <= 0:
            normal_cost.append(np.nan)
            adjusted_cost.append(np.nan)
            penalties.append(1.0)
            statuses.append("estimated isolation under current risk model")
        else:
            normal_cost.append(base)
            adjusted_cost.append(risk)
            penalties.append(float(np.clip((risk - base) / base, 0.0, 1.0)))
            statuses.append("estimated accessibility risk")
    return (
        np.asarray(normal_cost),
        np.asarray(adjusted_cost),
        np.asarray(penalties),
        statuses,
    )


def build_context_features(
    buildings: gpd.GeoDataFrame,
    *,
    worldpop_path: Path,
    roads: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    required = {
        "p_damaged",
        "p_destroyed",
        "expected_severity",
        "predictive_entropy",
    }
    missing = required.difference(buildings.columns)
    if missing:
        raise ValueError(f"Building predictions lack columns: {sorted(missing)}")
    if buildings.empty or buildings.crs is None:
        raise ValueError("Georeferenced building predictions with a CRS are required")
    _validate_config(config)
    projected_crs = buildings.estimate_utm_crs()
    if projected_crs is None:
        raise ValueError("Could not determine a local projected CRS for the event")
    if roads.crs is None or facilities.crs is None:
        raise ValueError("Real OSM roads and facilities must both declare a CRS")
    buildings = buildings.to_crs(projected_crs).copy()
    roads = roads.to_crs(projected_crs).copy()
    facilities = facilities.to_crs(projected_crs).copy()
    if roads.empty or facilities.empty:
        raise ValueError("Non-empty real OSM roads and facilities are required")
    grid_size_m = float(config["grid_size_m"])
    destroyed_multiplier = float(config["damage"]["destroyed_multiplier"])
    normalization_lower = float(config["normalization"]["lower_quantile"])
    normalization_upper = float(config["normalization"]["upper_quantile"])
    road_config = config["road_risk"]
    risk_buffer_m = max(
        float(road_config["minimum_buffer_m"]),
        grid_size_m * float(road_config["buffer_fraction_of_grid"]),
    )
    risk_cost_multiplier = float(road_config["cost_multiplier"])
    grid = _grid(tuple(buildings.total_bounds), projected_crs, grid_size_m)
    centroids = buildings.copy()
    centroids.geometry = buildings.geometry.representative_point()
    joined = gpd.sjoin(centroids, grid[["cell_id", "geometry"]], predicate="within", how="left")
    grouped = joined.dropna(subset=["cell_id"]).groupby("cell_id")
    aggregates = pd.DataFrame(
        {
            "number_buildings": grouped.size(),
            "expected_damaged_buildings": grouped["p_damaged"].sum(),
            "expected_destroyed_buildings": grouped["p_destroyed"].sum(),
            "mean_expected_severity": grouped["expected_severity"].mean(),
            "mean_model_entropy": grouped["predictive_entropy"].mean(),
        }
    )
    grid = grid.join(aggregates, on="cell_id")
    count_columns = (
        "number_buildings",
        "expected_damaged_buildings",
        "expected_destroyed_buildings",
        "mean_expected_severity",
        "mean_model_entropy",
    )
    grid[list(count_columns)] = grid[list(count_columns)].fillna(0.0)
    population, population_metadata = _population_totals(grid, worldpop_path)
    grid["population_total"] = population
    damage_raw = (
        grid["expected_damaged_buildings"]
        + destroyed_multiplier * grid["expected_destroyed_buildings"]
    )
    grid["normalized_damage_score"] = robust_scale(
        damage_raw.to_numpy(), lower=normalization_lower, upper=normalization_upper
    )
    grid["population_damage_exposure"] = population * grid["normalized_damage_score"]
    graph, risk_roads = _road_graph(
        roads,
        buildings,
        risk_buffer_m=risk_buffer_m,
        risk_cost_multiplier=risk_cost_multiplier,
        normalization_lower=normalization_lower,
        normalization_upper=normalization_upper,
    )
    normal, adjusted, penalty, status = _accessibility(grid, graph, facilities)
    grid["normal_facility_cost_m"] = normal
    grid["risk_adjusted_facility_cost_m"] = adjusted
    grid["accessibility_penalty"] = penalty
    grid["accessibility_status"] = status
    metadata = {
        "projected_crs": str(projected_crs),
        "grid_size_m": grid_size_m,
        "effective_assumptions": {
            "destroyed_multiplier": destroyed_multiplier,
            "normalization_lower_quantile": normalization_lower,
            "normalization_upper_quantile": normalization_upper,
            "road_risk_buffer_m": risk_buffer_m,
            "road_risk_cost_multiplier": risk_cost_multiplier,
        },
        "population": population_metadata,
        "population_semantics": "exposure estimate, not exact event-time population",
        "accessibility_semantics": "estimated risk, not confirmed road closure",
        "road_graph_nodes": graph.number_of_nodes(),
        "road_graph_edges": graph.number_of_edges(),
    }
    return grid, risk_roads, metadata
