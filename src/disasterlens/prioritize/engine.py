"""Relative priority score, Monte Carlo propagation, and weight sensitivity."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd


def robust_normalize(values: np.ndarray, *, lower: float, upper: float) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    low_value, high_value = np.quantile(array, [lower, upper])
    if high_value <= low_value:
        return np.zeros_like(array)
    return np.clip((array - low_value) / (high_value - low_value), 0.0, 1.0)


def _validate_config(config: dict[str, Any]) -> None:
    lower = float(config["normalization"]["lower_quantile"])
    upper = float(config["normalization"]["upper_quantile"])
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Normalization quantiles must satisfy 0 <= lower < upper <= 1")
    band_values = [
        float(config["bands"][name])
        for name in ("moderate_quantile", "high_quantile", "critical_quantile")
    ]
    if not 0.0 < band_values[0] < band_values[1] < band_values[2] < 1.0:
        raise ValueError("Priority-band quantiles must be strictly increasing within (0, 1)")
    top_fraction = float(config["top_fraction"])
    if not np.isclose(top_fraction, 0.10):
        raise ValueError(
            "top_fraction must remain 0.10 because the saved BRIGHT contract is "
            "prob_top_10_percent"
        )
    uncertainty = [
        float(config["uncertainty_quantiles"][name])
        for name in ("lower", "median", "upper")
    ]
    if not 0.0 <= uncertainty[0] < uncertainty[1] < uncertainty[2] <= 1.0:
        raise ValueError("Uncertainty quantiles must satisfy 0 <= lower < median < upper <= 1")
    if float(config["damage"]["destroyed_multiplier"]) < 0:
        raise ValueError("Destroyed-building multiplier must be non-negative")
    if int(config["monte_carlo"]["simulations"]) < 1:
        raise ValueError("Monte Carlo simulations must be positive")
    if int(config["sensitivity"]["samples"]) < 1:
        raise ValueError("Sensitivity samples must be positive")
    for name, bounds in config["sensitivity"]["ranges"].items():
        if len(bounds) != 2 or float(bounds[0]) < 0 or float(bounds[0]) > float(bounds[1]):
            raise ValueError(f"Invalid sensitivity weight range for {name!r}: {bounds!r}")


def available_weights(weights: dict[str, float], *, hazard_available: bool) -> dict[str, float]:
    selected = {
        name: float(value)
        for name, value in weights.items()
        if name != "hazard" or hazard_available
    }
    if any(value < 0 for value in selected.values()) or sum(selected.values()) <= 0:
        raise ValueError("Available priority weights must be non-negative with a positive sum")
    total = sum(selected.values())
    return {name: value / total for name, value in selected.items()}


def _components(
    cells: gpd.GeoDataFrame, config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], bool]:
    required = {
        "expected_damaged_buildings",
        "expected_destroyed_buildings",
        "population_total",
        "accessibility_penalty",
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"Context table lacks priority inputs: {sorted(missing)}")
    destroyed_multiplier = float(config["damage"]["destroyed_multiplier"])
    lower = float(config["normalization"]["lower_quantile"])
    upper = float(config["normalization"]["upper_quantile"])
    damage = (
        cells["expected_damaged_buildings"].to_numpy(float)
        + destroyed_multiplier * cells["expected_destroyed_buildings"].to_numpy(float)
    )
    components = {
        "damage": robust_normalize(damage, lower=lower, upper=upper),
        "population": robust_normalize(
            np.log1p(np.clip(cells["population_total"].to_numpy(float), 0, None)),
            lower=lower,
            upper=upper,
        ),
        "accessibility": np.clip(cells["accessibility_penalty"].to_numpy(float), 0, 1),
    }
    hazard_available = "hazard_score" in cells and bool(cells["hazard_score"].notna().any())
    if hazard_available:
        components["hazard"] = robust_normalize(
            cells["hazard_score"].fillna(0).to_numpy(float),
            lower=lower,
            upper=upper,
        )
    return components, hazard_available


def _score(components: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * components[name] for name in weights)


def _ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _bands(scores: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    band_config = config["bands"]
    q50, q75, q90 = np.quantile(
        scores,
        [
            float(band_config["moderate_quantile"]),
            float(band_config["high_quantile"]),
            float(band_config["critical_quantile"]),
        ],
    )
    return np.select(
        [scores >= q90, scores >= q75, scores >= q50],
        ["CRITICAL", "HIGH", "MODERATE"],
        default="LOW",
    )


def _building_cell_indices(cells: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> np.ndarray:
    if buildings.crs is None or cells.crs is None:
        raise ValueError("Cells and buildings require CRS metadata")
    buildings = buildings.to_crs(cells.crs).copy()
    points = buildings.copy()
    points.geometry = buildings.geometry.representative_point()
    joined = gpd.sjoin(points, cells[["cell_id", "geometry"]], predicate="within", how="left")
    lookup = {cell_id: index for index, cell_id in enumerate(cells["cell_id"])}
    return np.asarray([lookup.get(cell_id, -1) for cell_id in joined["cell_id"]], dtype=int)


def _monte_carlo(
    cells: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    static: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    simulations: int,
    seed: int,
    destroyed_multiplier: float,
    normalization_lower: float,
    normalization_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    if simulations < 1:
        raise ValueError("Monte Carlo simulations must be positive")
    probabilities = buildings[["p_intact", "p_damaged", "p_destroyed"]].to_numpy(float)
    if np.any(probabilities < 0) or np.any(~np.isfinite(probabilities)):
        raise ValueError("Building probabilities must be finite and non-negative")
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Building probability rows must have positive mass")
    probabilities /= totals
    cell_index = _building_cell_indices(cells, buildings)
    valid = cell_index >= 0
    probabilities, cell_index = probabilities[valid], cell_index[valid]
    rng = np.random.default_rng(seed)
    scores = np.empty((simulations, len(cells)), dtype=np.float32)
    ranks = np.empty_like(scores)
    cumulative = probabilities.cumsum(axis=1)
    for simulation in range(simulations):
        sampled = (rng.random(len(probabilities))[:, None] > cumulative).sum(axis=1)
        severity = np.choose(sampled, (0.0, 1.0, destroyed_multiplier))
        damage_raw = np.bincount(cell_index, weights=severity, minlength=len(cells))
        components = {
            **static,
            "damage": robust_normalize(
                damage_raw,
                lower=normalization_lower,
                upper=normalization_upper,
            ),
        }
        values = _score(components, weights)
        scores[simulation] = values
        ranks[simulation] = _ranks(values)
    return scores, ranks


def _sensitivity(
    components: dict[str, np.ndarray],
    default_score: np.ndarray,
    ranges: dict[str, list[float]],
    *,
    samples: int,
    seed: int,
    top_fraction: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    names = list(components)
    default_ranks = _ranks(default_score)
    top_count = max(1, int(np.ceil(top_fraction * len(default_score))))
    default_top = set(np.argsort(-default_score)[:top_count].tolist())
    rank_matrix = np.empty((samples, len(default_score)), dtype=np.float32)
    rows: list[dict[str, float]] = []
    for index in range(samples):
        drawn = np.asarray([rng.uniform(*ranges[name]) for name in names], dtype=float)
        drawn /= drawn.sum()
        values = _score(components, dict(zip(names, drawn, strict=True)))
        ranks = _ranks(values)
        rank_matrix[index] = ranks
        correlation = float(np.corrcoef(default_ranks, ranks)[0, 1]) if len(ranks) > 1 else 1.0
        overlap = len(default_top & set(np.argsort(-values)[:top_count].tolist())) / top_count
        rows.append(
            {
                "sample": index,
                **{f"weight_{name}": float(drawn[position]) for position, name in enumerate(names)},
                "spearman_rank_correlation": correlation,
                "top_decile_overlap": float(overlap),
            }
        )
    sensitivity_top_probability = (rank_matrix <= top_count).mean(axis=0)
    return pd.DataFrame(rows), rank_matrix.var(axis=0), sensitivity_top_probability


def priority_analysis(
    cells: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, np.ndarray | dict[str, float]]]:
    if cells.empty or buildings.empty:
        raise ValueError("Priority analysis requires non-empty real cells and building predictions")
    _validate_config(config)
    components, hazard_available = _components(cells, config)
    weights = available_weights(config["weights"], hazard_available=hazard_available)
    score = _score(components, weights)
    result = cells.copy()
    result["damage_score"] = components["damage"]
    result["population_score"] = components["population"]
    result["accessibility_score"] = components["accessibility"]
    result["hazard_available"] = hazard_available
    result["priority_score"] = score
    result["relative_priority_band"] = _bands(score, config)
    lower = float(config["normalization"]["lower_quantile"])
    upper = float(config["normalization"]["upper_quantile"])
    top_fraction = float(config["top_fraction"])
    monte_carlo = config["monte_carlo"]
    score_draws, rank_draws = _monte_carlo(
        result,
        buildings,
        components,
        weights,
        simulations=int(monte_carlo["simulations"]),
        seed=int(monte_carlo["seed"]),
        destroyed_multiplier=float(config["damage"]["destroyed_multiplier"]),
        normalization_lower=lower,
        normalization_upper=upper,
    )
    top_count = max(1, int(np.ceil(top_fraction * len(result))))
    quantiles = config["uncertainty_quantiles"]
    lower_quantile = float(quantiles["lower"])
    median_quantile = float(quantiles["median"])
    upper_quantile = float(quantiles["upper"])
    result["priority_mean"] = score_draws.mean(axis=0)
    result["priority_p05"] = np.quantile(score_draws, lower_quantile, axis=0)
    result["priority_p50"] = np.quantile(score_draws, median_quantile, axis=0)
    result["priority_p95"] = np.quantile(score_draws, upper_quantile, axis=0)
    result["rank_mean"] = rank_draws.mean(axis=0)
    result["rank_p05"] = np.quantile(rank_draws, lower_quantile, axis=0)
    result["rank_p95"] = np.quantile(rank_draws, upper_quantile, axis=0)
    result["prob_top_10_percent"] = (rank_draws <= top_count).mean(axis=0)
    sensitivity = config["sensitivity"]
    ranges = {name: sensitivity["ranges"][name] for name in components}
    sensitivity_rows, rank_variance, sensitivity_top = _sensitivity(
        components,
        score,
        ranges,
        samples=int(sensitivity["samples"]),
        seed=int(sensitivity["seed"]),
        top_fraction=top_fraction,
    )
    result["weight_rank_variance"] = rank_variance
    result["weight_prob_top_10_percent"] = sensitivity_top
    draws = {"priority_scores": score_draws, "ranks": rank_draws, "weights": weights}
    return result, sensitivity_rows, draws
