"""Building-level aggregation and validation-only temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage, optimize


EPSILON = 1e-12
CLASS_NAMES = ("intact", "damaged", "destroyed")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    values = np.exp(scaled)
    return values / values.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, *, bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == bins - 1 else (confidence < upper)
        )
        count = int(selected.sum())
        accuracy = float((prediction[selected] == targets[selected]).mean()) if count else 0.0
        mean_confidence = float(confidence[selected].mean()) if count else 0.0
        if count:
            ece += count / len(targets) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return float(ece), rows


def classification_metrics(
    logits: np.ndarray, targets: np.ndarray, *, temperature: float, bins: int = 15
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"Expected building logits [N,3], got {logits.shape}")
    if len(logits) == 0 or len(targets) != len(logits):
        raise ValueError("Building logits and targets must be equally sized and non-empty")
    probabilities = softmax(logits, temperature)
    prediction = probabilities.argmax(axis=1)
    nll = float(-np.log(np.clip(probabilities[np.arange(len(targets)), targets], EPSILON, 1)).mean())
    truth = np.eye(3, dtype=np.float64)[targets]
    brier = float(np.square(probabilities - truth).sum(axis=1).mean())
    ece, reliability = expected_calibration_error(probabilities, targets, bins=bins)
    f1: list[float] = []
    recall: list[float] = []
    for class_id in range(3):
        true_positive = int(((prediction == class_id) & (targets == class_id)).sum())
        false_positive = int(((prediction == class_id) & (targets != class_id)).sum())
        false_negative = int(((prediction != class_id) & (targets == class_id)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(2 * true_positive / denominator if denominator else 0.0)
        recall_denominator = true_positive + false_negative
        recall.append(true_positive / recall_denominator if recall_denominator else 0.0)
    return (
        {
            "temperature": float(temperature),
            "n_buildings": int(len(targets)),
            "nll": nll,
            "brier": brier,
            "ece": ece,
            "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(np.mean(recall)),
        },
        reliability,
    )


def bootstrap_classification_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    temperature: float,
    bins: int = 15,
    samples: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return building-level percentile intervals without bootstrapping pixels."""
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    if len(logits) == 0 or len(targets) != len(logits):
        raise ValueError("Building logits and targets must be equally sized and non-empty")
    generator = np.random.default_rng(seed)
    tracked = ("ece", "nll", "brier", "macro_f1", "balanced_accuracy")
    draws = {metric: np.empty(samples, dtype=np.float64) for metric in tracked}
    for draw in range(samples):
        indices = generator.integers(0, len(targets), size=len(targets))
        metrics, _ = classification_metrics(
            logits[indices], targets[indices], temperature=temperature, bins=bins
        )
        for metric in tracked:
            draws[metric][draw] = float(metrics[metric])
    return {
        "unit": "building",
        "n_units": int(len(targets)),
        "samples": samples,
        "seed": seed,
        "metrics": {
            metric: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in draws.items()
        },
    }


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit one scalar on validation data; callers must enforce partition provenance."""
    if len(logits) == 0:
        raise ValueError("Cannot calibrate an empty validation set")

    def objective(log_temperature: float) -> float:
        probabilities = softmax(logits, float(np.exp(log_temperature)))
        return float(
            -np.log(np.clip(probabilities[np.arange(len(targets)), targets], EPSILON, 1)).mean()
        )

    result = optimize.minimize_scalar(
        objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded"
    )
    if not result.success:
        raise RuntimeError(f"Temperature optimization failed: {result.message}")
    temperature = float(np.exp(result.x))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError(f"Invalid fitted temperature: {temperature}")
    return temperature


@dataclass
class BuildingAggregation:
    rows: list[dict[str, Any]]
    logits: np.ndarray
    targets: np.ndarray
    geometries: list[Any | None]
    crs_values: set[str]


def _scalar(data: Any) -> str:
    return str(np.asarray(data).item())


def aggregate_prediction_files(
    paths: Iterable[Path], *, partition: str, connectivity: int = 8,
    include_geometries: bool = False,
) -> BuildingAggregation:
    """Aggregate pixel logits over reference-mask connected components.

    Reference components are used only for evaluation/calibration. Event inference
    uses predicted building components and is kept separate to avoid leakage.
    """
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    targets: list[int] = []
    geometries: list[Any | None] = []
    crs_values: set[str] = set()
    ordered_paths = sorted(paths)
    total_paths = len(ordered_paths)
    for file_index, path in enumerate(ordered_paths, start=1):
        with np.load(path, allow_pickle=False) as saved:
            logits = np.asarray(saved["logits"], dtype=np.float64)
            mask = np.asarray(saved["mask"], dtype=np.int64)
            event_id, tile_id = _scalar(saved["event_id"]), _scalar(saved["tile_id"])
            crs = _scalar(saved["crs"]) if "crs" in saved else ""
            bounds = np.asarray(saved["bounds"], dtype=np.float64) if "bounds" in saved else np.empty(0)
        if logits.shape[0] != 4 or logits.shape[1:] != mask.shape:
            raise ValueError(f"Invalid prediction artifact shapes in {path}: {logits.shape}, {mask.shape}")
        labels, count = ndimage.label((mask > 0) & (mask != 255), structure=structure)
        geometry_by_label: dict[int, Any] = {}
        # Calibration only consumes logits and reference-component labels.  Do
        # not raster-vectorize every component unless a caller explicitly
        # needs geometries: on a CPU finalizer this otherwise dominates time
        # without changing any calibration metric or saved prediction table.
        if include_geometries and crs and bounds.size == 4:
            from rasterio.features import shapes
            from rasterio.transform import from_bounds
            from shapely.geometry import shape

            transform = from_bounds(*bounds.tolist(), width=mask.shape[1], height=mask.shape[0])
            for geometry, value in shapes(
                labels.astype(np.int32), mask=labels > 0, transform=transform
            ):
                geometry_by_label[int(value)] = shape(geometry)
            crs_values.add(crs)
        for component_id in range(1, count + 1):
            selected = labels == component_id
            class_pixels = mask[selected]
            class_pixels = class_pixels[(class_pixels >= 1) & (class_pixels <= 3)]
            if not len(class_pixels):
                continue
            target = int(np.bincount(class_pixels - 1, minlength=3).argmax())
            vector = logits[1:, selected].mean(axis=1)
            building_id = f"{event_id}:{tile_id}:{component_id}"
            rows.append(
                {
                    "building_id": building_id,
                    "event_id": event_id,
                    "tile_id": tile_id,
                    "partition": partition,
                    "target_class": CLASS_NAMES[target],
                    "target_class_id": target,
                    "pixel_count": int(selected.sum()),
                    "aggregation_source": "reference_mask_connected_component",
                }
            )
            vectors.append(vector)
            targets.append(target)
            geometries.append(geometry_by_label.get(component_id))
        if file_index == 1 or file_index % 25 == 0 or file_index == total_paths:
            print(
                f"[calibration] {partition}: aggregated {file_index}/{total_paths} tiles "
                f"({len(rows):,} buildings)",
                flush=True,
            )
    if not rows:
        raise ValueError(f"No building components found in {partition} predictions")
    return BuildingAggregation(
        rows=rows,
        logits=np.stack(vectors),
        targets=np.asarray(targets, dtype=np.int64),
        geometries=geometries,
        crs_values=crs_values,
    )
