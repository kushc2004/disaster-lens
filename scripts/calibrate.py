#!/usr/bin/env python3
"""Fit validation-only temperature scaling on saved real BRIGHT predictions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.eval import (  # noqa: E402
    aggregate_prediction_files,
    bootstrap_classification_metrics,
    classification_metrics,
    fit_temperature,
    softmax,
)


def arguments(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected key=value, received {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def resolve(raw: str, *, must_exist: bool = True) -> Path:
    path = Path(raw)
    path = path if path.is_absolute() else ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def prediction_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No saved prediction NPZ files in {directory}")
    return files


def reliability_figure(
    output: Path,
    before: list[dict[str, float]],
    after: list[dict[str, float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for axis, title, rows in zip(
        axes, ("Before temperature scaling", "After temperature scaling"), (before, after), strict=True
    ):
        occupied = [row for row in rows if row["count"]]
        axis.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
        axis.plot(
            [row["confidence"] for row in occupied],
            [row["accuracy"] for row in occupied],
            marker="o",
            label="building predictions",
        )
        axis.set(title=title, xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def add_probabilities(
    rows: list[dict[str, Any]], logits: np.ndarray, *, temperature: float
) -> list[dict[str, Any]]:
    probabilities = softmax(logits, temperature)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1) / np.log(3)
    predictions = probabilities.argmax(axis=1)
    complete: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        complete.append(
            {
                **row,
                "logit_intact": float(logits[index, 0]),
                "logit_damaged": float(logits[index, 1]),
                "logit_destroyed": float(logits[index, 2]),
                "p_intact": float(probabilities[index, 0]),
                "p_damaged": float(probabilities[index, 1]),
                "p_destroyed": float(probabilities[index, 2]),
                "predicted_class": ("intact", "damaged", "destroyed")[predictions[index]],
                "expected_severity": float(probabilities[index] @ np.asarray([0.0, 1.0, 2.0])),
                "predictive_entropy": float(entropy[index]),
                "temperature": float(temperature),
            }
        )
    return complete


def main() -> None:
    options = arguments(sys.argv[1:])
    missing = [
        name
        for name in ("validation_predictions", "test_predictions")
        if not options.get(name)
    ]
    if missing:
        raise ValueError(
            "Required saved prediction directories were not supplied: " + ", ".join(missing)
        )
    validation_dir = resolve(options["validation_predictions"])
    test_dir = resolve(options["test_predictions"])
    if not validation_dir.is_dir() or not test_dir.is_dir():
        raise NotADirectoryError("validation_predictions and test_predictions must be directories")
    if validation_dir.resolve() == test_dir.resolve():
        raise ValueError("Validation and test prediction directories must be different")
    output = resolve(options.get("output_dir", "outputs/calibration/m4_event_holdout"), must_exist=False)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/experiment/calibration.yaml").read_text(encoding="utf-8"))
    bins = int(options.get("ece_bins", config["metrics"]["ece_bins"]))
    connectivity = int(options.get("building_connectivity", config["building_connectivity"]))
    bootstrap_samples = int(options.get("bootstrap_samples", config.get("bootstrap_samples", 1000)))
    bootstrap_seed = int(options.get("bootstrap_seed", config.get("bootstrap_seed", 42)))

    validation = aggregate_prediction_files(
        prediction_files(validation_dir), partition="validation", connectivity=connectivity
    )
    test = aggregate_prediction_files(
        prediction_files(test_dir), partition="test", connectivity=connectivity
    )
    temperature = fit_temperature(validation.logits, validation.targets)
    metrics_rows: list[dict[str, Any]] = []
    reliability: dict[tuple[str, str], list[dict[str, float]]] = {}
    for partition, aggregate in (("validation", validation), ("test", test)):
        for calibration, value in (("uncalibrated", 1.0), ("temperature_scaled", temperature)):
            metrics, bins_rows = classification_metrics(
                aggregate.logits, aggregate.targets, temperature=value, bins=bins
            )
            intervals = bootstrap_classification_metrics(
                aggregate.logits,
                aggregate.targets,
                temperature=value,
                bins=bins,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            interval_columns = {
                f"{metric}_ci_lower": bounds["lower"]
                for metric, bounds in intervals["metrics"].items()
            } | {
                f"{metric}_ci_upper": bounds["upper"]
                for metric, bounds in intervals["metrics"].items()
            }
            metrics_rows.append(
                {
                    "partition": partition,
                    "calibration": calibration,
                    "bootstrap_unit": intervals["unit"],
                    "bootstrap_samples": intervals["samples"],
                    **metrics,
                    **interval_columns,
                }
            )
            reliability[(partition, calibration)] = bins_rows

    temperature_payload = {
        "temperature": temperature,
        "fit_partition": "validation",
        "validation_predictions": str(validation_dir),
        "test_predictions": str(test_dir),
        "n_validation_buildings": len(validation.rows),
        "n_test_buildings": len(test.rows),
        "method": "scalar_temperature_scaling",
        "aggregation": "mean_building_logits_from_reference_mask_connected_components",
        "bootstrap": {
            "unit": "building",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
    }
    (output / "temperature.json").write_text(
        json.dumps(temperature_payload, indent=2) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame(metrics_rows)
    table.to_csv(output / "calibration_table.csv", index=False)
    reports = ROOT / "outputs/reports"
    reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports / "calibration_table.csv", index=False)
    reliability_figure(
        output / "reliability_diagram.png",
        reliability[("test", "uncalibrated")],
        reliability[("test", "temperature_scaled")],
    )

    all_rows = add_probabilities(validation.rows, validation.logits, temperature=temperature)
    all_rows.extend(add_probabilities(test.rows, test.logits, temperature=temperature))
    # This combined calibration table can span several events and CRSs. Keep
    # it deliberately non-geospatial; label-free per-event inference writes
    # the authoritative GeoParquet/GeoJSON building artifacts in one CRS.
    frame = pd.DataFrame(all_rows)
    frame.to_parquet(output / "building_predictions.parquet", index=False)

    test_before = next(row for row in metrics_rows if row["partition"] == "test" and row["calibration"] == "uncalibrated")
    test_after = next(row for row in metrics_rows if row["partition"] == "test" and row["calibration"] == "temperature_scaled")
    report = [
        "# Building-level calibration report",
        "",
        "The scalar temperature was fit on validation buildings only. Test labels were used only for final evaluation.",
        "Reference-mask connected components define buildings for this evaluation artifact; event inference uses predicted components.",
        "",
        f"- Fitted temperature: `{temperature:.6f}`",
        f"- Validation buildings: `{len(validation.rows):,}`",
        f"- Test buildings: `{len(test.rows):,}`",
        "",
        "| Test metric | Before (95% CI) | After (95% CI) | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("ece", "nll", "brier", "macro_f1", "balanced_accuracy"):
        report.append(
            f"| {metric} | {test_before[metric]:.6f} "
            f"[{test_before[f'{metric}_ci_lower']:.6f}, {test_before[f'{metric}_ci_upper']:.6f}] | "
            f"{test_after[metric]:.6f} "
            f"[{test_after[f'{metric}_ci_lower']:.6f}, {test_after[f'{metric}_ci_upper']:.6f}] | "
            f"{test_after[metric] - test_before[metric]:+.6f} |"
        )
    report.extend(("", "Uncertainty is reported separately as normalized predictive entropy and is not automatically treated as priority."))
    (output / "calibration_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[calibration] fitted T={temperature:.6f} on validation only; artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
