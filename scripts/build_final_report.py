#!/usr/bin/env python3
"""Build the final DisasterLens report only from completed real-run artifacts.

This is deliberately fail-loud: it does not invent metrics, maps, data
provenance, or optional-extension results.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"


def options(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected key=value, received {value!r}")
        key, item = value.split("=", 1)
        parsed[key] = item
    return parsed


def require(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required completed artifact is missing or empty: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require(path).read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return "../" + str(path.relative_to(ROOT / "outputs"))


def metric(payload: dict[str, Any], name: str) -> float:
    for scope in (payload, payload.get("pooled", {})):
        if name in scope:
            return float(scope[name])
    raise KeyError(f"Metric {name!r} is absent from {payload.keys()}")


def summary(payload: dict[str, Any]) -> str:
    values = [
        f"macro_f1={metric(payload, 'macro_f1'):.4f}",
        f"miou={metric(payload, 'miou'):.4f}",
    ]
    for damage_key in ("f1_damage", "val_f1_damage"):
        if damage_key in payload or damage_key in payload.get("pooled", {}):
            values.append(f"f1_damage={metric(payload, damage_key):.4f}")
            break
    return ", ".join(values)


def main() -> None:
    event_id = options(sys.argv[1:]).get("event_id")
    if not event_id:
        raise ValueError("event_id=<held-out BRIGHT event> is required")

    runs = ROOT / "outputs/runs"
    m2_heldout = read_json(ROOT / "outputs/metrics/best_test.json")
    m3_standard = read_json(runs / "m3/standard_split/metrics.json")
    m3_heldout = read_json(runs / "m3/event_holdout/metrics.json")
    m4_standard = read_json(runs / "m4/standard_full/metrics.json")
    m4_heldout = read_json(runs / "m4/event_holdout_full/metrics.json")
    calibration_dir = ROOT / "outputs/calibration/m4_event_holdout"
    calibration = pd.read_csv(require(calibration_dir / "calibration_table.csv"))
    temperature = read_json(calibration_dir / "temperature.json")
    context_dir = ROOT / "outputs/geospatial" / event_id
    context = read_json(context_dir / "metadata.json")
    priority_dir = ROOT / "outputs/priority" / event_id
    priority = read_json(priority_dir / "metadata.json")
    features = gpd.read_parquet(require(context_dir / "features.parquet"))
    ranking = gpd.read_parquet(require(priority_dir / "priority.parquet"))
    sensitivity = pd.read_csv(require(priority_dir / "weight_sensitivity.csv"))
    require(priority_dir / "monte_carlo_draws.npz")
    cross_event = require(REPORTS / "cross_event_report.md")
    ablation = require(REPORTS / "ablation_table.csv")
    require(calibration_dir / "calibration_report.md")
    require(REPORTS / "priority_sensitivity.md")

    missing_provenance = [
        key
        for key in (
            "population_year", "population_source", "population_version",
            "population_license", "population_download_date",
        )
        if not context.get(key)
    ]
    if missing_provenance:
        raise ValueError(
            "Geospatial context lacks required population provenance: "
            + ", ".join(missing_provenance)
        )
    ablations = pd.read_csv(ablation)
    required_fusions = {"full", "gated_only", "sar_only"}
    observed_fusions = set(ablations.get("fusion", pd.Series(dtype=str)).dropna())
    if not required_fusions.issubset(observed_fusions):
        raise ValueError(
            "Ablation table is incomplete; required fusion modes are missing: "
            + ", ".join(sorted(required_fusions - observed_fusions))
        )

    figures = [
        runs / "m4/event_holdout_full/figures/test/pre_event_optical.png",
        runs / "m4/event_holdout_full/figures/test/post_event_sar.png",
        runs / "m4/event_holdout_full/figures/test/ground_truth.png",
        runs / "m4/event_holdout_full/figures/test/predicted_damage_map.png",
        runs / "m4/event_holdout_full/figures/test/entropy_map.png",
        runs / "m4/event_holdout_full/figures/test/confusion_matrix.png",
        runs / "m4/event_holdout_full/figures/test/per_event_metrics.png",
        calibration_dir / "reliability_diagram.png",
        ROOT / "outputs/figures" / event_id / "population_exposure_map.png",
        ROOT / "outputs/figures" / event_id / "road_risk_map.png",
        ROOT / "outputs/figures" / event_id / "priority_map.png",
        ROOT / "outputs/figures" / event_id / "rank_stability_map.png",
    ]
    for figure in figures:
        require(figure)

    test_calibration = calibration[calibration["partition"].eq("test")].set_index("calibration")
    before = test_calibration.loc["uncalibrated"]
    after = test_calibration.loc["temperature_scaled"]
    gap = metric(m4_standard, "macro_f1") - metric(m4_heldout, "macro_f1")
    exposure = float(features["population_damage_exposure"].sum())
    accessibility = float(features["accessibility_penalty"].mean())
    top_probability = float(ranking["prob_top_10_percent"].max())
    bands = ranking["priority_band"].value_counts().to_dict()

    matrix_rows = []
    matrix_rows.append({
        "model": "M2 early-fusion U-Net", "evaluation": "event held-out", 
        "calibration": "uncalibrated segmentation",
        "macro_f1": metric(m2_heldout, "macro_f1"), "miou": metric(m2_heldout, "miou"),
    })
    for model, evaluation, payload in (
        ("M3 pseudo-Siamese", "standard split", m3_standard),
        ("M3 pseudo-Siamese", "event held-out", m3_heldout),
        ("M4 fusion transformer", "standard split", m4_standard),
        ("M4 fusion transformer", "event held-out", m4_heldout),
    ):
        matrix_rows.append({
            "model": model, "evaluation": evaluation, "calibration": "uncalibrated segmentation",
            "macro_f1": metric(payload, "macro_f1"), "miou": metric(payload, "miou"),
        })
    for state, row in (("uncalibrated", before), ("temperature_scaled", after)):
        matrix_rows.append({
            "model": "M4 building aggregation", "evaluation": "event held-out test", "calibration": state,
            "macro_f1": row.get("macro_f1"), "miou": None, "ece": row["ece"],
            "nll": row["nll"], "brier": row["brier"],
        })
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows).to_csv(REPORTS / "experimental_matrix.csv", index=False)

    lines = [
        "# DisasterLens final report", "",
        "## Problem and motivation", "Building damage mapping is treated as multimodal segmentation followed by transparent, uncertainty-aware regional prioritization; outputs are decision support, not emergency-deployment claims.", "",
        "## Datasets and limitations", "The primary dataset is official BRIGHT VHR pre-event optical plus post-event SAR. Known limitations include label noise, class/event imbalance, residual registration error, and the absence of confirmed road closures or exact affected-population labels.", "",
        "## Multimodal model", "M3 uses a pseudo-Siamese architecture. M4 uses modality encoders, high-resolution gated fusion, coarse bidirectional cross-attention with optical/SAR/attention/difference/product features, and FPN multiscale decoding.", "",
        "## Standard evaluation", f"M3: {summary(m3_standard)}. M4: {summary(m4_standard)}.", "",
        "## Event-held-out evaluation", f"Held-out event `{event_id}` — M2 early-fusion U-Net: {summary(m2_heldout)}. M3: {summary(m3_heldout)}. M4: {summary(m4_heldout)}.", "",
        "## Cross-disaster generalization gap", f"M4 standard-to-held-out macro-F1 gap: `{gap:.4f}` (standard minus held-out). Detailed event analysis: [{cross_event.name}]({rel(cross_event)}).", "",
        "## Calibration and uncertainty", f"Scalar temperature `{float(temperature['temperature']):.6f}` was fit on validation buildings only. On held-out test buildings: ECE `{before['ece']:.4f}` → `{after['ece']:.4f}`, NLL `{before['nll']:.4f}` → `{after['nll']:.4f}`, Brier `{before['brier']:.4f}` → `{after['brier']:.4f}`.", "",
        "## Error analysis", "Use the confusion matrix, per-event chart, and saved prediction maps below to inspect error patterns. No causal claim is made beyond these observed artifacts.", "",
        "## Population exposure", f"Damage-weighted population exposure sums to `{exposure:.2f}` over analysis cells. This is a model-derived relative exposure estimate, not an exact count of affected people.", "",
        "## Accessibility-risk estimation", f"Mean estimated accessibility penalty is `{accessibility:.4f}`. It is a road-network risk heuristic, not a confirmed closure or routing guarantee.", "",
        "## Priority formulation", f"Relative event-level priority combines configured damage, population, and accessibility terms. Band counts: `{bands}`. Effective weights are recorded in [{priority_dir.name} metadata]({rel(priority_dir / 'metadata.json')}).", "",
        "## Monte Carlo ranking uncertainty", f"Monte Carlo uses `{priority['monte_carlo_simulations']}` draws; the largest cell-level probability of being top-decile priority is `{top_probability:.4f}`.", "",
        "## Weight sensitivity", f"Across `{len(sensitivity):,}` sampled weight settings, median Spearman rank correlation is `{sensitivity['spearman_rank_correlation'].median():.4f}` and median top-decile overlap is `{sensitivity['top_decile_overlap'].median():.4f}`.", "",
        "## Limitations", "Priority weights are policy assumptions requiring domain-expert validation. Population, accessibility, and uncertainty layers inherit geospatial alignment and model limitations. This work does not predict verified casualties, road closures, or operational response outcomes.", "",
        "## Optional Prithvi/xBD-S12 extension", "Not implemented in this V1 report. Prithvi is intentionally not forced onto BRIGHT VHR RGB imagery; any extension must use compatible aligned Sentinel/HLS-style inputs.", "",
        "## Future work", "Add xBD/xBD-S12 replication, compatible medium-resolution context, domain adaptation, few-shot event adaptation, and externally validated hazard/accessibility feeds.", "",
        "## Required visualizations", "",
    ]
    captions = ("Pre-event optical", "Post-event SAR", "Ground truth", "Predicted damage", "Predictive entropy", "Confusion matrix", "Per-event F1 and mIoU", "Reliability", "Population exposure", "Road accessibility risk", "Relative priority", "Rank stability")
    lines.extend(f"- {caption}: ![]({rel(path)})" for caption, path in zip(captions, figures, strict=True))
    lines.extend(["", "## Reproducibility artifacts", f"- [Experimental matrix]({rel(REPORTS / 'experimental_matrix.csv')})", f"- [Ablation table]({rel(ablation)})", f"- [Calibration report]({rel(calibration_dir / 'calibration_report.md')})", f"- [Priority sensitivity report]({rel(REPORTS / 'priority_sensitivity.md')})", ""])
    (REPORTS / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] complete real-artifact final report: {REPORTS / 'final_report.md'}", flush=True)


if __name__ == "__main__":
    main()
