#!/usr/bin/env python3
"""Resumable, fail-loud DisasterLens milestone runner.

This entry point orchestrates the real implementation scripts.  It never
downloads, creates, or substitutes training data, and it never treats a
missing milestone implementation as success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
MILESTONES = tuple(f"M{number}" for number in range(9))
STATE_VERSION = 1


class PipelineError(RuntimeError):
    """An actionable pipeline contract failure."""


@dataclass(frozen=True)
class Step:
    milestone: str
    name: str
    command: tuple[str, ...]
    artifacts: tuple[Path, ...] = ()
    validator: Callable[[], tuple[bool, str]] | None = None
    gpu: bool = False
    adopt_existing: bool = False
    context: str = ""

    @property
    def key(self) -> str:
        return f"{self.milestone}:{self.name}"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "state_version": STATE_VERSION,
                "command": self.command,
                "context": self.context,
            },
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def nonempty(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*"))
    return False


def json_file(path: Path) -> tuple[bool, str]:
    if not nonempty(path):
        return False, f"missing or empty: {path.relative_to(ROOT)}"
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid JSON {path.relative_to(ROOT)}: {exc}"
    return True, "valid JSON"


def validate_artifacts(step: Step) -> tuple[bool, str]:
    missing = [path for path in step.artifacts if not nonempty(path)]
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        return False, f"missing or empty artifacts: {names}"
    if step.validator is not None:
        return step.validator()
    return True, "required artifacts exist"


def validate_official_bright_root(root: Path) -> None:
    if not root.is_dir():
        raise PipelineError(f"Official BRIGHT root does not exist: {root}")
    required = ("pre-event", "post-event", "target")
    failures: list[str] = []
    for name in required:
        directory = root / name
        if not directory.is_dir():
            failures.append(f"missing directory {directory}")
        elif not next((item for item in directory.rglob("*") if item.is_file()), None):
            failures.append(f"no files below {directory}")
    if failures:
        raise PipelineError(
            "Attached official BRIGHT data is incomplete:\n- " + "\n- ".join(failures)
        )


def require_t4() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise PipelineError("A Kaggle Tesla T4 GPU is required but nvidia-smi failed") from exc
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not any("T4" in name.upper() for name in names):
        raise PipelineError(f"Tesla T4 required; detected: {names or ['no GPU']}")
    print(f"[preflight] GPU: {', '.join(names)}", flush=True)


def manifest_events() -> list[str]:
    path = ROOT / "data/manifests/bright_manifest.jsonl"
    if not nonempty(path):
        return []
    events: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)["event_id"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise PipelineError(f"Invalid manifest record at {path}:{number}") from exc
            events.add(str(event))
    return sorted(events)


def validate_m1() -> tuple[bool, str]:
    required = (
        ROOT / "data/manifests/bright_manifest.jsonl",
        ROOT / "data/manifests/bright_normalization.json",
        ROOT / "outputs/reports/bright_data_audit.md",
    )
    missing = [path for path in required if not nonempty(path)]
    if missing:
        return False, "missing M1 artifacts: " + ", ".join(
            str(path.relative_to(ROOT)) for path in missing
        )
    ok, message = json_file(required[1])
    if not ok:
        return ok, message
    events = manifest_events()
    if not events:
        return False, "manifest contains no events"
    return True, f"validated manifest with {len(events)} events"


def validate_tiny_gate() -> tuple[bool, str]:
    path = ROOT / "outputs/checkpoints/early_fusion_unet_tiny/tiny_overfit_result.json"
    ok, message = json_file(path)
    if not ok:
        return ok, message
    result = json.loads(path.read_text(encoding="utf-8"))
    score = float(result.get("best_damage_macro_f1", -1))
    required = float(result.get("required_damage_macro_f1", 0.95))
    if score < required:
        return False, f"tiny gate failed: {score:.4f} < {required:.4f}"
    return True, f"tiny gate passed: {score:.4f} >= {required:.4f}"


def validate_full_m2(minimum_epochs: int) -> tuple[bool, str]:
    metrics = ROOT / "outputs/checkpoints/early_fusion_unet_full/metrics.json"
    checkpoint = ROOT / "outputs/checkpoints/early_fusion_unet_full/best.pt"
    ok, message = json_file(metrics)
    if not ok:
        return ok, message
    if not nonempty(checkpoint):
        return False, "missing non-empty full M2 best.pt"
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if not payload.get("history") or "best" not in payload:
        return False, "full M2 metrics lacks best/history"
    if len(payload["history"]) < minimum_epochs:
        return False, f"full M2 has {len(payload['history'])}/{minimum_epochs} required epochs"
    return True, f"validated {len(payload['history'])} completed epochs and best.pt"


def validate_event_split(test_event: str) -> tuple[bool, str]:
    split_path = ROOT / "data/manifests/splits/event_holdout.json"
    ok, message = json_file(split_path)
    if not ok:
        return ok, message
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set(split) != {"train", "val", "test"} or any(not split[name] for name in split):
        return False, "split must have non-empty train, val, and test partitions"
    partitions = {name: set(values) for name, values in split.items()}
    if any(
        partitions[left] & partitions[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        return False, "split contains tile leakage"
    manifest_path = ROOT / "data/manifests/bright_manifest.jsonl"
    tile_events: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                tile_events[str(record["tile_id"])] = str(record["event_id"])
    unknown = [tile for tile in partitions["test"] if tile not in tile_events]
    if unknown:
        return False, f"split references {len(unknown)} tiles absent from the manifest"
    actual = {tile_events[tile] for tile in partitions["test"]}
    if actual != {test_event}:
        return False, f"test partition events {sorted(actual)} do not equal requested {test_event!r}"
    return True, f"leakage-free held-out split for {test_event}"


def validate_standard_split() -> tuple[bool, str]:
    split_path = ROOT / "data/manifests/splits/standard_split.json"
    ok, message = json_file(split_path)
    if not ok:
        return ok, message
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set(split) != {"train", "val", "test"} or any(not split[name] for name in split):
        return False, "standard split must have non-empty train, val, and test partitions"
    partitions = {name: set(values) for name, values in split.items()}
    if any(
        partitions[left] & partitions[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        return False, "standard split contains tile leakage"
    return True, "non-empty leakage-free standard split"


def validate_event_predictions(event_id: str) -> tuple[bool, str]:
    directory = ROOT / "outputs/predictions" / event_id
    required = (
        directory / "pre_event_optical.tif",
        directory / "post_event_sar.tif",
        directory / "semantic_mask.tif",
        directory / "damage_probabilities.tif",
        directory / "uncertainty.tif",
        directory / "building_predictions.parquet",
        directory / "buildings.geojson",
        directory / "metadata.json",
    )
    missing = [path for path in required if not nonempty(path)]
    if missing:
        return False, "missing event inference artifacts: " + ", ".join(
            str(path.relative_to(ROOT)) for path in missing
        )
    ok, message = json_file(directory / "metadata.json")
    if not ok:
        return ok, message
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("event_id") != event_id or metadata.get("label_usage") != "none; labels are not read during inference":
        return False, "event inference metadata violates the test-event/no-label contract"
    return True, f"complete label-free georeferenced predictions for {event_id}"


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolved_file(path: Path | None, option: str) -> Path | None:
    if path is None:
        return None
    result = path.expanduser()
    result = result if result.is_absolute() else ROOT / result
    result = result.resolve()
    if not result.is_file():
        raise PipelineError(f"{option} does not point to a real file: {result}")
    return result


def phase_scripts(milestone: str) -> tuple[str, ...]:
    mapping = {
        "M3": ("scripts/train_m3.py", "scripts/evaluate_m3.py"),
        "M4": ("scripts/train_m4.py", "scripts/evaluate_m4.py"),
        "M5": ("scripts/calibrate.py", "scripts/infer_event.py"),
        "M6": (
            "scripts/fetch_population.py",
            "scripts/fetch_osm.py",
            "scripts/build_geospatial_context.py",
        ),
        "M7": ("scripts/build_priority_outputs.py",),
        "M8": ("app/streamlit_app.py", "scripts/build_final_report.py", "scripts/smoke_app.py"),
    }
    return mapping.get(milestone, ())


def require_implemented(milestones: Iterable[str]) -> None:
    missing: dict[str, list[str]] = {}
    for milestone in milestones:
        absent = [path for path in phase_scripts(milestone) if not (ROOT / path).is_file()]
        if absent:
            missing[milestone] = absent
    if missing:
        details = "; ".join(f"{key}: {', '.join(value)}" for key, value in missing.items())
        raise PipelineError(
            "Requested milestones are not implemented in this checkout. "
            f"Refusing to fake completion ({details}). Implement and test them before running."
        )


def build_steps(args: argparse.Namespace, test_event: str) -> list[Step]:
    py = sys.executable
    heldout_split = "data/manifests/splits/event_holdout.json"
    standard_split = "data/manifests/splits/standard_split.json"
    m3_standard = ROOT / "outputs/runs/m3/standard_split"
    m3_heldout = ROOT / "outputs/runs/m3/event_holdout"
    m4_standard = ROOT / "outputs/runs/m4/standard_full"
    m4_heldout = ROOT / "outputs/runs/m4/event_holdout_full"
    m4_ablation = ROOT / "outputs/runs/m4/event_holdout_gated_only"
    m4_sar = ROOT / "outputs/runs/m4/event_holdout_sar_only"
    calibration = ROOT / "outputs/calibration/m4_event_holdout"
    predictions = ROOT / "outputs/predictions" / test_event
    context = ROOT / "outputs/geospatial" / test_event
    priority = ROOT / "outputs/priority" / test_event

    def training_artifacts(directory: Path) -> tuple[Path, ...]:
        return tuple(
            directory / name
            for name in (
                "config.yaml",
                "training_metrics.json",
                "split_manifest.parquet",
                "checkpoint.pt",
                "git_commit.txt",
                "environment.txt",
            )
        )

    def evaluation_artifacts(directory: Path, partition: str = "test") -> tuple[Path, ...]:
        return tuple(
            directory / name
            for name in ("metrics.json", "metrics_by_event.csv", "class_metrics.csv")
        ) + tuple(
            directory / "figures" / partition / name
            for name in (
                "pre_event_optical.png", "post_event_sar.png", "ground_truth.png",
                "predicted_damage_map.png", "entropy_map.png", "confusion_matrix.png",
                "per_event_metrics.png",
            )
        )

    def train_command(
        kind: str,
        split_path: str,
        split_name: str,
        directory: Path,
        epochs: int,
        *,
        fusion: str | None = None,
    ) -> tuple[str, ...]:
        command = [
            py,
            "-u",
            f"scripts/train_{kind}.py",
            f"split_path={split_path}",
            f"split_name={split_name}",
            f"run_dir={relative_or_absolute(directory)}",
            f"trainer.epochs={epochs}",
        ]
        if fusion is not None:
            command.append(f"model.fusion.mode={fusion}")
        return tuple(command)

    def evaluate_command(
        kind: str,
        split_path: str,
        split_name: str,
        directory: Path,
        *,
        partition: str = "test",
        fusion: str | None = None,
    ) -> tuple[str, ...]:
        command = [
            py,
            "-u",
            f"scripts/evaluate_{kind}.py",
            f"split_path={split_path}",
            f"split_name={split_name}",
            f"run_dir={relative_or_absolute(directory)}",
            f"checkpoint={relative_or_absolute(directory / 'checkpoint.pt')}",
            f"partition={partition}",
            "save_predictions=true",
        ]
        if fusion is not None:
            command.append(f"model.fusion.mode={fusion}")
        return tuple(command)

    steps = [
        Step("M0", "tests", (py, "-m", "pytest", "-q")),
        Step(
            "M1",
            "audit",
            (py, "-u", "scripts/inspect_bright.py", "data=bright"),
            validator=validate_m1,
            adopt_existing=True,
        ),
        Step(
            "M1",
            "loader",
            (py, "-u", "scripts/verify_bright_loader.py", "data=bright"),
        ),
        Step(
            "M2",
            "split",
            (py, "-u", "scripts/make_splits.py", "data=bright", f"split.test_events=[{test_event}]"),
            validator=lambda: validate_event_split(test_event),
            adopt_existing=True,
        ),
        Step(
            "M2",
            "tiny_overfit",
            (
                py, "-u", "scripts/train.py", f"split_path={heldout_split}", "overfit_tiles=8",
                f"epochs={args.tiny_epochs}", "crop_size=512", "learning_rate=0.001",
                "warmup_epochs=0", "use_class_weights=false",
                "checkpoint_dir=outputs/checkpoints/early_fusion_unet_tiny",
            ),
            validator=validate_tiny_gate,
            gpu=True,
            adopt_existing=True,
        ),
        Step(
            "M2",
            "full_training",
            (
                py, "-u", "scripts/train.py", f"split_path={heldout_split}",
                f"epochs={args.baseline_epochs}", "crop_size=512",
                "checkpoint_dir=outputs/checkpoints/early_fusion_unet_full",
            ),
            validator=lambda: validate_full_m2(args.baseline_epochs),
            gpu=True,
            adopt_existing=True,
        ),
        Step(
            "M2",
            "evaluation",
            (
                py, "-u", "scripts/evaluate.py",
                "checkpoint=outputs/checkpoints/early_fusion_unet_full/best.pt",
                f"split_path={heldout_split}", "partition=test",
            ),
            validator=lambda: json_file(ROOT / "outputs/metrics/best_test.json"),
            gpu=True,
            adopt_existing=True,
        ),
    ]

    # Every later command consumes saved real-data artifacts from the previous
    # command. No milestone has a synthetic or placeholder fallback.
    later = {
        "M3": [
            Step(
                "M3",
                "heldout_split",
                (
                    py,
                    "-u",
                    "scripts/make_splits.py",
                    "data=bright",
                    f"split.test_events=[{test_event}]",
                ),
                validator=lambda: validate_event_split(test_event),
                adopt_existing=True,
            ),
            Step(
                "M3",
                "standard_split",
                (py, "-u", "scripts/make_splits.py", "data=bright", "split=standard_split"),
                artifacts=(ROOT / standard_split,),
                validator=validate_standard_split,
            ),
            Step(
                "M3", "standard_training",
                train_command("m3", standard_split, "standard_split", m3_standard, args.m3_epochs),
                artifacts=training_artifacts(m3_standard), gpu=True,
            ),
            Step(
                "M3", "standard_evaluation",
                evaluate_command("m3", standard_split, "standard_split", m3_standard),
                artifacts=evaluation_artifacts(m3_standard), gpu=True,
            ),
            Step(
                "M3", "heldout_training",
                train_command("m3", heldout_split, "event_holdout", m3_heldout, args.m3_epochs),
                artifacts=training_artifacts(m3_heldout), gpu=True,
            ),
            Step(
                "M3", "heldout_evaluation",
                evaluate_command("m3", heldout_split, "event_holdout", m3_heldout),
                artifacts=evaluation_artifacts(m3_heldout) + (
                    ROOT / "outputs/reports/cross_event_report.md",
                ),
                gpu=True,
            ),
        ],
        "M4": [
            Step(
                "M4", "standard_training",
                train_command("m4", standard_split, "standard_full", m4_standard, args.m4_epochs, fusion="full"),
                artifacts=training_artifacts(m4_standard), gpu=True,
            ),
            Step(
                "M4", "standard_evaluation",
                evaluate_command("m4", standard_split, "standard_full", m4_standard, fusion="full"),
                artifacts=evaluation_artifacts(m4_standard), gpu=True,
            ),
            Step(
                "M4", "heldout_training",
                train_command("m4", heldout_split, "event_holdout_full", m4_heldout, args.m4_epochs, fusion="full"),
                artifacts=training_artifacts(m4_heldout), gpu=True,
            ),
            Step(
                "M4", "heldout_validation_predictions",
                evaluate_command(
                    "m4", heldout_split, "event_holdout_full", m4_heldout,
                    partition="val", fusion="full",
                ),
                artifacts=(m4_heldout / "predictions/val",) + evaluation_artifacts(m4_heldout, "val"), gpu=True,
            ),
            Step(
                "M4", "heldout_test_evaluation",
                evaluate_command("m4", heldout_split, "event_holdout_full", m4_heldout, fusion="full"),
                artifacts=evaluation_artifacts(m4_heldout) + (
                    m4_heldout / "predictions/test",
                    ROOT / "outputs/reports/cross_event_report.md",
                ),
                gpu=True,
            ),
            Step(
                "M4", "gated_only_ablation_training",
                train_command(
                    "m4", heldout_split, "event_holdout_gated_only", m4_ablation,
                    args.m4_epochs, fusion="gated_only",
                ),
                artifacts=training_artifacts(m4_ablation), gpu=True,
            ),
            Step(
                "M4", "gated_only_ablation_evaluation",
                evaluate_command(
                    "m4", heldout_split, "event_holdout_gated_only", m4_ablation,
                    fusion="gated_only",
                ),
                artifacts=evaluation_artifacts(m4_ablation) + (
                    ROOT / "outputs/reports/ablation_table.csv",
                ),
                gpu=True,
            ),
            Step(
                "M4", "sar_only_ablation_training",
                train_command(
                    "m4", heldout_split, "event_holdout_sar_only", m4_sar,
                    args.m4_epochs, fusion="sar_only",
                ),
                artifacts=training_artifacts(m4_sar), gpu=True,
            ),
            Step(
                "M4", "sar_only_ablation_evaluation",
                evaluate_command(
                    "m4", heldout_split, "event_holdout_sar_only", m4_sar,
                    fusion="sar_only",
                ),
                artifacts=evaluation_artifacts(m4_sar) + (
                    ROOT / "outputs/reports/ablation_table.csv",
                ),
                gpu=True,
            ),
        ],
        "M5": [
            Step(
                "M5", "calibration",
                (
                    py, "-u", "scripts/calibrate.py",
                    f"validation_predictions={relative_or_absolute(m4_heldout / 'predictions/val')}",
                    f"test_predictions={relative_or_absolute(m4_heldout / 'predictions/test')}",
                    f"output_dir={relative_or_absolute(calibration)}",
                ),
                artifacts=tuple(
                    calibration / name
                    for name in (
                        "temperature.json", "calibration_table.csv", "reliability_diagram.png",
                        "building_predictions.parquet", "calibration_report.md",
                    )
                ) + (ROOT / "outputs/reports/calibration_table.csv",),
            ),
            Step(
                "M5", "event_inference",
                (
                    py, "-u", "scripts/infer_event.py",
                    f"checkpoint={relative_or_absolute(m4_heldout / 'checkpoint.pt')}",
                    f"event_id={test_event}",
                    f"split_path={heldout_split}",
                    f"temperature_path={relative_or_absolute(calibration / 'temperature.json')}",
                    "model_kind=m4", "fusion_mode=full",
                    f"output_dir={relative_or_absolute(predictions)}",
                ),
                validator=lambda: validate_event_predictions(test_event), gpu=True,
            ),
        ],
        "M6": [],
        "M7": [
            Step(
                "M7", "priority",
                (
                    py, "-u", "scripts/build_priority_outputs.py",
                    f"event_id={test_event}",
                    f"features={relative_or_absolute(context / 'features.parquet')}",
                    f"buildings={relative_or_absolute(predictions / 'building_predictions.parquet')}",
                    f"output_dir={relative_or_absolute(priority)}",
                    f"simulations={args.monte_carlo_simulations}",
                    f"sensitivity_samples={args.sensitivity_samples}",
                    f"checkpoint={relative_or_absolute(m4_heldout / 'checkpoint.pt')}",
                ),
                artifacts=tuple(
                    priority / name
                    for name in (
                        "priority.parquet", "priority.geojson", "weight_sensitivity.csv",
                        "monte_carlo_draws.npz", "metadata.json",
                    )
                ) + (
                    ROOT / "outputs/figures/rank_stability_map.png",
                    ROOT / "outputs/figures" / test_event / "priority_map.png",
                    ROOT / "outputs/figures" / test_event / "rank_stability_map.png",
                    ROOT / "outputs/reports/priority_sensitivity.md",
                ),
            )
        ],
        "M8": [
            Step(
                "M8", "final_report",
                (py, "-u", "scripts/build_final_report.py", f"event_id={test_event}"),
                artifacts=(ROOT / "outputs/reports/final_report.md", ROOT / "outputs/reports/experimental_matrix.csv"),
            ),
            Step(
                "M8", "app_smoke",
                (py, "-u", "scripts/smoke_app.py"),
                artifacts=(ROOT / "app/streamlit_app.py", ROOT / "outputs/app/smoke.json"),
            )
        ],
    }

    # M3--M5 share stable output locations.  Include the held-out event in
    # their state fingerprint so a later run for another event cannot silently
    # reuse a checkpoint, calibration, or report produced for the previous one.
    for milestone in ("M3", "M4", "M5"):
        later[milestone] = [
            replace(step, context=f"heldout_event={test_event}")
            for step in later[milestone]
        ]

    if args.worldpop_fetch:
        source_argument = (
            f"source={relative_or_absolute(args.worldpop_source_resolved)}"
            if args.worldpop_source_resolved is not None
            else f"url={args.worldpop_url}"
        )
        later["M6"].append(
            Step(
                "M6", "population_acquisition",
                (
                    py, "-u", "scripts/fetch_population.py", f"event_id={test_event}",
                    f"year={args.population_year}", f"source_name={args.population_source}",
                    f"version={args.population_version}", f"license={args.population_license}",
                    f"download_date={args.population_download_date}",
                    source_argument, f"output={relative_or_absolute(args.worldpop_resolved)}",
                ),
                artifacts=(
                    args.worldpop_resolved,
                    args.worldpop_resolved.parent / "metadata.json",
                ),
            )
        )
    if args.osm_fetch:
        if args.osm_bbox:
            osm_source = [f"bbox={args.osm_bbox}"]
            if args.overpass_url:
                osm_source.append(f"overpass_url={args.overpass_url}")
        else:
            osm_source = [
                f"roads_source={relative_or_absolute(args.roads_source_resolved)}",
                f"facilities_source={relative_or_absolute(args.facilities_source_resolved)}",
            ]
        later["M6"].append(
            Step(
                "M6", "osm_acquisition",
                tuple(
                    [
                        py, "-u", "scripts/fetch_osm.py", f"event_id={test_event}",
                        *osm_source, f"output_dir={relative_or_absolute(args.osm_output_resolved)}",
                    ]
                ),
                artifacts=(
                    args.roads_resolved,
                    args.facilities_resolved,
                    args.osm_output_resolved / "graph.graphml",
                    args.osm_output_resolved / "metadata.json",
                ),
            )
        )
    later["M6"].append(
        Step(
            "M6", "geospatial_context",
            (
                py, "-u", "scripts/build_geospatial_context.py",
                f"event_id={test_event}",
                f"buildings={relative_or_absolute(predictions / 'building_predictions.parquet')}",
                f"worldpop={relative_or_absolute(args.worldpop_resolved)}",
                f"roads={relative_or_absolute(args.roads_resolved)}",
                f"facilities={relative_or_absolute(args.facilities_resolved)}",
                f"population_year={args.population_year}",
                f"population_source={args.population_source}",
                f"population_version={args.population_version}",
                f"population_license={args.population_license}",
                f"population_download_date={args.population_download_date}",
                f"output_dir={relative_or_absolute(context)}",
            ),
            artifacts=tuple(
                context / name
            for name in (
                "features.parquet", "features.geojson",
                "roads_with_estimated_risk.parquet", "metadata.json",
            )
            ) + (
                ROOT / "outputs/figures" / test_event / "population_exposure_map.png",
                ROOT / "outputs/figures" / test_event / "road_risk_map.png",
            ),
        )
    )
    for milestone in MILESTONES[3:]:
        steps.extend(later[milestone])
    return steps


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"version": STATE_VERSION, "steps": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read state file {path}: {exc}") from exc
    if state.get("version") != STATE_VERSION or not isinstance(state.get("steps"), dict):
        raise PipelineError(f"Unsupported or invalid state file: {path}")
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed(step: Step, state: dict, force: bool) -> tuple[bool, str]:
    if force:
        return False, "forced"
    valid, reason = validate_artifacts(step)
    record = state["steps"].get(step.key, {})
    # Only explicitly marked compatibility steps may adopt artifacts created
    # before this runner existed. Later milestones require the exact recorded
    # command fingerprint so stale checkpoints cannot be silently reused.
    if valid and (
        record.get("fingerprint") == step.fingerprint
        or (step.adopt_existing and step.validator is not None)
    ):
        return True, reason
    return False, reason


def run_step(step: Step, state: dict, state_path: Path, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step.milestone}_{step.name}.log"
    print(f"\n{'=' * 78}\n[{step.milestone}] {step.name}\n{'=' * 78}", flush=True)
    print("[command] " + " ".join(step.command), flush=True)
    started = time.monotonic()
    record = {"status": "running", "started_at": utc_now(), "fingerprint": step.fingerprint}
    state["steps"][step.key] = record
    save_state(state_path, state)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{record['started_at']}] {' '.join(step.command)}\n")
        process = subprocess.Popen(
            step.command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code:
        record.update(status="failed", finished_at=utc_now(), seconds=elapsed, return_code=return_code)
        save_state(state_path, state)
        raise PipelineError(f"{step.key} failed with exit code {return_code}; see {log_path}")
    valid, reason = validate_artifacts(step)
    if not valid:
        record.update(status="failed", finished_at=utc_now(), seconds=elapsed, error=reason)
        save_state(state_path, state)
        raise PipelineError(f"{step.key} command returned 0 but artifact validation failed: {reason}")
    record.update(status="complete", finished_at=utc_now(), seconds=elapsed, validation=reason)
    save_state(state_path, state)
    print(f"[complete] {step.key} in {elapsed:.1f}s ({reason})", flush=True)


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PipelineError(
            f"Pipeline lock exists at {path}. If no run is active, remove only that stale lock file."
        ) from exc


def unchecked_path(path: Path | None, fallback: Path) -> Path:
    if path is None:
        return fallback.resolve()
    candidate = path.expanduser()
    return (candidate if candidate.is_absolute() else ROOT / candidate).resolve()


def configure_external_sources(
    args: argparse.Namespace,
    event_id: str,
    *,
    required: bool,
) -> None:
    """Resolve M6 inputs without inventing population or OSM data."""
    population_default = ROOT / "data/external/worldpop" / event_id / "population.tif"
    osm_default = ROOT / "data/external/osm" / event_id

    args.worldpop_fetch = False
    args.osm_fetch = False
    args.worldpop_source_resolved = None
    args.roads_source_resolved = None
    args.facilities_source_resolved = None
    args.osm_output_resolved = osm_default.resolve()
    args.worldpop_resolved = unchecked_path(args.worldpop, population_default)
    args.roads_resolved = unchecked_path(args.roads, osm_default / "roads.gpkg")
    args.facilities_resolved = unchecked_path(
        args.facilities, osm_default / "facilities.gpkg"
    )

    if not required:
        return
    provenance = (
        args.population_year, args.population_source, args.population_version,
        args.population_license, args.population_download_date,
    )
    if not all(provenance):
        raise PipelineError(
            "M6 requires --population-year, --population-source, --population-version, "
            "--population-license, and --population-download-date so every "
            "population estimate has explicit provenance"
        )

    population_modes = sum(
        bool(value) for value in (args.worldpop, args.worldpop_source, args.worldpop_url)
    )
    if population_modes > 1:
        raise PipelineError(
            "Choose exactly one population mode: --worldpop, --worldpop-source, or --worldpop-url"
        )
    if args.worldpop is not None:
        args.worldpop_resolved = resolved_file(args.worldpop, "--worldpop")
    elif args.worldpop_source is not None:
        args.worldpop_source_resolved = resolved_file(
            args.worldpop_source, "--worldpop-source"
        )
        args.worldpop_resolved = population_default.resolve()
        args.worldpop_fetch = True
    elif args.worldpop_url:
        args.worldpop_resolved = population_default.resolve()
        args.worldpop_fetch = True
    elif population_default.is_file() and population_default.stat().st_size > 0:
        args.worldpop_resolved = population_default.resolve()
    else:
        raise PipelineError(
            "M6 has no real WorldPop input. Pass --worldpop for an existing raster, "
            "--worldpop-source to copy one, or --worldpop-url to download an official raster"
        )

    prepared_osm = bool(args.roads or args.facilities)
    source_osm = bool(args.roads_source or args.facilities_source)
    query_osm = bool(args.osm_bbox)
    if sum((prepared_osm, source_osm, query_osm)) > 1:
        raise PipelineError(
            "Choose one OSM mode: prepared --roads/--facilities, local "
            "--roads-source/--facilities-source, or --osm-bbox"
        )
    if prepared_osm:
        if args.roads is None or args.facilities is None:
            raise PipelineError("Prepared OSM mode requires both --roads and --facilities")
        args.roads_resolved = resolved_file(args.roads, "--roads")
        args.facilities_resolved = resolved_file(args.facilities, "--facilities")
    elif source_osm:
        if args.roads_source is None or args.facilities_source is None:
            raise PipelineError(
                "Local OSM acquisition requires both --roads-source and --facilities-source"
            )
        args.roads_source_resolved = resolved_file(args.roads_source, "--roads-source")
        args.facilities_source_resolved = resolved_file(
            args.facilities_source, "--facilities-source"
        )
        args.roads_resolved = (osm_default / "roads.gpkg").resolve()
        args.facilities_resolved = (osm_default / "facilities.gpkg").resolve()
        args.osm_fetch = True
    elif query_osm:
        args.roads_resolved = (osm_default / "roads.gpkg").resolve()
        args.facilities_resolved = (osm_default / "facilities.gpkg").resolve()
        args.osm_fetch = True
    elif (
        (osm_default / "roads.gpkg").is_file()
        and (osm_default / "facilities.gpkg").is_file()
    ):
        args.roads_resolved = (osm_default / "roads.gpkg").resolve()
        args.facilities_resolved = (osm_default / "facilities.gpkg").resolve()
    else:
        raise PipelineError(
            "M6 has no real OSM inputs. Pass prepared --roads and --facilities, "
            "local --roads-source and --facilities-source, or an --osm-bbox query"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-milestone", choices=MILESTONES, default="M0")
    parser.add_argument("--through", choices=MILESTONES, default="M2")
    parser.add_argument("--bright-root", type=Path, default=None)
    parser.add_argument("--test-event", default=os.environ.get("DISASTERLENS_TEST_EVENT"))
    parser.add_argument("--tiny-epochs", type=int, default=400)
    parser.add_argument("--baseline-epochs", type=int, default=30)
    parser.add_argument("--m3-epochs", type=int, default=60)
    parser.add_argument("--m4-epochs", type=int, default=60)
    parser.add_argument("--worldpop", type=Path, help="existing real WorldPop GeoTIFF")
    parser.add_argument("--worldpop-source", type=Path, help="real GeoTIFF to copy into the run")
    parser.add_argument("--worldpop-url", help="official WorldPop GeoTIFF URL")
    parser.add_argument("--population-year", default=os.environ.get("DISASTERLENS_POPULATION_YEAR"))
    parser.add_argument("--population-source", default=os.environ.get("DISASTERLENS_POPULATION_SOURCE"))
    parser.add_argument("--population-version", default=os.environ.get("DISASTERLENS_POPULATION_VERSION"))
    parser.add_argument("--population-license", default=os.environ.get("DISASTERLENS_POPULATION_LICENSE"))
    parser.add_argument("--population-download-date", default=os.environ.get("DISASTERLENS_POPULATION_DOWNLOAD_DATE"))
    parser.add_argument("--roads", type=Path, help="existing real OSM-derived roads dataset")
    parser.add_argument("--facilities", type=Path, help="existing real OSM-derived facilities dataset")
    parser.add_argument("--roads-source", type=Path, help="real OSM roads extract to normalize")
    parser.add_argument("--facilities-source", type=Path, help="real OSM facilities extract to normalize")
    parser.add_argument("--osm-bbox", help="Overpass bbox as west,south,east,north")
    parser.add_argument("--overpass-url", default=None)
    parser.add_argument("--monte-carlo-simulations", type=int, default=500)
    parser.add_argument("--sensitivity-samples", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="rerun steps even when valid artifacts exist")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-non-t4", action="store_true", help="development only; Kaggle production runs require T4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start, end = MILESTONES.index(args.from_milestone), MILESTONES.index(args.through)
    if start > end:
        raise PipelineError("--from-milestone must not be after --through")
    positive_values = {
        "--tiny-epochs": args.tiny_epochs,
        "--baseline-epochs": args.baseline_epochs,
        "--m3-epochs": args.m3_epochs,
        "--m4-epochs": args.m4_epochs,
        "--monte-carlo-simulations": args.monte_carlo_simulations,
        "--sensitivity-samples": args.sensitivity_samples,
    }
    invalid = [name for name, value in positive_values.items() if value < 1]
    if invalid:
        raise PipelineError("These values must be positive: " + ", ".join(invalid))
    selected_milestones = MILESTONES[start : end + 1]
    if end >= 3:
        require_implemented(selected_milestones)

    needs_data = any(milestone != "M0" for milestone in selected_milestones)
    if needs_data:
        bright_root = args.bright_root or (
            Path(os.environ["DISASTERLENS_BRIGHT_ROOT"])
            if os.environ.get("DISASTERLENS_BRIGHT_ROOT")
            else None
        )
        if bright_root is None:
            raise PipelineError("Set DISASTERLENS_BRIGHT_ROOT or pass --bright-root for official BRIGHT data")
        bright_root = bright_root.expanduser().resolve()
        validate_official_bright_root(bright_root)
        os.environ["DISASTERLENS_BRIGHT_ROOT"] = str(bright_root)
        print(f"[preflight] official BRIGHT root: {bright_root}", flush=True)

    requires_gpu = any(milestone in selected_milestones for milestone in ("M2", "M3", "M4"))
    if requires_gpu and not args.allow_non_t4 and not args.dry_run:
        require_t4()

    events = manifest_events()
    if args.test_event and events and args.test_event not in events:
        raise PipelineError(f"Unknown test event {args.test_event!r}; available events include {events[:10]}")
    if args.dry_run:
        if not events and not args.test_event and end > 1:
            raise PipelineError(
                "--test-event is required for a dry run until M1 has created the manifest"
            )
        test_event = args.test_event or (events[0] if events else "M1_PENDING")
        configure_external_sources(args, test_event, required="M6" in selected_milestones)
        steps = [
            step for step in build_steps(args, test_event)
            if step.milestone in selected_milestones
        ]
        print(f"[dry-run] milestones: {', '.join(selected_milestones)}", flush=True)
        for step in steps:
            print(f"[dry-run] {step.key}: {' '.join(step.command)}", flush=True)
        return 0

    state_path = ROOT / "outputs/pipeline/state.json"
    lock_path = ROOT / "outputs/pipeline/run.lock"
    lock_fd = acquire_lock(lock_path)
    try:
        state = load_state(state_path)
        if not events:
            if "M1" not in selected_milestones:
                raise PipelineError(
                    "The audited M1 manifest is missing. Start from M1 or provide the completed M1 artifacts."
                )
            staging_event = args.test_event or "M1_PENDING"
            configure_external_sources(args, staging_event, required=False)
            preliminary = [
                step for step in build_steps(args, staging_event)
                if step.milestone in selected_milestones and step.milestone in ("M0", "M1")
            ]
            for step in preliminary:
                is_complete, reason = completed(step, state, args.force)
                if is_complete:
                    print(f"[resume] skip {step.key}: {reason}", flush=True)
                    continue
                run_step(step, state, state_path, ROOT / "outputs/pipeline/logs")
            events = manifest_events()
            if not events:
                raise PipelineError("M1 completed without any audited manifest events")

        test_event = args.test_event or events[0]
        if test_event not in events:
            raise PipelineError(
                f"Unknown test event {test_event!r}; available events include {events[:10]}"
            )
        print(f"[preflight] held-out event: {test_event}", flush=True)
        configure_external_sources(args, test_event, required="M6" in selected_milestones)
        steps = [
            step for step in build_steps(args, test_event)
            if step.milestone in selected_milestones
        ]
        for step in steps:
            is_complete, reason = completed(step, state, args.force)
            if is_complete:
                print(f"[resume] skip {step.key}: {reason}", flush=True)
                continue
            run_step(step, state, state_path, ROOT / "outputs/pipeline/logs")
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
    print(f"[pipeline] completed through {args.through}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"[pipeline:error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
