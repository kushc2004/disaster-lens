#!/usr/bin/env python3
"""Reproducible E1--E4 runner for the focused BRIGHT cross-disaster project.

This is deliberately an orchestration layer: data parsing, datasets, models,
losses, training and event metrics remain in ``src/disasterlens``.  It never
creates data or alters root M0--M2 manifests/checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import BrightDataset, collate_samples, load_manifest, normalization_from_stats
from disasterlens.data.splits import event_holdout_split, official_split, standard_tile_split
from disasterlens.eval import evaluate_by_event, write_evaluation_figures
from disasterlens.eval.calibration import (
    aggregate_prediction_files,
    classification_metrics,
    fit_temperature,
)
from disasterlens.models import EarlyFusionUNet, PseudoSiameseUNet


OUTPUT_ROOT = ROOT / "outputs" / "cross_disaster_damage_assessment"
CLASSES = ("background", "intact", "damaged", "destroyed")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _root_data() -> tuple[dict[str, Any], list[Any]]:
    data = load_yaml(ROOT / "configs" / "data" / "bright.yaml")
    dataset_root = Path(os.environ.get("DISASTERLENS_BRIGHT_ROOT", data["root"])).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Official BRIGHT data is required. Set DISASTERLENS_BRIGHT_ROOT to the "
            "directory containing pre-event, post-event, and target."
        )
    manifest = ROOT / data["manifest_path"]
    if not manifest.is_file():
        raise FileNotFoundError(
            f"M1 manifest is missing: {manifest}. Run the root official-data audit first; "
            "this focused project will not fabricate a manifest."
        )
    return data, load_manifest(manifest, dataset_root=dataset_root)


def _split_payload(split: Any) -> dict[str, list[str]]:
    return {name: [sample.tile_id for sample in getattr(split, name)] for name in ("train", "val", "test")}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_split(path: Path, split: Any, metadata: dict[str, Any]) -> None:
    """Write immutable split bytes plus separate protocol metadata.

    The JSON itself intentionally has only train/val/test keys because the
    shared loaders reject unrecognised split fields.
    """
    payload = _split_payload(split)
    if len(set().union(*map(set, payload.values()))) != sum(map(len, payload.values())):
        raise RuntimeError("Refusing to write split with tile leakage")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_json(path.with_suffix(".metadata.json"), metadata)


def _event_assignments(samples: list[Any], split_payload: dict[str, list[str]]) -> dict[str, set[str]]:
    by_id = {sample.tile_id: sample for sample in samples}
    return {name: {by_id[tile].event_id for tile in tiles} for name, tiles in split_payload.items()}


def _choose_val_event(events: list[str], heldout: str) -> str:
    candidates = [event for event in sorted(events) if event != heldout]
    if len(candidates) < 2:
        raise ValueError("At least three independent BRIGHT events are required for train/val/test event holdout")
    # Stable choice means independently launched Kaggle runs use the same split.
    return candidates[0]


def prepare(args: argparse.Namespace) -> None:
    data, samples = _root_data()
    output = Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT
    audit_root = _ensure_dir(output / "audit")
    split_root = output / "splits"
    event_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[sample.event_id].append(sample)
    for event_id, rows in sorted(grouped.items()):
        types = sorted({item.disaster_type for item in rows})
        event_rows.append({
            "event_id": event_id,
            "disaster_type": types[0] if len(types) == 1 else ";".join(types),
            "n_tiles": len(rows),
            "pre_event_optical": all(item.pre_optical is not None and item.pre_optical.is_file() for item in rows),
            "post_event_sar": all(item.post_sar is not None and item.post_sar.is_file() for item in rows),
            "damage_labels": all(item.label is not None and item.label.is_file() for item in rows),
            "crs": ";".join(sorted({str(item.crs) for item in rows if item.crs})),
            "bounds_available": all(item.bounds is not None for item in rows),
            "location_metadata": "geospatial_bounds" if all(item.bounds is not None for item in rows) else "not_available",
        })
    pd.DataFrame(event_rows).to_csv(audit_root / "events.csv", index=False)
    type_counts = Counter(row["disaster_type"] for row in event_rows)
    type_rows = []
    for disaster_type, n_events in sorted(type_counts.items()):
        remaining = len(event_rows) - n_events
        type_rows.append({
            "disaster_type": disaster_type,
            "n_events": n_events,
            "other_events": remaining,
            "leave_one_type_out_eligible": remaining >= 2,
            "reason": "test type plus distinct train and validation events are non-empty" if remaining >= 2 else "insufficient non-test events for distinct train and validation groups",
        })
    pd.DataFrame(type_rows).to_csv(audit_root / "disaster_types.csv", index=False)
    _write_json(audit_root / "audit_summary.json", {
        "source": "official BRIGHT manifest from root M1",
        "manifest": str(ROOT / data["manifest_path"]),
        "dataset_root": str(Path(os.environ.get("DISASTERLENS_BRIGHT_ROOT", data["root"])).expanduser()),
        "n_tiles": len(samples), "n_events": len(event_rows), "n_disaster_types": len(type_rows),
        "modalities": ["pre_event_optical", "post_event_sar"], "label_ids": data["label_ids"],
    })

    official_root = data.get("official_split_root")
    if official_root and Path(official_root).is_dir():
        standard = official_split(samples, split_root=Path(official_root))
        standard_protocol = "official_tile_split"
    else:
        standard = standard_tile_split(samples, seed=int(args.seed))
        standard_protocol = "deterministic_event_stratified_tile_split"
    _write_split(split_root / "standard.json", standard, {
        "protocol": standard_protocol,
        "seed": int(args.seed),
        "event_isolation": False,
        "note": "Standard split is tile-disjoint; events may occur in multiple partitions by design.",
    })

    events = sorted(grouped)
    requested = events if args.heldout_event == "all" else [args.heldout_event]
    unknown = sorted(set(requested).difference(events))
    if unknown:
        raise ValueError(f"Unknown held-out event(s): {unknown}. See {output / 'audit' / 'events.csv'}")
    for heldout in requested:
        val_event = _choose_val_event(events, heldout)
        split = event_holdout_split(samples, train_events=[event for event in events if event not in {heldout, val_event}], val_events=[val_event], test_events=[heldout])
        payload = _split_payload(split)
        assignments = _event_assignments(samples, payload)
        if any(assignments[left] & assignments[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError("Event-isolation check failed; split was not written")
        _write_split(split_root / "event_holdout" / f"{heldout}.json", split, {
            "protocol": "event_held_out", "heldout_event": heldout, "validation_event": val_event,
            "train_events": sorted(assignments["train"]), "val_events": sorted(assignments["val"]), "test_events": sorted(assignments["test"]),
            "event_isolation": True,
        })
    print(f"[prepare] official BRIGHT audit and splits saved to {output}", flush=True)


def _run_dir(output_root: Path, model: str, split_path: Path) -> Path:
    return output_root / "runs" / model / split_path.stem


def _shared_overrides(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        f"trainer.epochs={args.epochs}", f"trainer.seed={args.seed}",
        f"trainer.batch_size={args.batch_size}", f"trainer.num_workers={args.workers}",
        f"trainer.crop_size={args.crop_size}", f"trainer.device={args.device}",
        "trainer.event_balanced_sampling=false",  # E1 and E2 use identical sampling/preprocessing.
    ]


def train(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT
    split_path = Path(args.split).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Split not found: {split_path}. Run prepare first.")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    data, samples = _root_data()
    assignments = _event_assignments(samples, payload)
    if "event_holdout" in split_path.parts and any(assignments[a] & assignments[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Refusing event-held-out training with event leakage")
    run_dir = _run_dir(output, args.model, split_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_path, run_dir / "split.json")
    metadata = split_path.with_suffix(".metadata.json")
    if metadata.is_file():
        shutil.copy2(metadata, run_dir / "split.metadata.json")
    by_id = {sample.tile_id: sample for sample in samples}
    pd.DataFrame(
        {
            "partition": partition,
            "tile_id": tile_id,
            "event_id": by_id[tile_id].event_id,
            "disaster_type": by_id[tile_id].disaster_type,
        }
        for partition, tile_ids in payload.items()
        for tile_id in tile_ids
    ).to_parquet(run_dir / "split_manifest.parquet", index=False)
    resolved = {
        "project": "cross_disaster_damage_assessment", "experiment": "E1" if args.model == "unet" else "E2",
        "model": args.model, "source_data": data, "split_source": str(split_path),
        "trainer": {"epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size, "num_workers": args.workers, "crop_size": args.crop_size, "device": args.device, "event_balanced_sampling": False},
        "evaluation": "shared evaluate_by_event on validation and test partitions",
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unavailable"
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    _write_json(run_dir / "environment.json", {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()})
    if args.model == "unet":
        command = [sys.executable, "-u", "scripts/train.py", f"split_path={split_path}", f"trainer.checkpoint_dir={run_dir}", *_shared_overrides(args, run_dir)]
    else:
        command = [sys.executable, "-u", "scripts/train_m3.py", f"split_path={split_path}", f"split_name={split_path.stem}", f"run_dir={run_dir}", "resume=true", *_shared_overrides(args, run_dir)]
    print("[train] " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    # M2 calls its training record metrics.json; reserve that name for event evaluation.
    if args.model == "unet" and (run_dir / "metrics.json").is_file():
        shutil.move(run_dir / "metrics.json", run_dir / "training_metrics.json")
    if not (run_dir / "checkpoint.pt").is_file() and (run_dir / "best.pt").is_file():
        shutil.copy2(run_dir / "best.pt", run_dir / "checkpoint.pt")
    if not (run_dir / "checkpoint.pt").is_file():
        raise RuntimeError("Training completed without checkpoint.pt")
    print(f"[train] complete: {run_dir}", flush=True)


def _model(model_name: str) -> torch.nn.Module:
    if model_name == "unet":
        config = load_yaml(ROOT / "configs" / "model" / "unet_baseline.yaml")
        return EarlyFusionUNet(in_channels=int(config["in_channels"]), num_classes=int(config["num_classes"]), base_channels=int(config["base_channels"]))
    if model_name == "siamese_resnet18":
        config = load_yaml(ROOT / "configs" / "model" / "siamese_baseline.yaml")
        return PseudoSiameseUNet(num_classes=int(config["num_classes"]), base_channels=int(config["base_channels"]), encoder=str(config["encoder"]))
    raise ValueError(f"Unknown model {model_name}")


def evaluate(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    config_path, split_path, checkpoint = run_dir / "resolved_config.yaml", run_dir / "split.json", run_dir / "checkpoint.pt"
    if not all(path.is_file() for path in (config_path, split_path, checkpoint)):
        raise FileNotFoundError("run_dir must contain resolved_config.yaml, split.json, and checkpoint.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    partition = args.partition
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if partition not in split:
        raise ValueError("partition must be train, val, or test")
    data, samples = _root_data()
    by_id = {sample.tile_id: sample for sample in samples}
    selected = [by_id[tile] for tile in split[partition]]
    normalization = normalization_from_stats(data["normalization"], ROOT / data["normalization_stats_path"])
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    model = _model(config["model"])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    loader = DataLoader(
        BrightDataset(selected, normalization=normalization),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_samples,
    )
    predictions = run_dir / "predictions" / partition
    result = evaluate_by_event(
        model,
        loader,
        device,
        disaster_types={sample.event_id: sample.disaster_type for sample in selected},
        predictions_dir=predictions,
        compress_predictions=args.prediction_compression == "deflated",
    )
    result.update({"project": "cross_disaster_damage_assessment", "experiment": config["experiment"], "model": config["model"], "partition": partition, "checkpoint": "checkpoint.pt", "split": "split.json"})
    target = run_dir / "evaluation" / partition
    _write_json(target / "metrics.json", result)
    _write_json(target / "confusion_matrix.json", {
        "class_order": list(CLASSES), "matrix": result["pooled"]["confusion_matrix"]
    })
    with (target / "metrics_by_event.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("event_id", "disaster_type", "n_tiles", "n_buildings", "miou", "macro_f1", "f1_localization", "f1_damage", "ece")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in result["per_event"])
    with (target / "class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(("class_id", "class_name", "iou", "f1"))
        writer.writerows((index, name, result["pooled"]["class_iou"][index], result["pooled"]["class_f1"][index]) for index, name in enumerate(CLASSES))
    write_evaluation_figures(result, predictions, target / "figures")
    print(f"[evaluate] {partition} metrics and predictions saved to {target}", flush=True)


def calibrate(args: argparse.Namespace) -> None:
    """Fit temperature on validation bundles only; held-out test labels remain untouched until scoring."""
    run_dir = Path(args.run_dir).resolve()
    val_dir, test_dir = run_dir / "predictions" / "val", run_dir / "predictions" / "test"
    val_files, test_files = sorted(val_dir.glob("*.npz")), sorted(test_dir.glob("*.npz"))
    if not val_files or not test_files:
        raise FileNotFoundError("Evaluate both val and test before calibration so prediction provenance is explicit")
    val = aggregate_prediction_files(val_files, partition="validation")
    test = aggregate_prediction_files(test_files, partition="heldout_test")
    temperature = fit_temperature(val.logits, val.targets)
    rows = []
    reliabilities: dict[str, list[dict[str, float]]] = {}
    for name, aggregate in (("validation", val), ("heldout_test", test)):
        for label, value in (("uncalibrated", 1.0), ("temperature_scaled", temperature)):
            metrics, reliability = classification_metrics(aggregate.logits, aggregate.targets, temperature=value)
            rows.append({"partition": name, "calibration": label, **metrics})
            reliabilities[f"{name}_{label}"] = reliability
    target = run_dir / "calibration"; target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "temperature.json", {"fit_partition": "validation", "temperature": temperature, "n_validation_buildings": len(val.targets), "heldout_test_labels_used_for_fit": False})
    pd.DataFrame(rows).to_csv(target / "calibration_metrics.csv", index=False)
    _write_json(target / "reliability.json", reliabilities)
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(6.5, 6.0)); axis.plot([0, 1], [0, 1], "--", color="black", label="perfect calibration")
    for name, color in (("heldout_test_uncalibrated", "#d95f02"), ("heldout_test_temperature_scaled", "#1b9e77")):
        rows_ = reliabilities[name]; axis.plot([r["confidence"] for r in rows_], [r["accuracy"] for r in rows_], marker="o", label=name.replace("heldout_test_", ""), color=color)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy", title="Held-out reliability diagram"); axis.legend(); axis.grid(alpha=.25); figure.tight_layout(); figure.savefig(target / "reliability_diagram.png", dpi=180); plt.close(figure)
    print(f"[calibrate] validation-only temperature fit saved to {target}", flush=True)


def report(args: argparse.Namespace) -> None:
    output = Path(args.output_root).resolve() if args.output_root else OUTPUT_ROOT
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "runs").glob("*/*/evaluation/test/metrics.json")):
        metric = json.loads(path.read_text(encoding="utf-8")); run_dir = path.parents[2]
        split_meta_path = run_dir / "split.metadata.json"; split_meta = json.loads(split_meta_path.read_text()) if split_meta_path.is_file() else {}
        for event in metric["per_event"]:
            rows.append({"model": metric["model"], "run": run_dir.name, "protocol": split_meta.get("protocol", "unknown"), "heldout_event": split_meta.get("heldout_event"), **event})
    reports = output / "reports"; reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reports / "per_event_results.csv", index=False)
    standard = {
        (row["model"], row["event_id"]): row
        for row in rows if row["protocol"] in {"official_tile_split", "deterministic_event_stratified_tile_split"}
    }
    comparisons = []
    for row in rows:
        if row["protocol"] != "event_held_out":
            continue
        reference = standard.get((row["model"], row["event_id"]))
        comparisons.append({
            "model": row["model"], "heldout_event": row["heldout_event"], "disaster_type": row["disaster_type"],
            "heldout_macro_f1": row["macro_f1"], "heldout_miou": row["miou"],
            "standard_macro_f1": reference["macro_f1"] if reference else None,
            "standard_miou": reference["miou"] if reference else None,
            "macro_f1_change_vs_standard": float(row["macro_f1"]) - float(reference["macro_f1"]) if reference else None,
            "miou_change_vs_standard": float(row["miou"]) - float(reference["miou"]) if reference else None,
        })
    pd.DataFrame(comparisons).to_csv(reports / "event_holdout_vs_standard.csv", index=False)
    lines = ["# Cross-disaster results", "", "Only completed saved test artifacts are listed. No conclusion is implied by missing rows.", "", "| Model | Protocol | Held-out event | Event | Disaster type | Macro F1 | mIoU |", "|---|---|---|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['protocol']} | {row['heldout_event'] or '-'} | {row['event_id']} | {row['disaster_type']} | {float(row['macro_f1']):.6f} | {float(row['miou']):.6f} |")
    lines.extend(("", "## Event-held-out change versus standard", "", "See `event_holdout_vs_standard.csv` for matched event-level deltas. A negative value means the held-out-event score was lower than the standard split score."))
    (reports / "cross_disaster_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] saved {reports / 'cross_disaster_report.md'}", flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="audit existing official manifest and write benchmark splits")
    prep.add_argument("--heldout-event", default="all", help="BRIGHT event ID or 'all' (default)"); prep.add_argument("--seed", type=int, default=42); prep.add_argument("--output-root"); prep.set_defaults(func=prepare)
    train_p = commands.add_parser("train", help="run E1 or E2 on one immutable split")
    train_p.add_argument("--model", choices=("unet", "siamese_resnet18"), required=True); train_p.add_argument("--split", required=True); train_p.add_argument("--output-root"); train_p.add_argument("--epochs", type=int, default=30); train_p.add_argument("--seed", type=int, default=42); train_p.add_argument("--batch-size", type=int, default=16); train_p.add_argument("--workers", type=int, default=4); train_p.add_argument("--crop-size", type=int, default=512); train_p.add_argument("--device", default="auto"); train_p.set_defaults(func=train)
    evaluate_p = commands.add_parser("evaluate", help="save event and class metrics for a trained run")
    evaluate_p.add_argument("--run-dir", required=True); evaluate_p.add_argument("--partition", choices=("val", "test"), required=True); evaluate_p.add_argument("--batch-size", type=int, default=16); evaluate_p.add_argument("--workers", type=int, default=4); evaluate_p.add_argument("--device", default="auto"); evaluate_p.add_argument("--prediction-compression", choices=("deflated", "stored"), default="deflated", help="Use 'stored' for lossless uncompressed bundles when GPU time matters."); evaluate_p.set_defaults(func=evaluate)
    cal = commands.add_parser("calibrate", help="fit validation-only temperature and score heldout predictions"); cal.add_argument("--run-dir", required=True); cal.set_defaults(func=calibrate)
    rep = commands.add_parser("report", help="compile completed saved test metrics only"); rep.add_argument("--output-root"); rep.set_defaults(func=report)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
