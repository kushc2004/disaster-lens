"""Shared, real-data M3/M4 training and evaluation entry points."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import (
    BrightDataset,
    EventBalancedSampler,
    SynchronizedGeometry,
    class_weights_from_training_samples,
    collate_samples,
    load_manifest,
    normalization_from_stats,
)
from disasterlens.eval import evaluate_by_event, write_evaluation_figures
from disasterlens.models import (
    DamageFusionFormer,
    DualHeadSegmentationLoss,
    PseudoSiameseUNet,
    SegmentationLoss,
)
from disasterlens.train import Trainer, set_seed


def value(overrides: list[str], key: str, default: str | None = None) -> str | None:
    prefix = f"{key}="
    return next((item[len(prefix) :] for item in overrides if item.startswith(prefix)), default)


def section_overrides(overrides: list[str], section: str) -> list[str]:
    selected: list[str] = []
    prefix = f"{section}."
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        if key.startswith(prefix):
            selected.append(f"{key[len(prefix):]}={raw}")
        elif "." not in key:
            selected.append(item)
    return selected


def device_from(value_: str) -> torch.device:
    if value_ != "auto":
        return torch.device(value_)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_and_loss(
    kind: str,
    model_config: dict[str, Any],
    class_weights: torch.Tensor | None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if kind == "m3":
        model = PseudoSiameseUNet(
            num_classes=int(model_config["num_classes"]),
            base_channels=int(model_config["base_channels"]),
            encoder=str(model_config["encoder"]),
        )
        return model, SegmentationLoss(
            class_weights=class_weights,
            lovasz_weight=float(model_config["lovasz_weight"]),
        )
    if kind != "m4":
        raise ValueError(f"Unknown experiment kind: {kind}")
    model = DamageFusionFormer(
        base_channels=int(model_config["encoder"]["base_channels"]),
        heads=int(model_config["fusion"]["heads"]),
        dropout=float(model_config["fusion"]["dropout"]),
        decoder_channels=int(model_config["decoder"]["channels"]),
        ablation=str(model_config["fusion"]["mode"]),
    )
    loss = model_config["loss"]
    damage_weights = class_weights[1:] if class_weights is not None else None
    if damage_weights is not None:
        damage_weights = damage_weights / damage_weights.mean()
    return model, DualHeadSegmentationLoss(
        damage_class_weights=damage_weights,
        lovasz_weight=float(loss["lovasz_weight"]),
        lambda_localization=float(loss["lambda_localization"]),
        lambda_damage=float(loss["lambda_damage"]),
    )


def _load_inputs(overrides: list[str]) -> tuple[dict[str, Any], list[Any], dict[str, list[str]], str]:
    data = load_yaml(ROOT / "configs/data/bright.yaml", section_overrides(overrides, "data"))
    split_path_raw = value(overrides, "split_path")
    if not split_path_raw:
        raise ValueError("split_path=<saved real BRIGHT split JSON> is required")
    split_path = Path(split_path_raw)
    if not split_path.is_absolute():
        split_path = ROOT / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"Saved BRIGHT split does not exist: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set(split) != {"train", "val", "test"} or any(not split[name] for name in split):
        raise ValueError("Split must contain non-empty train, val, and test arrays")
    if (set(split["train"]) & set(split["val"])) or (set(split["train"]) & set(split["test"])) or (set(split["val"]) & set(split["test"])):
        raise ValueError("Tile leakage detected in saved split")
    samples = load_manifest(ROOT / data["manifest_path"], dataset_root=Path(data["root"]))
    by_id = {sample.tile_id: sample for sample in samples}
    unknown = set().union(*map(set, split.values())).difference(by_id)
    if unknown:
        raise ValueError(f"Split references tiles absent from official manifest: {sorted(unknown)[:5]}")
    if split_path.stem == "event_holdout":
        partition_events = {
            name: {by_id[tile_id].event_id for tile_id in tile_ids}
            for name, tile_ids in split.items()
        }
        if any(
            partition_events[left] & partition_events[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError("Event leakage detected in saved event-held-out split")
    return data, samples, split, split_path_raw


def _run_dir(kind: str, split_name: str, model_config: dict[str, Any], overrides: list[str]) -> Path:
    explicit = value(overrides, "run_dir")
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else ROOT / path
    suffix = ""
    if kind == "m4":
        suffix = f"_{model_config['fusion']['mode']}"
    return ROOT / "outputs" / "runs" / kind / f"{split_name}{suffix}"


def _write_provenance(run_dir: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    (run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (run_dir / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    import pandas as pd

    pd.DataFrame(rows).to_parquet(run_dir / "split_manifest.parquet", index=False)


def _training_signature(config: dict[str, Any], split_file: Path) -> dict[str, Any]:
    """Bind resumable checkpoints to the complete experiment and exact split bytes."""
    canonical = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return {
        "schema_version": 1,
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
    }


def _validate_resume_checkpoint(state: dict[str, Any], expected: dict[str, Any], path: Path) -> None:
    actual = state.get("checkpoint_metadata", {}).get("training_signature")
    if actual != expected:
        raise RuntimeError(
            f"Refusing to resume incompatible checkpoint {path}. The model/config/split "
            "signature differs or predates signed checkpoints. Use resume=false to restart "
            "this run directory, or choose a new run_dir."
        )


def train(kind: str) -> None:
    overrides = sys.argv[1:]
    data, samples, split, split_path = _load_inputs(overrides)
    model_file = "siamese_baseline.yaml" if kind == "m3" else "damage_fusion_former.yaml"
    model_config = load_yaml(ROOT / "configs/model" / model_file, section_overrides(overrides, "model"))
    trainer = load_yaml(ROOT / "configs/trainer/multimodal.yaml", section_overrides(overrides, "trainer"))
    split_name = str(value(overrides, "split_name", Path(split_path).stem))
    run_dir = _run_dir(kind, split_name, model_config, overrides)
    by_id = {sample.tile_id: sample for sample in samples}
    unknown = set().union(*map(set, split.values())).difference(by_id)
    if unknown:
        raise ValueError(f"Split references tiles absent from official manifest: {sorted(unknown)[:5]}")
    partitions = {name: [by_id[tile] for tile in split[name]] for name in split}
    normalization = normalization_from_stats(
        data["normalization"], ROOT / data["normalization_stats_path"]
    )
    seed = int(trainer["seed"])
    set_seed(seed)
    crop_size = int(trainer["crop_size"])
    train_data = BrightDataset(
        partitions["train"],
        transform=SynchronizedGeometry(seed=seed, crop_size=crop_size, randomize=True),
        normalization=normalization,
    )
    val_data = BrightDataset(partitions["val"], normalization=normalization)
    sampler = None
    if bool(trainer["event_balanced_sampling"]):
        sampler = EventBalancedSampler(
            partitions["train"],
            samples_per_epoch=trainer.get("samples_per_epoch") or len(partitions["train"]),
            seed=seed,
        )
    loader_arguments = {
        "batch_size": int(trainer["batch_size"]),
        "num_workers": int(trainer["num_workers"]),
        "pin_memory": True,
        "collate_fn": collate_samples,
    }
    train_loader = DataLoader(
        train_data, sampler=sampler, shuffle=sampler is None, **loader_arguments
    )
    val_loader = DataLoader(val_data, shuffle=False, **loader_arguments)
    device = device_from(str(trainer["device"]))
    weights = None
    if bool(trainer["use_class_weights"]):
        weights = torch.from_numpy(class_weights_from_training_samples(partitions["train"])).to(device)
    model, criterion = model_and_loss(kind, model_config, weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    epochs, warmup = int(trainer["epochs"]), int(trainer["warmup_epochs"])
    if warmup:
        scheduler: Any = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1 / warmup, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, epochs - warmup)
                ),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
    split_file = Path(split_path)
    if not split_file.is_absolute():
        split_file = ROOT / split_file
    effective = {
        "kind": kind,
        "data": data,
        "model": model_config,
        "trainer": trainer,
        "split_path": split_path,
        "split_name": split_name,
        "run_dir": str(run_dir),
    }
    signature = _training_signature(effective, split_file)
    rows = []
    for partition, tile_ids in split.items():
        rows.extend(
            {
                "partition": partition,
                "tile_id": tile_id,
                "event_id": by_id[tile_id].event_id,
                "disaster_type": by_id[tile_id].disaster_type,
            }
            for tile_id in tile_ids
        )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume = str(value(overrides, "resume", "true")).lower() not in {"false", "0", "no"}
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.json"
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        _validate_resume_checkpoint(state, signature, last_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        start_epoch = int(state["epoch"]) + 1
        print(f"[training] resuming {kind.upper()} at epoch {start_epoch}/{epochs}", flush=True)
    _write_provenance(run_dir, effective, rows)
    if start_epoch <= epochs:
        print(
            f"[training] {kind.upper()} on {device}: {len(partitions['train']):,} train, "
            f"{len(partitions['val']):,} val, event-balanced={sampler is not None}",
            flush=True,
        )
        history = Trainer(
            model,
            optimizer,
            criterion,
            device,
            run_dir,
            amp=bool(trainer["amp"]),
            checkpoint_metadata={"training_signature": signature},
        ).fit(
            train_loader,
            val_loader,
            epochs=epochs,
            scheduler=scheduler,
            start_epoch=start_epoch,
            history=history,
        )
    if not history:
        raise RuntimeError("Training produced no epoch history")
    best = max(history, key=lambda row: float(row["val_f1_damage"]))
    (run_dir / "training_metrics.json").write_text(
        json.dumps({"best": best, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(run_dir / "best.pt", run_dir / "checkpoint.pt")
    print(f"[training] complete; source-of-truth artifacts: {run_dir}", flush=True)


def _write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_id", "disaster_type", "n_tiles", "n_buildings", "miou",
        "macro_f1", "f1_localization", "f1_damage", "ece",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def _write_class_csv(path: Path, metrics: dict[str, Any]) -> None:
    names = ("background", "intact", "damaged", "destroyed")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "class_name", "iou", "f1"))
        for index, name in enumerate(names):
            writer.writerow((index, name, metrics["class_iou"][index], metrics["class_f1"][index]))


def _ci(pooled: dict[str, Any], metric: str) -> str:
    interval = pooled.get("bootstrap_95_ci", {}).get("metrics", {}).get(metric)
    if not interval:
        return "not available"
    return f"[{float(interval['lower']):.6f}, {float(interval['upper']):.6f}]"


def _cross_event_report(_kind: str) -> None:
    records: list[tuple[str, str, dict[str, Any], Path]] = []
    for kind in ("m3", "m4"):
        for path in sorted((ROOT / "outputs" / "runs" / kind).glob("*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append((kind, path.parent.name, payload, path))
    report_path = ROOT / "outputs" / "reports" / "cross_event_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-event generalization report",
        "",
        "All values below are read from saved evaluation artifacts; no values are imputed. "
        "The 95% intervals resample complete events, never pixels or tiles.",
        "",
        "| Model/run | Split | Macro F1 (95% CI) | mIoU (95% CI) | Damage F1 (95% CI) | ECE | Generalization gap (macro F1) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    complete: dict[str, dict[str, bool]] = {}
    for kind, name, payload, _ in records:
        pooled = payload["pooled"]
        is_heldout = "holdout" in name or "heldout" in name
        complete.setdefault(kind, {"standard": False, "heldout": False})[
            "heldout" if is_heldout else "standard"
        ] = True
        standard = next(
            (
                row
                for row in records
                if row[0] == kind and row[1].startswith("standard")
                and row[2]["model"]["fusion"].get("mode")
                == payload["model"]["fusion"].get("mode")
            ),
            None,
        ) if kind == "m4" else next(
            (row for row in records if row[0] == kind and row[1].startswith("standard")),
            None,
        )
        gap = "TBD"
        if standard is not None and is_heldout:
            gap = f"{float(standard[2]['pooled']['macro_f1']) - float(pooled['macro_f1']):.6f}"
        split_type = "event-held-out" if is_heldout else "standard"
        lines.append(
            f"| {kind.upper()} / {name} | {split_type} | "
            f"{pooled['macro_f1']:.6f} {_ci(pooled, 'macro_f1')} | "
            f"{pooled['miou']:.6f} {_ci(pooled, 'miou')} | "
            f"{pooled['f1_damage']:.6f} {_ci(pooled, 'f1_damage')} | "
            f"{pooled['ece']:.6f} | {gap} |"
        )
    incomplete = [
        kind.upper()
        for kind, status in complete.items()
        if not status["standard"] or not status["heldout"]
    ]
    if incomplete or not records:
        missing = ", ".join(incomplete) if incomplete else "M3 and M4"
        lines.extend(("", f"> Report is incomplete for {missing} until real standard and held-out evaluations exist."))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ablation_table() -> None:
    base = ROOT / "outputs" / "runs" / "m4"
    rows = []
    for path in sorted(base.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pooled = payload["pooled"]
        rows.append(
            {
                "run": path.parent.name,
                "split": payload["split_name"],
                "fusion": payload["model"]["fusion"]["mode"],
                "modalities": (
                    "post-event SAR only"
                    if payload["model"]["fusion"]["mode"] == "sar_only"
                    else "pre-event optical + post-event SAR"
                ),
                "cross_attention": payload["model"]["fusion"]["mode"] == "full",
                "macro_f1": pooled["macro_f1"],
                "macro_f1_ci_lower": pooled.get("bootstrap_95_ci", {}).get("metrics", {}).get("macro_f1", {}).get("lower"),
                "macro_f1_ci_upper": pooled.get("bootstrap_95_ci", {}).get("metrics", {}).get("macro_f1", {}).get("upper"),
                "miou": pooled["miou"],
                "f1_damage": pooled["f1_damage"],
                "ece": pooled["ece"],
            }
        )
    target = ROOT / "outputs" / "reports" / "ablation_table.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run", "split", "fusion", "modalities", "cross_attention",
                "macro_f1", "macro_f1_ci_lower", "macro_f1_ci_upper",
                "miou", "f1_damage", "ece",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(kind: str) -> None:
    overrides = sys.argv[1:]
    data, samples, split, split_path = _load_inputs(overrides)
    model_file = "siamese_baseline.yaml" if kind == "m3" else "damage_fusion_former.yaml"
    model_config = load_yaml(ROOT / "configs/model" / model_file, section_overrides(overrides, "model"))
    split_name = str(value(overrides, "split_name", Path(split_path).stem))
    run_dir = _run_dir(kind, split_name, model_config, overrides)
    checkpoint_raw = value(overrides, "checkpoint", str(run_dir / "checkpoint.pt"))
    checkpoint = Path(str(checkpoint_raw))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {checkpoint}")
    partition = str(value(overrides, "partition", "test"))
    if partition not in split:
        raise ValueError(f"Unknown split partition: {partition}")
    by_id = {sample.tile_id: sample for sample in samples}
    selected = [by_id[tile] for tile in split[partition]]
    normalization = normalization_from_stats(
        data["normalization"], ROOT / data["normalization_stats_path"]
    )
    trainer = load_yaml(ROOT / "configs/trainer/multimodal.yaml", section_overrides(overrides, "trainer"))
    loader = DataLoader(
        BrightDataset(selected, normalization=normalization),
        batch_size=int(trainer["batch_size"]),
        shuffle=False,
        num_workers=int(trainer["num_workers"]),
        pin_memory=True,
        collate_fn=collate_samples,
    )
    device = device_from(str(trainer["device"]))
    model, _ = model_and_loss(kind, model_config, None)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    save_predictions = str(value(overrides, "save_predictions", "true")).lower() not in {"false", "0", "no"}
    predictions = run_dir / "predictions" / partition if save_predictions else None
    result = evaluate_by_event(
        model,
        loader,
        device,
        disaster_types={sample.event_id: sample.disaster_type for sample in selected},
        predictions_dir=predictions,
    )
    result.update(
        {
            "kind": kind,
            "model": model_config,
            "split_path": split_path,
            "split_name": split_name,
            "partition": partition,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_event_csv(run_dir / "metrics_by_event.csv", result["per_event"])
    _write_class_csv(run_dir / "class_metrics.csv", result["pooled"])
    if predictions is not None:
        write_evaluation_figures(result, predictions, run_dir / "figures" / partition)
    _cross_event_report(kind)
    if kind == "m4":
        _ablation_table()
    print(f"[evaluation] complete; artifacts: {run_dir}", flush=True)
