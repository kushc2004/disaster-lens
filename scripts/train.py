#!/usr/bin/env python3
"""Train M2 only on a saved, real-data BRIGHT split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import BrightDataset, SynchronizedGeometry, class_weights_from_training_samples, collate_samples, load_manifest, normalization_from_stats, select_tiny_overfit_samples
from disasterlens.models import EarlyFusionUNet, SegmentationLoss
from disasterlens.train import Trainer, set_seed


def _value(overrides: list[str], key: str, default: str | None = None) -> str | None:
    prefix = f"{key}="
    return next((item.removeprefix(prefix) for item in overrides if item.startswith(prefix)), default)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def _config_overrides(overrides: list[str], section: str) -> list[str]:
    """Apply unscoped and section-scoped overrides to one configuration file.

    The notebook historically used ``trainer.epochs=100``.  Passing that
    unchanged to a top-level trainer YAML silently created an unused ``trainer``
    mapping.  Normalising it here makes both ``epochs=100`` and
    ``trainer.epochs=100`` affect the effective trainer configuration.
    """
    selected: list[str] = []
    prefix = f"{section}."
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key.startswith(prefix):
            selected.append(f"{key.removeprefix(prefix)}={value}")
        elif "." not in key:
            selected.append(override)
    return selected


def main() -> None:
    overrides = sys.argv[1:]
    print("[training] loading configuration, official manifest, normalization, and saved split", flush=True)
    data = load_yaml(ROOT / "configs/data/bright.yaml", _config_overrides(overrides, "data"))
    model_config_path = _value(overrides, "model_config_path", "configs/model/unet_baseline.yaml")
    model_config_file = Path(model_config_path)
    if not model_config_file.is_absolute():
        model_config_file = ROOT / model_config_file
    if not model_config_file.is_file():
        raise FileNotFoundError(f"Model configuration not found: {model_config_file}")
    model_config = load_yaml(model_config_file, _config_overrides(overrides, "model"))
    trainer = load_yaml(ROOT / "configs/trainer/unet_baseline.yaml", _config_overrides(overrides, "trainer"))
    split_path = _value(overrides, "split_path")
    if not split_path:
        raise ValueError("A saved real split is required: split_path=data/manifests/splits/event_holdout.json")
    samples = load_manifest(ROOT / data["manifest_path"], dataset_root=Path(data["root"]))
    normalization = normalization_from_stats(data["normalization"], ROOT / data["normalization_stats_path"])
    by_id = {sample.tile_id: sample for sample in samples}
    split = json.loads((ROOT / split_path).read_text(encoding="utf-8"))
    train_samples = [by_id[tile] for tile in split["train"]]
    val_samples = [by_id[tile] for tile in split["val"]]
    overfit_tiles = int(_value(overrides, "overfit_tiles", "0") or 0)
    tiny_overfit = overfit_tiles > 0
    tiny_selection: dict[str, Any] | None = None
    if overfit_tiles:
        if overfit_tiles < 1:
            raise ValueError("overfit_tiles must be positive")
        selection_cache = ROOT / "outputs/cache" / f"tiny_overfit_selection_{overfit_tiles}_crop{int(trainer['crop_size'])}.json"
        train_samples, tiny_selection = select_tiny_overfit_samples(
            train_samples,
            count=overfit_tiles,
            crop_size=int(trainer["crop_size"]),
            cache_path=selection_cache,
        )
        val_samples = train_samples
    if not train_samples or not val_samples:
        raise ValueError("Training and validation partitions must both be non-empty")
    print(
        f"[training] split loaded: {len(train_samples):,} train tiles, {len(val_samples):,} validation tiles; building data loaders",
        flush=True,
    )
    set_seed(int(trainer["seed"]))
    crop_size = int(trainer["crop_size"])
    transform = SynchronizedGeometry(seed=int(trainer["seed"]), crop_size=crop_size, randomize=not tiny_overfit)
    train_data = BrightDataset(train_samples, transform=transform, normalization=normalization)
    # A tiny-overfit gate must validate the *same exact real crops* it trains
    # on.  Full training retains no random validation augmentation.
    val_transform = SynchronizedGeometry(seed=int(trainer["seed"]), crop_size=crop_size, randomize=False) if tiny_overfit else None
    val_data = BrightDataset(val_samples, transform=val_transform, normalization=normalization)
    batch_size, workers = int(trainer["batch_size"]), int(trainer["num_workers"])
    loader_options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "collate_fn": collate_samples,
    }
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_options)
    device = _device(str(trainer["device"]))
    use_class_weights = bool(trainer["use_class_weights"])
    checkpoint_dir = ROOT / trainer["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # A remote orchestrator may supply a durable train-split cache shared by
    # otherwise independent run directories.  The payload is still keyed by
    # the exact train IDs below, so it cannot leak validation/test labels or be
    # reused for a different split accidentally.
    supplied_weights_path = os.environ.get("DISASTERLENS_CLASS_WEIGHTS_PATH") or _value(overrides, "class_weights_path")
    weights_path = Path(supplied_weights_path) if supplied_weights_path else checkpoint_dir / "class_weights.json"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights = None
    if use_class_weights:
        tile_ids = [sample.tile_id for sample in train_samples]
        cached_weights = None
        if weights_path.exists():
            candidate = json.loads(weights_path.read_text(encoding="utf-8"))
            if candidate.get("train_tile_ids") == tile_ids:
                cached_weights = candidate.get("weights")
        if cached_weights is None:
            print(f"[training] computing class weights from {len(train_samples):,} official training masks", flush=True)
            computed = class_weights_from_training_samples(
                train_samples,
                progress=lambda completed, total: print(f"[training] class-weight masks {completed:,}/{total:,}", flush=True),
            )
            weights_path.write_text(json.dumps({"train_tile_ids": tile_ids, "weights": computed.tolist()}, indent=2) + "\n", encoding="utf-8")
            cached_weights = computed.tolist()
            print(f"[training] saved reusable class weights to {weights_path}", flush=True)
        else:
            print(f"[training] reusing cached class weights from {weights_path}", flush=True)
        weights = torch.tensor(cached_weights, dtype=torch.float32, device=device)
    model = EarlyFusionUNet(in_channels=int(model_config["in_channels"]), num_classes=int(model_config["num_classes"]), base_channels=int(model_config["base_channels"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trainer["learning_rate"]), weight_decay=float(trainer["weight_decay"]))
    epochs, warmup_epochs = int(trainer["epochs"]), int(trainer["warmup_epochs"])
    print(
        f"[training] real BRIGHT run on {device}: {len(train_samples):,} train tiles, "
        f"{len(val_samples):,} validation tiles, {epochs} epochs",
        flush=True,
    )
    print(
        "[training] loss: "
        + (f"Focal(gamma={float(model_config.get('focal_gamma', 0.0)):g})" if float(model_config.get("focal_gamma", 0.0)) else "CrossEntropy")
        + " + "
        + f"{float(model_config['lovasz_weight']):g} * Lovasz-Softmax; "
        f"class weights={'enabled' if use_class_weights else 'disabled'}",
        flush=True,
    )
    if warmup_epochs:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup_epochs, end_factor=1.0, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs)),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    run_config: dict[str, Any] = {
        "data": data,
        "model": model_config,
        "trainer": trainer,
        "split_path": split_path,
        "overfit_tiles": overfit_tiles,
        "tiny_overfit": tiny_overfit,
        "tiny_overfit_selection": tiny_selection,
        "device": str(device),
    }
    (checkpoint_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"[training] checkpoints will be written to {checkpoint_dir}", flush=True)
    history = Trainer(model, optimizer, SegmentationLoss(class_weights=weights, lovasz_weight=float(model_config["lovasz_weight"]), focal_gamma=float(model_config.get("focal_gamma", 0.0))), device, checkpoint_dir, amp=bool(trainer["amp"])).fit(train_loader, val_loader, epochs=epochs, scheduler=scheduler)
    best_record = max(history, key=lambda record: float(record["val_f1_damage"]))
    (checkpoint_dir / "metrics.json").write_text(json.dumps({"best": best_record, "history": history}, indent=2) + "\n", encoding="utf-8")
    if tiny_overfit:
        minimum = float(_value(overrides, "overfit_min_damage_f1", str(trainer["tiny_overfit_min_damage_f1"])) or trainer["tiny_overfit_min_damage_f1"])
        best_f1 = float(best_record["val_f1_damage"])
        result = {"tiles": overfit_tiles, "best_epoch": best_record["epoch"], "best_damage_macro_f1": best_f1, "required_damage_macro_f1": minimum}
        (checkpoint_dir / "tiny_overfit_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if best_f1 < minimum:
            raise RuntimeError(
                f"Tiny-set overfit failed: best validation damage macro-F1 {best_f1:.4f} is below {minimum:.4f}. "
                "Do not start full training; debug the baseline first."
            )
        print(f"[training] tiny-set overfit passed: damage macro-F1={best_f1:.4f} (threshold={minimum:.4f})", flush=True)


if __name__ == "__main__":
    main()
