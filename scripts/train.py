#!/usr/bin/env python3
"""Train M2 only on a saved, real-data BRIGHT split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import BrightDataset, SynchronizedGeometry, class_weights_from_training_samples, collate_samples, load_manifest, normalization_from_stats
from disasterlens.models import EarlyFusionUNet, SegmentationLoss
from disasterlens.train import Trainer, set_seed


def _value(overrides: list[str], key: str, default: str | None = None) -> str | None:
    prefix = f"{key}="
    return next((item.removeprefix(prefix) for item in overrides if item.startswith(prefix)), default)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    overrides = sys.argv[1:]
    data = load_yaml(ROOT / "configs/data/bright.yaml", overrides)
    model_config = load_yaml(ROOT / "configs/model/unet_baseline.yaml", overrides)
    trainer = load_yaml(ROOT / "configs/trainer/unet_baseline.yaml", overrides)
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
    if overfit_tiles:
        if overfit_tiles < 1:
            raise ValueError("overfit_tiles must be positive")
        train_samples = train_samples[:overfit_tiles]
        val_samples = train_samples
    if not train_samples or not val_samples:
        raise ValueError("Training and validation partitions must both be non-empty")
    set_seed(int(trainer["seed"]))
    crop_size = int(trainer["crop_size"])
    transform = SynchronizedGeometry(seed=int(trainer["seed"]), crop_size=crop_size)
    train_data = BrightDataset(train_samples, transform=transform, normalization=normalization)
    # Validation keeps full tiles; no random crop or geometry augmentation.
    val_data = BrightDataset(val_samples, normalization=normalization)
    batch_size, workers = int(trainer["batch_size"]), int(trainer["num_workers"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_samples)
    device = _device(str(trainer["device"]))
    weights = torch.from_numpy(class_weights_from_training_samples(train_samples)).to(device)
    model = EarlyFusionUNet(in_channels=int(model_config["in_channels"]), num_classes=int(model_config["num_classes"]), base_channels=int(model_config["base_channels"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(trainer["learning_rate"]), weight_decay=float(trainer["weight_decay"]))
    epochs, warmup_epochs = int(trainer["epochs"]), int(trainer["warmup_epochs"])
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
    checkpoint_dir = ROOT / trainer["checkpoint_dir"]
    run_config = {"data": data, "model": model_config, "trainer": trainer, "split_path": split_path, "overfit_tiles": overfit_tiles, "device": str(device)}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    Trainer(model, optimizer, SegmentationLoss(class_weights=weights, lovasz_weight=float(model_config["lovasz_weight"])), device, checkpoint_dir, amp=bool(trainer["amp"])).fit(train_loader, val_loader, epochs=epochs, scheduler=scheduler)


if __name__ == "__main__":
    main()
