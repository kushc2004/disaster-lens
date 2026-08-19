#!/usr/bin/env python3
"""Evaluate an M2 checkpoint on a named partition from a saved BRIGHT split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import BrightDataset, collate_samples, load_manifest, normalization_from_stats
from disasterlens.models import EarlyFusionUNet, SegmentationLoss
from disasterlens.train import evaluate_epoch


def _value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    value = next((item.removeprefix(prefix) for item in overrides if item.startswith(prefix)), None)
    if not value:
        raise ValueError(f"Missing required argument: {key}=...")
    return value


def main() -> None:
    overrides = sys.argv[1:]
    checkpoint_path = ROOT / _value(overrides, "checkpoint")
    split_path = ROOT / _value(overrides, "split_path")
    partition = _value(overrides, "partition")
    if partition not in {"train", "val", "test"}:
        raise ValueError("partition must be train, val, or test")
    data, model_config = load_yaml(ROOT / "configs/data/bright.yaml", overrides), load_yaml(ROOT / "configs/model/unet_baseline.yaml", overrides)
    samples = load_manifest(ROOT / data["manifest_path"], dataset_root=Path(data["root"]))
    by_id = {sample.tile_id: sample for sample in samples}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    normalization = normalization_from_stats(data["normalization"], ROOT / data["normalization_stats_path"])
    dataset = BrightDataset([by_id[tile] for tile in split[partition]], normalization=normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[evaluation] loading {checkpoint_path} on {device}; {len(dataset):,} real {partition} tiles", flush=True)
    model = EarlyFusionUNet(in_channels=int(model_config["in_channels"]), num_classes=int(model_config["num_classes"]), base_channels=int(model_config["base_channels"])).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state"])
    metrics = evaluate_epoch(model, DataLoader(dataset, batch_size=1, collate_fn=collate_samples), SegmentationLoss(), device)
    output = ROOT / "outputs/metrics" / f"{checkpoint_path.stem}_{partition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[evaluation] metrics saved to {output}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
