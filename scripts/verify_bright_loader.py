#!/usr/bin/env python3
"""Verify one batch from the official BRIGHT manifest without creating data."""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data import BrightDataset, collate_samples, load_manifest, normalization_from_stats


def main() -> None:
    overrides = sys.argv[1:]
    config = load_yaml(ROOT / "configs/data/bright.yaml", overrides)
    samples = load_manifest(ROOT / config["manifest_path"], dataset_root=Path(config["root"]))
    if not samples:
        raise RuntimeError("The official BRIGHT manifest is empty.")
    loader = DataLoader(
        BrightDataset(samples[:2], normalization=normalization_from_stats(config["normalization"], ROOT / config["normalization_stats_path"])),
        batch_size=min(2, len(samples)), shuffle=False, num_workers=0, collate_fn=collate_samples,
    )
    batch = next(iter(loader))
    pre, post, target = batch["images"]["pre_optical"], batch["images"]["post_sar"], batch["mask"]
    if pre.ndim != 4 or pre.shape[1] != 3 or post.ndim != 4 or post.shape[1] != 1 or target.ndim != 3:
        raise RuntimeError(f"Unexpected real-data batch shapes: pre={tuple(pre.shape)}, post={tuple(post.shape)}, target={tuple(target.shape)}")
    print(f"Verified official BRIGHT DataLoader batch: pre={tuple(pre.shape)}, post={tuple(post.shape)}, target={tuple(target.shape)}")


if __name__ == "__main__":
    main()
