from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import DataLoader

from disasterlens.data.augmentations import SynchronizedGeometry
from disasterlens.data.bright import BrightDataset, collate_samples
from disasterlens.data.manifest import build_bright_manifest
from disasterlens.data.schemas import BRIGHT_V1, LabelValidationError


def test_loader_batch_shape_and_raw_files_unchanged(bright_root):
    samples = build_bright_manifest(bright_root)
    before = {path: path.read_bytes() for path in bright_root.rglob("*.tif")}
    dataset = BrightDataset(samples, transform=SynchronizedGeometry(seed=7))
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_samples)))
    assert batch["images"]["pre_optical"].shape == (2, 3, 12, 10)
    assert batch["images"]["post_sar"].shape == (2, 1, 12, 10)
    assert batch["mask"].shape == (2, 12, 10)
    assert before == {path: path.read_bytes() for path in bright_root.rglob("*.tif")}


def test_unknown_labels_fail_loudly(bright_root):
    samples = build_bright_manifest(bright_root)
    mask = np.full((1, 12, 10), 99, dtype=np.uint8)
    from conftest import write_tif
    write_tif(samples[0].label, mask)
    with pytest.raises(LabelValidationError, match="unknown label IDs"):
        BrightDataset(samples)[0]
