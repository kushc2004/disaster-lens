from __future__ import annotations

import numpy as np
import pytest

from disasterlens.data.augmentations import SynchronizedGeometry
from disasterlens.data.manifest import build_bright_manifest
from disasterlens.data.splits import Split, event_holdout_split


def test_event_holdout_has_no_event_or_tile_leakage(bright_root):
    samples = build_bright_manifest(bright_root)
    split = event_holdout_split(samples, train_events=["bata-explosion"], val_events=["surat-flood"], test_events=["maui-wildfire"])
    split.validate()
    with pytest.raises(ValueError, match="Event leakage"):
        Split(split.train, split.train, split.test).validate()


def test_synchronized_transform_preserves_marker_alignment():
    marker = np.zeros((1, 6, 7), dtype=np.float32)
    marker[0, 1, 5] = 1
    images, mask = SynchronizedGeometry(seed=11)({"pre_optical": marker.copy(), "post_sar": marker.copy()}, marker[0].astype(np.int64), index=0)
    assert np.array_equal(images["pre_optical"][0], images["post_sar"][0])
    assert np.array_equal(images["pre_optical"][0], mask)
