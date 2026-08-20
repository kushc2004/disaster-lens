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


def test_crop_size_returns_fixed_shape_for_variable_tiles():
    marker = np.arange(6 * 7, dtype=np.float32).reshape(1, 6, 7)
    images, mask = SynchronizedGeometry(seed=11, crop_size=4)(
        {"pre_optical": marker.copy(), "post_sar": marker.copy()},
        marker[0].astype(np.int64),
        index=0,
    )
    assert images["pre_optical"].shape == (1, 4, 4)
    assert images["post_sar"].shape == (1, 4, 4)
    assert mask.shape == (4, 4)
    assert np.array_equal(images["pre_optical"][0], images["post_sar"][0])
    assert np.array_equal(images["pre_optical"][0], mask)


def test_crop_size_pads_undersized_tiles_to_fixed_shape():
    marker = np.zeros((1, 6, 7), dtype=np.float32)
    images, mask = SynchronizedGeometry(seed=11, crop_size=8)(
        {"pre_optical": marker.copy(), "post_sar": marker.copy()},
        marker[0].astype(np.int64),
        index=0,
    )
    assert images["pre_optical"].shape == (1, 8, 8)
    assert images["post_sar"].shape == (1, 8, 8)
    assert mask.shape == (8, 8)


def test_non_random_geometry_reuses_the_same_center_crop_across_epochs():
    geometry = SynchronizedGeometry(seed=11, crop_size=4, randomize=False)
    first = geometry.plan_for(0, height=6, width=7)
    geometry.set_epoch(99)
    second = geometry.plan_for(0, height=6, width=7)
    assert first == second == type(first)(top=1, left=1, crop_size=4)
