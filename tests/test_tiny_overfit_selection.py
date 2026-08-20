from pathlib import Path

import numpy as np

from disasterlens.data import BrightDataset, DisasterSample, select_tiny_overfit_samples


def _sample(tile_id: str) -> DisasterSample:
    return DisasterSample(tile_id=tile_id, event_id="event", disaster_type="test", label=Path(tile_id))


def test_tiny_overfit_selection_prefers_damage_class_coverage(monkeypatch, tmp_path):
    masks = {
        "a": np.array([[0, 1], [1, 1]]),
        "b": np.array([[0, 2], [2, 2]]),
        "c": np.array([[0, 3], [3, 3]]),
        "d": np.array([[0, 1], [2, 3]]),
    }

    def read(path, *, channels=None):
        return masks[path.name][None].astype(np.float32)

    monkeypatch.setattr(BrightDataset, "_read", staticmethod(read))
    selected, metadata = select_tiny_overfit_samples([_sample(key) for key in masks], count=2, crop_size=2, cache_path=tmp_path / "selection.json")

    assert {sample.tile_id for sample in selected} == {"d", "c"}
    assert all(value > 0 for key, value in metadata["aggregate_centre_crop_pixels"].items() if key != "0")
    restored, _ = select_tiny_overfit_samples([_sample(key) for key in masks], count=2, crop_size=2, cache_path=tmp_path / "selection.json")
    assert [sample.tile_id for sample in restored] == [sample.tile_id for sample in selected]
