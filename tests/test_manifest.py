from __future__ import annotations

import pytest

from disasterlens.data.manifest import BrightLayoutError, build_bright_manifest, load_manifest, write_manifest


def test_manifest_has_expected_modalities_and_relative_paths(bright_root, tmp_path):
    samples = build_bright_manifest(bright_root)
    assert len(samples) == 3
    assert {sample.event_id for sample in samples} == {"bata-explosion", "surat-flood", "maui-wildfire"}
    path = write_manifest(samples, tmp_path / "manifest.jsonl", dataset_root=bright_root)
    assert load_manifest(path, dataset_root=bright_root) == samples


def test_manifest_rejects_missing_modality(bright_root):
    (bright_root / "post-event" / "bata-explosion_00000000_post_disaster.tif").unlink()
    with pytest.raises(BrightLayoutError, match="Modalities do not align"):
        build_bright_manifest(bright_root)
