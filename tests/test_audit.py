from __future__ import annotations

import rasterio
from rasterio.transform import from_origin

from disasterlens.data.audit import audit_bright
from disasterlens.data.schemas import BRIGHT_V1


def test_audit_writes_required_outputs(bright_root, tmp_path):
    audit_bright(bright_root, BRIGHT_V1, tmp_path)
    assert (tmp_path / "reports/bright_data_audit.md").exists()
    for name in ("event_distribution.png", "class_distribution.png", "modality_examples.png"):
        assert (tmp_path / "figures" / name).exists()


def test_audit_reports_geospatial_bounds_warning_without_resampling(bright_root, tmp_path):
    post_path = bright_root / "post-event/bata-explosion_00000000_post_disaster.tif"
    with rasterio.open(post_path, "r+") as dataset:
        dataset.transform = from_origin(500010, 4100000, 10, 10)

    audit_bright(bright_root, BRIGHT_V1, tmp_path)

    report = (tmp_path / "reports/bright_data_audit.md").read_text(encoding="utf-8")
    assert "Geospatial bounds: differ for 1 tiles" in report
    assert "bata-explosion_00000000: bounds differ" in report
