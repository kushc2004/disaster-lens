from __future__ import annotations

from disasterlens.data.audit import audit_bright
from disasterlens.data.schemas import BRIGHT_V1


def test_audit_writes_required_outputs(bright_root, tmp_path):
    audit_bright(bright_root, BRIGHT_V1, tmp_path)
    assert (tmp_path / "reports/bright_data_audit.md").exists()
    for name in ("event_distribution.png", "class_distribution.png", "modality_examples.png"):
        assert (tmp_path / "figures" / name).exists()
