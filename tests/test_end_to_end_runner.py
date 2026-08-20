from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_end_to_end", ROOT / "scripts/run_end_to_end.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_official_root_requires_all_three_nonempty_modalities(tmp_path: Path) -> None:
    for name in ("pre-event", "post-event", "target"):
        directory = tmp_path / name / name
        directory.mkdir(parents=True)
        (directory / "real.tif").write_bytes(b"not-empty")
    runner.validate_official_bright_root(tmp_path)


def test_tiny_gate_requires_recorded_threshold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    result = tmp_path / "outputs/checkpoints/early_fusion_unet_tiny/tiny_overfit_result.json"
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps({"best_damage_macro_f1": 0.94, "required_damage_macro_f1": 0.95}),
        encoding="utf-8",
    )
    ok, reason = runner.validate_tiny_gate()
    assert not ok
    assert "0.9400 < 0.9500" in reason


def test_later_milestones_fail_preflight_when_not_implemented(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    try:
        runner.require_implemented(["M3"])
    except runner.PipelineError as exc:
        assert "Refusing to fake completion" in str(exc)
    else:
        raise AssertionError("missing M3 scripts must not be accepted")


def test_event_split_rejects_a_different_cached_holdout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest = tmp_path / "data/manifests/bright_manifest.jsonl"
    split = tmp_path / "data/manifests/splits/event_holdout.json"
    split.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(
            json.dumps({"tile_id": tile, "event_id": event})
            for tile, event in (("a", "train"), ("b", "val"), ("c", "flood"))
        ) + "\n",
        encoding="utf-8",
    )
    split.write_text(json.dumps({"train": ["a"], "val": ["b"], "test": ["c"]}), encoding="utf-8")
    ok, reason = runner.validate_event_split("wildfire")
    assert not ok
    assert "do not equal requested" in reason
