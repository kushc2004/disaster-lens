#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data.manifest import build_bright_manifest, load_manifest
from disasterlens.data.splits import event_holdout_split, official_split


def _records(samples):
    return [sample.tile_id for sample in samples]


def main() -> None:
    data = load_yaml(ROOT / "configs/data/bright.yaml", sys.argv[1:])
    experiment = load_yaml(ROOT / "configs/experiment/event_holdout.yaml", sys.argv[1:]).get("split", {})
    manifest_path = ROOT / data["manifest_path"]
    if manifest_path.exists():
        samples = load_manifest(manifest_path, dataset_root=Path(data["root"]))
        print(f"[split] reusing manifest {manifest_path} ({len(samples):,} samples)", flush=True)
    else:
        print("[split] manifest not found; building it from the official dataset", flush=True)
        samples = build_bright_manifest(Path(data["root"]))
    if any(item == "split=standard_split" for item in sys.argv[1:]):
        if not data.get("official_split_root"):
            raise ValueError("official_split_root must be configured for split=standard_split")
        split = official_split(samples, split_root=Path(data["official_split_root"]))
        name = "standard_split"
    else:
        groups = {key: list(experiment.get(key, [])) for key in ("train_events", "val_events", "test_events")}
        known = sorted({sample.event_id for sample in samples})
        if not groups["test_events"]:
            raise ValueError("event_holdout requires split.test_events")
        remaining = [event for event in known if event not in groups["test_events"]]
        if not groups["val_events"]:
            groups["val_events"] = remaining[-1:]
        if not groups["train_events"]:
            groups["train_events"] = [event for event in remaining if event not in groups["val_events"]]
        split = event_holdout_split(samples, **groups)
        name = "event_holdout"
    path = ROOT / "data/manifests/splits" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train": _records(split.train), "val": _records(split.val), "test": _records(split.test)}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {name} split to {path} (train={len(split.train)}, val={len(split.val)}, test={len(split.test)})")


if __name__ == "__main__":
    main()
