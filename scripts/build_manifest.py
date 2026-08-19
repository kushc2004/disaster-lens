#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data.manifest import build_bright_manifest, write_manifest


def main() -> None:
    config = load_yaml(ROOT / "configs/data/bright.yaml", sys.argv[1:])
    dataset_root = Path(config["root"])
    samples = build_bright_manifest(dataset_root)
    path = write_manifest(samples, ROOT / config["manifest_path"], dataset_root=dataset_root)
    print(f"Wrote {len(samples)} BRIGHT samples to {path}")


if __name__ == "__main__":
    main()
