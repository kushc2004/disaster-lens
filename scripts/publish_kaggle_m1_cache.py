#!/usr/bin/env python3
"""Publish completed M1 derived artifacts as a private Kaggle Dataset.

The cache intentionally contains no BRIGHT rasters.  It holds only results
derived from the official attached dataset, so later Kaggle runs can resume at
M2 without repeating the full audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


REQUIRED = (
    "data/manifests/bright_manifest.jsonl",
    "data/manifests/bright_normalization.json",
    "outputs/reports/bright_data_audit.md",
)
FIGURES = ("class_distribution.png", "event_distribution.png", "modality_examples.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Kernel-output repository directory containing completed M1 artifacts")
    parser.add_argument("--slug", default="kushchaudhari/disaster-lens-m1-cache")
    parser.add_argument("--version", action="store_true", help="Create a new version of an existing cache dataset")
    args = parser.parse_args()

    source = args.source.resolve()
    required = [source / relative for relative in REQUIRED]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot publish incomplete M1 cache; missing: " + ", ".join(missing))
    figures = [source / "outputs/figures" / name for name in FIGURES]
    missing_figures = [str(path) for path in figures if not path.is_file()]
    if missing_figures:
        raise FileNotFoundError("Cannot publish incomplete M1 cache; missing figures: " + ", ".join(missing_figures))

    manifest = required[0]
    tile_count = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
    if tile_count == 0:
        raise ValueError("Cannot publish an empty M1 manifest")

    with tempfile.TemporaryDirectory(prefix="disasterlens-m1-cache-") as temporary:
        stage = Path(temporary)
        for path in required:
            shutil.copy2(path, stage / path.name)
        for path in figures:
            shutil.copy2(path, stage / path.name)
        cache = {
            "schema_version": 1,
            "source_dataset": "kushchaudhari/bright-dataset",
            "tile_count": tile_count,
            "files": {path.name: sha256(path) for path in [*required, *figures]},
        }
        (stage / "m1_cache.json").write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        (stage / "dataset-metadata.json").write_text(json.dumps({
            "title": "DisasterLens M1 Audit Cache",
            "id": args.slug,
            "licenses": [{"name": "other"}],
        }, indent=2) + "\n", encoding="utf-8")
        command = ["kaggle", "datasets", "version" if args.version else "create", "-p", str(stage)]
        if args.version:
            command.extend(["-m", f"M1 audit cache for {tile_count:,} official BRIGHT tiles"])
        print("[M1 cache] publishing derived artifacts only; no raw BRIGHT files", flush=True)
        subprocess.run(command, check=True)
    print(f"[M1 cache] published {args.slug} ({tile_count:,} official BRIGHT tiles)", flush=True)


if __name__ == "__main__":
    main()
