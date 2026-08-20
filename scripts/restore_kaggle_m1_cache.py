#!/usr/bin/env python3
"""Restore and validate a persistent M1 cache mounted by Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED = ("bright_manifest.jsonl", "bright_normalization.json", "bright_data_audit.md")
FIGURES = ("class_distribution.png", "event_distribution.png", "modality_examples.png")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    cache_root, dataset_root, repo_root = (path.resolve() for path in (args.cache_root, args.dataset_root, args.repo_root))
    cache_path = cache_root / "m1_cache.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"M1 cache marker missing: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("schema_version") != 1 or cache.get("source_dataset") != "kushchaudhari/bright-dataset":
        raise ValueError("Attached M1 cache does not identify the expected official BRIGHT source")
    expected_files = set(REQUIRED + FIGURES)
    if set(cache.get("files", {})) != expected_files:
        raise ValueError("Attached M1 cache has an unexpected artifact set")
    for name in expected_files:
        path = cache_root / name
        if not path.is_file() or sha256(path) != cache["files"][name]:
            raise ValueError(f"M1 cache artifact failed integrity validation: {name}")
    records = [json.loads(line) for line in (cache_root / "bright_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != cache.get("tile_count") or not records:
        raise ValueError("M1 cache manifest count does not match its metadata")
    for record in records:
        for key in ("pre_optical", "post_sar", "label"):
            relative = record.get(key)
            if not isinstance(relative, str) or not (dataset_root / relative).is_file():
                raise ValueError(f"M1 cache does not match attached official BRIGHT data: {key}={relative!r}")
    targets = {
        "bright_manifest.jsonl": repo_root / "data/manifests/bright_manifest.jsonl",
        "bright_normalization.json": repo_root / "data/manifests/bright_normalization.json",
        "bright_data_audit.md": repo_root / "outputs/reports/bright_data_audit.md",
    }
    targets.update({name: repo_root / "outputs/figures" / name for name in FIGURES})
    for name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_root / name, target)
    print(f"[M1 cache] restored {len(records):,} validated official BRIGHT records; audit skipped", flush=True)


if __name__ == "__main__":
    main()
