#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disasterlens.config import load_yaml
from disasterlens.data.audit import audit_bright
from disasterlens.data.schemas import BRIGHT_V1


def main() -> None:
    config = load_yaml(ROOT / "configs/data/bright.yaml", sys.argv[1:])
    samples = audit_bright(config["root"], BRIGHT_V1, ROOT / "outputs", ROOT / config["normalization_stats_path"])
    print(f"Audited {len(samples)} BRIGHT samples; report: outputs/reports/bright_data_audit.md")


if __name__ == "__main__":
    main()
