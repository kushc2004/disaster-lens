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
    print(f"[M1] auditing official BRIGHT data under {config['root']}", flush=True)
    samples = audit_bright(config["root"], BRIGHT_V1, ROOT / "outputs", ROOT / config["normalization_stats_path"])
    events = sorted({sample.event_id for sample in samples})
    print(f"[M1] audited {len(samples):,} BRIGHT samples; report: outputs/reports/bright_data_audit.md", flush=True)
    print("[M1] events available for TEST_EVENT: " + ", ".join(events), flush=True)


if __name__ == "__main__":
    main()
