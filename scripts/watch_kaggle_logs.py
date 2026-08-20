#!/usr/bin/env python3
"""Poll a Kaggle notebook run and stream new logs to stdout and a local file.

Run this locally after Kaggle Studio has started Push & Run. Authentication is
handled by the Kaggle CLI; this script never reads or prints the API token.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time


TERMINAL_STATUSES = {
    "COMPLETE",
    "ERROR",
    "CANCELLED",
    "CANCELED",
    "KERNELWORKERSTATUS.COMPLETE",
    "KERNELWORKERSTATUS.ERROR",
    "KERNELWORKERSTATUS.CANCELLED",
    "KERNELWORKERSTATUS.CANCELED",
}


def _run_kaggle(*args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["kaggle", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout.rstrip()


def _status(slug: str) -> str:
    code, output = _run_kaggle("kernels", "status", slug)
    if code:
        return f"STATUS_COMMAND_FAILED: {output}"
    marker = 'status "'
    if marker in output:
        return output.split(marker, 1)[1].split('"', 1)[0]
    return output.splitlines()[-1] if output else "UNKNOWN"


def _new_lines(previous: list[str], current: list[str]) -> list[str]:
    """Return only lines not already emitted, tolerating refreshed log output."""
    if current[: len(previous)] == previous:
        return current[len(previous) :]
    common = 0
    for old, new in zip(previous, current):
        if old != new:
            break
        common += 1
    return current[common:]


def watch(slug: str, output_path: Path, interval: float, once: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_lines: list[str] = []
    started = datetime.now(timezone.utc).isoformat()
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"\n===== Kaggle log watcher started {started} ({slug}) =====\n")
        output.flush()
        print(f"[watch] {slug}", flush=True)
        print(f"[watch] storing logs in {output_path}", flush=True)
        while True:
            current_status = _status(slug)
            timestamp = datetime.now(timezone.utc).isoformat()
            status_line = f"[{timestamp}] status: {current_status}"
            print(status_line, flush=True)
            output.write(status_line + "\n")

            code, logs = _run_kaggle("kernels", "logs", slug)
            if code:
                line = f"[{timestamp}] log command failed ({code}): {logs}"
                print(line, flush=True)
                output.write(line + "\n")
            else:
                current_lines = logs.splitlines()
                for line in _new_lines(previous_lines, current_lines):
                    print(line, flush=True)
                    output.write(line + "\n")
                previous_lines = current_lines
            output.flush()

            normalized = current_status.upper()
            if once or normalized in TERMINAL_STATUSES or any(
                normalized.endswith(f".{status}")
                for status in ("COMPLETE", "ERROR", "CANCELLED", "CANCELED")
            ):
                print(f"[watch] stopped with status {current_status}", flush=True)
                return 0 if "ERROR" not in normalized else 1
            time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="kushchaudhari/disaster-lens")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between polls (default: 10)")
    parser.add_argument("--output", type=Path, default=Path(".kaggle-run.log"))
    parser.add_argument("--once", action="store_true", help="fetch one status/log snapshot and exit")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        return watch(args.slug, args.output, args.interval, args.once)
    except KeyboardInterrupt:
        print("\n[watch] stopped by user", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
