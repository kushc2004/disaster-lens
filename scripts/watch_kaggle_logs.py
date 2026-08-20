#!/usr/bin/env python3
"""Stream a Kaggle notebook run's live logs to stdout and a local file.

Run this locally after Kaggle Studio has started Push & Run. Authentication is
handled by the Kaggle CLI; this script never reads or prints the API token.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess

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


def _write_line(output, line: str) -> None:
    print(line, flush=True)
    output.write(line + "\n")
    output.flush()


def watch(slug: str, output_path: Path, once: bool) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"\n===== Kaggle log watcher started {started} ({slug}) =====\n")
        output.flush()
        print(f"[watch] {slug}", flush=True)
        print(f"[watch] storing logs in {output_path}", flush=True)
        if once:
            current_status = _status(slug)
            timestamp = datetime.now(timezone.utc).isoformat()
            _write_line(output, f"[{timestamp}] status: {current_status}")
            code, logs = _run_kaggle("kernels", "logs", slug)
            if code:
                _write_line(output, f"[{timestamp}] log command failed ({code}): {logs}")
                return 1
            for line in logs.splitlines():
                _write_line(output, line)
            _write_line(output, f"[watch] stopped with status {current_status}")
            return 0 if "ERROR" not in current_status.upper() else 1

        _write_line(output, "[watch] following Kaggle live log stream")
        process = subprocess.Popen(
            ["kaggle", "kernels", "logs", "--follow", slug],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                _write_line(output, raw_line.rstrip("\n"))
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            _write_line(output, "[watch] stopped by user (Kaggle run was not cancelled)")
            return 130

        final_status = _status(slug)
        _write_line(output, f"[watch] final status: {final_status}")
        if return_code:
            _write_line(output, f"[watch] log stream exited with code {return_code}")
            return return_code
        _write_line(output, f"[watch] stopped with status {final_status}")
        return 0 if "ERROR" not in final_status.upper() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="kushchaudhari/disaster-lens")
    parser.add_argument("--output", type=Path, default=Path(".kaggle-run.log"))
    parser.add_argument("--once", action="store_true", help="fetch one status/log snapshot and exit")
    args = parser.parse_args()
    try:
        return watch(args.slug, args.output, args.once)
    except KeyboardInterrupt:
        print("\n[watch] stopped by user", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
