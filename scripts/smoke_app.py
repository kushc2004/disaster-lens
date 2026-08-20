#!/usr/bin/env python3
"""Launch the artifact-only Streamlit app and verify its health endpoint."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/app/smoke.json"
PORT = 8501


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.headless=true",
        f"--server.port={PORT}",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    started = time.monotonic()
    status = None
    try:
        while time.monotonic() - started < 45:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Streamlit exited early ({process.returncode}):\n{output}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2
                ) as response:
                    status = response.status
                    if status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        if status != 200:
            raise TimeoutError("Streamlit health endpoint did not become ready within 45 seconds")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(
                {
                    "status": "healthy",
                    "http_status": status,
                    "checked_at": datetime.now(UTC).isoformat(),
                    "app": "app/streamlit_app.py",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[app] health check passed; artifact: {OUTPUT}", flush=True)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
