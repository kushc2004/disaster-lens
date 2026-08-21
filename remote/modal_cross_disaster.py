#!/usr/bin/env python3
"""Persistent Modal GPU runner for the focused BRIGHT cross-disaster benchmark.

Inputs are deliberately read-only.  The official BRIGHT files and the M1
manifest are uploaded once to ``disasterlens-bright-v1``.  Each invocation
creates a unique run directory in ``disasterlens-results-v1`` and streams the
underlying training output both to Modal's live logs and to ``run.log``.

Run locally with, for example:
    uv run --group modal modal run remote/modal_cross_disaster.py --epochs 30
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Iterable

import modal


APP_NAME = "disaster-lens-cross-disaster"
INPUT_VOLUME_NAME = "disasterlens-bright-v1"
RESULTS_VOLUME_NAME = "disasterlens-results-v1"
REMOTE_REPO = "/root/disaster-lens"
INPUT_MOUNT = "/mnt/disasterlens-input"
RESULTS_MOUNT = "/mnt/disasterlens-results"

# L40S is the default cost/performance choice for the 512px BRIGHT runs.  Set
# this single constant to "A100-40GB" if an A100 is preferred for a run.
GPU = "L40S"

app = modal.App(APP_NAME)
input_volume = modal.Volume.from_name(INPUT_VOLUME_NAME, version=2)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True, version=2)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install(
        "numpy>=1.26,<3",
        "pandas>=2.2",
        "pyarrow>=15",
        "pyyaml>=6.0",
        "rasterio>=1.3",
        "matplotlib>=3.8",
        "scipy>=1.12",
    )
    .add_local_dir("src", remote_path=f"{REMOTE_REPO}/src", copy=True)
    .add_local_dir("configs", remote_path=f"{REMOTE_REPO}/configs", copy=True)
    .add_local_dir("scripts", remote_path=f"{REMOTE_REPO}/scripts", copy=True)
    .add_local_dir(
        "cross_disaster_damage_assessment",
        remote_path=f"{REMOTE_REPO}/cross_disaster_damage_assessment",
        copy=True,
    )
    .add_local_file("pyproject.toml", remote_path=f"{REMOTE_REPO}/pyproject.toml", copy=True)
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _safe_run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("run_name must be 1-80 characters using letters, digits, '.', '_' or '-'.")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_inputs() -> Path:
    """Make root configuration paths resolve without copying official data."""
    bright_root = Path(INPUT_MOUNT) / "bright"
    manifest_root = Path(INPUT_MOUNT) / "m1-cache" / "manifests"
    required = (
        bright_root / "pre-event",
        bright_root / "post-event",
        bright_root / "target",
        manifest_root / "bright_manifest.jsonl",
        manifest_root / "bright_normalization.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Modal input volume is incomplete. Missing: " + "; ".join(missing) + ". "
            "Follow remote/MODAL.md exactly; no synthetic data is accepted."
        )

    # A smoke manifest is useful for local tests but must never drive a paid
    # remote run. The official M1 audit contains 4,246 aligned BRIGHT tiles.
    manifest_lines = sum(1 for line in (manifest_root / "bright_manifest.jsonl").open(encoding="utf-8") if line.strip())
    if manifest_lines < 1_000:
        raise RuntimeError(
            f"Refusing to train from a non-official/smoke manifest ({manifest_lines} records). "
            "Upload the official M1 BRIGHT manifest with 4,246 records."
        )

    data_dir = Path(REMOTE_REPO) / "data"
    if data_dir.is_symlink() or data_dir.exists():
        if data_dir.is_symlink() or data_dir.is_file():
            data_dir.unlink()
        else:
            shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifests").symlink_to(manifest_root, target_is_directory=True)
    os.environ["DISASTERLENS_BRIGHT_ROOT"] = str(bright_root)
    return bright_root


def _run_command(command: Iterable[str], *, log_handle, commit_after_epoch: bool = False) -> None:
    printable = " ".join(str(part) for part in command)
    message = f"[command] {printable}"
    print(message, flush=True)
    log_handle.write(message + "\n"); log_handle.flush()
    process = subprocess.Popen(
        list(command), cwd=REMOTE_REPO, env=os.environ.copy(), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line); log_handle.flush()
        # Checkpoints are written at epoch boundaries.  Committing here makes
        # completed epochs recoverable even if a later stage is interrupted.
        if commit_after_epoch and "[training] epoch" in line and "complete:" in line:
            results_volume.commit()
            print("[modal] committed completed epoch checkpoint and log", flush=True)
            log_handle.write("[modal] committed completed epoch checkpoint and log\n"); log_handle.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 12,
    volumes={
        INPUT_MOUNT: input_volume.with_mount_options(read_only=True),
        RESULTS_MOUNT: results_volume,
    },
)
def run_pipeline(
    run_name: str,
    model: str = "unet",
    split_name: str = "standard",
    epochs: int = 30,
    batch_size: int = 4,
    workers: int = 2,
    calibrate: bool = True,
    force_prepare: bool = False,
) -> dict[str, object]:
    """Prepare once, then train/evaluate/calibrate with durable artifacts."""
    if model not in {"unet", "siamese_resnet18"}:
        raise ValueError("model must be 'unet' or 'siamese_resnet18'.")
    if split_name != "standard":
        raise ValueError("This runner currently accepts the persisted standard split only.")
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs and batch_size must be positive; workers cannot be negative.")

    run_name = _safe_run_name(run_name)
    results_root = Path(RESULTS_MOUNT)
    run_root = results_root / "runs" / run_name
    status_path = run_root / "status.json"
    if run_root.exists():
        raise FileExistsError(f"Run directory already exists: {run_root}. Choose a new --run-name.")

    run_root.mkdir(parents=True)
    log_path = run_root / "run.log"
    started = _utc_now()
    _write_json(run_root / "run_spec.json", {
        "run_name": run_name, "model": model, "split": split_name, "epochs": epochs,
        "batch_size": batch_size, "workers": workers, "calibrate": calibrate,
        "gpu_request": GPU, "started_at_utc": started,
        "bright_root": f"{INPUT_MOUNT}/bright", "m1_manifest_root": f"{INPUT_MOUNT}/m1-cache/manifests",
    })
    _write_json(status_path, {"state": "running", "started_at_utc": started})
    results_volume.commit()

    try:
        bright_root = _link_inputs()
        print(f"[modal] GPU request: {GPU}; official BRIGHT: {bright_root}", flush=True)
        prepare_root = results_root / "prepared"
        split_path = prepare_root / "splits" / f"{split_name}.json"
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            if force_prepare or not split_path.is_file():
                _run_command(
                    [sys.executable, "-u", "cross_disaster_damage_assessment/run.py", "prepare", "--output-root", str(prepare_root)],
                    log_handle=log_handle,
                )
                results_volume.commit()
                print(f"[modal] persisted shared audit and splits: {prepare_root}", flush=True)
            else:
                message = f"[modal] reusing persisted prepared split: {split_path}"
                print(message, flush=True); log_handle.write(message + "\n")

            _run_command(
                [
                    sys.executable, "-u", "cross_disaster_damage_assessment/run.py", "train",
                    "--model", model, "--split", str(split_path), "--output-root", str(run_root),
                    "--epochs", str(epochs), "--batch-size", str(batch_size), "--workers", str(workers),
                ],
                log_handle=log_handle, commit_after_epoch=True,
            )
            model_run_dir = run_root / "runs" / model / split_name
            for partition in ("val", "test"):
                _run_command(
                    [
                        sys.executable, "-u", "cross_disaster_damage_assessment/run.py", "evaluate",
                        "--run-dir", str(model_run_dir), "--partition", partition,
                        "--batch-size", str(batch_size), "--workers", str(workers),
                    ],
                    log_handle=log_handle,
                )
                results_volume.commit()
            if calibrate:
                _run_command(
                    [sys.executable, "-u", "cross_disaster_damage_assessment/run.py", "calibrate", "--run-dir", str(model_run_dir)],
                    log_handle=log_handle,
                )
                results_volume.commit()
            _run_command(
                [sys.executable, "-u", "cross_disaster_damage_assessment/run.py", "report", "--output-root", str(results_root)],
                log_handle=log_handle,
            )
            log_handle.write("[modal] pipeline completed successfully\n"); log_handle.flush()
        result = {
            "state": "completed", "started_at_utc": started, "completed_at_utc": _utc_now(),
            "run_root": str(run_root), "model_run_dir": str(model_run_dir), "log": str(log_path),
        }
        _write_json(status_path, result)
        results_volume.commit()
        print(f"[modal] completed; durable artifacts: {run_root}", flush=True)
        return result
    except Exception as exc:
        failure = {
            "state": "failed", "started_at_utc": started, "failed_at_utc": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "run_root": str(run_root), "log": str(log_path),
        }
        _write_json(status_path, failure)
        results_volume.commit()
        print(f"[modal] failed; log and status were persisted at {run_root}", flush=True)
        raise


@app.local_entrypoint()
def main(
    model: str = "unet",
    epochs: int = 30,
    batch_size: int = 4,
    workers: int = 2,
    run_name: str = "",
    no_calibrate: bool = False,
    force_prepare: bool = False,
) -> None:
    """Launch one uniquely named remote run and stream its logs locally."""
    if not run_name:
        run_name = f"{model}-standard-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    result = run_pipeline.remote(
        run_name=run_name, model=model, epochs=epochs, batch_size=batch_size,
        workers=workers, calibrate=not no_calibrate, force_prepare=force_prepare,
    )
    print("[modal] result:\n" + json.dumps(result, indent=2, sort_keys=True), flush=True)
