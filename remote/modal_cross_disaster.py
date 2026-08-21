#!/usr/bin/env python3
"""Persistent Modal GPU runner for the focused BRIGHT cross-disaster benchmark.

Inputs are deliberately read-only during training. The official public BRIGHT
archive is downloaded directly from Kaggle once into ``disasterlens-bright-v1``
and paired with the verified M1 manifest. Each invocation
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
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import modal


APP_NAME = "disaster-lens-cross-disaster"
INPUT_VOLUME_NAME = "disasterlens-bright-v1"
RESULTS_VOLUME_NAME = "disasterlens-results-v1"
REMOTE_REPO = "/root/disaster-lens"
INPUT_MOUNT = "/mnt/disasterlens-input"
RESULTS_MOUNT = "/mnt/disasterlens-results"
KAGGLE_DATASET_SLUG = "kushchaudhari/bright-dataset"
KAGGLE_ARCHIVE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET_SLUG}"
IMAGE_M1_CACHE = "/root/modal-m1-cache"
EXPECTED_TILE_COUNT = 4_246
ARCHIVE_NAME = "bright-dataset.zip"
LOCAL_BRIGHT_ROOT = Path("/tmp/disasterlens-bright")

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
        "shapely>=2.0",
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
    # These are the already completed M1 artifacts. They are copied into the
    # input Volume by the bootstrap function, so M0/M1 never rerun remotely.
    .add_local_file(
        ".kaggle-outputs/latest/disaster-lens/data/manifests/bright_manifest.jsonl",
        remote_path=f"{IMAGE_M1_CACHE}/bright_manifest.jsonl",
        copy=True,
    )
    .add_local_file(
        ".kaggle-outputs/latest/disaster-lens/data/manifests/bright_normalization.json",
        remote_path=f"{IMAGE_M1_CACHE}/bright_normalization.json",
        copy=True,
    )
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


def _source_tile_count(directory: Path, suffix: str) -> int:
    return sum(1 for path in directory.glob(f"*{suffix}") if path.is_file())


def _input_is_complete(input_root: Path) -> bool:
    bright_root = input_root / "bright"
    expected = {
        "pre-event": "_pre_disaster.tif",
        "post-event": "_post_disaster.tif",
        "target": "_building_damage.tif",
    }
    return (
        all(_source_tile_count(bright_root / name, suffix) == EXPECTED_TILE_COUNT for name, suffix in expected.items())
        and (input_root / "m1-cache/manifests/bright_manifest.jsonl").is_file()
        and (input_root / "m1-cache/manifests/bright_normalization.json").is_file()
    )


def _archive_path(input_root: Path) -> Path:
    return input_root / ARCHIVE_NAME


def _download_archive(destination: Path) -> int:
    """Download the public archive with durable, useful progress logs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"[bootstrap] downloading public Kaggle dataset archive: {KAGGLE_DATASET_SLUG}", flush=True)
    downloaded = 0
    next_report = 512 * 1024 * 1024
    request = urllib.request.Request(KAGGLE_ARCHIVE_URL, headers={"User-Agent": "disaster-lens-modal/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(32 * 1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"[bootstrap] archive download: {downloaded / (1024 ** 3):.1f} GiB", flush=True)
                next_report += 512 * 1024 * 1024
    temporary.replace(destination)
    print(f"[bootstrap] archive download complete: {downloaded / (1024 ** 3):.2f} GiB", flush=True)
    return downloaded


def _find_unique_modality_directory(extract_root: Path, suffix: str) -> Path:
    """Locate the one directory holding every official tile for one modality."""
    candidates = sorted(
        {
            file_path.parent
            for file_path in extract_root.rglob(f"*{suffix}")
            if file_path.is_file()
        }
    )
    valid = [path for path in candidates if _source_tile_count(path, suffix) == EXPECTED_TILE_COUNT]
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected exactly one directory with {EXPECTED_TILE_COUNT} '*{suffix}' files in "
            f"the official Kaggle archive; found {[str(path) for path in valid] or candidates}."
        )
    return valid[0]


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Extract only archive members whose resolved paths remain in staging."""
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        member_path = (destination / member.filename).resolve()
        if not member_path.is_relative_to(destination_resolved):
            raise RuntimeError(f"Refusing unsafe archive path: {member.filename}")
    archive.extractall(destination)


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    volumes={INPUT_MOUNT: input_volume},
)
def bootstrap_official_bright(force: bool = False) -> dict[str, object]:
    """Download public official BRIGHT directly into the persistent input Volume."""
    input_root = Path(INPUT_MOUNT)
    archive_path = _archive_path(input_root)
    if _input_is_complete(input_root) and archive_path.is_file() and not force:
        result = {"state": "already_ready", "input_root": str(input_root), "dataset": KAGGLE_DATASET_SLUG, "archive": str(archive_path)}
        print(f"[bootstrap] reusing validated official BRIGHT input and archive: {input_root}", flush=True)
        return result

    # The original extracted Volume is valid, but opening thousands of TIFFs
    # across a network mount is slow. Keep one verified ZIP beside it so each
    # GPU container can make one sequential local-SSD copy and extract locally.
    if _input_is_complete(input_root) and not force:
        downloaded = _download_archive(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise RuntimeError(f"Persisted Kaggle archive is corrupt at: {corrupt_member}")
        _write_json(input_root / "bootstrap.json", {
            "state": "ready", "dataset": KAGGLE_DATASET_SLUG, "input_root": str(input_root),
            "archive": str(archive_path), "archive_bytes": downloaded,
        })
        input_volume.commit()
        print("[bootstrap] committed reusable verified BRIGHT archive", flush=True)
        return {"state": "archive_added", "input_root": str(input_root), "dataset": KAGGLE_DATASET_SLUG, "archive": str(archive_path)}

    bright_root = input_root / "bright"
    if bright_root.exists() and not force:
        raise RuntimeError(
            "Input Volume contains an incomplete BRIGHT directory. Refusing to overwrite it. "
            "Inspect it, then rerun bootstrap with force=True if replacement is intended."
        )

    staging = input_root / ".bootstrap-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    archive_path = staging / ARCHIVE_NAME
    extract_root = staging / "extracted"
    extract_root.mkdir()
    try:
        downloaded = _download_archive(archive_path)
        print("[bootstrap] validating archive", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member:
                raise RuntimeError(f"Downloaded Kaggle archive is corrupt at: {corrupt_member}")
            _safe_extract(archive, extract_root)

        modalities = {
            "pre-event": "_pre_disaster.tif",
            "post-event": "_post_disaster.tif",
            "target": "_building_damage.tif",
        }
        if bright_root.exists():
            shutil.rmtree(bright_root)
        bright_root.mkdir(parents=True)
        counts: dict[str, int] = {}
        for modality, suffix in modalities.items():
            source = _find_unique_modality_directory(extract_root, suffix)
            destination = bright_root / modality
            shutil.move(str(source), str(destination))
            counts[modality] = _source_tile_count(destination, suffix)
            print(f"[bootstrap] validated {modality}: {counts[modality]:,} official tiles", flush=True)

        cache_destination = input_root / "m1-cache" / "manifests"
        cache_destination.mkdir(parents=True, exist_ok=True)
        for filename in ("bright_manifest.jsonl", "bright_normalization.json"):
            shutil.copy2(Path(IMAGE_M1_CACHE) / filename, cache_destination / filename)
        if not _input_is_complete(input_root):
            raise RuntimeError("Bootstrap validation failed; the input Volume was not committed.")
        result = {
            "state": "ready", "dataset": KAGGLE_DATASET_SLUG, "input_root": str(input_root),
            "tile_counts": counts, "m1_manifest": str(cache_destination / "bright_manifest.jsonl"),
        }
        _write_json(input_root / "bootstrap.json", result)
        # Preserve a single sequential source for local GPU staging.  It avoids
        # repeated high-latency Volume opens during every training epoch.
        shutil.move(str(archive_path), str(_archive_path(input_root)))
        input_volume.commit()
        print("[bootstrap] committed validated official BRIGHT and existing M1 cache", flush=True)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _stage_bright_to_local_ssd() -> Path:
    """Copy one archive sequentially, then extract official BRIGHT on local SSD."""
    archive_source = _archive_path(Path(INPUT_MOUNT))
    if not archive_source.is_file():
        raise FileNotFoundError(f"Missing persisted BRIGHT archive: {archive_source}")
    if LOCAL_BRIGHT_ROOT.exists():
        shutil.rmtree(LOCAL_BRIGHT_ROOT)
    free_bytes = shutil.disk_usage("/tmp").free
    required_bytes = archive_source.stat().st_size * 3
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient local SSD for BRIGHT staging: free={free_bytes / (1024 ** 3):.1f} GiB, "
            f"need approximately {required_bytes / (1024 ** 3):.1f} GiB."
        )
    staging = Path("/tmp/disasterlens-bright-stage")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    local_archive = staging / ARCHIVE_NAME
    total = archive_source.stat().st_size
    copied = 0
    next_report = 1024 * 1024 * 1024
    print(f"[modal] staging {total / (1024 ** 3):.2f} GiB BRIGHT archive to local SSD", flush=True)
    with archive_source.open("rb") as source, local_archive.open("wb") as destination:
        while chunk := source.read(64 * 1024 * 1024):
            destination.write(chunk)
            copied += len(chunk)
            if copied >= next_report:
                print(f"[modal] local staging: {copied / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f} GiB", flush=True)
                next_report += 1024 * 1024 * 1024
    extract_root = staging / "extracted"
    extract_root.mkdir()
    print("[modal] extracting official BRIGHT to local SSD", flush=True)
    with zipfile.ZipFile(local_archive) as archive:
        _safe_extract(archive, extract_root)
    modalities = {"pre-event": "_pre_disaster.tif", "post-event": "_post_disaster.tif", "target": "_building_damage.tif"}
    LOCAL_BRIGHT_ROOT.mkdir(parents=True)
    for modality, suffix in modalities.items():
        source = _find_unique_modality_directory(extract_root, suffix)
        destination = LOCAL_BRIGHT_ROOT / modality
        shutil.move(str(source), str(destination))
        count = _source_tile_count(destination, suffix)
        if count != EXPECTED_TILE_COUNT:
            raise RuntimeError(f"Local staging validation failed for {modality}: {count} tiles")
        print(f"[modal] local {modality}: {count:,} official tiles", flush=True)
    shutil.rmtree(staging)
    print(f"[modal] local SSD staging complete: {LOCAL_BRIGHT_ROOT}", flush=True)
    return LOCAL_BRIGHT_ROOT


def _link_inputs(bright_root: Path) -> Path:
    """Make root configuration paths resolve against the local staged data."""
    volume_bright_root = Path(INPUT_MOUNT) / "bright"
    manifest_root = Path(INPUT_MOUNT) / "m1-cache" / "manifests"
    required = (
        volume_bright_root / "pre-event",
        volume_bright_root / "post-event",
        volume_bright_root / "target",
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
    # ``scripts/train.py`` imports the package directly, whereas the focused
    # runner adds this itself.  Set it once here so both execution paths are
    # valid in the image without relying on an editable local install.
    source_root = str(Path(REMOTE_REPO) / "src")
    os.environ["PYTHONPATH"] = source_root + os.pathsep + os.environ.get("PYTHONPATH", "")
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
    batch_size: int = 16,
    workers: int = 4,
    force_prepare: bool = False,
) -> dict[str, object]:
    """Prepare, train, and score on GPU; persist everything before release."""
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
        "batch_size": batch_size, "workers": workers,
        "gpu_request": GPU, "started_at_utc": started,
        "bright_root": f"{INPUT_MOUNT}/bright", "m1_manifest_root": f"{INPUT_MOUNT}/m1-cache/manifests",
    })
    _write_json(status_path, {"state": "running", "started_at_utc": started})
    results_volume.commit()

    try:
        local_bright_root = _stage_bright_to_local_ssd()
        bright_root = _link_inputs(local_bright_root)
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

            # This file is intentionally outside the unique run directory.
            # It contains only counts/weights from the immutable standard
            # training IDs and prevents a 2,972-TIFF scan at every rerun.
            class_weight_cache = prepare_root / "class_weights" / f"{split_name}.json"
            os.environ["DISASTERLENS_CLASS_WEIGHTS_PATH"] = str(class_weight_cache)
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
                        # Lossless stored bundles avoid per-tile Deflate work
                        # holding the L40S after model inference is complete.
                        "--prediction-compression", "stored",
                    ],
                    log_handle=log_handle,
                )
                results_volume.commit()
            log_handle.write("[modal] GPU training and scoring completed; artifacts committed before CPU finalization\n"); log_handle.flush()
        result = {
            "state": "gpu_scoring_completed", "started_at_utc": started, "gpu_completed_at_utc": _utc_now(),
            "run_root": str(run_root), "model_run_dir": str(model_run_dir), "log": str(log_path),
        }
        _write_json(status_path, result)
        results_volume.commit()
        print(f"[modal] GPU scoring complete; durable artifacts: {run_root}", flush=True)
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


@app.function(
    image=image,
    timeout=60 * 60 * 2,
    volumes={RESULTS_MOUNT: results_volume},
)
def finalize_run(run_name: str, *, calibrate: bool = True) -> dict[str, object]:
    """Run calibration, figures, and reports after the GPU is no longer allocated.

    The GPU stage commits ``checkpoint.pt``, training history, metrics, and
    lossless prediction bundles first.  This CPU function intentionally reads
    only the results Volume, so a finalization failure cannot discard or alter
    the already-durable GPU artifacts.
    """
    run_name = _safe_run_name(run_name)
    results_root = Path(RESULTS_MOUNT)
    run_root = results_root / "runs" / run_name
    status_path = run_root / "status.json"
    gpu_stage_path = run_root / "gpu_stage.json"
    if not run_root.is_dir() or not status_path.is_file():
        raise FileNotFoundError(f"No durable GPU run found at {run_root}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    prior_state = str(status.get("state"))
    recoverable_states = {"gpu_scoring_completed", "failed", "cpu_finalization_failed"}
    if prior_state not in recoverable_states:
        raise RuntimeError(
            f"Cannot finalize GPU state {prior_state!r}; expected a scored or recoverable run."
        )
    model_run_dir = Path(str(status["model_run_dir"]))
    required = (model_run_dir / "checkpoint.pt", model_run_dir / "evaluation" / "val" / "metrics.json", model_run_dir / "evaluation" / "test" / "metrics.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Refusing CPU finalization; GPU artifacts are incomplete: " + ", ".join(missing))

    source_root = str(Path(REMOTE_REPO) / "src")
    os.environ["PYTHONPATH"] = source_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    # A previous failure is recoverable only after the required immutable GPU
    # artifacts above are present.  This covers an old calibration/report-only
    # failure without ever treating a failed training run as successful.
    _write_json(gpu_stage_path, {**status, "state": "gpu_scoring_completed"})
    started = _utc_now()
    _write_json(status_path, {**status, "state": "cpu_finalizing", "cpu_finalization_started_at_utc": started})
    results_volume.commit()
    log_path = run_root / "run.log"
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            message = "[modal] CPU finalization started after GPU release"
            print(message, flush=True); log_handle.write(message + "\n")
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
            log_handle.write("[modal] CPU finalization completed successfully\n"); log_handle.flush()
        result = {
            **{key: value for key, value in status.items() if key not in {"error", "traceback", "failed_at_utc"}},
            "state": "completed",
            "cpu_finalization_started_at_utc": started,
            "completed_at_utc": _utc_now(),
            "calibration_completed": calibrate,
            "recovered_from_prior_state": prior_state if prior_state != "gpu_scoring_completed" else None,
        }
        _write_json(status_path, result)
        results_volume.commit()
        print(f"[modal] completed; all durable artifacts: {run_root}", flush=True)
        return result
    except Exception as exc:
        failure = {
            **status,
            "state": "cpu_finalization_failed",
            "cpu_finalization_started_at_utc": started,
            "failed_at_utc": _utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gpu_artifacts_preserved": True,
        }
        _write_json(status_path, failure)
        results_volume.commit()
        print(f"[modal] CPU finalization failed; GPU artifacts remain durable at {run_root}", flush=True)
        raise


@app.local_entrypoint()
def main(
    model: str = "unet",
    epochs: int = 30,
    batch_size: int = 16,
    workers: int = 4,
    run_name: str = "",
    no_calibrate: bool = False,
    force_prepare: bool = False,
    skip_bootstrap: bool = False,
    force_bootstrap: bool = False,
) -> None:
    """Launch one uniquely named remote run and stream its logs locally."""
    if not skip_bootstrap:
        bootstrap = bootstrap_official_bright.remote(force=force_bootstrap)
        print("[modal] input bootstrap:\n" + json.dumps(bootstrap, indent=2, sort_keys=True), flush=True)
    if not run_name:
        run_name = f"{model}-standard-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    result = run_pipeline.remote(
        run_name=run_name, model=model, epochs=epochs, batch_size=batch_size,
        workers=workers, force_prepare=force_prepare,
    )
    print("[modal] GPU result:\n" + json.dumps(result, indent=2, sort_keys=True), flush=True)
    finalized = finalize_run.remote(run_name, calibrate=not no_calibrate)
    print("[modal] final result:\n" + json.dumps(finalized, indent=2, sort_keys=True), flush=True)
