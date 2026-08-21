# Modal GPU pipeline

`modal_cross_disaster.py` is a separate remote execution path for the focused
BRIGHT cross-disaster benchmark. It does not change Kaggle, the M0--M2
artifacts, or the official dataset.

It uses two Modal Volume v2 volumes:

- `disasterlens-bright-v1` — official BRIGHT input plus the existing M1
  manifest and normalization statistics. Bootstrap writes this Volume once;
  training mounts it read-only.
- `disasterlens-results-v1` — shared prepared audit/splits and every run's
  checkpoints, metrics, figures, predictions, logs, and status file.

The GPU default is an L40S. Change the `GPU` constant in
`remote/modal_cross_disaster.py` to `A100-40GB` only when an A100 is wanted.

## One-time setup

From the repository root, authenticate and create the volumes:

```bash
uv sync --group modal
uv run --group modal modal setup
uv run --group modal modal volume create --version=2 disasterlens-bright-v1
uv run --group modal modal volume create --version=2 disasterlens-results-v1
```

The project downloads the public official Kaggle dataset
`kushchaudhari/bright-dataset` straight into Modal. It never uploads the raw
dataset from the laptop and it never generates data. The runner also copies
the completed M1 manifest and normalization statistics from the local
`.kaggle-outputs/latest/...` cache into the input Volume.

On the first launch, bootstrap downloads the 13.26 GB Kaggle archive, extracts
it, confirms exactly 4,246 tiles for each source modality, copies the M1 cache,
and commits the Volume. Later launches validate and reuse it immediately. The
committed layout is:

```text
/bright/pre-event/<event>/*_pre_disaster.tif
/bright/post-event/<event>/*_post_disaster.tif
/bright/target/<event>/*_building_damage.tif
/m1-cache/manifests/bright_manifest.jsonl
/m1-cache/manifests/bright_normalization.json
```

You can bootstrap without starting a GPU run:

```bash
uv run --group modal modal run remote/modal_cross_disaster.py::bootstrap_official_bright
```

The normal training command runs that same safe bootstrap automatically. Use
`--skip-bootstrap` only after an already validated input Volume is present.

## Launch and monitor

Launch GPU work detached. This is required for long jobs: an attached
`modal run` is cancelled if the terminal, VS Code task, or local client
disconnects. The command prints a Modal App URL; use that URL or `modal app
logs` for live batch/epoch output. Every line is also mirrored to a durable
`run.log` in the results Volume. At every completed epoch it commits the
Volume, so the latest completed checkpoint and log are not dependent on
container lifetime. After GPU training plus validation/test inference have
been committed, a separate CPU Modal function runs temperature calibration,
figures, and report generation. A CPU finalization failure therefore cannot
discard a checkpoint, evaluation metrics, or prediction bundles.

The default U-Net settings are `--batch-size 16 --workers 4` on the L40S.
They reduce a standard-split epoch from 743 to about 186 training batches and
each 638-tile evaluation from 160 to about 40 batches. The pseudo-Siamese
ResNet-18 is more memory-intensive; start it with `--batch-size 8 --workers 4`.
GPU inference is retained on the GPU. Only CPU-bound calibration, figure, and
report work moves to CPU; stored (lossless, uncompressed) prediction bundles
avoid Deflate compression holding the GPU allocation.

```bash
uv run --group modal modal run --detach remote/modal_cross_disaster.py --epochs 30 --batch-size 16 --workers 4
```

Give an explicit unique name when you want a memorable output directory:

```bash
uv run --group modal modal run --detach remote/modal_cross_disaster.py \
  --run-name unet-standard-30e-l40s --epochs 30 --batch-size 16 --workers 4
```

The first launch creates `prepared/` once. Later launches reuse its immutable
splits and skip preparation. `--force-prepare` is only for a deliberate split
regeneration.

Monitor a detached run with the App ID printed at launch:

```bash
uv run --group modal modal app logs <app-id> --tail 100 --timestamps
```

## Retrieve outputs

Each completed run has this layout in the results Volume:

```text
/runs/<run-name>/run.log
/runs/<run-name>/status.json
/runs/<run-name>/runs/unet/standard/checkpoint.pt
/runs/<run-name>/runs/unet/standard/training_metrics.json
/runs/<run-name>/runs/unet/standard/evaluation/{val,test}/metrics.json
/runs/<run-name>/runs/unet/standard/predictions/{val,test}/
/runs/<run-name>/runs/unet/standard/calibration/
/runs/<run-name>/gpu_stage.json
```

Download all durable artifacts after the run:

```bash
uv run --group modal modal volume get disasterlens-results-v1 runs/unet-standard-30e-l40s ./modal-outputs/unet-standard-30e-l40s
```

If a run fails, download the same directory. `status.json` contains the
exception and traceback; `run.log` retains the streamed command output up to
the failure. If the status is `failed` but the checkpoint and both evaluation
metric files exist, it was a post-scoring failure and its GPU artifacts are
safe. After pulling this code, recover calibration/report generation without a
GPU rerun:

```bash
uv run --group modal modal run \
  remote/modal_cross_disaster.py::finalize_run \
  --run-name <run-name>
```
