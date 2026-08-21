# Modal GPU pipeline

`modal_cross_disaster.py` is a separate remote execution path for the focused
BRIGHT cross-disaster benchmark. It does not change Kaggle, the M0--M2
artifacts, or the official dataset.

It uses two Modal Volume v2 volumes:

- `disasterlens-bright-v1` — read-only official BRIGHT input plus the existing
  M1 manifest and normalization statistics.
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

Upload only the official, extracted BRIGHT files. The volume must contain this
exact layout; it intentionally does not accept zip files or generated samples:

```text
/bright/pre-event/<event>/*_pre_disaster.tif
/bright/post-event/<event>/*_post_disaster.tif
/bright/target/<event>/*_building_damage.tif
/m1-cache/manifests/bright_manifest.jsonl
/m1-cache/manifests/bright_normalization.json
```

For a local extracted BRIGHT directory and the M1 cache directory, upload the
three dataset folders and the two M1 artifacts. Replace the two placeholder
paths below; do not upload `data/raw` unless it is the official BRIGHT root.

```bash
uv run --group modal modal volume put disasterlens-bright-v1 /absolute/path/to/bright/pre-event /bright/pre-event
uv run --group modal modal volume put disasterlens-bright-v1 /absolute/path/to/bright/post-event /bright/post-event
uv run --group modal modal volume put disasterlens-bright-v1 /absolute/path/to/bright/target /bright/target
uv run --group modal modal volume put disasterlens-bright-v1 /absolute/path/to/m1/manifests/bright_manifest.jsonl /m1-cache/manifests/bright_manifest.jsonl
uv run --group modal modal volume put disasterlens-bright-v1 /absolute/path/to/m1/manifests/bright_normalization.json /m1-cache/manifests/bright_normalization.json
```

Inspect before spending GPU time:

```bash
uv run --group modal modal volume ls disasterlens-bright-v1 /bright
uv run --group modal modal volume ls disasterlens-bright-v1 /m1-cache/manifests
```

## Launch and monitor

The command prints live batch/epoch output in the terminal and mirrors every
line to a durable `run.log` in the results Volume. At every completed epoch it
commits the Volume, so the latest completed checkpoint and log are not
dependent on Modal container lifetime.

```bash
uv run --group modal modal run remote/modal_cross_disaster.py --epochs 30 --batch-size 4 --workers 2
```

Give an explicit unique name when you want a memorable output directory:

```bash
uv run --group modal modal run remote/modal_cross_disaster.py \
  --run-name unet-standard-30e-l40s --epochs 30 --batch-size 4 --workers 2
```

The first launch creates `prepared/` once. Later launches reuse its immutable
splits and skip preparation. `--force-prepare` is only for a deliberate split
regeneration.

## Retrieve outputs

Each completed run has this layout in the results Volume:

```text
/runs/<run-name>/run.log
/runs/<run-name>/status.json
/runs/<run-name>/runs/unet/standard/checkpoint.pt
/runs/<run-name>/runs/unet/standard/training_metrics.json
/runs/<run-name>/runs/unet/standard/evaluation/{val,test}/metrics.json
/runs/<run-name>/runs/unet/standard/calibration/
```

Download all durable artifacts after the run:

```bash
uv run --group modal modal volume get disasterlens-results-v1 runs/unet-standard-30e-l40s ./modal-outputs/unet-standard-30e-l40s
```

If a run fails, download the same directory. `status.json` contains the
exception and traceback; `run.log` retains the streamed command output up to
the failure.
