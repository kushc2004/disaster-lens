# DisasterLens

Research pipeline for multimodal building-damage mapping with BRIGHT. It implements M0–M2 source code: project setup, BRIGHT discovery/audit, manifests, label validation, leakage-safe splits, synchronized transforms, and an early-fusion U-Net baseline.

## Setup

```bash
uv sync --python 3.11 --group dev
```

Point `configs/data/bright.yaml` at an extracted official BRIGHT dataset. Raw data is read-only; manifests and reports are written outside it.

```bash
uv run python scripts/inspect_bright.py data=bright
uv run python scripts/build_manifest.py data=bright
uv run python scripts/make_splits.py data=bright split=event_holdout split.test_events='[bata-explosion]'
uv run pytest
```

In Colab, mount the Drive holding the official BRIGHT data, then run the real-data workflow below. No generated or substitute dataset is used.

```bash
export DISASTERLENS_BRIGHT_ROOT=/content/drive/MyDrive/disaster-lens/data/raw/bright
python scripts/inspect_bright.py data=bright
python scripts/build_manifest.py data=bright
python scripts/make_splits.py data=bright split.test_events='[<real-event-id>]'
# This must pass before starting the full run; it uses eight official BRIGHT tiles.
python scripts/train.py split_path=data/manifests/splits/event_holdout.json overfit_tiles=8 epochs=400 crop_size=512 learning_rate=0.001 warmup_epochs=0 use_class_weights=false checkpoint_dir=outputs/checkpoints/early_fusion_unet_tiny
python scripts/train.py split_path=data/manifests/splits/event_holdout.json epochs=30 crop_size=512 checkpoint_dir=outputs/checkpoints/early_fusion_unet_full
python scripts/evaluate.py checkpoint=outputs/checkpoints/early_fusion_unet_full/best.pt split_path=data/manifests/splits/event_holdout.json partition=test
```

## Kaggle Studio GPU workflow

The preferred remote workflow is Kaggle Studio from VS Code. `kaggle.yml` attaches the official `kushchaudhari/bright-dataset`, selects a GPU runtime, and runs [`notebooks/train.ipynb`](notebooks/train.ipynb). The notebook requires and verifies an NVIDIA Tesla T4, audits the attached BRIGHT files, extracts ZIP archives when needed, trains the M2 baseline with visible epoch output, evaluates the held-out real event, and leaves artifacts under `/kaggle/working/outputs` for download into `.kaggle-outputs/`.

After `Kaggle: Sign In`, `Kaggle: Init Project`, and `Kaggle: Attach Dataset`, run `Kaggle: Run Current Notebook`. Do not add a Drive mount or a GitHub clone to the Kaggle notebook; the Kaggle Studio project uploads the repository files and the dataset is provided by Kaggle.

To watch the remote run locally after Push & Run, use:

~~~bash
uv run python scripts/watch_kaggle_logs.py
~~~

The watcher attaches to Kaggle's live log stream, prints every training/data-progress line as it arrives, and appends the same output to `.kaggle-run.log`. Use `--output logs/kaggle-run.log` for another file; it stops automatically when Kaggle reports completion or an error. Press Ctrl-C to stop watching without cancelling the Kaggle run.

## Scope and limitations

M1 and M2 are not accepted until the commands above run against the extracted official BRIGHT copy and save their artifacts. The included `create_smoke_bright.py` is legacy scaffolding and is not part of the BRIGHT workflow. Results and performance claims are intentionally absent. BRIGHT masks must be audited before real-data loading: unknown label IDs fail loudly. Population and road data are not yet part of this milestone.
