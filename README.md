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
python scripts/train.py split_path=data/manifests/splits/event_holdout.json overfit_tiles=8 trainer.epochs=100 trainer.crop_size=512
python scripts/evaluate.py checkpoint=outputs/checkpoints/early_fusion_unet/best.pt split_path=data/manifests/splits/event_holdout.json partition=test
```

## Scope and limitations

M1 and M2 are not accepted until the commands above run against the extracted official BRIGHT copy and save their artifacts. The included `create_smoke_bright.py` is legacy scaffolding and is not part of the BRIGHT workflow. Results and performance claims are intentionally absent. BRIGHT masks must be audited before real-data loading: unknown label IDs fail loudly. Population and road data are not yet part of this milestone.
