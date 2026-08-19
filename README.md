# DisasterLens

Research pipeline for multimodal building-damage mapping with BRIGHT. This checkout implements **M0–M1 only**: project setup, BRIGHT discovery/audit, manifests, label validation, leakage-safe splits, synchronized geometry transforms, and a tiny synthetic smoke dataset.

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

For a self-contained verification run:

```bash
uv run python scripts/create_smoke_bright.py
uv run python scripts/inspect_bright.py data=bright dataset.root=data/samples/bright_smoke
uv run pytest
```

## Scope and limitations

No model, training, prioritization, external-context, or Streamlit code has been implemented yet. Results and performance claims are intentionally absent. BRIGHT masks must be audited before real-data loading: unknown label IDs fail loudly. Population and road data are not yet part of this milestone.

