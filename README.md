# DisasterLens

End-to-end implementation of the BRIGHT multimodal building-damage pipeline described in [`DisasterLens_IMPLEMENTATION_SPEC.md`](DisasterLens_IMPLEMENTATION_SPEC.md). The code covers M0–M8: data audit, leakage-safe splits, the M2 baseline, M3 pseudo-Siamese training, M4 fusion and ablations, M5 validation-only calibration and label-free inference, M6 real WorldPop/OSM context, M7 uncertainty-aware prioritization, and an artifact-only M8 Streamlit application.

The pipeline never creates or substitutes training, population, road, or facility data. Missing real inputs fail loudly.

## Local setup

```bash
uv sync --python 3.11 --group dev
export DISASTERLENS_BRIGHT_ROOT=/absolute/path/to/bright
```

The BRIGHT root must contain populated `pre-event`, `post-event`, and `target` directories. Raw data is read-only; all generated manifests and artifacts stay in the repository's `data/manifests/` and `outputs/` directories.

## Kaggle Studio T4 workflow

`kernel-metadata.json` attaches `kushchaudhari/bright-dataset` and the persistent validated M1 cache. [`notebooks/train.ipynb`](notebooks/train.ipynb) locates the nested Kaggle mounts, requires an NVIDIA Tesla T4, restores the M1 cache, installs the current repository checkout, and streams every phase and epoch to the Kaggle log.

The notebook defaults to M3–M4 so the completed M1 audit and M2 baseline are not repeated:

```text
DISASTERLENS_FROM_MILESTONE=M3
DISASTERLENS_THROUGH_MILESTONE=M4
DISASTERLENS_TEST_EVENT=<audited-event-id>  # optional; defaults to first event
DISASTERLENS_M3_EPOCHS=60
DISASTERLENS_M4_EPOCHS=60
```

Set `DISASTERLENS_THROUGH_MILESTONE=M8` to continue through the application. M6 additionally requires explicit real-data provenance and one input/acquisition mode for each source:

```text
DISASTERLENS_POPULATION_YEAR=2025
DISASTERLENS_POPULATION_SOURCE=WorldPop
DISASTERLENS_POPULATION_VERSION=<official-release-version>
DISASTERLENS_POPULATION_LICENSE=<official-license>
DISASTERLENS_POPULATION_DOWNLOAD_DATE=YYYY-MM-DD

# Choose one WorldPop mode:
DISASTERLENS_WORLDPOP=/kaggle/input/.../population.tif
DISASTERLENS_WORLDPOP_SOURCE=/kaggle/input/.../population.tif
DISASTERLENS_WORLDPOP_URL=https://.../official-worldpop.tif

# Choose one OSM mode:
DISASTERLENS_ROADS=/kaggle/input/.../roads.gpkg
DISASTERLENS_FACILITIES=/kaggle/input/.../facilities.gpkg

# Or local source files to normalize:
DISASTERLENS_ROADS_SOURCE=/kaggle/input/.../roads.gpkg
DISASTERLENS_FACILITIES_SOURCE=/kaggle/input/.../facilities.gpkg

# Or an Overpass query:
DISASTERLENS_OSM_BBOX=west,south,east,north
```

The notebook does not need a GitHub token. Kaggle Studio uploads project files; its fallback clone uses the public repository.

To stream and persist the latest Kaggle run log locally after Push & Run:

```bash
uv run python scripts/watch_kaggle_logs.py
```

The watcher runs `kaggle kernels logs kushchaudhari/disaster-lens`, prints new log lines, appends them to `.kaggle-run.log`, and stops on terminal status.

## Resumable end-to-end runner

The notebook delegates to the same fail-loud entry point available from any T4 environment:

```bash
python -u scripts/run_end_to_end.py \
  --bright-root /path/to/official/bright \
  --from-milestone M0 \
  --through M8 \
  --test-event <audited-event-id> \
  --population-year 2025 \
  --population-source WorldPop \
  --population-version <official-release-version> \
  --population-license <official-license> \
  --population-download-date YYYY-MM-DD \
  --worldpop /path/to/population.tif \
  --roads /path/to/roads.gpkg \
  --facilities /path/to/facilities.gpkg
```

For the current post-M2 continuation:

```bash
python -u scripts/run_end_to_end.py \
  --bright-root /path/to/official/bright \
  --from-milestone M3 \
  --through M4 \
  --test-event <audited-event-id>
```

The runner:

- verifies official BRIGHT structure and the Tesla T4 before GPU phases;
- streams child output and epoch/batch progress;
- records an atomic command fingerprint and status per step in `outputs/pipeline/state.json`;
- adopts existing validated M1/M2 artifacts, avoiding a repeated audit or completed baseline run;
- resumes later steps only when both the exact command fingerprint and required artifacts match;
- writes each step's full output to `outputs/pipeline/logs/`;
- refuses missing inputs, incompatible checkpoints, stale commands, unknown labels, and synthetic fallbacks.

Do not use `--force` unless a deliberate full rerun is intended.

## Output contract

Training runs write the resolved config, epoch metrics, event metrics, class metrics, split manifest, checkpoint, Git commit, and environment. M5 event inference writes georeferenced semantic/probability/uncertainty rasters, building-level Parquet and GeoJSON, and provenance metadata without reading held-out labels. M6–M7 add context features, accessibility, priority ranks, Monte Carlo draws, and sensitivity reports. M8 reads saved artifacts only; it never performs model inference in the UI.

No unexecuted training result or performance claim is implied by the source implementation. Remote completion is established only by the saved Kaggle artifacts and terminal run status.
