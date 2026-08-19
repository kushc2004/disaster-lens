# DisasterLens — Implementation Specification

## Multimodal AI for Post-Disaster Damage Assessment, Cross-Disaster Generalization, Uncertainty, and Relief Prioritization

**Audience:** Coding agent / Codex  
**Goal:** Build a research-grade, CV-ready computer vision + geospatial data science project.  
**Primary dataset:** BRIGHT  
**Secondary datasets:** xBD, xBD-S12  
**Primary research contribution:** multimodal building-damage mapping with event-held-out generalization and calibrated uncertainty  
**Decision-support contribution:** transparent, uncertainty-aware relief prioritization from damage, population exposure, accessibility risk, and optional hazard context

---

# 1. Project Definition

DisasterLens is an end-to-end disaster-response decision-support system. It takes multimodal Earth-observation imagery and geospatial context and produces:

1. building localization,
2. building damage severity,
3. calibrated uncertainty,
4. cross-disaster performance analysis,
5. population exposure estimates,
6. estimated road-accessibility risk,
7. optional hazard/weather context,
8. an interpretable relative relief-priority ranking.

The project must **not** be implemented as generic satellite image classification.

The central research question is:

> Can a multimodal damage model trained on some disaster events generalize to completely unseen events, and can calibrated damage uncertainty be propagated into transparent relief-priority rankings?

The final project should demonstrate the complete chain:

```text
multimodal Earth observation
        ↓
building damage mapping
        ↓
event-held-out evaluation
        ↓
probability calibration
        ↓
population + road context
        ↓
uncertainty-aware priority ranking
```

---

# 2. Critical Design Decision

## 2.1 Use BRIGHT for the main VHR damage model

BRIGHT is the recommended primary dataset because it is specifically designed for multimodal building-damage assessment using very-high-resolution optical and SAR imagery.

The core model should operate on BRIGHT's native multimodal setting, especially:

```text
pre-event VHR optical
+
post-event VHR SAR
```

where available.

## 2.2 Do not force Prithvi onto incompatible VHR imagery

Prithvi-EO-2.0 was pretrained on NASA HLS imagery at approximately 30 m resolution using six spectral bands:

```text
Blue
Green
Red
Narrow NIR
SWIR1
SWIR2
```

Therefore:

```text
BRIGHT/xBD VHR RGB → Prithvi
```

must **not** be the default architecture.

Prithvi is an optional later extension using:

- xBD-S12 Sentinel-2,
- aligned HLS,
- or other correctly formatted Sentinel/HLS context.

The core project must work without Prithvi.

---

# 3. Research Questions

The implementation and final report must answer:

### RQ1 — Multimodal utility
Does pre-event optical + post-event SAR outperform SAR-only or simple early-fusion models?

### RQ2 — Cross-disaster generalization
How much does performance degrade when an entire disaster event is absent during training?

### RQ3 — Fusion architecture
Does a custom cross-modal fusion model improve held-out-event performance over simpler baselines?

### RQ4 — Calibration
How reliable are damage probabilities, and does temperature scaling improve ECE, NLL, or Brier score?

### RQ5 — Decision support
Can calibrated damage probabilities be combined with population and accessibility data into an auditable regional ranking?

### RQ6 — Ranking robustness
How stable are priority rankings to model uncertainty and scoring-weight choices?

### RQ7 — Optional medium-resolution context
Do aligned Sentinel-1/Sentinel-2 or Prithvi features provide useful additional regional context?

---

# 4. Scope

## 4.1 Required V1

The minimum complete project must contain:

- BRIGHT data pipeline,
- strong segmentation baseline,
- multimodal proposed model,
- standard split evaluation,
- event-held-out evaluation,
- per-event metrics,
- calibrated building-level probabilities,
- population exposure layer,
- road accessibility-risk layer,
- relief-priority scoring,
- Monte Carlo uncertainty propagation,
- priority-weight sensitivity analysis,
- interactive Streamlit demo.

## 4.2 Optional V2

Only after V1 works:

- xBD replication,
- xBD-S12,
- Prithvi-EO-2.0,
- post-event optical + SAR triple-modality,
- domain adaptation,
- few-shot adaptation,
- hazard-specific data feeds.

## 4.3 Non-goals

Do not:

- claim actual emergency deployment,
- claim predicted road risk is a confirmed closure,
- invent supervised "relief priority" labels,
- claim exact affected population,
- mix train and test tiles from the same held-out event,
- hide heuristic policy weights,
- invent performance numbers,
- require external APIs for model training.

---

# 5. Data Sources

## 5.1 BRIGHT — Primary

Use the official BRIGHT paper/repository as the source of truth for:

- directory structure,
- label encoding,
- modality definitions,
- split definitions,
- normalization,
- benchmark methodology.

Important characteristics to preserve:

- VHR optical and SAR,
- roughly 0.3–1 m imagery,
- multiple disaster types/events,
- strong class imbalance,
- event imbalance,
- residual multimodal registration error,
- damage label noise,
- standard and event-transfer evaluation.

The BRIGHT paper reports categories corresponding to:

```text
background
intact building
damaged building
destroyed building
```

However, **Codex must inspect the actual official mask encoding before hard-coding integer IDs.**

## 5.2 xBD — Secondary

Use xBD after BRIGHT is stable.

Keep xBD's label schema separate:

```text
background
no-damage
minor-damage
major-damage
destroyed
```

Implement a dataset-specific `LabelSchema`.

Never silently map xBD and BRIGHT classes into each other.

## 5.3 xBD-S12 — Optional

Use the official xBD-S12 repository and dataset for:

- Sentinel-1,
- Sentinel-2,
- aligned medium-resolution context,
- optional Prithvi branch.

## 5.4 Population

Use WorldPop gridded population data.

For each event store:

```text
source
population year
native resolution
CRS
download date
event used
```

Population values are exposure estimates, not exact counts of people physically present during the event.

## 5.5 Roads and facilities

Use OpenStreetMap.

Extract/cache:

- roads,
- road class,
- bridges where present,
- hospitals,
- clinics,
- emergency facilities where present.

Do not repeatedly query OSM during inference/demo.

## 5.6 Rainfall / hazard

Optional for V1.

For floods/cyclones, support NASA GPM IMERG through a generic hazard adapter.

Possible features:

```text
6-hour rainfall
24-hour rainfall
72-hour rainfall
local percentile/anomaly
```

Earthquake events should not be assigned meaningless rainfall features merely to fill a model input.

---

# 6. Repository Structure

Create:

```text
disasterlens/
├── README.md
├── IMPLEMENTATION_SPEC.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── Makefile
├── configs/
│   ├── data/
│   │   ├── bright.yaml
│   │   ├── xbd.yaml
│   │   └── xbd_s12.yaml
│   ├── model/
│   │   ├── unet_baseline.yaml
│   │   ├── siamese_baseline.yaml
│   │   └── damage_fusion_former.yaml
│   ├── experiment/
│   │   ├── standard_split.yaml
│   │   ├── event_holdout.yaml
│   │   └── calibration.yaml
│   └── priority.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   ├── samples/
│   └── external/
│       ├── worldpop/
│       ├── osm/
│       └── gpm/
├── src/
│   └── disasterlens/
│       ├── data/
│       │   ├── schemas.py
│       │   ├── manifest.py
│       │   ├── bright.py
│       │   ├── xbd.py
│       │   ├── xbd_s12.py
│       │   ├── augmentations.py
│       │   ├── samplers.py
│       │   └── splits.py
│       ├── models/
│       │   ├── common.py
│       │   ├── baselines.py
│       │   ├── encoders.py
│       │   ├── fusion.py
│       │   ├── decoders.py
│       │   ├── damage_fusion_former.py
│       │   └── losses.py
│       ├── training/
│       │   ├── engine.py
│       │   ├── optim.py
│       │   └── checkpointing.py
│       ├── evaluation/
│       │   ├── segmentation.py
│       │   ├── building_metrics.py
│       │   ├── event_metrics.py
│       │   ├── calibration.py
│       │   └── reports.py
│       ├── inference/
│       │   ├── predictor.py
│       │   ├── tiling.py
│       │   ├── vectorize.py
│       │   └── uncertainty.py
│       ├── context/
│       │   ├── worldpop.py
│       │   ├── roads.py
│       │   ├── gpm.py
│       │   └── alignment.py
│       ├── prioritization/
│       │   ├── features.py
│       │   ├── normalization.py
│       │   ├── scoring.py
│       │   ├── monte_carlo.py
│       │   └── sensitivity.py
│       ├── visualization/
│       │   ├── maps.py
│       │   ├── overlays.py
│       │   └── calibration_plots.py
│       └── utils/
│           ├── geo.py
│           ├── seed.py
│           ├── logging.py
│           └── io.py
├── scripts/
│   ├── inspect_bright.py
│   ├── build_manifest.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── infer_event.py
│   ├── fetch_population.py
│   ├── fetch_osm.py
│   ├── build_priority_map.py
│   └── export_report.py
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_results.ipynb
│   ├── 03_cross_event_analysis.ipynb
│   └── 04_priority_analysis.ipynb
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_schemas.py
│   ├── test_splits.py
│   ├── test_dataset.py
│   ├── test_model_shapes.py
│   ├── test_metrics.py
│   ├── test_calibration.py
│   ├── test_geo_alignment.py
│   └── test_priority_scoring.py
└── outputs/
    ├── checkpoints/
    ├── predictions/
    ├── metrics/
    ├── figures/
    ├── maps/
    └── reports/
```

---

# 7. Environment

Use Python 3.11 unless upstream official code requires another supported version.

Recommended dependencies:

```text
torch
torchvision
timm
segmentation-models-pytorch
torchmetrics
albumentations
opencv-python-headless
numpy
pandas
scikit-learn
scipy
rasterio
rioxarray
xarray
geopandas
shapely
pyproj
fiona
networkx
osmnx
matplotlib
plotly
streamlit
hydra-core
omegaconf
pydantic
pyarrow
tqdm
rich
pytest
```

Optional:

```text
terratorch
mlflow
wandb
captum
earthaccess
```

Use `uv` or equivalent and lock versions **after** the environment works.

Support:

- CUDA for full training,
- Apple MPS where practical for smoke tests,
- CPU for tests.

---

# 8. Canonical Data Model

Implement:

```python
@dataclass
class DisasterSample:
    event_id: str
    disaster_type: str
    tile_id: str
    pre_optical: Path | None
    post_optical: Path | None
    pre_sar: Path | None
    post_sar: Path | None
    label: Path | None
    crs: str | None
    bounds: tuple[float, float, float, float] | None
    metadata: dict
```

Dataset output:

```python
{
    "images": {
        "pre_optical": Tensor | None,
        "post_optical": Tensor | None,
        "pre_sar": Tensor | None,
        "post_sar": Tensor | None,
    },
    "mask": LongTensor,
    "event_id": str,
    "tile_id": str,
    "geo": {...}
}
```

Modalities must be normalized separately.

Transforms that change geometry must be synchronized across all images and masks.

---

# 9. Data Governance

Rules:

1. never modify `data/raw`,
2. derived files go to `data/processed`,
3. generated metadata goes to `data/manifests`,
4. preserve georeferencing,
5. every split must be reproducible,
6. external context must be cached,
7. every experiment must save configuration,
8. every result must record the split used,
9. metrics shown in README/CV must come from saved artifacts,
10. unknown label IDs or missing CRS should fail loudly.

---

# 10. Mandatory Data Audit

Implement:

```bash
python scripts/inspect_bright.py data=bright
```

Produce:

```text
outputs/reports/bright_data_audit.md
outputs/figures/event_distribution.png
outputs/figures/class_distribution.png
outputs/figures/modality_examples.png
```

Audit:

- actual directory structure,
- events,
- disaster types,
- tiles/event,
- label values,
- global class frequencies,
- per-event class frequencies,
- missing/corrupt images,
- image sizes,
- modality availability,
- per-modality intensity distributions,
- CRS/bounds,
- geospatial metadata,
- optical/SAR alignment examples.

The data loader should not be implemented from filename assumptions if official metadata provides reliable mapping.

---

# 11. Split Strategy

## 11.1 Standard benchmark split

Reproduce/follow BRIGHT's official split as closely as possible.

Purpose:

- sanity check,
- baseline comparison,
- implementation verification.

## 11.2 Event-held-out split

This is the most important experimental split.

All samples from a target event must be absent from training.

Example:

```yaml
split:
  strategy: event_holdout
  train_events: [...]
  val_events: [...]
  test_events: [...]
```

Assertions:

```python
assert train_events.isdisjoint(val_events)
assert train_events.isdisjoint(test_events)
assert val_events.isdisjoint(test_events)
```

Also assert no tile ID/hash overlap.

## 11.3 Minimum held-out experiments

If data permits, test at least three qualitatively different targets:

- earthquake,
- flood,
- wildfire / explosion / another event class.

If compute is limited:

- standard split once,
- event-held-out on three events,
- one architecture ablation.

---

# 12. Baselines

Do not implement the proposed model before a baseline works.

## 12.1 Baseline A — Early-fusion U-Net

Inputs:

```text
pre optical
post SAR
```

Normalize independently, then concatenate.

Output:

```text
background
intact
damaged
destroyed
```

Loss:

```text
CrossEntropy + Lovasz-Softmax
```

This keeps the initial baseline conceptually close to BRIGHT's benchmark loss.

## 12.2 Baseline B — Pseudo-Siamese model

Use separate optical and SAR encoders.

```text
pre optical -> encoder A --\
                            fusion -> decoder -> damage
post SAR   -> encoder B --/
```

Start with ResNet-18 or ResNet-50.

Do not share weights by default because optical and SAR distributions are very different.

## 12.3 Optional upstream baseline

After the local baseline is stable, optionally wrap an official BRIGHT baseline such as:

- U-Net,
- DeepLabV3+,
- DamageFormer,
- MambaBDA / ChangeMamba.

Keep third-party code attributed and isolated.

---

# 13. Proposed Model: DamageFusionFormer

The custom model should be interesting but not unnecessarily huge.

## 13.1 Inputs

Core:

```text
X_o = pre-event VHR optical
X_s = post-event VHR SAR
```

Optional controlled subset:

```text
X_po = post-event VHR optical
```

## 13.2 Encoders

Recommended:

```text
Swin-Tiny
```

or another reasonably sized hierarchical encoder from `timm`.

Use separate encoders:

```text
O_l = E_o^l(X_o)
S_l = E_s^l(X_s)
```

Do not weight-share in the default configuration.

## 13.3 Fusion

Use lightweight fusion at high spatial resolution and cross-attention at coarse resolution.

### Stages 1–2

At 1/4 and 1/8 scale:

```text
g = sigmoid(conv(concat(O,S)))
F = g*O + (1-g)*S
```

Optionally concatenate:

```text
abs(O-S)
O*S
```

before projection.

### Stages 3–4

At 1/16 and 1/32:

```text
A_os = MultiHeadAttention(Q=O, K=S, V=S)
A_so = MultiHeadAttention(Q=S, K=O, V=O)

F = concat(
    O,
    S,
    A_os,
    A_so,
    abs(O-S),
    O*S
)

F = projection(F)
```

Cross-attention only at coarse scales prevents memory explosion.

## 13.4 Decoder

Use FPN or UPerNet-style multiscale decoding.

Create two heads.

### Localization head

```text
background
building
```

### Damage head

```text
intact
damaged
destroyed
```

Final semantic predictions combine the localization and severity heads.

## 13.5 Loss

Use a decoupled objective:

```text
L_loc = CE_loc + Lovasz_loc
L_damage = weighted_CE_damage + Lovasz_damage

L_total = λ_loc L_loc + λ_damage L_damage
```

Initial:

```yaml
lambda_loc: 1.0
lambda_damage: 1.0
```

Class weights must be computed **only from the training split**.

Do not alter architecture and loss simultaneously in the primary comparison; otherwise attribution is unclear.

---

# 14. Training

## 14.1 Optimizer

Start:

```text
AdamW
lr = 1e-4
weight_decay = 5e-3
```

## 14.2 Scheduler

Use cosine decay with warmup.

Suggested:

```text
warmup = 3 epochs
```

## 14.3 Augmentation

Geometry must be synchronized:

- horizontal flip,
- vertical flip,
- 90° rotations,
- random crop.

Optical-only:

- brightness,
- contrast,
- color jitter.

Do not apply RGB color transforms to SAR.

## 14.4 Imbalance

Implement one or both:

- tile weighting by rare-damage content,
- event-aware sampling.

Keep a maximum weight cap.

Log results under natural evaluation distribution regardless of sampling.

## 14.5 Mixed precision

Use AMP on CUDA.

## 14.6 Checkpointing

Primary checkpoint metric:

```text
validation damage macro-F1
```

Also log:

```text
mIoU
F1 localization
F1 damage
per-class F1
```

## 14.7 Tiny-set overfit test

Before full training, overfit roughly 8 tiles.

If the model cannot strongly overfit a tiny sample, debug first.

---

# 15. Evaluation

## 15.1 Pixel level

Report:

- overall accuracy,
- mIoU,
- class IoU,
- macro F1,
- class F1,
- confusion matrix.

## 15.2 Task-decomposed

Report:

```text
F1_loc
F1_damage
```

## 15.3 Building level

Where polygons are available:

1. aggregate pixel logits/probabilities within each building,
2. produce building class probabilities,
3. evaluate:
   - macro F1,
   - balanced accuracy,
   - confusion matrix,
   - Brier score,
   - negative log likelihood,
   - expected calibration error.

Use building-level probabilities for downstream prioritization.

## 15.4 Event level

Save:

```text
event_id
disaster_type
n_tiles
n_buildings
mIoU
macro_F1
F1_localization
F1_damage
ECE
```

Report:

- pooled score,
- macro average over events,
- worst-event performance,
- standard deviation over events.

---

# 16. Cross-Disaster Generalization

Generate:

```text
outputs/reports/cross_event_report.md
```

For each model compare:

```text
standard split
vs
event-held-out split
```

Define:

```text
generalization_gap =
standard_metric - heldout_metric
```

Analyze gap by:

- disaster type,
- target event,
- class prevalence,
- building size,
- modality quality if available.

Required plots:

- model × event performance,
- per-event confusion matrix,
- performance vs destroyed-class prevalence,
- calibration by event.

Avoid causal interpretation of observational correlations.

---

# 17. Probability Calibration

Calibration is required.

## 17.1 Building probabilities

For building `b`:

```text
p_b = aggregate(pixel softmax probabilities within building)
```

Start with mean probability.

## 17.2 Temperature scaling

Fit scalar `T` on validation logits only:

```text
p = softmax(logits / T)
```

Never fit on test events.

## 17.3 Metrics

Before and after:

- ECE,
- Brier,
- NLL,
- macro F1.

Produce reliability diagrams.

## 17.4 Uncertainty

Use normalized predictive entropy:

```text
H(p) = -Σ p_c log(p_c)
```

Keep uncertainty as a separate output.

Do not automatically treat uncertainty as priority.

---

# 18. Inference Outputs

Command:

```bash
python scripts/infer_event.py \
  checkpoint=<path> \
  event_id=<event>
```

Output:

```text
outputs/predictions/<event>/
├── semantic_mask.tif
├── damage_probabilities.tif
├── building_predictions.parquet
├── buildings.geojson
└── metadata.json
```

`building_predictions.parquet`:

```text
building_id
event_id
tile_id
p_intact
p_damaged
p_destroyed
predicted_class
expected_severity
predictive_entropy
geometry reference
```

Expected severity:

```text
intact = 0
damaged = 1
destroyed = 2

E[severity] = Σ severity(c) p(c)
```

Use a separate severity map for xBD.

---

# 19. Population Exposure

## 19.1 Reprojection

Never intersect data with mismatched CRS.

Workflow:

1. load event bounds,
2. select local projected CRS / UTM,
3. reproject buildings/decision grid,
4. reproject or window population raster appropriately,
5. use count-preserving aggregation where possible.

## 19.2 Decision grid

Use a configurable local grid.

Start:

```yaml
grid_size_m: 500
```

For each grid cell calculate:

```text
population_total
number_buildings
expected_damaged_buildings
expected_destroyed_buildings
mean_expected_severity
mean_model_entropy
```

## 19.3 Exposure

Example:

```text
population_damage_exposure =
population_total *
normalized_damage_score
```

This is an exposure proxy, not casualty prediction.

---

# 20. Road Accessibility Risk

Never output "road definitely inaccessible" unless an external confirmed road-closure source exists.

Output:

```text
estimated accessibility risk
```

## 20.1 Road graph

Create/cache:

```text
data/external/osm/<event>/
├── roads.gpkg
├── facilities.gpkg
└── graph.graphml
```

Facility candidates:

- hospital,
- clinic,
- emergency facility.

## 20.2 Road-segment risk

Estimate from:

- nearby predicted destroyed-building density,
- overlap with optional flood/hazard mask,
- any external confirmed road-damage data if later available.

Example:

```text
road_risk =
a * local_destroyed_density +
b * hazard_overlap
```

Coefficients are configurable heuristic assumptions.

## 20.3 Accessibility penalty

For each cell:

1. shortest normal path to nearest relevant facility,
2. shortest risk-adjusted path,
3. calculate:

```text
A =
clip(
    (risk_adjusted_cost - normal_cost) / normal_cost,
    0,
    1
)
```

If no path remains, set:

```text
A = 1
```

and label it:

```text
estimated isolation under current risk model
```

not confirmed isolation.

---

# 21. Hazard Adapter

Interface:

```python
class HazardProvider(Protocol):
    def features_for_event(self, event_metadata, grid):
        ...
```

Initial optional provider:

```text
GPMIMERGProvider
```

Possible flood/cyclone features:

```text
rain_6h
rain_24h
rain_72h
rain_percentile
```

If unavailable:

```text
hazard_available = false
```

Priority weights must renormalize over remaining features.

---

# 22. Relief-Priority Scoring

This is a transparent multi-criteria decision-support layer.

It is **not** a supervised classifier unless genuine expert priority labels are later obtained.

For cell `i`:

## 22.1 Damage score

Example:

```text
D_raw =
1.0 * E[number damaged] +
2.0 * E[number destroyed]
```

Robust-normalize to `[0,1]`.

## 22.2 Population score

```text
P_raw = log1p(population_total)
```

Robust-normalize.

## 22.3 Accessibility score

Use normalized accessibility penalty.

## 22.4 Hazard score

Optional event-specific value.

## 22.5 Overall score

```text
R_i =
w_D D_i +
w_P P_i +
w_A A_i +
w_H H_i
```

Demonstration defaults:

```yaml
weights:
  damage: 0.40
  population: 0.30
  accessibility: 0.20
  hazard: 0.10
```

These are not ground-truth humanitarian weights.

Every report/UI must state:

> Priority weights are configurable decision-support assumptions. Operational weights require domain-expert validation.

If hazard is absent, renormalize available weights.

## 22.6 Relative bands

Define within-event bands from percentiles:

```text
LOW      0–50%
MODERATE 50–75%
HIGH     75–90%
CRITICAL 90–100%
```

Label them:

```text
relative event-level priority bands
```

not universal emergency classifications.

---

# 23. Uncertainty Propagation

Do not use only argmax labels.

## 23.1 Monte Carlo ranking

For `N = 500` initially:

For every simulation:

1. sample building damage class from calibrated probabilities,
2. aggregate damage to grid cells,
3. recompute `D`,
4. recompute priority score,
5. rank cells.

Store:

```text
priority_mean
priority_p05
priority_p50
priority_p95
rank_mean
rank_p05
rank_p95
prob_top_10_percent
```

Interpret:

- high priority + narrow interval = stable,
- high priority + wide interval = important but uncertain.

---

# 24. Priority-Weight Sensitivity

Sample many valid weight combinations, e.g.:

```text
damage:        0.25–0.55
population:    0.15–0.40
accessibility: 0.10–0.30
hazard:        0.00–0.20
```

Normalize each vector to sum to one.

Measure:

- Spearman rank correlation,
- top-k overlap,
- rank variance,
- probability each cell stays in top decile.

Produce:

```text
outputs/reports/priority_sensitivity.md
outputs/figures/rank_stability_map.png
```

This analysis is required because priority weights are policy assumptions.

---

# 25. Statistical Analysis

## 25.1 Bootstrap confidence intervals

Use appropriate unit:

- event-level bootstrap for broad generalization,
- building-level bootstrap for building metrics.

Do not bootstrap pixels as independent observations.

Report 95% intervals for:

- macro F1,
- mIoU,
- Brier/ECE where sensible,
- generalization gap.

## 25.2 Paired comparisons

For two models evaluated on the same buildings:

```text
Δ metric = proposed - baseline
```

Bootstrap paired differences.

Report point estimate + CI.

## 25.3 Error slices

Analyze by:

- event,
- disaster type,
- class,
- building size,
- geography,
- modality quality if supported.

---

# 26. Optional Prithvi Extension

Only implement after the full V1 pipeline works.

## 26.1 Role

Prithvi should be a **medium-resolution contextual encoder**.

Preferred input:

- xBD-S12 Sentinel-2,
- or correctly formatted HLS.

## 26.2 Feature generation

Example:

```text
Z_prithvi =
Prithvi(pre/post Sentinel/HLS sequence)
```

Aggregate features to regional cells.

Test:

```text
main damage/priority pipeline
vs
main pipeline + Prithvi regional context
```

Possible questions:

- Does it help wide-area damage context?
- Does it improve event-held-out ranking?
- Does it add useful flood/land-surface context?

Do not claim value until evaluated.

---

# 27. Ablations

Required matrix:

| Experiment | Purpose |
|---|---|
| Early-fusion U-Net | simple baseline |
| Pseudo-Siamese | modality-specific baseline |
| DamageFusionFormer without cross-attention | fusion ablation |
| DamageFusionFormer with cross-attention | main model |
| SAR-only | modality ablation |
| Optical + SAR | primary multimodal setting |
| Standard split | benchmark |
| Event-held-out | generalization |
| Uncalibrated | calibration baseline |
| Temperature-scaled | calibration |

Optional:

- shared vs separate encoders,
- natural vs event-aware sampling,
- gated vs attention fusion,
- focal vs benchmark-style loss,
- Prithvi context.

---

# 28. Visualizations

Generate:

1. pre-event optical,
2. post-event SAR,
3. ground truth,
4. predicted damage map,
5. entropy/uncertainty map,
6. confusion matrix,
7. per-event F1/mIoU,
8. reliability diagram,
9. population exposure map,
10. road-risk map,
11. priority map,
12. rank uncertainty/stability map.

All maps must include meaningful legends.

---

# 29. Streamlit Demo

After the research pipeline works, build:

```bash
streamlit run app/streamlit_app.py
```

## Event controls

- dataset,
- event,
- model checkpoint,
- hazard on/off,
- priority-weight sliders.

## Map layers

- pre optical,
- post SAR,
- predicted damage,
- uncertainty,
- population,
- roads/accessibility,
- priority.

## Grid-cell inspector

Show:

```text
estimated population
expected damaged buildings
expected destroyed buildings
accessibility penalty
hazard feature if present
priority score
90% priority interval
probability of top-decile priority
```

## Building inspector

Show:

```text
P(intact)
P(damaged)
P(destroyed)
expected severity
entropy
```

Use cautious wording:

```text
Estimated damage
Estimated accessibility risk
Population exposure estimate
Relative relief priority
Model uncertainty
```

---

# 30. CLI Contract

## Audit

```bash
python scripts/inspect_bright.py data=bright
```

## Manifest

```bash
python scripts/build_manifest.py data=bright
```

## Split

```bash
python scripts/make_splits.py \
  data=bright \
  split=event_holdout \
  split.test_events='[<EVENT_ID>]'
```

## Baseline

```bash
python scripts/train.py \
  data=bright \
  model=unet_baseline \
  experiment=standard_split
```

## Main model

```bash
python scripts/train.py \
  data=bright \
  model=damage_fusion_former \
  experiment=event_holdout
```

## Evaluate

```bash
python scripts/evaluate.py \
  checkpoint=outputs/checkpoints/<CHECKPOINT>.pt \
  split=test
```

## Calibrate

```bash
python scripts/calibrate.py \
  checkpoint=outputs/checkpoints/<CHECKPOINT>.pt
```

## Infer event

```bash
python scripts/infer_event.py \
  checkpoint=outputs/checkpoints/<CHECKPOINT>.pt \
  event_id=<EVENT_ID>
```

## Context

```bash
python scripts/fetch_population.py event_id=<EVENT_ID>
python scripts/fetch_osm.py event_id=<EVENT_ID>
```

## Priority

```bash
python scripts/build_priority_map.py \
  event_id=<EVENT_ID> \
  predictions=outputs/predictions/<EVENT_ID>
```

---

# 31. Configuration

No experiment constants should be buried in Python.

Example:

```yaml
name: damage_fusion_former

encoder:
  name: swin_tiny_patch4_window7_224
  pretrained: true
  share_weights: false

fusion:
  gated_stages: [1, 2]
  cross_attention_stages: [3, 4]
  hidden_dim: 256
  heads: 8
  dropout: 0.1

decoder:
  type: fpn
  channels: 256

heads:
  localization_classes: 2
  damage_classes: 3

loss:
  lambda_loc: 1.0
  lambda_damage: 1.0
```

Training:

```yaml
seed: 42
epochs: 60
batch_size: 8
num_workers: 8
precision: 16

optimizer:
  name: adamw
  lr: 1e-4
  weight_decay: 5e-3

scheduler:
  name: cosine
  warmup_epochs: 3

checkpoint_metric: val/damage_macro_f1
```

Adjust batch size to hardware rather than silently changing semantic resolution.

---

# 32. Logging

Each run saves:

```text
config.yaml
metrics.json
metrics_by_event.csv
class_metrics.csv
split_manifest.parquet
checkpoint.pt
git_commit.txt
environment.txt
```

Optional MLflow/W&B is fine, but local artifacts remain the source of truth.

---

# 33. Tests

## Unit

### Data
- tensor shapes,
- labels valid,
- transforms synchronized,
- raw data not modified.

### Splits
- no event leakage,
- no tile leakage,
- deterministic.

### Models
- forward shape,
- batch size 1,
- mixed-modality handling.

### Metrics
- verify on hand-computed toy examples.

### Calibration
- positive temperature,
- fit uses validation only.

### Geospatial
- CRS transformations,
- raster/vector overlap,
- approximate population-count preservation.

### Priority
- score rises with damage holding other variables fixed,
- weight renormalization works,
- Monte Carlo reproducible with seed.

## Smoke integration

Test:

```text
load → augment → forward → loss → backward → save → reload → infer
```

on a tiny sample.

---

# 34. Milestones

## M0 — Bootstrap

Deliver:

- project structure,
- environment,
- configs,
- tests,
- README shell.

Done when:

```text
pytest passes
package imports
```

## M1 — BRIGHT pipeline

Deliver:

- data audit,
- parser,
- manifest,
- label schema,
- splits,
- synchronized augmentation.

Done when:

- valid DataLoader batch,
- event-leakage tests pass.

## M2 — Simple baseline

Deliver:

- U-Net,
- training engine,
- losses,
- evaluator.

Done when:

- tiny-set overfit succeeds,
- full training can run.

## M3 — Strong multimodal baseline

Deliver:

- pseudo-Siamese model,
- event-aware sampling,
- event metrics.

Done when:

- standard + held-out report generated.

## M4 — DamageFusionFormer

Deliver:

- separate encoders,
- gated fusion,
- cross-attention,
- dual heads.

Done when:

- training is stable,
- fusion ablation exists.

## M5 — Calibration

Deliver:

- building aggregation,
- temperature scaling,
- reliability plots,
- ECE/Brier/NLL.

Done when:

- pre/post calibration report exists.

## M6 — Geospatial context

Deliver:

- WorldPop,
- OSM,
- decision grid,
- accessibility features.

Done when:

- one event yields a complete geospatial feature table.

## M7 — Priority engine

Deliver:

- score,
- Monte Carlo,
- sensitivity analysis,
- map exports.

Done when:

- priority GeoJSON/Parquet + uncertainty outputs exist.

## M8 — Demo

Deliver Streamlit application.

Done when:

- documented fresh launch works from saved artifacts.

## M9 — Extensions

Only now:

- xBD,
- xBD-S12,
- Prithvi,
- few-shot/domain adaptation.

---

# 35. Codex Rules

Treat these as mandatory instructions.

1. **Inspect upstream data/code before parsing.** Do not invent BRIGHT filenames or labels.
2. **Preserve label semantics.** xBD and BRIGHT stay separate.
3. **Baseline first.** No novel model before a working baseline.
4. **No fake metrics.** Use `TBD` until produced.
5. **Prevent event leakage.** Test it.
6. **Auxiliary context is optional.** Damage inference must work without WorldPop/OSM/GPM.
7. **Separate prediction from prioritization.**
8. **Cache external data.**
9. **Fail loudly on unknown labels/missing critical metadata.**
10. **Provide sample/smoke mode.**
11. **Keep training/evaluation config-driven.**
12. **Every reported metric must be reproducible from artifacts.**
13. **Do not overclaim real humanitarian impact.**
14. **Do not implement optional extensions before core milestones pass.**

---

# 36. Strong Experimental Matrix

If compute permits:

| ID | Model | Modalities | Split | Calibration |
|---|---|---|---|---|
| E1 | U-Net | optical + SAR | standard | no |
| E2 | Pseudo-Siamese | optical + SAR | standard | no |
| E3 | DamageFusionFormer | optical + SAR | standard | no |
| E4 | U-Net | SAR-only | event-held-out | no |
| E5 | Pseudo-Siamese | optical + SAR | event-held-out | no |
| E6 | DamageFusionFormer | optical + SAR | event-held-out | no |
| E7 | DamageFusionFormer | optical + SAR | event-held-out | yes |
| E8 | Optional Prithvi/S1/S2 context | multimodal | event-held-out | yes |

Repeat E4–E7 on multiple held-out events if feasible.

---

# 37. Definition of Done

- [ ] environment installs,
- [ ] tests pass,
- [ ] BRIGHT audit generated,
- [ ] loader validated,
- [ ] no split leakage,
- [ ] baseline trains,
- [ ] proposed model trains,
- [ ] standard result saved,
- [ ] event-held-out result saved,
- [ ] per-event metrics saved,
- [ ] building-level calibration evaluated,
- [ ] one event enriched with population,
- [ ] one event enriched with OSM roads,
- [ ] priority map built,
- [ ] priority uncertainty propagated,
- [ ] weight sensitivity analyzed,
- [ ] Streamlit demo works,
- [ ] README commands reproducible,
- [ ] all CV claims trace to saved metrics.

---

# 38. Final Report

Structure:

1. Problem and motivation
2. Datasets and limitations
3. Multimodal model
4. Standard evaluation
5. Event-held-out evaluation
6. Cross-disaster generalization gap
7. Calibration and uncertainty
8. Error analysis
9. Population exposure
10. Accessibility-risk estimation
11. Priority formulation
12. Monte Carlo ranking uncertainty
13. Weight sensitivity
14. Limitations
15. Optional Prithvi/xBD-S12 extension
16. Future work

---

# 39. Resume Positioning

Recommended project title:

**DisasterLens — Multimodal Disaster Damage Assessment & Relief Prioritization**

Potential bullets after real results exist:

- Developed multimodal optical–SAR damage-assessment pipeline using cross-attention fusion, localizing buildings and classifying structural damage across diverse disaster events.
- Evaluated cross-disaster generalization under event-held-out testing, benchmarking architectures using macro-F1, mIoU, calibration, and per-event error analysis.
- Integrated calibrated damage probabilities with population and road-network data to generate uncertainty-aware regional relief-priority rankings with sensitivity analysis.

Do not add percentages until experiments support them.

---

# 40. Known Limitations to Preserve

- optical/SAR residual registration error,
- visual damage-label ambiguity/noise,
- strong class imbalance,
- event imbalance,
- geographic/domain shift,
- limited sensor coverage,
- approximate population data,
- estimated rather than confirmed road disruption,
- heuristic priority weights,
- retrospective benchmark rather than live disaster deployment.

These limitations should appear in the final README/report.

---

# 41. Exact Implementation Order

Codex should work in this order:

```text
1. Bootstrap
2. Inspect official BRIGHT structure
3. Build data audit
4. Implement BRIGHT loader
5. Implement event splits + leakage tests
6. Implement U-Net
7. Tiny-set overfit
8. Train baseline
9. Implement pseudo-Siamese baseline
10. Evaluate standard + held-out
11. Implement DamageFusionFormer
12. Run architecture ablation
13. Add building-level aggregation
14. Add temperature scaling
15. Add WorldPop
16. Add OSM road graph
17. Add priority features
18. Add Monte Carlo uncertainty
19. Add weight sensitivity
20. Build Streamlit demo
21. Only then add Prithvi/xBD-S12
```

---

# 42. First Prompt to Give Codex

```text
Read IMPLEMENTATION_SPEC.md completely before editing anything.

Implement only Milestones M0 and M1 first.

Before writing the BRIGHT loader, inspect the actual BRIGHT dataset directory and the official BRIGHT source repository if present. Verify real filenames, modalities, metadata, label IDs, and geospatial information. Do not assume them from memory.

Then:
1. bootstrap the Python project/configuration structure,
2. implement common data and label schemas,
3. implement the BRIGHT manifest and data-audit pipeline,
4. implement deterministic standard and event-held-out split infrastructure,
5. add tests for label parsing, tensor shapes, synchronized transforms, duplicate tiles, and event leakage,
6. add a tiny sample/smoke-test mode.

Do not implement DamageFusionFormer, Prithvi, the relief-priority engine, or Streamlit yet.

At the end, report:
- files created/modified,
- install commands,
- audit command,
- test command/results,
- verified dataset structure,
- assumptions or information that could not be verified.
```

---

# 43. Upstream References

## BRIGHT

Paper:  
https://essd.copernicus.org/articles/17/6217/2025/

Official repository:  
https://github.com/ChenHongruixuan/BRIGHT

Dataset record:  
https://zenodo.org/records/14619798

## xBD

Paper:  
https://arxiv.org/abs/1911.09296

xView2 project:  
https://www.sei.cmu.edu/projects/xview-2-challenge/

Unseen-location/generalization baseline:  
https://arxiv.org/abs/2401.17271

## xBD-S12

Official repository:  
https://github.com/prs-eth/xbd-s12

Dataset:  
https://zenodo.org/records/18960454

Paper:  
https://arxiv.org/abs/2511.05461

## Prithvi-EO-2.0

Official model:  
https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M

Official NASA examples:  
https://github.com/NASA-IMPACT/Prithvi-EO-2.0

## Population

WorldPop:  
https://www.worldpop.org/

## Roads

OpenStreetMap:  
https://www.openstreetmap.org/

## Rainfall

NASA GPM IMERG:  
https://gpm.nasa.gov/data/imerg

---

# 44. Final Product Statement

The project should ultimately be described as:

> A multimodal geospatial decision-support system that evaluates cross-disaster building-damage generalization, calibrates predictive uncertainty, and combines damage estimates with population exposure and accessibility risk for transparent relief prioritization.

The project's strongest feature is not the use of a Transformer by itself.

Its strongest feature is the methodological completeness:

```text
multimodal CV
+ domain shift
+ uncertainty calibration
+ geospatial analytics
+ transparent decision science
```

That is the standard Codex should optimize for.
