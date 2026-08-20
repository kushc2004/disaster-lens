# Cross-Disaster Building Damage Assessment — Focused Implementation Spec

## 1. Objective

Build and evaluate a multimodal building-damage segmentation system using paired BRIGHT inputs:

- pre-event optical imagery;
- post-event SAR imagery; and
- four damage labels: no damage, minor, major, and destroyed.

The primary question is whether performance remains reliable on disaster events excluded from training. If the audited metadata contains enough independent events per disaster type, run an additional leave-one-disaster-type-out experiment. Do not describe an event-held-out result as an unseen-type result.

## 2. Relationship to the root specification

`../DisasterLens_IMPLEMENTATION_SPEC.md` remains the authoritative technical source. This document narrows it to the CV project and maps the reused implementation:

| Focused phase | Root milestone | Reused implementation |
| --- | --- | --- |
| Foundation | M0 | package, configuration, tests, environment |
| Data integrity | M1 | audit, BRIGHT parser, manifest, schemas, event-safe splits, transforms |
| Baseline | M2 | early-fusion U-Net, losses, training engine, evaluator |
| Cross-disaster baseline | M3 | pseudo-Siamese ResNet-18, event-aware sampling, event metrics |
| Calibration | M5 | validation-only temperature scaling and reliability metrics |

Do not fork or modify the root M0–M2 artifacts for this project. Store all newly generated training outputs under a clearly named output directory and preserve the exact split manifest and resolved configuration beside every result.

## 3. Experiment design

### E1 — Early-fusion U-Net

Train the existing U-Net on the standard split and event-held-out split. Report macro F1, class-wise F1, mIoU, confusion matrix, and metrics for each held-out event.

### E2 — Pseudo-Siamese ResNet-18

Train the existing pseudo-Siamese baseline on the exact same manifests. It uses separate branches for the two modalities before combining their features. Compare it against E1 with identical preprocessing, epochs, and evaluation protocol.

### E3 — Cross-disaster analysis

For each held-out event, report the change from the standard split. Slice errors by event and by audited disaster type. Only add a leave-one-type-out split after the audit proves that each type has enough events to leave train, validation, and test groups non-empty.

### E4 — Calibration

Fit temperature scaling on validation predictions only. Then evaluate the untouched held-out event predictions with ECE, Brier score, NLL, and a reliability diagram. Report pre- and post-calibration metrics together.

### Optional E5 — Transformer extension

Add a ChangeFormer-style Siamese Transformer only after E1–E4 have saved artifacts. Keep its data manifests, parameter count, training budget, and augmentation policy comparable to E1/E2. It is an extension, not a prerequisite for a complete CV project.

## 4. Acceptance criteria

- The BRIGHT audit records event identifiers, locations, modalities, labels, and disaster-type metadata when available.
- No event appears in more than one split; tests enforce this invariant.
- Each experiment saves its resolved config, immutable split manifest, checkpoint, training history, class metrics, event metrics, and confusion matrix.
- Calibration never reads held-out labels while fitting the temperature.
- A conclusion on generalization is supported by saved held-out metrics, not a random-split score.
- No model-quality, calibration, or generalization result is claimed until the corresponding run completes and artifacts exist.

## 5. Deliverable and CV wording

The finished project should present one question, one reproducible evaluation protocol, and one bounded conclusion. Use this wording only after E1–E4 complete:

> Evaluated bi-temporal optical/SAR building-damage segmentation under event-held-out testing, comparing U-Net and pseudo-Siamese models with calibrated uncertainty estimates.

Do not claim unseen disaster-type generalization unless the leave-one-type-out experiment was actually run.
