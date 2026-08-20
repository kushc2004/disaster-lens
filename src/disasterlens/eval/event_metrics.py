"""Natural-distribution, event-stratified BRIGHT evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from disasterlens.models.multimodal import dual_heads_to_four_class

from .segmentation import confusion_matrix, metrics_from_confusion


class CalibrationBins:
    def __init__(self, bins: int = 15) -> None:
        self.edges = np.linspace(0.0, 1.0, bins + 1)
        self.count = np.zeros(bins, dtype=np.int64)
        self.confidence = np.zeros(bins, dtype=np.float64)
        self.correct = np.zeros(bins, dtype=np.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = logits.softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
        valid = target != 255
        values = confidence[valid].detach().cpu().numpy()
        correct = (prediction[valid] == target[valid]).detach().cpu().numpy()
        indices = np.minimum(np.digitize(values, self.edges[1:-1]), len(self.count) - 1)
        for index in range(len(self.count)):
            selected = indices == index
            if selected.any():
                self.count[index] += int(selected.sum())
                self.confidence[index] += float(values[selected].sum())
                self.correct[index] += float(correct[selected].sum())

    def ece(self) -> float:
        total = int(self.count.sum())
        if not total:
            return 0.0
        occupied = self.count > 0
        accuracy = np.divide(self.correct, self.count, where=occupied)
        confidence = np.divide(self.confidence, self.count, where=occupied)
        return float(np.sum(self.count[occupied] * np.abs(accuracy[occupied] - confidence[occupied])) / total)


def _semantic(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs if isinstance(outputs, torch.Tensor) else dual_heads_to_four_class(outputs)


def _building_count(mask: np.ndarray) -> int:
    structure = np.ones((3, 3), dtype=np.uint8)
    return int(ndimage.label((mask > 0) & (mask != 255), structure=structure)[1])


def _event_bootstrap(
    matrices: dict[str, torch.Tensor], *, samples: int = 1_000, seed: int = 42
) -> dict[str, Any]:
    """Bootstrap pooled segmentation metrics using events as the sampling unit.

    Pixels and tiles within an event are correlated, so they are deliberately
    never treated as independent bootstrap observations.
    """
    event_matrices = [matrices[event_id].cpu() for event_id in sorted(matrices)]
    generator = np.random.default_rng(seed)
    tracked = ("macro_f1", "miou", "f1_localization", "f1_damage")
    draws = {metric: np.empty(samples, dtype=np.float64) for metric in tracked}
    for draw in range(samples):
        indices = generator.integers(0, len(event_matrices), size=len(event_matrices))
        matrix = sum(
            (event_matrices[int(index)] for index in indices),
            torch.zeros_like(event_matrices[0]),
        )
        metrics = metrics_from_confusion(matrix)
        for metric in tracked:
            draws[metric][draw] = float(metrics[metric])
    return {
        "unit": "event",
        "n_units": len(event_matrices),
        "samples": samples,
        "seed": seed,
        "metrics": {
            metric: {
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in draws.items()
        },
    }


@torch.no_grad()
def evaluate_by_event(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    disaster_types: dict[str, str],
    predictions_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate pooled and event metrics, optionally persisting raw logits."""
    model.eval()
    matrices: dict[str, torch.Tensor] = {}
    calibration: dict[str, CalibrationBins] = defaultdict(CalibrationBins)
    tile_counts: dict[str, int] = defaultdict(int)
    building_counts: dict[str, int] = defaultdict(int)
    pooled_bins = CalibrationBins()
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
    saved_visual_example = False

    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        pre = batch["images"]["pre_optical"].to(device, non_blocking=True)
        sar = batch["images"]["post_sar"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        logits = _semantic(model(pre, sar))
        for row, (event_id, tile_id) in enumerate(
            zip(batch["event_id"], batch["tile_id"], strict=True)
        ):
            event_logits = logits[row : row + 1]
            event_target = targets[row : row + 1]
            matrix = confusion_matrix(event_logits.cpu(), event_target.cpu())
            matrices[event_id] = matrices.get(event_id, torch.zeros_like(matrix)) + matrix
            calibration[event_id].update(event_logits, event_target)
            pooled_bins.update(event_logits, event_target)
            tile_counts[event_id] += 1
            target_numpy = event_target[0].detach().cpu().numpy()
            building_counts[event_id] += _building_count(target_numpy)
            if predictions_dir is not None:
                geo = batch["geo"][row]
                crs = geo.get("crs") or ""
                bounds = geo.get("bounds")
                payload: dict[str, np.ndarray] = {
                    "logits": event_logits[0].float().cpu().numpy(),
                    "mask": target_numpy,
                    "event_id": np.asarray(event_id),
                    "tile_id": np.asarray(tile_id),
                    "crs": np.asarray(crs),
                    "bounds": np.asarray(
                        bounds if bounds is not None else (), dtype=np.float64
                    ),
                }
                if not saved_visual_example:
                    payload.update(
                        pre_optical=pre[row].float().cpu().numpy(),
                        post_sar=sar[row].float().cpu().numpy(),
                    )
                    saved_visual_example = True
                np.savez_compressed(
                    predictions_dir / f"{tile_id}.npz",
                    **payload,
                )
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_batches:
            print(f"[event-evaluation] batch {batch_index}/{total_batches}", flush=True)

    if not matrices:
        raise ValueError("Evaluation partition is empty")
    pooled = sum(matrices.values(), torch.zeros_like(next(iter(matrices.values()))))
    pooled_metrics = metrics_from_confusion(pooled)
    pooled_metrics["ece"] = pooled_bins.ece()
    # An event-level bootstrap needs more than one independent event.  With a
    # singleton held-out event every resample is identical, so percentile
    # bounds would look precise while conveying no uncertainty information.
    if len(matrices) >= 2:
        pooled_metrics["bootstrap_95_ci"] = _event_bootstrap(matrices)
    else:
        pooled_metrics["bootstrap_95_ci"] = {
            "available": False,
            "reason": "event-level bootstrap requires at least two held-out events",
            "unit": "event",
            "n_units": len(matrices),
        }
    per_event: list[dict[str, Any]] = []
    for event_id in sorted(matrices):
        values = metrics_from_confusion(matrices[event_id])
        per_event.append(
            {
                "event_id": event_id,
                "disaster_type": disaster_types.get(event_id, "unknown"),
                "n_tiles": tile_counts[event_id],
                "n_buildings": building_counts[event_id],
                "miou": values["miou"],
                "macro_f1": values["macro_f1"],
                "f1_localization": values["f1_localization"],
                "f1_damage": values["f1_damage"],
                "ece": calibration[event_id].ece(),
                "class_iou": values["class_iou"],
                "class_f1": values["class_f1"],
                "confusion_matrix": values["confusion_matrix"],
            }
        )
    summary: dict[str, Any] = {}
    for metric in ("miou", "macro_f1", "f1_localization", "f1_damage", "ece"):
        numbers = np.asarray([float(row[metric]) for row in per_event])
        summary[metric] = {
            "event_macro": float(numbers.mean()),
            "worst_event": float(numbers.max() if metric == "ece" else numbers.min()),
            "std": float(numbers.std(ddof=0)),
        }
    return {"pooled": pooled_metrics, "event_summary": summary, "per_event": per_event}
