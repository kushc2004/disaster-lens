"""Pixel-level segmentation metrics with no hidden third-party state."""

from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(logits: torch.Tensor, target: torch.Tensor, *, num_classes: int = 4, ignore_index: int = 255) -> torch.Tensor:
    prediction = logits.argmax(dim=1).reshape(-1)
    target = target.reshape(-1)
    valid = target != ignore_index
    prediction, target = prediction[valid], target[valid]
    encoded = target * num_classes + prediction
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes).float()


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float | list[float] | list[list[int]]]:
    tp = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    iou = tp / (support + predicted - tp).clamp_min(1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    nonempty = support > 0
    loc_tp = confusion[1:, 1:].diag().sum()
    loc_f1 = 2 * loc_tp / (confusion[1:, :].sum() + confusion[:, 1:].sum()).clamp_min(1)
    damage_nonempty = support[1:] > 0
    damage_f1 = f1[1:][damage_nonempty].mean() if damage_nonempty.any() else torch.tensor(0.0, device=confusion.device)
    return {
        "accuracy": float(tp.sum() / confusion.sum().clamp_min(1)),
        "miou": float(iou[nonempty].mean()) if nonempty.any() else 0.0,
        "macro_f1": float(f1[nonempty].mean()) if nonempty.any() else 0.0,
        "class_iou": [float(value) for value in iou], "class_f1": [float(value) for value in f1],
        "f1_localization": float(loc_f1), "f1_damage": float(damage_f1),
        "confusion_matrix": [[int(value) for value in row] for row in confusion],
    }


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, *, num_classes: int = 4, ignore_index: int = 255) -> dict[str, float | list[float] | list[list[int]]]:
    return metrics_from_confusion(confusion_matrix(logits, target, num_classes=num_classes, ignore_index=ignore_index))
