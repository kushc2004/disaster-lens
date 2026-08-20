from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def lovasz_softmax_flat(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Multiclass Lovasz-Softmax loss over valid pixels."""
    if probabilities.numel() == 0:
        return probabilities.sum() * 0.0
    losses: list[torch.Tensor] = []
    for class_id in range(probabilities.shape[1]):
        foreground = (labels == class_id).to(probabilities.dtype)
        if not foreground.any():
            continue
        errors = (foreground - probabilities[:, class_id]).abs()
        errors, permutation = torch.sort(errors, descending=True)
        foreground = foreground[permutation]
        intersection = foreground.sum() - foreground.cumsum(0)
        union = foreground.sum() + (1 - foreground).cumsum(0)
        gradient = 1 - intersection / union.clamp_min(1)
        # Do not use ``gradient[1:] -= gradient[:-1]``: recent PyTorch
        # releases reject the overlapping source and destination views.
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
        losses.append(torch.dot(errors, gradient))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0


class SegmentationLoss(nn.Module):
    def __init__(self, *, class_weights: torch.Tensor | None = None, ignore_index: int = 255, lovasz_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index, self.lovasz_weight = ignore_index, lovasz_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        probabilities = logits.softmax(dim=1).permute(0, 2, 3, 1)[valid]
        return ce + self.lovasz_weight * lovasz_softmax_flat(probabilities, target[valid])


class DualHeadSegmentationLoss(nn.Module):
    """Localization plus conditional building-severity loss for BRIGHT."""

    def __init__(
        self,
        *,
        damage_class_weights: torch.Tensor | None = None,
        ignore_index: int = 255,
        lovasz_weight: float = 1.0,
        lambda_localization: float = 1.0,
        lambda_damage: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("damage_class_weights", damage_class_weights)
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        self.lambda_localization = lambda_localization
        self.lambda_damage = lambda_damage

    def _component(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = target != self.ignore_index
        # A synchronized crop can legitimately contain no labelled buildings.
        # Cross-entropy over an all-ignore target returns NaN, so make that
        # head contribute a differentiable zero instead.
        if not bool(valid.any()):
            return logits.sum() * 0.0
        ce = F.cross_entropy(
            logits, target, weight=weights, ignore_index=self.ignore_index
        )
        probabilities = logits.softmax(dim=1).permute(0, 2, 3, 1)[valid]
        return ce + self.lovasz_weight * lovasz_softmax_flat(
            probabilities, target[valid]
        )

    def forward(
        self, outputs: dict[str, torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        if set(outputs) < {"localization", "damage"}:
            raise ValueError("Dual-head output lacks localization or damage logits")
        valid = target != self.ignore_index
        localization_target = torch.where(
            valid, (target > 0).long(), torch.full_like(target, self.ignore_index)
        )
        damage_target = torch.where(
            valid & (target > 0), target - 1, torch.full_like(target, self.ignore_index)
        )
        localization = self._component(
            outputs["localization"], localization_target
        )
        damage = self._component(
            outputs["damage"], damage_target, self.damage_class_weights
        )
        return self.lambda_localization * localization + self.lambda_damage * damage
