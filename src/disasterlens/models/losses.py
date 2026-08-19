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
        gradient[1:] -= gradient[:-1]
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
