from __future__ import annotations

import torch

from disasterlens.models.losses import SegmentationLoss


def test_segmentation_loss_supports_backward_pass():
    logits = torch.randn(2, 4, 3, 3, requires_grad=True)
    target = torch.tensor(
        [
            [[0, 1, 2], [3, 0, 1], [2, 3, 0]],
            [[1, 2, 3], [0, 1, 2], [3, 0, 1]],
        ]
    )
    loss = SegmentationLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
