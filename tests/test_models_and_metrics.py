from __future__ import annotations

import torch
import pytest

from disasterlens.eval import segmentation_metrics
from disasterlens.models import EarlyFusionUNet


def test_early_fusion_unet_preserves_spatial_shape_for_batch_one():
    model = EarlyFusionUNet(in_channels=4, num_classes=4, base_channels=4)
    logits = model(torch.randn(1, 3, 17, 19), torch.randn(1, 1, 17, 19))
    assert logits.shape == (1, 4, 17, 19)


def test_segmentation_metrics_match_a_hand_computed_example():
    # Targets: [background, intact, damaged, destroyed].  The prediction gets
    # exactly the first three pixels right and calls destroyed background.
    target = torch.tensor([[[0, 1, 2, 3]]])
    logits = torch.tensor(
        [[[[8.0, 0.0, 0.0, 8.0]], [[0.0, 8.0, 0.0, 0.0]], [[0.0, 0.0, 8.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]]
    )
    metrics = segmentation_metrics(logits, target)
    assert metrics["accuracy"] == 0.75
    assert metrics["class_f1"] == pytest.approx([2 / 3, 1.0, 1.0, 0.0])
    assert metrics["f1_damage"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
