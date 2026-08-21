from __future__ import annotations

import torch
import pytest

from disasterlens.eval import segmentation_metrics
from disasterlens.models import EarlyFusionUNet, PseudoSiameseUNet


def test_early_fusion_unet_preserves_spatial_shape_for_batch_one():
    model = EarlyFusionUNet(in_channels=4, num_classes=4, base_channels=4)
    logits = model(torch.randn(1, 3, 17, 19), torch.randn(1, 1, 17, 19))
    assert logits.shape == (1, 4, 17, 19)


def test_pseudo_siamese_resnet18_uses_independent_optical_and_sar_branches():
    model = PseudoSiameseUNet(num_classes=4, base_channels=4, encoder="resnet18").eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 32, 40), torch.randn(1, 1, 32, 40))
    assert logits.shape == (1, 4, 32, 40)
    assert model.optical_encoder.stem[0].in_channels == 3
    assert model.sar_encoder.stem[0].in_channels == 1
    assert {id(parameter) for parameter in model.optical_encoder.parameters()}.isdisjoint(
        {id(parameter) for parameter in model.sar_encoder.parameters()}
    )


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
