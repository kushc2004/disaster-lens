"""Small, dependency-light segmentation baselines for M2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class EarlyFusionUNet(nn.Module):
    """Early-fusion U-Net for pre-event RGB plus post-event single-band SAR."""

    def __init__(self, in_channels: int = 4, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes < 2 or base_channels <= 0:
            raise ValueError("in_channels, num_classes, and base_channels must be positive")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.enc1, self.enc2 = DoubleConv(in_channels, channels[0]), DoubleConv(channels[0], channels[1])
        self.enc3, self.bottleneck = DoubleConv(channels[1], channels[2]), DoubleConv(channels[2], channels[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0])
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    @staticmethod
    def _up_to(up: nn.Module, encoded: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoded = up(encoded)
        if decoded.shape[-2:] != skip.shape[-2:]:
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, decoded), dim=1)

    def forward(self, pre_optical: torch.Tensor, post_sar: torch.Tensor | None = None) -> torch.Tensor:
        x = pre_optical if post_sar is None else torch.cat((pre_optical, post_sar), dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(self._up_to(self.up3, b, e3))
        d2 = self.dec2(self._up_to(self.up2, d3, e2))
        d1 = self.dec1(self._up_to(self.up1, d2, e1))
        return self.head(d1)
