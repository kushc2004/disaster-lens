"""M3 pseudo-Siamese baseline and M4 DamageFusionFormer."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .baselines import DoubleConv


class ModalityEncoder(nn.Module):
    """Compact four-stage encoder; instances never share weights."""

    def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        previous = in_channels
        for width in channels:
            blocks.append(DoubleConv(previous, width))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        value = image
        for index, block in enumerate(self.blocks):
            if index:
                value = self.pool(value)
            value = block(value)
            features.append(value)
        return features


class PseudoSiameseUNet(nn.Module):
    """Separate optical/SAR encoders with stage-wise learned fusion."""

    def __init__(self, *, num_classes: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.optical_encoder = ModalityEncoder(3, channels)
        self.sar_encoder = ModalityEncoder(1, channels)
        self.fuse = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for width in channels])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DoubleConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DoubleConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DoubleConv(channels[0] * 2, channels[0])
        self.head = nn.Conv2d(channels[0], num_classes, 1)

    @staticmethod
    def _up(up: nn.Module, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((value, skip), dim=1)

    def forward(self, pre_optical: torch.Tensor, post_sar: torch.Tensor) -> torch.Tensor:
        optical = self.optical_encoder(pre_optical)
        sar = self.sar_encoder(post_sar)
        fused = [layer(torch.cat(pair, dim=1)) for layer, pair in zip(self.fuse, zip(optical, sar), strict=True)]
        value = self.dec3(self._up(self.up3, fused[3], fused[2]))
        value = self.dec2(self._up(self.up2, value, fused[1]))
        value = self.dec1(self._up(self.up1, value, fused[0]))
        return self.head(value)


class GatedFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.project = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        pair = torch.cat((optical, sar), dim=1)
        gate = self.gate(pair)
        mixed = gate * optical + (1.0 - gate) * sar
        return mixed + self.project(pair)


class SpatialCrossAttention(nn.Module):
    """Bidirectional cross-attention with bounded token count."""

    def __init__(self, channels: int, heads: int = 8, dropout: float = 0.1, max_tokens: int = 1024) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("attention channels must be divisible by heads")
        self.optical_to_sar = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.sar_to_optical = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        # Preserve both modality features as well as the two directed
        # attention responses.  Difference and product terms make explicit
        # the two most useful change-detection interactions before the learned
        # projection, matching the M4 fusion contract.
        self.project = nn.Sequential(
            nn.Conv2d(channels * 6, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.max_tokens = max_tokens

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        height, width = optical.shape[-2:]
        scale = min(1.0, math.sqrt(self.max_tokens / max(1, height * width)))
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        left = F.adaptive_avg_pool2d(optical, size).flatten(2).transpose(1, 2)
        right = F.adaptive_avg_pool2d(sar, size).flatten(2).transpose(1, 2)
        left_attended, _ = self.optical_to_sar(left, right, right, need_weights=False)
        right_attended, _ = self.sar_to_optical(right, left, left, need_weights=False)
        left = self.norm(left + left_attended).transpose(1, 2).reshape(optical.shape[0], optical.shape[1], *size)
        right = self.norm(right + right_attended).transpose(1, 2).reshape(sar.shape[0], sar.shape[1], *size)
        left = F.interpolate(left, size=(height, width), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(height, width), mode="bilinear", align_corners=False)
        return self.project(
            torch.cat(
                (optical, sar, left, right, torch.abs(optical - sar), optical * sar),
                dim=1,
            )
        )


class FPNDecoder(nn.Module):
    """Top-down multiscale decoder with a full-resolution feature pyramid."""

    def __init__(self, encoder_channels: tuple[int, ...], output_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(width, output_channels, 1) for width in encoder_channels]
        )
        self.smooth = nn.ModuleList(
            [DoubleConv(output_channels, output_channels) for _ in encoder_channels]
        )
        self.aggregate = DoubleConv(
            output_channels * len(encoder_channels), output_channels
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral):
            raise ValueError(
                f"Decoder expected {len(self.lateral)} feature stages, found {len(features)}"
            )
        pyramid: list[torch.Tensor] = [features[0]] * len(features)
        top_down: torch.Tensor | None = None
        for index in reversed(range(len(features))):
            lateral = self.lateral[index](features[index])
            if top_down is not None:
                top_down = F.interpolate(
                    top_down,
                    size=lateral.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                lateral = lateral + top_down
            top_down = self.smooth[index](lateral)
            pyramid[index] = top_down
        target_size = pyramid[0].shape[-2:]
        aligned = [
            feature
            if feature.shape[-2:] == target_size
            else F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            for feature in pyramid
        ]
        return self.aggregate(torch.cat(aligned, dim=1))


def _compatible_heads(channels: int, requested: int) -> int:
    """Choose the greatest requested-or-smaller divisor of the channel width."""
    for heads in range(min(channels, requested), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class DamageFusionFormer(nn.Module):
    """Separate encoders, shallow gated fusion, deep cross-attention, dual heads."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        heads: int = 8,
        dropout: float = 0.1,
        decoder_channels: int = 128,
        ablation: str = "full",
    ) -> None:
        super().__init__()
        if ablation not in {"full", "gated_only", "attention_only", "concat", "sar_only"}:
            raise ValueError(f"unknown fusion ablation: {ablation}")
        channels = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        self.ablation = ablation
        self.optical_encoder = ModalityEncoder(3, channels)
        self.sar_encoder = ModalityEncoder(1, channels)
        self.gates = nn.ModuleList([GatedFusion(width) for width in channels])
        self.attention = nn.ModuleList(
            [
                SpatialCrossAttention(
                    width,
                    heads=_compatible_heads(width, heads),
                    dropout=dropout,
                )
                for width in channels
            ]
        )
        self.concat = nn.ModuleList([nn.Conv2d(width * 2, width, 1) for width in channels])
        self.decoder = FPNDecoder(channels, decoder_channels)
        self.localization_head = nn.Conv2d(decoder_channels, 2, 1)
        self.damage_head = nn.Conv2d(decoder_channels, 3, 1)

    def _fuse(self, stage: int, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        if self.ablation == "concat":
            return self.concat[stage](torch.cat((optical, sar), dim=1))
        if self.ablation == "gated_only" or (self.ablation == "full" and stage < 2):
            return self.gates[stage](optical, sar)
        if self.ablation == "attention_only" or self.ablation == "full":
            return self.attention[stage](optical, sar)
        raise AssertionError("unreachable fusion mode")

    def forward(self, pre_optical: torch.Tensor, post_sar: torch.Tensor) -> dict[str, torch.Tensor]:
        sar = self.sar_encoder(post_sar)
        if self.ablation == "sar_only":
            # A true modality ablation: optical pixels and optical encoder
            # parameters cannot influence the output.
            fused = sar
        else:
            optical = self.optical_encoder(pre_optical)
            fused = [self._fuse(index, left, right) for index, (left, right) in enumerate(zip(optical, sar, strict=True))]
        value = self.decoder(fused)
        return {"localization": self.localization_head(value), "damage": self.damage_head(value)}


def dual_heads_to_four_class(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compose background and conditional building-damage log probabilities."""
    localization = F.log_softmax(outputs["localization"], dim=1)
    damage = F.log_softmax(outputs["damage"], dim=1)
    return torch.cat((localization[:, :1], localization[:, 1:2] + damage), dim=1)
