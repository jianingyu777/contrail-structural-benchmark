from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, self.dropout_skip(skip)], dim=1)
        return self.conv2(self.conv1(x))


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: tuple[int, ...] = (384, 192, 96, 64),
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        in_channels = [encoder_channels[0], *decoder_channels[:-1]]
        skip_channels = [*encoder_channels[1:], 0]
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm, dropout)
                for in_ch, skip_ch, out_ch in zip(
                    in_channels, skip_channels, decoder_channels
                )
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]
        x = self.center(features[0])
        skips = features[1:]
        for index, block in enumerate(self.blocks):
            x = block(x, skips[index] if index < len(skips) else None)
        return x
