from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.initialization import initialize_decoder

from .decoder import UnetDecoder


def _check_reductions(reductions: list[int]) -> None:
    previous = 1
    for reduction in reductions:
        if reduction / previous != 2:
            raise ValueError(
                f"Encoder reductions must double at every stage: {reductions}"
            )
        previous = reduction


class ContrailMaxViTUNet(nn.Module):
    """MaxViT-Base encoder with a four-stage U-Net-like decoder."""

    def __init__(
        self,
        encoder_name: str = "maxvit_base_tf_512.in21k_ft_in1k",
        pretrained: bool = False,
        decoder_channels: tuple[int, ...] = (384, 192, 96, 64),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        dropout: float = 0.0,
        auxiliary_centerline: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=pretrained,
            out_indices=out_indices,
        )
        encoder_channels = self.encoder.feature_info.channels()
        _check_reductions(self.encoder.feature_info.reduction())
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError(
                "Encoder and decoder must expose the same number of stages"
            )

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=None,
            kernel_size=3,
        )
        self.centerline_head = (
            smp.base.SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=1,
                activation=None,
                kernel_size=3,
            )
            if auxiliary_centerline
            else None
        )
        initialize_decoder(self.decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        decoded = self.decoder(self.encoder(x))
        mask_logits = self.segmentation_head(decoded)
        if self.centerline_head is None:
            return mask_logits
        return {
            "mask": mask_logits,
            "centerline": self.centerline_head(decoded),
        }


def mask_logits(output: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if "mask" in output:
        return output["mask"]
    if "out" in output:
        return output["out"]
    raise KeyError("Model output does not contain 'mask' or 'out' logits")


def clean_state_dict(
    payload: object, include_centerline: bool = False
) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], Mapping):
                payload = payload[key]
                break
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must be a state dictionary or contain one")

    cleaned: dict[str, torch.Tensor] = {}
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("centerline_head") and not include_centerline:
            continue
        if isinstance(value, torch.Tensor):
            cleaned[key] = value
    return cleaned


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    include_centerline = getattr(model, "centerline_head", None) is not None
    return model.load_state_dict(
        clean_state_dict(payload, include_centerline=include_centerline),
        strict=strict,
    )
