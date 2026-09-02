from __future__ import annotations

from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError("Soft morphology expects NCHW tensors")
    vertical = -F.max_pool2d(-image, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeleton(image: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    work = image
    for _ in range(iterations):
        work = _soft_erode(work)
        opened = _soft_open(work)
        delta = F.relu(work - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 20,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predicted_skeleton = soft_skeleton(probabilities, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    dimensions = (1, 2, 3)
    topology_precision = (predicted_skeleton * target).sum(dimensions) / (
        predicted_skeleton.sum(dimensions) + smooth
    )
    topology_sensitivity = (target_skeleton * probabilities).sum(dimensions) / (
        target_skeleton.sum(dimensions) + smooth
    )
    cldice = (2.0 * topology_precision * topology_sensitivity + smooth) / (
        topology_precision + topology_sensitivity + smooth
    )
    return 1.0 - cldice.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 1.0,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    true_positive = (probabilities * target).sum(dimensions)
    false_positive = (probabilities * (1.0 - target)).sum(dimensions)
    false_negative = ((1.0 - probabilities) * target).sum(dimensions)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - score


def curriculum_weight(progress: float, start: float, maximum: float) -> float:
    if progress <= start:
        return 0.0
    fraction = min(1.0, (progress - start) / max(1e-12, 1.0 - start))
    return float(maximum * fraction)


@dataclass(frozen=True)
class LossConfig:
    positive_class_weight: float = 5.0
    background_sample_scale: float = 2.5
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_weight: float = 1.0
    curriculum_start_fraction: float = 0.2
    maximum_cldice_weight: float = 0.5
    maximum_centerline_weight: float = 0.3
    skeleton_iterations: int = 20


class CompositeContrailLoss:
    """StageA batch loss recovered from the historical training source."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config
        self.tversky = smp.losses.TverskyLoss(
            mode="binary",
            from_logits=True,
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
        )

    def __call__(
        self,
        mask_logits: torch.Tensor,
        mask_target: torch.Tensor,
        centerline_logits: torch.Tensor,
        centerline_target: torch.Tensor,
        progress: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pos_weight = torch.as_tensor(
            [self.config.positive_class_weight],
            dtype=mask_logits.dtype,
            device=mask_logits.device,
        )
        bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            mask_target,
            pos_weight=pos_weight,
        )
        if mask_target.sum() == 0:
            segmentation = self.config.background_sample_scale * bce
            tversky = mask_logits.sum() * 0.0
        else:
            tversky = self.tversky(mask_logits, mask_target)
            segmentation = bce + self.config.tversky_weight * tversky

        centerline_bce = F.binary_cross_entropy_with_logits(
            centerline_logits,
            centerline_target,
        )
        cldice_weight = curriculum_weight(
            progress,
            self.config.curriculum_start_fraction,
            self.config.maximum_cldice_weight,
        )
        centerline_weight = curriculum_weight(
            progress,
            self.config.curriculum_start_fraction,
            self.config.maximum_centerline_weight,
        )
        if cldice_weight > 0:
            cldice = soft_cldice_loss(
                mask_logits,
                mask_target,
                self.config.skeleton_iterations,
            )
        else:
            cldice = mask_logits.sum() * 0.0
        total = segmentation + cldice_weight * cldice
        total = total + centerline_weight * centerline_bce
        components = {
            "total": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "tversky": float(tversky.detach().cpu()),
            "cldice": float(cldice.detach().cpu()),
            "centerline_bce": float(centerline_bce.detach().cpu()),
            "cldice_weight": cldice_weight,
            "centerline_weight": centerline_weight,
        }
        return total, components
