from __future__ import annotations

from collections.abc import Iterable

import torch


class ThresholdMetrics:
    """Accumulate pixel and patch metrics for one or more thresholds."""

    def __init__(self, thresholds: Iterable[float]) -> None:
        self.thresholds = [float(value) for value in thresholds]
        self.counts: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, probabilities: torch.Tensor, masks: torch.Tensor) -> None:
        if self.counts is None:
            self.counts = {
                key: torch.zeros(
                    len(self.thresholds),
                    dtype=torch.float64,
                    device=probabilities.device,
                )
                for key in (
                    "tp",
                    "fp",
                    "tn",
                    "fn",
                    "positive_tp",
                    "positive_fp",
                    "positive_fn",
                    "positive_iou_sum",
                    "positive_patches",
                    "negative_patches",
                    "negative_fp_patches",
                )
            }
        threshold_tensor = torch.as_tensor(
            self.thresholds,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        target = masks > 0.5
        positive = target.flatten(1).any(1)
        predictions = (
            probabilities.unsqueeze(0) >= threshold_tensor[:, None, None, None, None]
        )
        expanded_target = target.unsqueeze(0)
        all_dimensions = (1, 2, 3, 4)
        self.counts["tp"] += (predictions & expanded_target).sum(all_dimensions)
        self.counts["fp"] += (predictions & ~expanded_target).sum(all_dimensions)
        self.counts["tn"] += (~predictions & ~expanded_target).sum(all_dimensions)
        self.counts["fn"] += (~predictions & expanded_target).sum(all_dimensions)

        if positive.any():
            positive_prediction = predictions[:, positive]
            positive_target = expanded_target[:, positive]
            intersection = (positive_prediction & positive_target).flatten(2).sum(2)
            union = (positive_prediction | positive_target).flatten(2).sum(2)
            self.counts["positive_tp"] += intersection.sum(1)
            self.counts["positive_fp"] += (positive_prediction & ~positive_target).sum(
                (1, 2, 3, 4)
            )
            self.counts["positive_fn"] += (~positive_prediction & positive_target).sum(
                (1, 2, 3, 4)
            )
            self.counts["positive_iou_sum"] += (intersection / union.clamp_min(1)).sum(
                1
            )
            self.counts["positive_patches"] += positive.sum()

        negative = ~positive
        if negative.any():
            has_false_positive = predictions[:, negative].flatten(2).any(2)
            self.counts["negative_patches"] += negative.sum()
            self.counts["negative_fp_patches"] += has_false_positive.sum(1)

    def results(self) -> dict[float, dict[str, float]]:
        if self.counts is None:
            raise RuntimeError("No samples were accumulated")
        result: dict[float, dict[str, float]] = {}
        for index, threshold in enumerate(self.thresholds):
            raw = {
                key: float(values[index].item()) for key, values in self.counts.items()
            }
            tp, fp, tn, fn = (raw[key] for key in ("tp", "fp", "tn", "fn"))
            positive_union = (
                raw["positive_tp"] + raw["positive_fp"] + raw["positive_fn"]
            )
            result[threshold] = {
                "threshold": threshold,
                "iou": tp / max(1.0, tp + fp + fn),
                "dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
                "precision": tp / max(1.0, tp + fp),
                "recall": tp / max(1.0, tp + fn),
                "specificity": tn / max(1.0, tn + fp),
                "accuracy": (tp + tn) / max(1.0, tp + tn + fp + fn),
                "positive_patch_iou_micro": raw["positive_tp"]
                / max(1.0, positive_union),
                "positive_patch_iou_mean": raw["positive_iou_sum"]
                / max(1.0, raw["positive_patches"]),
                "negative_patch_fp_rate": raw["negative_fp_patches"]
                / max(1.0, raw["negative_patches"]),
                "positive_patches": int(raw["positive_patches"]),
                "negative_patches": int(raw["negative_patches"]),
                "tp_pixels": int(tp),
                "fp_pixels": int(fp),
                "tn_pixels": int(tn),
                "fn_pixels": int(fn),
            }
        return result


def choose_reporting_threshold(
    results: dict[float, dict[str, float]],
) -> tuple[float, dict[str, float]]:
    threshold = max(
        results,
        key=lambda value: (
            results[value]["dice"],
            -results[value]["negative_patch_fp_rate"],
            value,
        ),
    )
    return threshold, results[threshold]


def choose_training_threshold(
    results: dict[float, dict[str, float]],
    target_negative_fp: float = 0.15,
    lambda_fp: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Apply the recovered StageA validation-threshold selection rule."""
    eligible = [
        threshold
        for threshold, metrics in results.items()
        if metrics["negative_patch_fp_rate"] <= target_negative_fp
    ]
    if eligible:
        threshold = max(
            eligible,
            key=lambda value: results[value]["positive_patch_iou_micro"],
        )
    else:
        threshold = max(
            results,
            key=lambda value: checkpoint_score(results[value], lambda_fp=lambda_fp),
        )
    return threshold, results[threshold]


def checkpoint_score(metrics: dict[str, float], lambda_fp: float = 1.0) -> float:
    return float(
        metrics["positive_patch_iou_micro"]
        - lambda_fp * metrics["negative_patch_fp_rate"]
    )
