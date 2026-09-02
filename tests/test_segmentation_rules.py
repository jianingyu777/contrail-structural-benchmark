from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from contrail_benchmark.segmentation.losses import (  # noqa: E402
    CompositeContrailLoss,
    LossConfig,
    curriculum_weight,
)
from contrail_benchmark.segmentation.metrics import (  # noqa: E402
    ThresholdMetrics,
    checkpoint_score,
    choose_reporting_threshold,
    choose_training_threshold,
)


def test_curriculum_is_zero_then_linear() -> None:
    assert curriculum_weight(0.2, 0.2, 0.5) == 0.0
    assert curriculum_weight(0.6, 0.2, 0.5) == pytest.approx(0.25)
    assert curriculum_weight(1.0, 0.2, 0.5) == pytest.approx(0.5)


def test_composite_loss_is_finite_for_mixed_batch() -> None:
    target = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    target[0, 0, 4:12, 7:9] = 1.0
    centerline = torch.zeros_like(target)
    centerline[0, 0, 4:12, 8] = 1.0
    mask_logits = torch.zeros_like(target, requires_grad=True)
    centerline_logits = torch.zeros_like(target, requires_grad=True)
    loss, components = CompositeContrailLoss(LossConfig(skeleton_iterations=2))(
        mask_logits, target, centerline_logits, centerline, progress=0.6
    )
    assert torch.isfinite(loss)
    assert components["cldice_weight"] > 0
    loss.backward()
    assert mask_logits.grad is not None


def test_threshold_and_checkpoint_rules() -> None:
    probabilities = torch.tensor(
        [[[[0.9, 0.8], [0.2, 0.1]]], [[[0.6, 0.4], [0.2, 0.1]]]]
    )
    masks = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    accumulator = ThresholdMetrics([0.5, 0.85])
    accumulator.update(probabilities, masks)
    results = accumulator.results()
    threshold, selected = choose_reporting_threshold(results)
    assert threshold == 0.5
    assert selected["negative_patch_fp_rate"] == 1.0
    assert checkpoint_score(selected, lambda_fp=1.0) == pytest.approx(
        selected["positive_patch_iou_micro"] - 1.0
    )


def test_training_threshold_respects_negative_patch_constraint() -> None:
    results = {
        0.4: {
            "positive_patch_iou_micro": 0.90,
            "negative_patch_fp_rate": 0.30,
        },
        0.5: {
            "positive_patch_iou_micro": 0.82,
            "negative_patch_fp_rate": 0.10,
        },
        0.6: {
            "positive_patch_iou_micro": 0.78,
            "negative_patch_fp_rate": 0.02,
        },
    }
    threshold, selected = choose_training_threshold(
        results, target_negative_fp=0.15, lambda_fp=1.0
    )
    assert threshold == 0.5
    assert selected["positive_patch_iou_micro"] == pytest.approx(0.82)


def test_training_threshold_falls_back_to_penalized_score() -> None:
    results = {
        0.4: {
            "positive_patch_iou_micro": 0.90,
            "negative_patch_fp_rate": 0.40,
        },
        0.5: {
            "positive_patch_iou_micro": 0.80,
            "negative_patch_fp_rate": 0.20,
        },
    }
    threshold, _ = choose_training_threshold(
        results, target_negative_fp=0.15, lambda_fp=1.0
    )
    assert threshold == 0.5
