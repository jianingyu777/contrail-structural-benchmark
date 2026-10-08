#!/usr/bin/env python3
"""Evaluate frozen core fusion methods after the geographic test is legitimately unsealed."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GEO_ROOT = Path(os.environ.get("TGRS_GEO_ROOT", str(ROOT / "outputs" / "p1_tgrs_geographic_holdout_v1"))).resolve()
PROTOCOL = GEO_ROOT / "protocol"
OUT = GEO_ROOT / "test_evaluation"
PROTOCOL_ID = os.environ.get("TGRS_PROTOCOL_ID", "TGRS_GEOGRAPHIC_HOLDOUT_V1_20260819")
SEEDS = (42, 123, 2025)
D4_VARIANTS = (
    "FULL_SPR_RETRAINED",
    "SPR_V2_ALPHA_INIT",
    "UNCONSTRAINED_SPATIAL_GATE",
    "SPR_NO_PERMISSION_MASK",
    "SPR_NO_RESIDUAL_BOUND",
    "SPR_NO_NEGATIVE_PENALTY",
)
PROBABILITY_BINS = 4096


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fusion_lib = load_module("tgrs_geo_fusion_eval_lib", SCRIPTS / "544_fit_tgrs_geo_fusion_and_spr_v1.py")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assert_unsealed() -> None:
    marker = PROTOCOL / "TEST_UNSEALED.json"
    if not marker.exists():
        raise PermissionError("Geographic test remains sealed")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "UNSEALED_AFTER_ALL_VALIDATION_DECISIONS_FROZEN":
        raise PermissionError("Invalid geographic-test unseal marker")


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "pixel_fpr": fp / max(fp + tn, 1),
        "pred_positive_pixels": int(prediction.sum()),
        "gt_positive_pixels": int(target.sum()),
    }


def topology_metrics(prediction: np.ndarray, target: np.ndarray, target_skeleton: np.ndarray) -> dict[str, float]:
    if not target.any():
        return {"cldice": math.nan, "centerline_recall": math.nan, "fragmentation_count": math.nan}
    prediction_skeleton = skeletonize(prediction) if prediction.any() else np.zeros_like(prediction, dtype=bool)
    predicted_skeleton_pixels = int(prediction_skeleton.sum())
    target_skeleton_pixels = int(target_skeleton.sum())
    topology_precision = (
        0.0 if predicted_skeleton_pixels == 0 else float(np.logical_and(prediction_skeleton, target).sum()) / predicted_skeleton_pixels
    )
    centerline_recall = (
        0.0 if target_skeleton_pixels == 0 else float(np.logical_and(target_skeleton, prediction).sum()) / target_skeleton_pixels
    )
    denominator = topology_precision + centerline_recall
    cldice = 0.0 if denominator == 0 else 2.0 * topology_precision * centerline_recall / denominator
    _, components = ndimage.label(prediction_skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    return {"cldice": cldice, "centerline_recall": centerline_recall, "fragmentation_count": float(components)}


def false_positive_components(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    false_positive = np.logical_and(prediction, ~target)
    labels, count = ndimage.label(false_positive, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        largest = 0
    else:
        largest = int(np.bincount(labels.ravel())[1:].max())
    return {"false_positive_component_count": int(count), "largest_false_positive_component_area": largest}


def new_probability_histogram() -> dict[str, np.ndarray]:
    return {
        "target": np.zeros(PROBABILITY_BINS, dtype=np.int64),
        "background": np.zeros(PROBABILITY_BINS, dtype=np.int64),
        "negative_patch": np.zeros(PROBABILITY_BINS, dtype=np.int64),
    }


def update_probability_histogram(histogram: dict[str, np.ndarray], probability: np.ndarray, target: np.ndarray) -> None:
    probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    truth = np.asarray(target, dtype=bool)
    levels = np.minimum((probability * (PROBABILITY_BINS - 1)).astype(np.int32), PROBABILITY_BINS - 1)
    histogram["target"] += np.bincount(levels[truth], minlength=PROBABILITY_BINS)
    histogram["background"] += np.bincount(levels[~truth], minlength=PROBABILITY_BINS)
    negative_patch = ~truth.reshape(len(truth), -1).any(axis=1)
    if negative_patch.any():
        histogram["negative_patch"] += np.bincount(levels[negative_patch].reshape(-1), minlength=PROBABILITY_BINS)


def finish_probability_histogram(histogram: dict[str, np.ndarray]) -> tuple[list[dict[str, float]], float]:
    target = histogram["target"][::-1].cumsum(dtype=np.float64)
    background = histogram["background"][::-1].cumsum(dtype=np.float64)
    negative_patch = histogram["negative_patch"][::-1].cumsum(dtype=np.float64)
    target_total = max(float(histogram["target"].sum()), 1.0)
    negative_total = max(float(histogram["negative_patch"].sum()), 1.0)
    recall = target / target_total
    precision = target / np.maximum(target + background, 1.0)
    negative_fpr = negative_patch / negative_total
    recall_increment = np.diff(np.concatenate(([0.0], recall)))
    average_precision = float(np.sum(recall_increment * precision))
    thresholds = np.arange(PROBABILITY_BINS - 1, -1, -1, dtype=np.float64) / (PROBABILITY_BINS - 1)
    rows = [
        {
            "threshold": float(threshold),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "negative_pixel_fpr": float(negative_fpr[index]),
            "cumulative_true_positive_pixels": float(target[index]),
            "cumulative_false_positive_pixels": float(background[index]),
        }
        for index, threshold in enumerate(thresholds)
    ]
    return rows, average_precision


def subgroup_membership(records: list[dict[str, Any]], targets: np.ndarray) -> list[dict[str, Any]]:
    definition = json.loads((GEO_ROOT / "subgroups" / "TRAIN_DERIVED_SUBGROUP_DEFINITIONS.json").read_text(encoding="utf-8"))
    low_threshold = float(definition["low_contrast_definition"]["threshold"])
    extreme_threshold = float(definition["extreme_low_contrast_definition"]["threshold"])
    candidates = pd.read_csv(
        GEO_ROOT / "hard_negatives" / "hard_negative_candidates_deduplicated.csv", encoding="utf-8-sig"
    ).fillna("")
    hard_names = set(candidates["image_basename_from_source"].astype(str)) | set(candidates["label_basename_from_source"].astype(str))
    rows = []
    for index, (record, target_raw) in enumerate(zip(records, targets)):
        target = np.asarray(target_raw, dtype=bool)
        area_ratio = float(target.mean())
        if target.any():
            skeleton = skeletonize(target)
            distance = ndimage.distance_transform_edt(target)
            width = float(np.median(np.maximum(2.0 * distance[skeleton] - 1.0, 1.0))) if skeleton.any() else math.nan
            stack_path = Path(str(record["bt_stack_path"]))
            if not stack_path.is_absolute():
                stack_path = ROOT / stack_path
            with np.load(stack_path, allow_pickle=False) as archive:
                btd23 = np.asarray(archive["stack"], dtype=np.float32)[4]
            outside_distance = ndimage.distance_transform_edt(~target)
            ring = (outside_distance >= 1.0) & (outside_distance <= 15.0)
            if not ring.any():
                ring = ~target
            contrast = abs(float(np.mean(btd23[target])) - float(np.mean(btd23[ring])))
        else:
            width = math.nan
            contrast = math.nan
        image_name = Path(str(record.get("image8_path", ""))).name
        label_name = Path(str(record.get("label_path", ""))).name
        rows.append(
            {
                "cache_index": index,
                "sample_id": record.get("sample_id", ""),
                "sample_key": record.get("sample_key", ""),
                "scene_key": record.get("scene_key", ""),
                "positive_negative": record.get("positive_negative", ""),
                "mask_area_ratio_recomputed": area_ratio,
                "skeleton_median_width_px": width,
                "btd23_absolute_target_background_contrast": contrast,
                "low_contrast_flag": bool(target.any() and contrast <= low_threshold),
                "extreme_low_contrast_flag": bool(target.any() and contrast <= extreme_threshold),
                "thin_target_flag": bool(target.any() and width <= 3.0),
                "small_target_flag": bool(target.any() and area_ratio < 0.005),
                "legacy_hard_negative_candidate": bool(image_name in hard_names or label_name in hard_names),
            }
        )
    return rows


def load_mlp(path: Path):
    import torch

    class PixelMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(7, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))

        def forward(self, value):
            return self.net(value)

    state = torch.load(path, map_location="cpu", weights_only=False)
    model = PixelMLP().eval()
    model.load_state_dict(state["model_state"], strict=True)
    return model


def baseline_probabilities(
    pair_d4: np.ndarray,
    pair_d0: np.ndarray,
    baseline: dict[str, Any],
    logistic: dict[str, Any],
    mlp,
):
    import torch

    decisions = baseline["decisions"]
    coefficient = np.asarray([logistic["coefficient_h1"], logistic["coefficient_d4"]], dtype=np.float32)
    features = fusion_lib.feature_stack(pair_d4).reshape(-1, 7)
    with torch.inference_mode():
        mlp_probability = torch.sigmoid(mlp(torch.from_numpy(features))).reshape(pair_d4.shape[0], pair_d4.shape[2], pair_d4.shape[3]).numpy()
    alpha = float(decisions["FIXED_LOGIT_FUSION"]["alpha"])
    return {
        "H1_VISUAL": fusion_lib.sigmoid(pair_d4[:, 0]),
        "D0_RENDERED_AUXILIARY": fusion_lib.sigmoid(pair_d0[:, 1]),
        "D4_RADIOMETRIC_AUXILIARY": fusion_lib.sigmoid(pair_d4[:, 1]),
        "PROBABILITY_AVERAGE": 0.5 * (fusion_lib.sigmoid(pair_d4[:, 0]) + fusion_lib.sigmoid(pair_d4[:, 1])),
        "FIXED_LOGIT_FUSION": fusion_lib.sigmoid((1.0 - alpha) * pair_d4[:, 0] + alpha * pair_d4[:, 1]),
        "LOGISTIC_STACKING": fusion_lib.sigmoid(
            logistic["intercept"] + coefficient[0] * pair_d4[:, 0] + coefficient[1] * pair_d4[:, 1]
        ),
        "MLP_1X1_STACKING": mlp_probability,
    }


def load_gates(seed: int, device):
    import torch

    models: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    fixed_fpr_decisions: dict[str, Any] = {}
    specifications = [("D0_RENDERED_AUXILIARY", "FULL_SPR_RETRAINED", "SPR_D0_RENDERED_CONTROL")]
    specifications.extend(("D4_RADIOMETRIC_AUXILIARY", variant, variant) for variant in D4_VARIANTS)
    for auxiliary, variant, method in specifications:
        run_dir = GEO_ROOT / "fusion" / "gates" / auxiliary / variant / f"seed_{seed}"
        decision = json.loads((run_dir / "VALIDATION_DECISION.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(run_dir / "best_gate_checkpoint.pt", map_location=device, weights_only=False)
        model = fusion_lib.ablation.build_gate(float(checkpoint["h1_threshold"]), variant).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[method] = (model, auxiliary)
        decisions[method] = decision["selected_validation"]
        fixed_fpr_decisions[method] = decision["fixed_negative_pixel_fpr_operating_points"]
    return models, decisions, fixed_fpr_decisions


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    positive = frame[frame["gt_positive_pixels"] > 0]
    negative = frame[frame["gt_positive_pixels"] == 0]
    tp, fp, fn, tn = (float(frame[column].sum()) for column in ("tp", "fp", "fn", "tn"))

    def mean(column: str, subset=positive):
        values = pd.to_numeric(subset[column], errors="coerce")
        return float(values.mean()) if values.notna().any() else math.nan

    def subgroup(flag: str, column: str):
        subset = positive[positive[flag].astype(bool)]
        return mean(column, subset), len(subset)

    thin_recall, thin_n = subgroup("thin_target_flag", "recall")
    small_recall, small_n = subgroup("small_target_flag", "recall")
    low_recall, low_n = subgroup("low_contrast_flag", "recall")
    extreme_recall, extreme_n = subgroup("extreme_low_contrast_flag", "recall")
    hard = negative[negative["legacy_hard_negative_candidate"].astype(bool)]
    return {
        "rows": len(frame),
        "scenes": int(frame["scene_key"].nunique()),
        "positive_rows": len(positive),
        "negative_rows": len(negative),
        "positive_patch_macro_dice": mean("dice"),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "negative_pixel_fpr": float(negative["fp"].sum() / max(negative["fp"].sum() + negative["tn"].sum(), 1)),
        "negative_patch_any_prediction_rate": float((negative["pred_positive_pixels"] > 0).mean()),
        "thin_recall": thin_recall,
        "thin_rows": thin_n,
        "small_recall": small_recall,
        "small_rows": small_n,
        "low_contrast_recall": low_recall,
        "low_contrast_rows": low_n,
        "extreme_low_contrast_recall": extreme_recall,
        "extreme_low_contrast_rows": extreme_n,
        "cldice": mean("cldice"),
        "centerline_recall": mean("centerline_recall"),
        "fragmentation_count": mean("fragmentation_count"),
        "hard_negative_rows_in_geographic_test": len(hard),
        "hard_negative_any_prediction_rate": float((hard["pred_positive_pixels"] > 0).mean()) if len(hard) else math.nan,
        "hard_negative_pixel_fpr": float(hard["fp"].sum() / max(hard["fp"].sum() + hard["tn"].sum(), 1)) if len(hard) else math.nan,
        "hard_negative_component_count_mean": mean("false_positive_component_count", hard),
        "hard_negative_largest_component_area_mean": mean("largest_false_positive_component_area", hard),
    }


def evaluate_seed(seed: int, device, batch_size: int) -> None:
    import torch

    assert_unsealed()
    raw, targets, records = fusion_lib.cache(seed, "test")
    parameters = fusion_lib.calibration(seed)
    seed_dir = OUT / "core" / f"seed_{seed}"
    complete = seed_dir / "EVALUATION_COMPLETE.json"
    if complete.exists():
        raise FileExistsError(f"Test evaluation already exists: {complete}")
    seed_dir.mkdir(parents=True, exist_ok=True)
    membership_path = OUT / "test_subgroup_membership.csv"
    if membership_path.exists():
        membership = pd.read_csv(membership_path, encoding="utf-8-sig").fillna("").to_dict("records")
    else:
        membership = subgroup_membership(records, targets)
        write_csv(membership_path, membership)
    if len(membership) != len(records):
        raise ValueError("Subgroup membership and test cache do not align")
    baseline_dir = GEO_ROOT / "fusion" / "baselines" / f"seed_{seed}"
    baseline = json.loads((baseline_dir / "VALIDATION_DECISIONS.json").read_text(encoding="utf-8"))
    logistic = json.loads((baseline_dir / "logistic_stacking.json").read_text(encoding="utf-8"))
    mlp = load_mlp(baseline_dir / "mlp_1x1_stacking.pt")
    baseline_thresholds = {method: float(row["threshold"]) for method, row in baseline["decisions"].items()}
    gates, gate_decisions, gate_fixed_fpr = load_gates(seed, device)
    thresholds = {**baseline_thresholds, **{method: float(row["threshold"]) for method, row in gate_decisions.items()}}
    fixed_fpr_thresholds = {
        method: rows
        for method, rows in baseline["fixed_negative_pixel_fpr_operating_points"].items()
        if method in {"H1_VISUAL", "D4_RADIOMETRIC_AUXILIARY", "FIXED_LOGIT_FUSION"}
    }
    fixed_fpr_thresholds.update(
        {
            method: rows
            for method, rows in gate_fixed_fpr.items()
            if method in {"SPR_V2_ALPHA_INIT", "FULL_SPR_RETRAINED", "SPR_D0_RENDERED_CONTROL"}
        }
    )
    metric_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in thresholds}
    fixed_fpr_rows: list[dict[str, Any]] = []
    probability_histograms = {method: new_probability_histogram() for method in thresholds}
    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        raw_batch = np.asarray(raw[start:stop], dtype=np.float32)
        pair_d4 = fusion_lib.calibrated_pair(raw_batch, parameters, "D4_RADIOMETRIC_AUXILIARY")
        pair_d0 = fusion_lib.calibrated_pair(raw_batch, parameters, "D0_RENDERED_AUXILIARY")
        probabilities = baseline_probabilities(pair_d4, pair_d0, baseline, logistic, mlp)
        with torch.inference_mode():
            for method, (model, auxiliary) in gates.items():
                frozen = pair_d0 if auxiliary == "D0_RENDERED_AUXILIARY" else pair_d4
                probability = torch.sigmoid(model(torch.from_numpy(frozen).to(device))["final_logit"])[:, 0]
                probabilities[method] = probability.float().cpu().numpy()
        target_batch_for_curve = np.asarray(targets[start:stop], dtype=bool)
        for method, probability_batch in probabilities.items():
            update_probability_histogram(probability_histograms[method], probability_batch, target_batch_for_curve)
        for offset, target_raw in enumerate(np.asarray(targets[start:stop])):
            record = dict(records[start + offset])
            group = dict(membership[start + offset])
            target = target_raw.astype(bool)
            target_skeleton = skeletonize(target) if target.any() else np.zeros_like(target, dtype=bool)
            for method, probability_batch in probabilities.items():
                prediction = probability_batch[offset] >= thresholds[method]
                row = {
                    **record,
                    **group,
                    "method": method,
                    "seed": seed,
                    "threshold": thresholds[method],
                }
                row.update(binary_metrics(prediction, target))
                row.update(topology_metrics(prediction, target, target_skeleton))
                row.update(false_positive_components(prediction, target))
                metric_rows[method].append(row)
                for operating_point in fixed_fpr_thresholds.get(method, []):
                    fixed_threshold = float(operating_point["threshold"])
                    fixed_prediction = probability_batch[offset] >= fixed_threshold
                    fixed_row = {
                        "cache_index": record.get("cache_index", start + offset),
                        "sample_id": record.get("sample_id", ""),
                        "sample_key": record.get("sample_key", ""),
                        "scene_key": record.get("scene_key", ""),
                        "positive_negative": record.get("positive_negative", ""),
                        "thin_target_flag": group.get("thin_target_flag", False),
                        "small_target_flag": group.get("small_target_flag", False),
                        "low_contrast_flag": group.get("low_contrast_flag", False),
                        "extreme_low_contrast_flag": group.get("extreme_low_contrast_flag", False),
                        "method": method,
                        "seed": seed,
                        "target_negative_pixel_fpr": float(operating_point["target_negative_pixel_fpr"]),
                        "validation_selected_threshold": fixed_threshold,
                        "validation_achieved_negative_pixel_fpr": float(operating_point["validation_negative_pixel_fpr"]),
                        "validation_target_unattainable": bool(operating_point["fallback_target_unattainable"]),
                    }
                    fixed_row.update(binary_metrics(fixed_prediction, target))
                    fixed_fpr_rows.append(fixed_row)
        if start == 0 or stop == len(raw) or stop % 512 == 0:
            print(f"test seed={seed}: {stop}/{len(raw)}", flush=True)
    summaries = []
    for method, rows in metric_rows.items():
        path = seed_dir / f"{method.lower()}_per_sample_metrics.csv"
        write_csv(path, rows)
        curve, pixel_auprc = finish_probability_histogram(probability_histograms[method])
        write_csv(seed_dir / f"{method.lower()}_pixel_operating_curve_4096bin.csv", curve)
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "threshold": thresholds[method],
                "pixel_auprc_4096bin": pixel_auprc,
                **summarize(pd.DataFrame(rows)),
            }
        )
    write_csv(seed_dir / "core_test_summary.csv", summaries)
    write_csv(seed_dir / "fixed_fpr_per_sample_metrics.csv", fixed_fpr_rows)
    fixed_summary = []
    fixed_frame = pd.DataFrame(fixed_fpr_rows)
    for (method, target_fpr), group in fixed_frame.groupby(["method", "target_negative_pixel_fpr"], sort=False):
        tp, fp, fn, tn = (float(group[column].sum()) for column in ("tp", "fp", "fn", "tn"))
        negative = group[group["gt_positive_pixels"] == 0]
        fixed_summary.append(
            {
                "method": method,
                "seed": seed,
                "target_negative_pixel_fpr": float(target_fpr),
                "validation_selected_threshold": float(group["validation_selected_threshold"].iloc[0]),
                "test_recall": tp / max(tp + fn, 1.0),
                "test_precision": tp / max(tp + fp, 1.0),
                "test_negative_pixel_fpr": float(negative["fp"].sum() / max(negative["fp"].sum() + negative["tn"].sum(), 1)),
                "positive_patch_macro_dice": float(group.loc[group["gt_positive_pixels"] > 0, "dice"].mean()),
            }
        )
    write_csv(seed_dir / "fixed_fpr_test_summary.csv", fixed_summary)
    complete.write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol_id": PROTOCOL_ID,
                "seed": seed,
                "rows": len(records),
                "methods": sorted(metric_rows),
                "test_accessed": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(complete)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    evaluate_seed(args.seed, device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
