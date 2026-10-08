"""Shared metric extraction and paired cluster bootstrap for matched baselines."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize


METRICS = (
    "positive_patch_macro_dice", "positive_patch_macro_iou", "cldice", "centerline_recall",
    "pixel_pooled_dice", "pixel_pooled_iou", "precision", "recall",
    "negative_patch_any_prediction_rate", "negative_pixel_fpr",
)


def original_metrics(path: Path):
    """Execute only the unmodified pure functions, not legacy pipeline imports."""
    names = {"binary_metrics", "topology_metrics", "false_positive_components",
             "new_probability_histogram", "update_probability_histogram", "finish_probability_histogram"}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("Original metric functions not found")
    namespace = {"np": np, "math": math, "ndimage": ndimage, "skeletonize": skeletonize,
                 "Any": object, "PROBABILITY_BINS": 4096}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def sufficient_statistics(frame: pd.DataFrame) -> np.ndarray:
    positive = frame.gt_positive_pixels.to_numpy() > 0
    negative = ~positive
    return np.column_stack([
        np.where(positive, frame.dice, 0), np.where(positive, frame.iou, 0),
        np.where(positive, frame.cldice, 0), np.where(positive, frame.centerline_recall, 0),
        positive, frame.tp, frame.fp, frame.fn,
        negative & (frame.pred_positive_pixels.to_numpy() > 0), negative,
        np.where(negative, frame.fp, 0), np.where(negative, frame.fp + frame.tn, 0),
    ]).astype(np.float64)


def metrics_from_sums(sums: np.ndarray) -> np.ndarray:
    def divide(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)

    dice, iou, cldice, center, pos, tp, fp, fn, any_neg, neg, neg_fp, neg_pixels = np.moveaxis(sums, -1, 0)
    return np.stack([
        divide(dice, pos), divide(iou, pos), divide(cldice, pos), divide(center, pos),
        divide(2 * tp, 2 * tp + fp + fn), divide(tp, tp + fp + fn),
        divide(tp, tp + fp), divide(tp, tp + fn), divide(any_neg, neg), divide(neg_fp, neg_pixels),
    ], axis=-1)


def summarize(frame: pd.DataFrame) -> dict:
    values = metrics_from_sums(sufficient_statistics(frame).sum(axis=0))
    return {**dict(zip(METRICS, map(float, values))), "rows": len(frame),
            "positive_rows": int((frame.gt_positive_pixels > 0).sum()),
            "negative_rows": int((frame.gt_positive_pixels == 0).sum()),
            "scenes": int(frame.scene_key.nunique()), "dates": int(frame.observation_date.nunique())}


def paired_bootstrap(frames: dict, models: tuple, seeds: tuple, cluster: str, repeats: int,
                     random_seed: int = 20260916) -> pd.DataFrame:
    reference = frames[(models[0], seeds[0])]
    groups = sorted(reference[cluster].unique())
    group_index = pd.Categorical(reference[cluster], categories=groups).codes
    if np.any(group_index < 0):
        raise ValueError("Missing cluster identifiers")
    rng = np.random.default_rng(random_seed)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repeats)
    boot, points = {}, {}
    for model in models:
        runs, estimates = [], []
        for seed in seeds:
            frame = frames[(model, seed)]
            if not np.array_equal(frame.sample_id, reference.sample_id) or not np.array_equal(frame[cluster], reference[cluster]):
                raise ValueError("Paired bootstrap sample/cluster order mismatch")
            stats = sufficient_statistics(frame)
            aggregated = np.zeros((len(groups), stats.shape[1]))
            np.add.at(aggregated, group_index, stats)
            runs.append(metrics_from_sums(weights @ aggregated))
            estimates.append(metrics_from_sums(stats.sum(axis=0)))
        # Resample clusters identically across models/seeds; seeds are not independent test samples.
        boot[model] = np.mean(runs, axis=0)
        points[model] = np.mean(estimates, axis=0)
    records = []
    for model in models:
        for comparison in ("estimate", "difference_vs_maxvit"):
            if model == models[0] and comparison != "estimate":
                continue
            values = boot[model] if comparison == "estimate" else boot[model] - boot[models[0]]
            point = points[model] if comparison == "estimate" else points[model] - points[models[0]]
            for index, metric in enumerate(METRICS):
                valid = values[:, index][np.isfinite(values[:, index])]
                low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
                records.append({"model": model, "metric": metric, "comparison": comparison,
                                "estimate": float(point[index]), "ci_low": float(low), "ci_high": float(high),
                                "cluster": cluster, "clusters": len(groups), "replicates": repeats,
                                "valid_replicates": len(valid), "seed": random_seed,
                                "seed_aggregation": "mean of three per-seed estimates"})
    return pd.DataFrame(records)
