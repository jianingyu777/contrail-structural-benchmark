"""Statistical functions from the frozen experiment implementation, unchanged."""
import numpy as np
import pandas as pd

def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    totals = frame[["tp_pixels", "fp_pixels", "tn_pixels", "fn_pixels"]].sum()
    tp, fp, tn, fn = (float(totals[key]) for key in totals.index)
    positive_iou = frame.loc[frame["positive_reference"].eq(1), "positive_patch_iou"]
    difficult = frame.loc[frame["difficult_negative"].eq(1)]
    return {
        "iou": tp / max(1.0, tp + fp + fn),
        "dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
        "precision": tp / max(1.0, tp + fp),
        "recall": tp / max(1.0, tp + fn),
        "mean_positive_patch_iou": float(positive_iou.mean()) if len(positive_iou) else np.nan,
        "difficult_negative_activation_rate": (
            float(difficult["prediction_nonempty"].mean()) if len(difficult) else np.nan
        ),
    }

def date_cluster_bootstrap(
    frame: pd.DataFrame, seed: int, replicates: int = 2000
) -> dict[str, dict[str, float]]:
    metric_names = (
        "iou",
        "dice",
        "precision",
        "recall",
        "mean_positive_patch_iou",
        "difficult_negative_activation_rate",
    )
    if frame.empty:
        return {}

    sufficient = frame.assign(
        positive_iou_sum=np.where(
            frame["positive_reference"].eq(1), frame["positive_patch_iou"], 0.0
        ),
        positive_count=frame["positive_reference"].eq(1).astype(np.int64),
        difficult_activation_sum=np.where(
            frame["difficult_negative"].eq(1), frame["prediction_nonempty"], 0.0
        ),
        difficult_count=frame["difficult_negative"].eq(1).astype(np.int64),
    ).groupby("acquisition_date", sort=True)[
        [
            "tp_pixels",
            "fp_pixels",
            "tn_pixels",
            "fn_pixels",
            "positive_iou_sum",
            "positive_count",
            "difficult_activation_sum",
            "difficult_count",
        ]
    ].sum()

    cluster_values = sufficient.to_numpy(dtype=np.float64)
    cluster_count = len(cluster_values)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    multiplicities = np.zeros((replicates, cluster_count), dtype=np.int32)
    rows = np.repeat(np.arange(replicates), cluster_count)
    np.add.at(multiplicities, (rows, chosen.ravel()), 1)
    totals = multiplicities @ cluster_values

    tp, fp, _tn, fn, positive_iou_sum, positive_count, difficult_sum, difficult_count = (
        totals[:, index] for index in range(totals.shape[1])
    )
    samples = {
        "iou": tp / np.maximum(1.0, tp + fp + fn),
        "dice": 2.0 * tp / np.maximum(1.0, 2.0 * tp + fp + fn),
        "precision": tp / np.maximum(1.0, tp + fp),
        "recall": tp / np.maximum(1.0, tp + fn),
        "mean_positive_patch_iou": np.divide(
            positive_iou_sum,
            positive_count,
            out=np.full(replicates, np.nan),
            where=positive_count > 0,
        ),
        "difficult_negative_activation_rate": np.divide(
            difficult_sum,
            difficult_count,
            out=np.full(replicates, np.nan),
            where=difficult_count > 0,
        ),
    }
    return {
        name: {
            "lower_95": float(np.quantile(values[np.isfinite(values)], 0.025)),
            "upper_95": float(np.quantile(values[np.isfinite(values)], 0.975)),
        }
        for name, values in samples.items()
        if name in metric_names and np.isfinite(values).any()
    }
