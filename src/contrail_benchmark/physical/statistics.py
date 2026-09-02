from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import packaged_config_path


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int = 3_000
    confidence_level: float = 0.95
    seed: int = 20_260_901
    date_column: str = "date_yyyymmdd"
    scene_column: str = "scene_id"

    @classmethod
    def from_json(cls, path: str | Path) -> "BootstrapConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("statistics", payload)
        return cls(
            replicates=int(
                values.get("bootstrap_replicates", values.get("replicates", 3_000))
            ),
            confidence_level=float(values.get("confidence_level", 0.95)),
            seed=int(values.get("seed", 20_260_901)),
            date_column=str(values.get("date_column", "date_yyyymmdd")),
            scene_column=str(values.get("scene_column", "scene_id")),
        )


def _validate_config(config: BootstrapConfig) -> None:
    if config.replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("Confidence level must lie between zero and one")


def _cluster_index(
    frame: pd.DataFrame,
    date_column: str,
    scene_column: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    dates = frame[date_column].astype(str)
    scenes = frame[scene_column].astype(str)
    unique_dates = dates.drop_duplicates().to_numpy()
    scenes_by_date: dict[str, np.ndarray] = {}
    rows_by_cluster: dict[tuple[str, str], np.ndarray] = {}
    for date in unique_dates:
        date_rows = frame.loc[dates.eq(date)]
        scene_values = date_rows[scene_column].astype(str).drop_duplicates().to_numpy()
        scenes_by_date[str(date)] = scene_values
        for scene in scene_values:
            rows_by_cluster[(str(date), str(scene))] = frame.index[
                dates.eq(date) & scenes.eq(scene)
            ].to_numpy(int)
    return unique_dates.astype(str), scenes_by_date, rows_by_cluster


def hierarchical_draw_indices(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    date_column: str = "date_yyyymmdd",
    scene_column: str = "scene_id",
) -> np.ndarray:
    """Sample dates, then complete scene clusters within each sampled date."""
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty table")
    for column in (date_column, scene_column):
        if column not in frame:
            raise ValueError(f"Bootstrap table has no '{column}' column")
    work = frame.reset_index(drop=True)
    dates, scenes_by_date, rows_by_cluster = _cluster_index(
        work, date_column, scene_column
    )
    selected: list[np.ndarray] = []
    for date in rng.choice(dates, size=len(dates), replace=True):
        scenes = scenes_by_date[str(date)]
        for scene in rng.choice(scenes, size=len(scenes), replace=True):
            selected.append(rows_by_cluster[(str(date), str(scene))])
    return np.concatenate(selected)


def mean_statistic(column: str) -> Callable[[pd.DataFrame], float]:
    def calculate(frame: pd.DataFrame) -> float:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        return float(np.nanmean(values))

    return calculate


def median_statistic(column: str) -> Callable[[pd.DataFrame], float]:
    def calculate(frame: pd.DataFrame) -> float:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        return float(np.nanmedian(values))

    return calculate


def ratio_of_sums_statistic(
    numerator: str,
    denominator: str,
) -> Callable[[pd.DataFrame], float]:
    def calculate(frame: pd.DataFrame) -> float:
        top = float(pd.to_numeric(frame[numerator], errors="coerce").sum())
        bottom = float(pd.to_numeric(frame[denominator], errors="coerce").sum())
        return top / bottom if abs(bottom) > 1e-12 else math.nan

    return calculate


def spearman_statistic(first: str, second: str) -> Callable[[pd.DataFrame], float]:
    def calculate(frame: pd.DataFrame) -> float:
        x = pd.to_numeric(frame[first], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[second], errors="coerce").to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if (
            valid.sum() < 3
            or np.unique(x[valid]).size < 2
            or np.unique(y[valid]).size < 2
        ):
            return math.nan
        return float(spearmanr(x[valid], y[valid]).statistic)

    return calculate


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    config: BootstrapConfig,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    _validate_config(config)
    for column in (config.date_column, config.scene_column):
        if column not in frame:
            raise ValueError(f"Bootstrap table has no '{column}' column")
    work = frame.reset_index(drop=True)
    point = float(statistic(work))
    rng = np.random.default_rng(config.seed)
    draws = np.empty(config.replicates, dtype=float)
    for replicate in range(config.replicates):
        indices = hierarchical_draw_indices(
            work, rng, config.date_column, config.scene_column
        )
        draws[replicate] = statistic(work.iloc[indices])
    finite = draws[np.isfinite(draws)]
    if not len(finite):
        raise RuntimeError("All bootstrap replicates were non-finite")
    alpha = 1.0 - config.confidence_level
    summary: dict[str, float | int | str] = {
        "estimate": point,
        "ci_low": float(np.quantile(finite, alpha / 2.0)),
        "ci_median": float(np.quantile(finite, 0.5)),
        "ci_high": float(np.quantile(finite, 1.0 - alpha / 2.0)),
        "confidence_level": config.confidence_level,
        "bootstrap_replicates_requested": config.replicates,
        "bootstrap_replicates_finite": int(len(finite)),
        "seed": config.seed,
        "dates": int(work[config.date_column].astype(str).nunique()),
        "scenes": int(
            work[[config.date_column, config.scene_column]]
            .astype(str)
            .drop_duplicates()
            .shape[0]
        ),
        "rows": int(len(work)),
        "resampling_rule": "date, then complete scene cluster within sampled date",
        "interval_method": "percentile",
    }
    replicate_frame = pd.DataFrame(
        {"replicate": np.arange(config.replicates), "estimate": draws}
    )
    return summary, replicate_frame


def paired_sensitivity_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    scenario_column: str,
    baseline: str,
    config: BootstrapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired scene-level sensitivity changes from one baseline scenario."""
    required = {
        value_column,
        scenario_column,
        config.date_column,
        config.scene_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sensitivity table is missing: {', '.join(sorted(missing))}")
    scene = frame.groupby(
        [config.date_column, config.scene_column, scenario_column], as_index=False
    )[value_column].mean()
    wide = scene.pivot(
        index=[config.date_column, config.scene_column],
        columns=scenario_column,
        values=value_column,
    ).reset_index()
    if baseline not in wide:
        raise ValueError(f"Baseline scenario '{baseline}' is absent")
    summary_rows: list[dict[str, Any]] = []
    replicate_rows: list[pd.DataFrame] = []
    scenarios = [
        column
        for column in wide.columns
        if column not in {config.date_column, config.scene_column, baseline}
    ]
    if not scenarios:
        raise ValueError("No non-baseline sensitivity scenario is available")
    for offset, scenario in enumerate(scenarios):
        paired = (
            wide[[config.date_column, config.scene_column, baseline, scenario]]
            .dropna()
            .copy()
        )
        paired["difference"] = paired[scenario] - paired[baseline]
        scenario_config = BootstrapConfig(
            replicates=config.replicates,
            confidence_level=config.confidence_level,
            seed=config.seed + offset,
            date_column=config.date_column,
            scene_column=config.scene_column,
        )
        summary, replicates = hierarchical_bootstrap(
            paired, mean_statistic("difference"), scenario_config
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "baseline": baseline,
                "paired_scene_mean": float(paired[scenario].mean()),
                "baseline_paired_scene_mean": float(paired[baseline].mean()),
                "mean_difference": summary["estimate"],
                "ci_low": summary["ci_low"],
                "ci_high": summary["ci_high"],
                "paired_scenes": int(len(paired)),
                "dates": int(paired[config.date_column].astype(str).nunique()),
            }
        )
        replicates.insert(1, "scenario", scenario)
        replicates.insert(2, "baseline", baseline)
        replicate_rows.append(replicates)
    return pd.DataFrame(summary_rows), pd.concat(replicate_rows, ignore_index=True)


def _config_with_columns(
    source: BootstrapConfig,
    date_column: str | None,
    scene_column: str | None,
    replicates: int | None,
    seed: int | None,
) -> BootstrapConfig:
    return BootstrapConfig(
        replicates=replicates if replicates is not None else source.replicates,
        confidence_level=source.confidence_level,
        seed=seed if seed is not None else source.seed,
        date_column=date_column or source.date_column,
        scene_column=scene_column or source.scene_column,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Date-then-scene hierarchical bootstrap"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates-output", default=None)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    parser.add_argument(
        "--statistic", choices=("mean", "median", "ratio", "spearman"), default="mean"
    )
    parser.add_argument("--value", default=None)
    parser.add_argument("--numerator", default=None)
    parser.add_argument("--denominator", default=None)
    parser.add_argument("--x", default=None)
    parser.add_argument("--y", default=None)
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--scene-column", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    source_config = BootstrapConfig.from_json(args.config)
    config = _config_with_columns(
        source_config,
        args.date_column,
        args.scene_column,
        args.replicates,
        args.seed,
    )
    if args.statistic == "mean" and args.value:
        statistic = mean_statistic(args.value)
    elif args.statistic == "median" and args.value:
        statistic = median_statistic(args.value)
    elif args.statistic == "ratio" and args.numerator and args.denominator:
        statistic = ratio_of_sums_statistic(args.numerator, args.denominator)
    elif args.statistic == "spearman" and args.x and args.y:
        statistic = spearman_statistic(args.x, args.y)
    else:
        raise ValueError(
            "The selected statistic is missing its required column arguments"
        )
    summary, replicates = hierarchical_bootstrap(
        pd.read_csv(args.input, low_memory=False), statistic, config
    )
    summary["statistic"] = args.statistic
    summary["value_column"] = args.value
    summary["numerator_column"] = args.numerator
    summary["denominator_column"] = args.denominator
    summary["x_column"] = args.x
    summary["y_column"] = args.y
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.replicates_output:
        replicate_path = Path(args.replicates_output)
        replicate_path.parent.mkdir(parents=True, exist_ok=True)
        compression = "gzip" if replicate_path.suffix.lower() == ".gz" else None
        replicates.to_csv(replicate_path, index=False, compression=compression)


def sensitivity_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Paired hierarchical sensitivity summary"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--scenario-column", default="sensitivity_scenario")
    parser.add_argument("--baseline", default="primary")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--replicates-output", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    parser.add_argument("--date-column", default=None)
    parser.add_argument("--scene-column", default=None)
    args = parser.parse_args()
    source = BootstrapConfig.from_json(args.config)
    config = _config_with_columns(
        source, args.date_column, args.scene_column, None, None
    )
    summary, replicates = paired_sensitivity_bootstrap(
        pd.read_csv(args.input, low_memory=False),
        args.value,
        args.scenario_column,
        args.baseline,
        config,
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    replicate_path = Path(args.replicates_output)
    replicate_path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if replicate_path.suffix.lower() == ".gz" else None
    replicates.to_csv(replicate_path, index=False, compression=compression)


if __name__ == "__main__":
    cli()
