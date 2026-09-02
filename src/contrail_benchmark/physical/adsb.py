from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import packaged_config_path


BROAD_STATUSES = {
    "broad_geometric_candidate",
    "geometrically_consistent_ambiguous_candidate",
    "geometrically_consistent_unambiguous_candidate",
}
STRICT_STATUSES = {
    "geometrically_consistent_ambiguous_candidate",
    "geometrically_consistent_unambiguous_candidate",
}
NO_COVERAGE_STATUS = "no_high_altitude_adsb_points_in_scene_window"


@dataclass(frozen=True)
class ADSBConfig:
    minimum_altitude_ft: float = 25_000.0
    maximum_absolute_time_offset_min: float = 30.0
    broad_distance_km: float = 10.0
    broad_heading_difference_deg: float = 45.0
    strict_distance_km: float = 5.0
    strict_heading_difference_deg: float = 20.0
    unambiguous_score_gap: float = 0.25
    permutations: int = 5_000
    seed: int = 20_260_810

    @classmethod
    def from_json(cls, path: str | Path) -> "ADSBConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("adsb", payload)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _as_bool(value: Any, default: bool = True) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def axial_difference_deg(
    first: np.ndarray | float, second: np.ndarray | float
) -> np.ndarray:
    """Return the unsigned difference between axial angles in [0, 90] degrees."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    return np.abs((first_array - second_array + 90.0) % 180.0 - 90.0)


def _coordinates(geometry: Any) -> np.ndarray:
    if geometry.geom_type == "LineString":
        return np.asarray(geometry.coords, dtype=float)[:, :2]
    if geometry.geom_type == "MultiLineString":
        parts = [np.asarray(part.coords, dtype=float)[:, :2] for part in geometry.geoms]
        return np.concatenate(parts, axis=0)
    raise ValueError(
        f"Expected LineString or MultiLineString, received {geometry.geom_type}"
    )


def axial_bearing_deg(geometry: Any) -> float:
    """Estimate an axial bearing from all vertices by principal-axis analysis."""
    coordinates = _coordinates(geometry)
    coordinates = coordinates[np.isfinite(coordinates).all(axis=1)]
    if len(coordinates) < 2:
        return math.nan
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return math.nan
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)


def _load_line(value: str) -> Any:
    try:
        from shapely import wkt
        from shapely.ops import linemerge
    except ImportError as exc:
        raise RuntimeError(
            "ADS-B matching requires the 'geo' optional dependencies"
        ) from exc

    geometry = wkt.loads(str(value))
    if geometry.is_empty:
        raise ValueError("Empty WKT geometry")
    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if merged.geom_type in {"LineString", "MultiLineString"}:
            geometry = merged
    if geometry.geom_type not in {"LineString", "MultiLineString"}:
        raise ValueError(f"Unsupported skeleton geometry: {geometry.geom_type}")
    return geometry


def normalized_candidate_score(
    distance_km: float,
    heading_difference_deg: float,
    config: ADSBConfig,
) -> float:
    """Transparent [0, 1] score used only to rank candidates for ambiguity."""
    distance_closeness = max(0.0, 1.0 - distance_km / config.broad_distance_km)
    heading_closeness = max(
        0.0, 1.0 - heading_difference_deg / config.broad_heading_difference_deg
    )
    return float(0.5 * (distance_closeness + heading_closeness))


def classify_candidate(
    distance_km: float,
    heading_difference_deg: float,
    runner_up_score_gap: float,
    config: ADSBConfig,
) -> str:
    strict = (
        distance_km <= config.strict_distance_km
        and heading_difference_deg <= config.strict_heading_difference_deg
    )
    broad = (
        distance_km <= config.broad_distance_km
        and heading_difference_deg <= config.broad_heading_difference_deg
    )
    if strict:
        unambiguous = (
            not np.isfinite(runner_up_score_gap)
            or runner_up_score_gap > config.unambiguous_score_gap
        )
        return (
            "geometrically_consistent_unambiguous_candidate"
            if unambiguous
            else "geometrically_consistent_ambiguous_candidate"
        )
    if broad:
        return "broad_geometric_candidate"
    return "no_geometrically_consistent_candidate"


def _trajectory_candidates(
    tracks: pd.DataFrame,
    scene_id: str,
    acquisition_epoch_utc: float,
    config: ADSBConfig,
) -> tuple[list[dict[str, Any]], int]:
    try:
        from shapely.geometry import LineString
    except ImportError as exc:
        raise RuntimeError(
            "ADS-B matching requires the 'geo' optional dependencies"
        ) from exc

    scene = tracks.loc[tracks["scene_id"].astype(str).eq(str(scene_id))].copy()
    scene["epoch_utc"] = pd.to_numeric(scene["epoch_utc"], errors="coerce")
    scene["altitude_ft"] = pd.to_numeric(scene["altitude_ft"], errors="coerce")
    scene["x_m"] = pd.to_numeric(scene["x_m"], errors="coerce")
    scene["y_m"] = pd.to_numeric(scene["y_m"], errors="coerce")
    eligible = (
        scene["altitude_ft"].ge(config.minimum_altitude_ft)
        & scene["epoch_utc"]
        .sub(float(acquisition_epoch_utc))
        .abs()
        .le(60.0 * config.maximum_absolute_time_offset_min)
        & np.isfinite(scene["x_m"])
        & np.isfinite(scene["y_m"])
    )
    scene = scene.loc[eligible].copy()
    candidates: list[dict[str, Any]] = []
    for candidate_id, group in scene.groupby("candidate_id", sort=False):
        group = group.sort_values("epoch_utc", kind="mergesort")
        coordinates = group[["x_m", "y_m"]].to_numpy(float)
        if len(np.unique(coordinates, axis=0)) < 2:
            continue
        line = LineString(coordinates)
        candidates.append(
            {
                "candidate_id": str(candidate_id),
                "geometry": line,
                "track_bearing_deg": axial_bearing_deg(line),
                "minimum_absolute_time_offset_min": float(
                    np.min(
                        np.abs(
                            group["epoch_utc"].to_numpy(float) - acquisition_epoch_utc
                        )
                    )
                    / 60.0
                ),
                "points": int(len(group)),
            }
        )
    return candidates, int(len(scene))


def match_structures(
    structures: pd.DataFrame,
    tracks: pd.DataFrame,
    config: ADSBConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match complete structure skeletons to high-altitude trajectory candidates."""
    _require_columns(
        structures,
        {
            "scene_id",
            "structure_id",
            "date_yyyymmdd",
            "acquisition_epoch_utc",
            "skeleton_wkt",
        },
        "structure table",
    )
    _require_columns(
        tracks,
        {"scene_id", "candidate_id", "epoch_utc", "x_m", "y_m", "altitude_ft"},
        "trajectory table",
    )

    audit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for structure in structures.to_dict(orient="records"):
        scene_id = str(structure["scene_id"])
        structure_id = str(structure["structure_id"])
        acquisition = float(structure["acquisition_epoch_utc"])
        coverage_supported = _as_bool(structure.get("coverage_supported", True))
        base = {
            "scene_id": scene_id,
            "structure_id": structure_id,
            "date_yyyymmdd": str(structure["date_yyyymmdd"]),
            "acquisition_epoch_utc": acquisition,
            "coverage_supported": coverage_supported,
        }
        if not coverage_supported:
            audit_rows.append(
                {
                    **base,
                    "component_bearing_deg": math.nan,
                    "best_candidate_id": "",
                    "best_distance_km": math.nan,
                    "best_track_bearing_deg": math.nan,
                    "best_heading_difference_deg": math.nan,
                    "best_time_offset_min": math.nan,
                    "best_candidate_score": math.nan,
                    "runner_up_score_gap": math.nan,
                    "eligible_track_points": 0,
                    "trajectory_candidates": 0,
                    "adsb_status": NO_COVERAGE_STATUS,
                }
            )
            continue

        skeleton = _load_line(str(structure["skeleton_wkt"]))
        supplied_bearing = pd.to_numeric(
            pd.Series([structure.get("component_bearing_deg", np.nan)]), errors="coerce"
        ).iloc[0]
        component_bearing = (
            float(supplied_bearing)
            if np.isfinite(supplied_bearing)
            else axial_bearing_deg(skeleton)
        )
        trajectories, eligible_points = _trajectory_candidates(
            tracks, scene_id, acquisition, config
        )
        if not trajectories:
            audit_rows.append(
                {
                    **base,
                    "component_bearing_deg": component_bearing,
                    "best_candidate_id": "",
                    "best_distance_km": math.nan,
                    "best_track_bearing_deg": math.nan,
                    "best_heading_difference_deg": math.nan,
                    "best_time_offset_min": math.nan,
                    "best_candidate_score": math.nan,
                    "runner_up_score_gap": math.nan,
                    "eligible_track_points": eligible_points,
                    "trajectory_candidates": 0,
                    "adsb_status": NO_COVERAGE_STATUS,
                }
            )
            continue

        ranked: list[dict[str, Any]] = []
        for trajectory in trajectories:
            distance_km = float(skeleton.distance(trajectory["geometry"]) / 1_000.0)
            heading_difference = float(
                axial_difference_deg(component_bearing, trajectory["track_bearing_deg"])
            )
            score = normalized_candidate_score(distance_km, heading_difference, config)
            row = {
                **base,
                "component_bearing_deg": component_bearing,
                "candidate_id": trajectory["candidate_id"],
                "distance_km": distance_km,
                "track_bearing_deg": trajectory["track_bearing_deg"],
                "heading_difference_deg": heading_difference,
                "minimum_absolute_time_offset_min": trajectory[
                    "minimum_absolute_time_offset_min"
                ],
                "trajectory_points": trajectory["points"],
                "candidate_score": score,
            }
            ranked.append(row)
        ranked.sort(
            key=lambda row: (
                -row["candidate_score"],
                row["distance_km"],
                row["heading_difference_deg"],
                row["candidate_id"],
            )
        )
        best = ranked[0]
        runner_up_gap = (
            float(best["candidate_score"] - ranked[1]["candidate_score"])
            if len(ranked) > 1
            else math.nan
        )
        status = classify_candidate(
            best["distance_km"], best["heading_difference_deg"], runner_up_gap, config
        )
        for rank, row in enumerate(ranked, start=1):
            candidate_rows.append({**row, "candidate_rank": rank})
        audit_rows.append(
            {
                **base,
                "component_bearing_deg": component_bearing,
                "best_candidate_id": best["candidate_id"],
                "best_distance_km": best["distance_km"],
                "best_track_bearing_deg": best["track_bearing_deg"],
                "best_heading_difference_deg": best["heading_difference_deg"],
                "best_time_offset_min": best["minimum_absolute_time_offset_min"],
                "best_candidate_score": best["candidate_score"],
                "runner_up_score_gap": runner_up_gap,
                "eligible_track_points": eligible_points,
                "trajectory_candidates": len(ranked),
                "adsb_status": status,
            }
        )
    return pd.DataFrame(audit_rows), pd.DataFrame(candidate_rows)


def _observed_rates(covered: pd.DataFrame) -> dict[str, float]:
    if covered.empty:
        raise ValueError("No coverage-supported structures are available")
    return {
        "broad_or_better": float(covered["adsb_status"].isin(BROAD_STATUSES).mean()),
        "strict_geometry": float(covered["adsb_status"].isin(STRICT_STATUSES).mean()),
        "strict_unambiguous": float(
            covered["adsb_status"]
            .eq("geometrically_consistent_unambiguous_candidate")
            .mean()
        ),
    }


def permutation_nulls(
    audit: pd.DataFrame,
    config: ADSBConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen random-heading and within-date reassignment nulls."""
    _require_columns(
        audit,
        {
            "date_yyyymmdd",
            "component_bearing_deg",
            "best_distance_km",
            "best_track_bearing_deg",
            "runner_up_score_gap",
            "adsb_status",
        },
        "ADS-B audit table",
    )
    covered = audit.loc[audit["adsb_status"].ne(NO_COVERAGE_STATUS)].copy()
    covered = covered.reset_index(drop=True)
    for column in (
        "component_bearing_deg",
        "best_distance_km",
        "best_track_bearing_deg",
        "runner_up_score_gap",
    ):
        covered[column] = pd.to_numeric(covered[column], errors="coerce")
    descriptor = (
        np.isfinite(covered["component_bearing_deg"])
        & np.isfinite(covered["best_distance_km"])
        & np.isfinite(covered["best_track_bearing_deg"])
    )
    observed = _observed_rates(covered)
    candidate_columns = [
        "best_distance_km",
        "best_track_bearing_deg",
        "runner_up_score_gap",
    ]
    date_groups = [
        np.asarray(indices, dtype=int)
        for indices in covered.loc[descriptor]
        .groupby(covered.loc[descriptor, "date_yyyymmdd"].astype(str), sort=False)
        .groups.values()
    ]
    rng = np.random.default_rng(config.seed)
    draws: list[dict[str, Any]] = []
    valid = descriptor.to_numpy(bool)
    component = covered["component_bearing_deg"].to_numpy(float)
    for replicate in range(config.permutations):
        reassigned = covered[candidate_columns].copy()
        for indices in date_groups:
            source = covered.loc[indices, candidate_columns].to_numpy().copy()
            reassigned.loc[indices, candidate_columns] = source[
                rng.permutation(len(source))
            ]
        null_inputs = (
            (
                "within_date_candidate_reassignment",
                reassigned["best_distance_km"].to_numpy(float),
                reassigned["best_track_bearing_deg"].to_numpy(float),
                reassigned["runner_up_score_gap"].to_numpy(float),
            ),
            (
                "random_heading_rotation",
                covered["best_distance_km"].to_numpy(float),
                rng.uniform(0.0, 180.0, size=len(covered)),
                covered["runner_up_score_gap"].to_numpy(float),
            ),
        )
        for null_name, distance, heading, gap in null_inputs:
            heading_difference = axial_difference_deg(heading, component)
            strict = (
                valid
                & (distance <= config.strict_distance_km)
                & (heading_difference <= config.strict_heading_difference_deg)
            )
            broad = (
                valid
                & (distance <= config.broad_distance_km)
                & (heading_difference <= config.broad_heading_difference_deg)
            )
            unambiguous = strict & (
                (gap > config.unambiguous_score_gap) | ~np.isfinite(gap)
            )
            draws.append(
                {
                    "replicate": replicate,
                    "null": null_name,
                    "broad_or_better": float(broad.mean()),
                    "strict_geometry": float(strict.mean()),
                    "strict_unambiguous": float(unambiguous.mean()),
                }
            )

    replicates = pd.DataFrame(draws)
    summary_rows: list[dict[str, Any]] = []
    for null_name, group in replicates.groupby("null", sort=True):
        for metric, estimate in observed.items():
            values = group[metric].to_numpy(float)
            summary_rows.append(
                {
                    "null": null_name,
                    "metric": metric,
                    "observed": estimate,
                    "null_ci_low": float(np.quantile(values, 0.025)),
                    "null_median": float(np.quantile(values, 0.5)),
                    "null_ci_high": float(np.quantile(values, 0.975)),
                    "upper_tail_p": float(
                        (1 + np.sum(values >= estimate)) / (len(values) + 1)
                    ),
                    "coverage_supported_structures": int(len(covered)),
                    "structures_with_candidate_descriptor": int(descriptor.sum()),
                    "permutations": int(config.permutations),
                    "seed": int(config.seed),
                }
            )
    return pd.DataFrame(summary_rows), replicates


def _write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if destination.suffix.lower() == ".gz" else None
    frame.to_csv(destination, index=False, compression=compression)


def match_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Match complete contrail-structure skeletons to high-altitude ADS-B tracks"
    )
    parser.add_argument("--structures", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-output", default=None)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    args = parser.parse_args()
    config = ADSBConfig.from_json(args.config)
    audit, candidates = match_structures(
        pd.read_csv(args.structures, low_memory=False),
        pd.read_csv(args.tracks, low_memory=False),
        config,
    )
    _write_csv(audit, args.output)
    if args.candidate_output:
        _write_csv(candidates, args.candidate_output)


def permutation_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run ADS-B geometric permutation nulls"
    )
    parser.add_argument("--audit", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--replicates", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    args = parser.parse_args()
    config = ADSBConfig.from_json(args.config)
    summary, replicates = permutation_nulls(
        pd.read_csv(args.audit, low_memory=False), config
    )
    _write_csv(summary, args.summary)
    _write_csv(replicates, args.replicates)


if __name__ == "__main__":
    match_cli()
