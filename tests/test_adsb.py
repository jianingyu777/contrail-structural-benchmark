from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")

from contrail_benchmark.physical.adsb import (
    ADSBConfig,
    axial_difference_deg,
    match_structures,
    permutation_nulls,
)


def test_axial_difference_wraps_at_180_degrees() -> None:
    assert axial_difference_deg(179.0, 1.0) == pytest.approx(2.0)
    assert axial_difference_deg(10.0, 100.0) == pytest.approx(90.0)


def test_full_skeleton_match_finds_parallel_track() -> None:
    structures = pd.DataFrame(
        {
            "scene_id": ["S1"],
            "structure_id": ["C1"],
            "date_yyyymmdd": ["20240101"],
            "acquisition_epoch_utc": [1_000.0],
            "skeleton_wkt": ["LINESTRING (0 0, 5000 0, 10000 0)"],
            "coverage_supported": [True],
        }
    )
    tracks = pd.DataFrame(
        {
            "scene_id": ["S1", "S1", "S1", "S1"],
            "candidate_id": ["parallel", "parallel", "cross", "cross"],
            "epoch_utc": [950.0, 1_050.0, 950.0, 1_050.0],
            "x_m": [0.0, 10_000.0, 5_000.0, 5_000.0],
            "y_m": [1_000.0, 1_000.0, -5_000.0, 5_000.0],
            "altitude_ft": [35_000.0] * 4,
        }
    )
    audit, candidates = match_structures(structures, tracks, ADSBConfig())
    row = audit.iloc[0]
    assert row["best_candidate_id"] == "parallel"
    assert row["best_distance_km"] == pytest.approx(1.0)
    assert row["best_heading_difference_deg"] == pytest.approx(0.0)
    assert row["adsb_status"] == "geometrically_consistent_unambiguous_candidate"
    assert len(candidates) == 2


def test_permutation_null_is_seed_reproducible() -> None:
    count = 20
    audit = pd.DataFrame(
        {
            "date_yyyymmdd": [f"202401{1 + index // 5:02d}" for index in range(count)],
            "component_bearing_deg": np.linspace(0, 171, count),
            "best_distance_km": np.full(count, 1.0),
            "best_track_bearing_deg": np.linspace(0, 171, count),
            "runner_up_score_gap": np.full(count, 0.5),
            "adsb_status": ["geometrically_consistent_unambiguous_candidate"] * count,
        }
    )
    config = ADSBConfig(permutations=25, seed=123)
    summary_a, draws_a = permutation_nulls(audit, config)
    summary_b, draws_b = permutation_nulls(audit, config)
    pd.testing.assert_frame_equal(draws_a, draws_b)
    pd.testing.assert_frame_equal(summary_a, summary_b)
    assert len(draws_a) == 50
    assert set(summary_a["null"]) == {
        "random_heading_rotation",
        "within_date_candidate_reassignment",
    }
