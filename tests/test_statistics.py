from __future__ import annotations

import pandas as pd
import pytest

from contrail_benchmark.physical.statistics import (
    BootstrapConfig,
    hierarchical_bootstrap,
    mean_statistic,
    paired_sensitivity_bootstrap,
)


def test_hierarchical_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "date_yyyymmdd": ["d1", "d1", "d2", "d2"],
            "scene_id": ["s1", "s2", "s3", "s4"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    config = BootstrapConfig(replicates=50, seed=9)
    summary_a, draws_a = hierarchical_bootstrap(frame, mean_statistic("value"), config)
    summary_b, draws_b = hierarchical_bootstrap(frame, mean_statistic("value"), config)
    assert summary_a == summary_b
    pd.testing.assert_frame_equal(draws_a, draws_b)
    assert summary_a["estimate"] == pytest.approx(2.5)


def test_paired_sensitivity_uses_common_scenes() -> None:
    rows = []
    for date, scene, baseline in (
        ("d1", "s1", 1.0),
        ("d1", "s2", 2.0),
        ("d2", "s3", 3.0),
    ):
        rows.append(
            {
                "date_yyyymmdd": date,
                "scene_id": scene,
                "scenario": "primary",
                "value": baseline,
            }
        )
        rows.append(
            {
                "date_yyyymmdd": date,
                "scene_id": scene,
                "scenario": "plus",
                "value": baseline + 0.5,
            }
        )
    summary, _ = paired_sensitivity_bootstrap(
        pd.DataFrame(rows),
        "value",
        "scenario",
        "primary",
        BootstrapConfig(replicates=20, seed=3),
    )
    plus = summary.loc[summary["scenario"].eq("plus")].iloc[0]
    assert plus["mean_difference"] == pytest.approx(0.5)
    assert plus["paired_scenes"] == 3
