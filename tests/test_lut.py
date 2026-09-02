from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xarray")

from contrail_benchmark.physical.lut import ContrailLUT


LUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "height_lut"
    / "contrail_lut_final_v5.nc"
)


@pytest.mark.skipif(
    not LUT_PATH.exists(), reason="frozen LUT artifact is not installed"
)
def test_lut_recovers_an_exact_node() -> None:
    with ContrailLUT(LUT_PATH) as lut:
        tau, anomalies, _ = lut.model_delta_radiance(10.0, 0.5, 10.0, 8.0)
        index = int(np.flatnonzero(np.isclose(tau, 0.08))[0])
        result = lut.retrieve(
            anomalies[index],
            cth_km=10.0,
            thickness_km=0.5,
            effective_radius_um=10.0,
            viewing_zenith_angle_deg=8.0,
            band_weights=np.ones(3),
        )
    assert result["tau"] == pytest.approx(0.08)
    assert result["retrieval_cost"] == pytest.approx(0.0, abs=1e-14)
