from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contrail_benchmark.physical.height import (
    HeightPriorConfig,
    add_caliop_feature_aliases,
    height_bin_from_adsb_prior,
    morphology_gate_weight,
    relative_humidity_over_ice_percent,
    select_height_layer,
)


def test_relative_humidity_is_finite() -> None:
    value = relative_humidity_over_ice_percent(220.0, 1e-4, 25_000.0)
    assert np.isfinite(value)
    assert float(value) > 0


def test_issr_first_layer_selection() -> None:
    structure = {
        "H_adsb_p50_km": 10.0,
        "H_caliop_p50_km": 10.4,
        "gate_w": 0.5,
    }
    layers = pd.DataFrame(
        {
            "level_hPa": [300.0, 250.0, 225.0, 200.0],
            "z_m": [9_000.0, 10_000.0, 10_500.0, 11_500.0],
            "RHi": [90.0, 99.0, 105.0, 101.0],
        }
    )
    selected = select_height_layer(structure, layers, HeightPriorConfig())
    assert selected["selected_from_ice_supersaturated_candidates"] is True
    assert selected["selected_level_hPa"] == 225.0
    assert selected["selected_cth_km"] == pytest.approx(10.5)


def test_morphology_gate_prefers_narrow_linear_state() -> None:
    early, narrow_weight = morphology_gate_weight(5.0, 100.0, -2.0)
    _, diffuse_weight = morphology_gate_weight(30.0, 100.0, -2.0)
    assert early == 1
    assert narrow_weight > diffuse_weight


def test_caliop_aliases_reproduce_frozen_feature_definitions() -> None:
    features = add_caliop_feature_aliases(
        {
            "image_id": "scene_001",
            "head_lon": 120.0,
            "head_lat": 35.0,
            "H_adsb_p50_km": 10.2,
            "BT1_core_med": 235.0,
            "BT2_core_med": 232.0,
            "BT3_core_med": 231.0,
            "BT1_bg_med": 240.0,
            "BT2_bg_med": 239.0,
            "BT3_bg_med": 238.0,
        }
    )
    assert features["BT1_c"] == 235.0
    assert features["dBT2"] == -7.0
    assert features["BTD32_c"] == -1.0
    assert features["dBTD32"] == 0.0
    assert features["z_bin"] == 3.0
    assert features["sceneid"] == "scene_001"
    assert height_bin_from_adsb_prior(np.nan) != height_bin_from_adsb_prior(np.nan)


def test_missing_heights_fail_without_runtime_warning() -> None:
    layers = pd.DataFrame(
        {
            "level_hPa": [300.0, 250.0],
            "z_m": [np.nan, np.nan],
            "RHi": [90.0, 105.0],
        }
    )
    with pytest.raises(ValueError, match="No finite height prior"):
        select_height_layer({}, layers, HeightPriorConfig())
