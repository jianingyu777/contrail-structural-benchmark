from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("joblib")
pytest.importorskip("xgboost")

from contrail_benchmark.physical.height import (  # noqa: E402
    _load_model_bundle,
    predict_height_km,
)


RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources" / "height_lut"


def test_frozen_height_regressors_load_and_predict() -> None:
    features = {
        "image_id": "integration_scene",
        "head_lon": 120.0,
        "head_lat": 35.0,
        "BT1_core_med": 235.0,
        "BT2_core_med": 232.0,
        "BT3_core_med": 231.0,
        "BT1_bg_med": 240.0,
        "BT2_bg_med": 239.0,
        "BT3_bg_med": 238.0,
        "dBT1_med": -5.0,
        "dBT2_med": -7.0,
        "dBT3_med": -7.0,
        "BTD23_core_med": 1.0,
        "BTD23_bg_med": 1.0,
        "dBTD23_med": 0.0,
        "area_px": 500.0,
        "major_axis_px": 100.0,
        "minor_axis_px": 5.0,
    }
    feature_columns = json.loads(
        (RESOURCE_DIR / "adsb_height_feature_cols.json").read_text(encoding="utf-8")
    )
    adsb_model, adsb_metadata = _load_model_bundle(
        RESOURCE_DIR / "adsb_calibrated_height_model_p50.joblib"
    )
    adsb_height = predict_height_km(
        adsb_model, features, feature_columns, adsb_metadata
    )
    assert np.isfinite(adsb_height)
    assert 5.0 < adsb_height < 16.0

    features["H_adsb_p50_km"] = adsb_height
    caliop_model, caliop_metadata = _load_model_bundle(
        RESOURCE_DIR / "caliop_calibrated_height_model_p50.joblib"
    )
    caliop_height = predict_height_km(caliop_model, features, None, caliop_metadata)
    assert np.isfinite(caliop_height)
    assert 5.0 < caliop_height < 16.0
