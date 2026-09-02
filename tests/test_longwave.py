from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contrail_benchmark.physical.longwave import (
    LongwaveConfig,
    calculate_table,
    fit_shared_response_coefficient,
    longwave_contribution,
    response_shape,
)


def test_frozen_response_formula_and_units() -> None:
    tau = 0.2
    amplitude = 12.0
    area_m2 = 2_000_000.0
    response, contribution = longwave_contribution(tau, amplitude, area_m2)
    expected = amplitude * (1.0 - np.exp(-2.0969536 * tau))
    assert float(response) == pytest.approx(expected)
    assert float(contribution) == pytest.approx(expected * area_m2)


def test_km2_input_is_converted_to_square_metres() -> None:
    frame = pd.DataFrame(
        {
            "structure_id": ["a"],
            "tau": [0.1],
            "response_amplitude_W_m2": [10.0],
            "area_km2": [2.0],
        }
    )
    output = calculate_table(frame, LongwaveConfig())
    expected_mw = 2.0 * 10.0 * float(response_shape(0.1))
    assert output.loc[
        0, "positive_instantaneous_nighttime_longwave_MW"
    ] == pytest.approx(expected_mw)


def test_shared_coefficient_can_be_recovered_from_synthetic_states() -> None:
    coefficient = 1.7
    rows = []
    for structure, amplitude in (("a", 8.0), ("b", 14.0), ("c", 20.0)):
        for tau in (0.02, 0.08, 0.2, 0.5):
            rows.append(
                {
                    "structure_id": structure,
                    "tau": tau,
                    "direct_longwave_response_W_m2": amplitude
                    * float(response_shape(tau, coefficient)),
                }
            )
    fitted, amplitudes, _ = fit_shared_response_coefficient(pd.DataFrame(rows))
    assert fitted == pytest.approx(coefficient, rel=1e-3)
    assert len(amplitudes) == 3
