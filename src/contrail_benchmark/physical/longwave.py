from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..config import packaged_config_path


@dataclass(frozen=True)
class LongwaveConfig:
    shared_response_coefficient: float = 2.0969536
    positive_only: bool = True
    interpretation: str = "positive instantaneous nighttime longwave contribution"

    @classmethod
    def from_json(cls, path: str | Path) -> "LongwaveConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("longwave", payload)
        return cls(
            shared_response_coefficient=float(
                values.get("shared_response_coefficient", 2.0969536)
            ),
            positive_only=bool(values.get("positive_only", True)),
            interpretation=str(
                values.get(
                    "interpretation",
                    "positive instantaneous nighttime longwave contribution",
                )
            ),
        )


def response_shape(
    optical_depth: np.ndarray | float,
    coefficient: float = 2.0969536,
) -> np.ndarray:
    tau = np.asarray(optical_depth, dtype=float)
    if (tau < 0).any():
        raise ValueError("Optical depth cannot be negative")
    if coefficient <= 0:
        raise ValueError("Response coefficient must be positive")
    return 1.0 - np.exp(-float(coefficient) * tau)


def longwave_response_w_m2(
    optical_depth: np.ndarray | float,
    response_amplitude_w_m2: np.ndarray | float,
    coefficient: float = 2.0969536,
    positive_only: bool = True,
) -> np.ndarray:
    response = np.asarray(response_amplitude_w_m2, dtype=float) * response_shape(
        optical_depth, coefficient
    )
    return np.maximum(response, 0.0) if positive_only else response


def longwave_contribution(
    optical_depth: np.ndarray | float,
    response_amplitude_w_m2: np.ndarray | float,
    area_m2: np.ndarray | float,
    coefficient: float = 2.0969536,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    area = np.asarray(area_m2, dtype=float)
    if (area < 0).any():
        raise ValueError("Area cannot be negative")
    response = longwave_response_w_m2(
        optical_depth,
        response_amplitude_w_m2,
        coefficient,
        positive_only,
    )
    return response, response * area


def fit_shared_response_coefficient(
    frame: pd.DataFrame,
    structure_column: str = "structure_id",
    tau_column: str = "tau",
    response_column: str = "direct_longwave_response_W_m2",
    bounds: tuple[float, float] = (0.05, 10.0),
) -> tuple[float, pd.Series, pd.DataFrame]:
    """Fit one response-shape coefficient and one amplitude per structure."""
    required = {structure_column, tau_column, response_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Direct-response table is missing: {', '.join(sorted(missing))}"
        )
    work = frame[[structure_column, tau_column, response_column]].copy()
    work[structure_column] = work[structure_column].astype(str)
    work[tau_column] = pd.to_numeric(work[tau_column], errors="coerce")
    work[response_column] = pd.to_numeric(work[response_column], errors="coerce")
    work = work.loc[
        np.isfinite(work[tau_column])
        & np.isfinite(work[response_column])
        & work[tau_column].ge(0)
    ].copy()
    if work.empty:
        raise ValueError("No finite direct-response states are available")

    def amplitudes(coefficient: float, source: pd.DataFrame = work) -> pd.Series:
        shape = response_shape(source[tau_column].to_numpy(float), coefficient)
        terms = pd.DataFrame(
            {
                structure_column: source[structure_column].to_numpy(),
                "shape_y": shape * source[response_column].to_numpy(float),
                "shape_squared": shape * shape,
            }
        )
        sums = terms.groupby(structure_column, sort=False)[
            ["shape_y", "shape_squared"]
        ].sum()
        return sums["shape_y"] / sums["shape_squared"].clip(lower=1e-12)

    def objective(coefficient: float) -> float:
        shape = response_shape(work[tau_column].to_numpy(float), coefficient)
        amplitude = amplitudes(coefficient)
        prediction = shape * work[structure_column].map(amplitude).to_numpy(float)
        observed = work[response_column].to_numpy(float)
        scale = (
            work.groupby(structure_column)[response_column]
            .transform("mean")
            .abs()
            .clip(lower=1e-12)
            .to_numpy(float)
        )
        return float(np.mean(((prediction - observed) / scale) ** 2))

    optimization = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not optimization.success:
        raise RuntimeError(
            f"Response-shape optimization failed: {optimization.message}"
        )
    coefficient = float(optimization.x)
    amplitude = amplitudes(coefficient)
    output = work.copy()
    output["fitted_amplitude_W_m2"] = output[structure_column].map(amplitude)
    output["predicted_longwave_response_W_m2"] = response_shape(
        output[tau_column].to_numpy(float), coefficient
    ) * output["fitted_amplitude_W_m2"].to_numpy(float)
    output["relative_error"] = (
        output["predicted_longwave_response_W_m2"] - output[response_column]
    ) / output[response_column].abs().clip(lower=1e-12)
    return coefficient, amplitude, output


def calculate_table(frame: pd.DataFrame, config: LongwaveConfig) -> pd.DataFrame:
    required = {"structure_id", "tau", "response_amplitude_W_m2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Longwave table is missing: {', '.join(sorted(missing))}")
    if "area_m2" in frame:
        area_m2 = pd.to_numeric(frame["area_m2"], errors="coerce").to_numpy(float)
        area_source = "area_m2"
    elif "area_km2" in frame:
        area_m2 = 1_000_000.0 * pd.to_numeric(
            frame["area_km2"], errors="coerce"
        ).to_numpy(float)
        area_source = "area_km2"
    else:
        raise ValueError("Longwave table requires area_m2 or area_km2")
    tau = pd.to_numeric(frame["tau"], errors="coerce").to_numpy(float)
    amplitude = pd.to_numeric(
        frame["response_amplitude_W_m2"], errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(tau).all()
        or not np.isfinite(amplitude).all()
        or not np.isfinite(area_m2).all()
    ):
        raise ValueError("tau, response amplitude and area must all be finite")
    response, contribution = longwave_contribution(
        tau,
        amplitude,
        area_m2,
        config.shared_response_coefficient,
        config.positive_only,
    )
    output = frame.copy()
    output["response_shape"] = response_shape(tau, config.shared_response_coefficient)
    output["positive_instantaneous_nighttime_longwave_W_m2"] = response
    output["positive_instantaneous_nighttime_longwave_W"] = contribution
    output["positive_instantaneous_nighttime_longwave_MW"] = contribution / 1_000_000.0
    output["area_input_column"] = area_source
    output["response_coefficient"] = config.shared_response_coefficient
    return output


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the bounded positive instantaneous nighttime longwave endpoint"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    args = parser.parse_args()
    config = LongwaveConfig.from_json(args.config)
    result = calculate_table(pd.read_csv(args.input, low_memory=False), config)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    record: dict[str, Any] = {
        "shared_response_coefficient": config.shared_response_coefficient,
        "positive_only": config.positive_only,
        "interpretation": config.interpretation,
        "equation": "amplitude * (1 - exp(-coefficient * tau)) * area_m2",
    }
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def fit_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the shared optical-depth response shape to direct RT states"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--structure-column", default="structure_id")
    parser.add_argument("--tau-column", default="tau")
    parser.add_argument("--response-column", default="direct_longwave_response_W_m2")
    args = parser.parse_args()
    coefficient, amplitudes, rows = fit_shared_response_coefficient(
        pd.read_csv(args.input, low_memory=False),
        args.structure_column,
        args.tau_column,
        args.response_column,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(destination, index=False)
    summary = {
        "shared_response_coefficient": coefficient,
        "structures": int(len(amplitudes)),
        "states": int(len(rows)),
        "median_absolute_relative_error_percent": float(
            100.0 * rows["relative_error"].abs().median()
        ),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    cli()
