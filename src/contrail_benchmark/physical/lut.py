from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import packaged_config_path


K1 = np.asarray([1655.627802, 838.7063188, 543.0579522], dtype=np.float64)
K2 = np.asarray([1542.760922, 1342.718697, 1232.021436], dtype=np.float64)
BANDS = ("TIR1", "TIR2", "TIR3")


@dataclass(frozen=True)
class RetrievalConfig:
    thickness_km: float = 0.5
    effective_radius_um: float = 10.0
    solar_zenith_angle_deg: float = 0.0
    atmosphere_index: int = 0
    band_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    height_sensitivity_offsets_km: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)
    thickness_sensitivity_km: tuple[float, ...] = (0.2, 1.0)
    effective_radius_sensitivity_um: tuple[float, ...] = (5.0, 20.0)
    optical_depth_nodes: tuple[float, ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> "RetrievalConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("retrieval", payload)
        return cls(
            thickness_km=float(values.get("thickness_km", 0.5)),
            effective_radius_um=float(values.get("effective_radius_um", 10.0)),
            solar_zenith_angle_deg=float(values.get("solar_zenith_angle_deg", 0.0)),
            atmosphere_index=int(values.get("atmosphere_index", 0)),
            band_weights=tuple(
                float(value) for value in values.get("band_weights", [1, 1, 1])
            ),
            height_sensitivity_offsets_km=tuple(
                float(value)
                for value in values.get(
                    "height_sensitivity_offsets_km", [-1.0, -0.5, 0.5, 1.0]
                )
            ),
            thickness_sensitivity_km=tuple(
                float(value)
                for value in values.get("thickness_sensitivity_km", [0.2, 1.0])
            ),
            effective_radius_sensitivity_um=tuple(
                float(value)
                for value in values.get("effective_radius_sensitivity_um", [5.0, 20.0])
            ),
            optical_depth_nodes=tuple(
                float(value) for value in values.get("optical_depth_nodes", [])
            ),
        )


def brightness_temperature_to_radiance(
    brightness_temperature: np.ndarray,
    band_axis: int = -1,
) -> np.ndarray:
    temperature = np.asarray(brightness_temperature, dtype=np.float64)
    if band_axis == -1:
        k1 = K1.reshape((1,) * (temperature.ndim - 1) + (3,))
        k2 = K2.reshape((1,) * (temperature.ndim - 1) + (3,))
    elif band_axis == 0:
        k1 = K1.reshape((3,) + (1,) * (temperature.ndim - 1))
        k2 = K2.reshape((3,) + (1,) * (temperature.ndim - 1))
    else:
        raise ValueError("band_axis must be -1 or 0")
    return k1 / (np.exp(k2 / np.maximum(temperature, 1e-6)) - 1.0)


class ContrailLUT:
    """Radiance-domain interface to the frozen three-band ice-layer LUT."""

    REQUIRED_VARIABLES = {"bt_clear", "bt_cloudy", "optical_depth"}

    def __init__(self, path: str | Path) -> None:
        try:
            import xarray as xr
        except ImportError as exc:
            raise RuntimeError(
                "LUT retrieval requires the 'geo' optional dependencies"
            ) from exc
        self.path = Path(path)
        self.dataset = xr.open_dataset(self.path, engine="h5netcdf")
        missing = self.REQUIRED_VARIABLES - set(self.dataset.variables)
        if missing:
            raise ValueError(f"LUT is missing variables: {', '.join(sorted(missing))}")
        clear = brightness_temperature_to_radiance(
            self.dataset["bt_clear"].values, band_axis=-1
        )
        cloudy = brightness_temperature_to_radiance(
            self.dataset["bt_cloudy"].values, band_axis=-1
        )
        self.dataset["L_clear"] = xr.DataArray(
            clear,
            dims=self.dataset["bt_clear"].dims,
            coords=self.dataset["bt_clear"].coords,
        )
        self.dataset["L_cloudy"] = xr.DataArray(
            cloudy,
            dims=self.dataset["bt_cloudy"].dims,
            coords=self.dataset["bt_cloudy"].coords,
        )

    def close(self) -> None:
        self.dataset.close()

    def __enter__(self) -> "ContrailLUT":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def coordinate(self, name: str) -> np.ndarray:
        return np.asarray(self.dataset.coords[name].values, dtype=float)

    def clipped_coordinate(self, name: str, value: float) -> tuple[float, bool]:
        coordinate = self.coordinate(name)
        clipped = float(
            np.clip(float(value), np.nanmin(coordinate), np.nanmax(coordinate))
        )
        return clipped, not math.isclose(
            clipped, float(value), rel_tol=0.0, abs_tol=1e-12
        )

    @staticmethod
    def _interpolate(data_array: Any, coordinates: dict[str, float]) -> Any:
        output = data_array
        for dimension, requested in coordinates.items():
            values = np.asarray(output.coords[dimension].values, dtype=float)
            if values.size == 1:
                output = output.isel({dimension: 0})
                continue
            value = float(np.clip(requested, np.nanmin(values), np.nanmax(values)))
            exact = np.flatnonzero(np.isclose(values, value, rtol=0.0, atol=1e-12))
            if exact.size:
                output = output.isel({dimension: int(exact[0])})
            else:
                output = output.interp({dimension: value})
        return output

    def model_delta_radiance(
        self,
        cth_km: float,
        thickness_km: float,
        effective_radius_um: float,
        viewing_zenith_angle_deg: float,
        solar_zenith_angle_deg: float = 0.0,
        atmosphere_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        requested = {
            "cth": cth_km,
            "thickness": thickness_km,
            "reff": effective_radius_um,
            "vza": viewing_zenith_angle_deg,
            "sza": solar_zenith_angle_deg,
            "atmosphere_idx": atmosphere_index,
        }
        selected: dict[str, float] = {}
        clipped: dict[str, bool] = {}
        for name, value in requested.items():
            selected[name], clipped[name] = self.clipped_coordinate(name, float(value))
        selected["atmosphere_idx"] = int(selected["atmosphere_idx"])
        clear = self._interpolate(
            self.dataset["L_clear"],
            {
                "vza": selected["vza"],
                "sza": selected["sza"],
                "atmosphere_idx": selected["atmosphere_idx"],
            },
        ).values
        cloudy = self._interpolate(
            self.dataset["L_cloudy"],
            selected,
        ).values
        tau = self.coordinate("optical_depth")
        metadata_record = {
            "selected_coordinates": selected,
            "coordinate_clipped": clipped,
        }
        return tau, np.asarray(cloudy - clear[None, :], dtype=float), metadata_record

    def retrieve(
        self,
        observed_delta_radiance: np.ndarray,
        cth_km: float,
        thickness_km: float,
        effective_radius_um: float,
        viewing_zenith_angle_deg: float,
        band_weights: np.ndarray,
        solar_zenith_angle_deg: float = 0.0,
        atmosphere_index: int = 0,
        tau_prior: float | None = None,
        smooth_lambda: float = 0.0,
    ) -> dict[str, Any]:
        observed = np.asarray(observed_delta_radiance, dtype=float).reshape(-1)
        weights = np.asarray(band_weights, dtype=float).reshape(-1)
        if observed.shape != (3,) or weights.shape != (3,):
            raise ValueError(
                "Three observed anomalies and three band weights are required"
            )
        if not np.isfinite(observed).all():
            raise ValueError("Observed radiance anomalies must be finite")
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "Band weights must be finite, non-negative and not all zero"
            )
        tau, model, coordinate_record = self.model_delta_radiance(
            cth_km,
            thickness_km,
            effective_radius_um,
            viewing_zenith_angle_deg,
            solar_zenith_angle_deg,
            atmosphere_index,
        )
        residual = model - observed[None, :]
        cost = np.sum(weights[None, :] * residual**2, axis=1)
        if tau_prior is not None and smooth_lambda > 0:
            cost = cost + float(smooth_lambda) * (tau - float(tau_prior)) ** 2
        index = int(np.argmin(cost))
        return {
            "tau": float(tau[index]),
            "retrieval_cost": float(cost[index]),
            "tau_node_index": index,
            "model_delta_L_TIR1": float(model[index, 0]),
            "model_delta_L_TIR2": float(model[index, 1]),
            "model_delta_L_TIR3": float(model[index, 2]),
            **coordinate_record,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "sha256": _sha256(self.path),
            "title": self.dataset.attrs.get("title"),
            "description": self.dataset.attrs.get("description"),
            "creation_date": self.dataset.attrs.get("creation_date"),
            "libradtran_version": self.dataset.attrs.get("libradtran_version"),
            "dimensions": {
                key: int(value) for key, value in self.dataset.sizes.items()
            },
            "coordinates": {
                name: self.coordinate(name).tolist()
                for name in (
                    "cth",
                    "thickness",
                    "reff",
                    "optical_depth",
                    "cloud_fraction",
                    "vza",
                    "sza",
                    "atmosphere_idx",
                )
                if name in self.dataset.coords
            },
        }


def _value(record: dict[str, Any], name: str, default: float) -> float:
    value = pd.to_numeric(pd.Series([record.get(name, np.nan)]), errors="coerce").iloc[
        0
    ]
    return float(value) if np.isfinite(value) else float(default)


def retrieve_table(
    frame: pd.DataFrame,
    lut: ContrailLUT,
    config: RetrievalConfig,
) -> pd.DataFrame:
    required = {
        "structure_id",
        "delta_L_TIR1",
        "delta_L_TIR2",
        "delta_L_TIR3",
        "cth_km",
        "vza_deg",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Optical-depth table is missing: {', '.join(sorted(missing))}"
        )
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        result = lut.retrieve(
            np.asarray([record[f"delta_L_{band}"] for band in BANDS], dtype=float),
            cth_km=float(record["cth_km"]),
            thickness_km=_value(record, "thickness_km", config.thickness_km),
            effective_radius_um=_value(
                record, "effective_radius_um", config.effective_radius_um
            ),
            viewing_zenith_angle_deg=float(record["vza_deg"]),
            band_weights=np.asarray(config.band_weights, dtype=float),
            solar_zenith_angle_deg=_value(
                record, "solar_zenith_angle_deg", config.solar_zenith_angle_deg
            ),
            atmosphere_index=int(
                _value(record, "atmosphere_index", config.atmosphere_index)
            ),
        )
        selected = result.pop("selected_coordinates")
        clipped = result.pop("coordinate_clipped")
        rows.append(
            {
                **record,
                **result,
                **{f"lut_{key}": value for key, value in selected.items()},
                "any_lut_coordinate_clipped": any(clipped.values()),
                "clipped_lut_coordinates": ";".join(
                    key for key, value in clipped.items() if value
                ),
            }
        )
    return pd.DataFrame(rows)


def retrieval_sensitivities(
    frame: pd.DataFrame,
    lut: ContrailLUT,
    config: RetrievalConfig,
) -> pd.DataFrame:
    scenarios = [("primary", 0.0, config.thickness_km, config.effective_radius_um)]
    scenarios.extend(
        (
            f"height_{'plus' if offset > 0 else 'minus'}_{abs(offset):g}km",
            offset,
            config.thickness_km,
            config.effective_radius_um,
        )
        for offset in config.height_sensitivity_offsets_km
        if offset != 0
    )
    scenarios.extend(
        (f"thickness_{thickness:g}km", 0.0, thickness, config.effective_radius_um)
        for thickness in config.thickness_sensitivity_km
        if thickness != config.thickness_km
    )
    scenarios.extend(
        (f"effective_radius_{radius:g}um", 0.0, config.thickness_km, radius)
        for radius in config.effective_radius_sensitivity_um
        if radius != config.effective_radius_um
    )
    rows: list[pd.DataFrame] = []
    for name, height_offset, thickness, radius in scenarios:
        scenario = frame.copy()
        scenario["cth_km"] = (
            pd.to_numeric(scenario["cth_km"], errors="coerce") + height_offset
        )
        scenario["thickness_km"] = thickness
        scenario["effective_radius_um"] = radius
        result = retrieve_table(scenario, lut, config)
        result.insert(1, "sensitivity_scenario", name)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve discrete effective thermal-infrared optical depth"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    parser.add_argument("--sensitivity-output", default=None)
    parser.add_argument("--provenance-output", default=None)
    args = parser.parse_args()
    config = RetrievalConfig.from_json(args.config)
    frame = pd.read_csv(args.input, low_memory=False)
    with ContrailLUT(args.lut) as lut:
        if config.optical_depth_nodes:
            actual = lut.coordinate("optical_depth")
            expected = np.asarray(config.optical_depth_nodes, dtype=float)
            if actual.shape != expected.shape or not np.allclose(actual, expected):
                raise RuntimeError(
                    "Configured optical-depth nodes do not match the LUT"
                )
        output = retrieve_table(frame, lut, config)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(destination, index=False)
        if args.sensitivity_output:
            sensitivity = retrieval_sensitivities(frame, lut, config)
            sensitivity_path = Path(args.sensitivity_output)
            sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
            sensitivity.to_csv(sensitivity_path, index=False)
        provenance = {
            "lut": lut.provenance(),
            "retrieval_config": {
                "thickness_km": config.thickness_km,
                "effective_radius_um": config.effective_radius_um,
                "solar_zenith_angle_deg": config.solar_zenith_angle_deg,
                "atmosphere_index": config.atmosphere_index,
                "band_weights": list(config.band_weights),
                "height_sensitivity_offsets_km": list(
                    config.height_sensitivity_offsets_km
                ),
                "thickness_sensitivity_km": list(config.thickness_sensitivity_km),
                "effective_radius_sensitivity_um": list(
                    config.effective_radius_sensitivity_um
                ),
                "optical_depth_nodes": list(config.optical_depth_nodes),
            },
            "runtime_versions": {
                "numpy": _version("numpy"),
                "pandas": _version("pandas"),
                "xarray": _version("xarray"),
                "h5netcdf": _version("h5netcdf"),
                "h5py": _version("h5py"),
            },
            "interpretation": (
                "tau is a discrete effective thermal-infrared LUT coordinate conditional "
                "on the supplied height and fixed ice-layer assumptions."
            ),
        }
    provenance_path = (
        Path(args.provenance_output)
        if args.provenance_output
        else destination.with_suffix(".provenance.json")
    )
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    cli()
