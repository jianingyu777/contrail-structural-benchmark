from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..config import packaged_config_path


EPSILON = 0.622
STANDARD_GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class HeightPriorConfig:
    candidate_window_km: float = 1.5
    fallback_pressure_levels_hpa: tuple[float, ...] = (300.0, 250.0, 225.0, 200.0)
    rhi_threshold_percent: float = 100.0
    distance_penalty_rhi_points_per_km: float = 2.0

    @classmethod
    def from_json(cls, path: str | Path) -> "HeightPriorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("height_prior", payload)
        fallback = values.get(
            "fallback_pressure_levels_hpa", cls().fallback_pressure_levels_hpa
        )
        return cls(
            candidate_window_km=float(values.get("candidate_window_km", 1.5)),
            fallback_pressure_levels_hpa=tuple(float(value) for value in fallback),
            rhi_threshold_percent=float(values.get("rhi_threshold_percent", 100.0)),
            distance_penalty_rhi_points_per_km=float(
                values.get("distance_penalty_rhi_points_per_km", 2.0)
            ),
        )


def saturation_vapor_pressure_ice_pa(temperature_k: np.ndarray | float) -> np.ndarray:
    """Murphy-Koop saturation vapour pressure over ice in Pa."""
    temperature = np.asarray(temperature_k, dtype=float)
    log_pressure = (
        9.550426
        - 5723.265 / temperature
        + 3.53068 * np.log(temperature)
        - 0.00728332 * temperature
    )
    return np.exp(log_pressure)


def vapor_pressure_from_specific_humidity_pa(
    specific_humidity: np.ndarray | float,
    pressure_pa: np.ndarray | float,
) -> np.ndarray:
    humidity = np.asarray(specific_humidity, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    return humidity * pressure / (EPSILON + (1.0 - EPSILON) * humidity)


def relative_humidity_over_ice_percent(
    temperature_k: np.ndarray | float,
    specific_humidity: np.ndarray | float,
    pressure_pa: np.ndarray | float,
) -> np.ndarray:
    vapor_pressure = vapor_pressure_from_specific_humidity_pa(
        specific_humidity, pressure_pa
    )
    return 100.0 * vapor_pressure / saturation_vapor_pressure_ice_pa(temperature_k)


def layer_table_from_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Standardize an ERA5-like pressure-level profile and derive RHi and wind speed."""
    frame = profile.copy()
    if "level_hPa" not in frame and "isobaricInhPa" in frame:
        frame["level_hPa"] = frame["isobaricInhPa"]
    if "z_m" not in frame:
        if "z" not in frame:
            raise ValueError("Profile requires z_m or geopotential z")
        frame["z_m"] = (
            pd.to_numeric(frame["z"], errors="coerce") / STANDARD_GRAVITY_M_S2
        )
    if "RHi" not in frame:
        required = {"t", "q", "level_hPa"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Cannot derive RHi; missing: {', '.join(sorted(missing))}"
            )
        frame["RHi"] = relative_humidity_over_ice_percent(
            pd.to_numeric(frame["t"], errors="coerce").to_numpy(float),
            pd.to_numeric(frame["q"], errors="coerce").to_numpy(float),
            100.0 * pd.to_numeric(frame["level_hPa"], errors="coerce").to_numpy(float),
        )
    if "ws" not in frame and {"u", "v"}.issubset(frame.columns):
        frame["ws"] = np.hypot(
            pd.to_numeric(frame["u"], errors="coerce"),
            pd.to_numeric(frame["v"], errors="coerce"),
        )
    return frame


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def blended_height_prior_km(row: pd.Series | dict[str, Any]) -> float | None:
    adsb = row.get("H_adsb_p50_km", np.nan)
    caliop = row.get("H_caliop_p50_km", np.nan)
    gate = row.get("gate_w", np.nan)
    if _finite(adsb) and _finite(caliop) and _finite(gate):
        gate_value = float(np.clip(float(gate), 0.0, 1.0))
        return gate_value * float(adsb) + (1.0 - gate_value) * float(caliop)
    if _finite(adsb):
        return float(adsb)
    if _finite(caliop):
        return float(caliop)
    if _finite(row.get("H_mix_km", np.nan)):
        return float(row["H_mix_km"])
    return None


def select_height_layer(
    structure: pd.Series | dict[str, Any],
    layers: pd.DataFrame,
    config: HeightPriorConfig,
) -> dict[str, Any]:
    """Select a meteorological layer using the frozen prior-window and ISSR-first rule."""
    if layers.empty:
        raise ValueError("At least one pressure-level row is required")
    frame = layer_table_from_profile(layers)
    for column in ("z_m", "level_hPa", "RHi"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    height_km = frame["z_m"].to_numpy(float) / 1_000.0
    pressure = frame["level_hPa"].to_numpy(float)
    mixed = blended_height_prior_km(structure)
    if mixed is None:
        existing = structure.get("best_z_m", np.nan)
        finite_heights = height_km[np.isfinite(height_km)]
        if _finite(existing):
            mixed = float(existing) / 1_000.0
        elif finite_heights.size:
            mixed = float(np.median(finite_heights))
        else:
            raise ValueError(
                "No finite height prior or geopotential-height layer is available"
            )

    candidate = np.zeros(len(frame), dtype=bool)
    for column in ("H_adsb_p50_km", "H_caliop_p50_km"):
        prior = structure.get(column, np.nan)
        if _finite(prior):
            candidate |= np.isfinite(height_km) & (
                np.abs(height_km - float(prior)) <= config.candidate_window_km
            )
    for level in config.fallback_pressure_levels_hpa:
        candidate |= np.isfinite(pressure) & np.isclose(pressure, level)
    if not candidate.any():
        candidate = np.isfinite(height_km)
    if not candidate.any():
        raise ValueError("No finite geopotential-height layer is available")

    candidates = frame.loc[candidate].copy()
    candidate_heights = candidates["z_m"].to_numpy(float) / 1_000.0
    candidate_rhi = candidates["RHi"].to_numpy(float)
    issr = np.isfinite(candidate_rhi) & (candidate_rhi >= config.rhi_threshold_percent)
    selected_from_issr = bool(issr.any())
    if selected_from_issr:
        candidates = candidates.loc[issr].copy()
        candidate_heights = candidates["z_m"].to_numpy(float) / 1_000.0
        candidate_rhi = candidates["RHi"].to_numpy(float)

    finite_height = np.isfinite(candidate_heights)
    score = np.where(np.isfinite(candidate_rhi), candidate_rhi, -1.0e6)
    score = score - config.distance_penalty_rhi_points_per_km * np.abs(
        candidate_heights - float(mixed)
    )
    score = np.where(finite_height, score, np.nan)
    if np.isfinite(score).any():
        selected_position = int(np.nanargmax(score))
    else:
        selected_position = int(
            np.nanargmin(
                np.where(finite_height, np.abs(candidate_heights - mixed), np.nan)
            )
        )
    selected = candidates.iloc[selected_position]
    return {
        "H_mix_km": float(mixed),
        "selected_cth_km": float(selected["z_m"]) / 1_000.0,
        "selected_level_hPa": float(selected["level_hPa"])
        if _finite(selected["level_hPa"])
        else math.nan,
        "selected_RHi_percent": float(selected["RHi"])
        if _finite(selected["RHi"])
        else math.nan,
        "selected_score": float(score[selected_position])
        if np.isfinite(score[selected_position])
        else math.nan,
        "selected_from_ice_supersaturated_candidates": selected_from_issr,
        "candidate_levels_after_issr_gate": int(len(candidates)),
    }


def morphology_gate_weight(
    width_px: float,
    length_px: float,
    contrast_proxy: float,
    width_threshold_px: float = 12.0,
    aspect_threshold: float = 0.25,
) -> tuple[int, float]:
    """Reproduce the frozen morphology gate used to blend the two height priors."""
    if not _finite(width_px) or not _finite(length_px) or float(length_px) <= 0:
        return 0, 0.5
    aspect = float(width_px) / (float(length_px) + 1e-6)
    early_stage_proxy = int(
        float(width_px) <= width_threshold_px and aspect <= aspect_threshold
    )
    contrast = abs(float(contrast_proxy)) if _finite(contrast_proxy) else 0.0
    logit = (
        0.55 * (width_threshold_px - float(width_px))
        + 8.0 * (aspect_threshold - aspect)
        + 0.8 * contrast
    )
    gate = 1.0 / (1.0 + math.exp(-float(np.clip(logit, -700.0, 700.0))))
    return early_stage_proxy, gate


def add_height_feature_aliases(features: dict[str, Any]) -> dict[str, Any]:
    output = dict(features)
    for band in ("BT1", "BT2", "BT3"):
        core = output.get(f"{band}_core_med", np.nan)
        if f"{band}_c_med" not in output and _finite(core):
            output[f"{band}_c_med"] = float(core)
    for location in ("core", "bg"):
        source = output.get(f"BTD23_{location}_med", np.nan)
        target = "BTD32_c_med" if location == "core" else "BTD32_bg_med"
        if target not in output and _finite(source):
            output[target] = -float(source)
    if "dBTD32_med" not in output:
        delta = output.get("dBTD23_med", np.nan)
        if _finite(delta):
            output["dBTD32_med"] = -float(delta)
        elif _finite(output.get("BTD32_c_med")) and _finite(output.get("BTD32_bg_med")):
            output["dBTD32_med"] = float(output["BTD32_c_med"]) - float(
                output["BTD32_bg_med"]
            )
    length = output.get("major_axis_px", output.get("branch_len_px", np.nan))
    width = output.get("minor_axis_px", np.nan)
    if _finite(length):
        output["length_px"] = float(length)
    if _finite(width):
        output["width_px"] = float(width)
    if _finite(length) and float(length) > 0 and _finite(width):
        output["aspect"] = float(width) / (float(length) + 1e-6)
    return output


def height_bin_from_adsb_prior(height_km: Any) -> float:
    """Map the ADS-B-calibrated prior to the bins used by the CALIOP model."""
    if not _finite(height_km):
        return math.nan
    height = float(height_km)
    if height < 8.34:
        return 0.0
    if height < 9.98:
        return 2.0
    if height < 11.03:
        return 3.0
    return 4.0


def add_caliop_feature_aliases(features: dict[str, Any]) -> dict[str, Any]:
    """Align structure descriptors with the frozen CALIOP pipeline schema."""
    output = add_height_feature_aliases(features)
    for band in (1, 2, 3):
        core = output.get(f"BT{band}_c_med", output.get(f"BT{band}_core_med", np.nan))
        background = output.get(f"BT{band}_bg_med", np.nan)
        if f"BT{band}_c" not in output and _finite(core):
            output[f"BT{band}_c"] = float(core)
        if f"BT{band}_bg" not in output and _finite(background):
            output[f"BT{band}_bg"] = float(background)
        if f"dBT{band}" not in output:
            delta = output.get(f"dBT{band}_med", np.nan)
            if _finite(delta):
                output[f"dBT{band}"] = float(delta)
            elif _finite(output.get(f"BT{band}_c")) and _finite(
                output.get(f"BT{band}_bg")
            ):
                output[f"dBT{band}"] = float(output[f"BT{band}_c"]) - float(
                    output[f"BT{band}_bg"]
                )

    for label, first, second in (("BTD21", "BT2", "BT1"), ("BTD32", "BT3", "BT2")):
        for location in ("c", "bg"):
            target = f"{label}_{location}"
            first_value = output.get(f"{first}_{location}", np.nan)
            second_value = output.get(f"{second}_{location}", np.nan)
            if target not in output and _finite(first_value) and _finite(second_value):
                output[target] = float(first_value) - float(second_value)
        delta_target = f"d{label}"
        if (
            delta_target not in output
            and _finite(output.get(f"{label}_c"))
            and _finite(output.get(f"{label}_bg"))
        ):
            output[delta_target] = float(output[f"{label}_c"]) - float(
                output[f"{label}_bg"]
            )

    image_id = output.get("image_id", output.get("scene_id", "unknown"))
    output.setdefault("sceneid", str(image_id))
    output.setdefault("filename_root", str(image_id))
    if "profile_lon" not in output:
        longitude = output.get("head_lon", output.get("centroid_lon", np.nan))
        if _finite(longitude):
            output["profile_lon"] = float(longitude)
    if "profile_lat" not in output:
        latitude = output.get("head_lat", output.get("centroid_lat", np.nan))
        if _finite(latitude):
            output["profile_lat"] = float(latitude)
    output.setdefault("dt_min", 0.4)
    output.setdefault("mode_c", "raw")
    output.setdefault("mode_bg", "raw")
    if not _finite(output.get("z_bin", np.nan)):
        output["z_bin"] = height_bin_from_adsb_prior(
            output.get("H_adsb_p50_km", np.nan)
        )
    return output


def _load_model_bundle(path: str | Path) -> tuple[Any, dict[str, Any]]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Height-regressor loading requires joblib") from exc
    warnings.warn(
        "Joblib files execute pickle payloads. Load only the trusted frozen artifacts "
        "distributed with this repository.",
        RuntimeWarning,
        stacklevel=2,
    )
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], {
            key: value for key, value in payload.items() if key != "model"
        }
    if (
        isinstance(payload, (tuple, list))
        and len(payload) == 2
        and hasattr(payload[0], "predict")
        and isinstance(payload[1], dict)
    ):
        return payload[0], payload[1]
    return payload, {}


def _pipeline_columns(model: Any) -> tuple[list[str], list[str]] | None:
    preprocessor = getattr(model, "named_steps", {}).get("pre")
    if preprocessor is None or not hasattr(preprocessor, "transformers_"):
        return None
    numerical: list[str] = []
    categorical: list[str] = []
    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            numerical = list(columns)
        elif name == "cat":
            categorical = list(columns)
    return numerical, categorical


def predict_height_km(
    model: Any,
    features: dict[str, Any],
    feature_columns: Sequence[str] | None = None,
    metadata_record: dict[str, Any] | None = None,
    fill_values: dict[str, Any] | None = None,
) -> float:
    features = add_caliop_feature_aliases(features)
    columns = _pipeline_columns(model)
    if columns is not None:
        numerical, categorical = columns
        frame = pd.DataFrame([features])
        for column in numerical:
            if column not in frame:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in categorical:
            if column not in frame:
                frame[column] = "unknown"
            frame[column] = frame[column].fillna("unknown").astype(str)
        return float(model.predict(frame[numerical + categorical])[0])
    if feature_columns is None:
        feature_columns = getattr(model, "feature_names_in_", None)
    if feature_columns is None:
        raise ValueError(
            "A bare height model requires an explicit feature-column order"
        )
    values: list[float] = []
    for column in feature_columns:
        value = features.get(column, np.nan)
        if not _finite(value) and fill_values is not None:
            value = fill_values.get(column, np.nan)
        values.append(float(value) if _finite(value) else math.nan)
    array = np.asarray([values], dtype=np.float32)
    prediction = float(model.predict(array)[0])
    record = metadata_record or {}
    if "y_mean" in record and "y_std" in record:
        prediction = prediction * float(record["y_std"]) + float(record["y_mean"])
    elif "y_min" in record and "y_max" in record:
        prediction = prediction * (
            float(record["y_max"]) - float(record["y_min"])
        ) + float(record["y_min"])
    elif "target_scaler" in record:
        prediction = float(
            record["target_scaler"].inverse_transform([[prediction]])[0, 0]
        )
    return prediction


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_height_selection(
    structures: pd.DataFrame,
    layers: pd.DataFrame,
    config: HeightPriorConfig,
    adsb_model_path: str | Path | None = None,
    caliop_model_path: str | Path | None = None,
    adsb_feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if "structure_id" not in structures or "structure_id" not in layers:
        raise ValueError("Both input tables require structure_id")
    work = structures.copy()
    work["structure_id"] = work["structure_id"].astype(str)
    layers = layers.copy()
    layers["structure_id"] = layers["structure_id"].astype(str)

    adsb_bundle = _load_model_bundle(adsb_model_path) if adsb_model_path else None
    caliop_bundle = _load_model_bundle(caliop_model_path) if caliop_model_path else None
    records = [
        add_height_feature_aliases(record) for record in work.to_dict(orient="records")
    ]
    adsb_fill_values: dict[str, float] | None = None
    if adsb_bundle is not None and adsb_feature_columns is not None:
        adsb_fill_values = {}
        for column in adsb_feature_columns:
            values = [
                float(record[column])
                for record in records
                if _finite(record.get(column))
            ]
            adsb_fill_values[column] = float(np.median(values)) if values else math.nan

    rows: list[dict[str, Any]] = []
    for record in records:
        if adsb_bundle is not None and not _finite(record.get("H_adsb_p50_km")):
            record["H_adsb_p50_km"] = predict_height_km(
                adsb_bundle[0],
                record,
                adsb_feature_columns,
                adsb_bundle[1],
                adsb_fill_values,
            )
        record = add_caliop_feature_aliases(record)
        if caliop_bundle is not None and not _finite(record.get("H_caliop_p50_km")):
            record["H_caliop_p50_km"] = predict_height_km(
                caliop_bundle[0], record, None, caliop_bundle[1]
            )
        if not _finite(record.get("gate_w")):
            early, gate = morphology_gate_weight(
                record.get("width_px", record.get("minor_axis_px", np.nan)),
                record.get("length_px", record.get("major_axis_px", np.nan)),
                record.get("dBT2_med", np.nan),
            )
            record["early_stage_proxy"] = early
            record["gate_w"] = gate
        subset = layers.loc[layers["structure_id"].eq(str(record["structure_id"]))]
        selection = select_height_layer(record, subset, config)
        rows.append({**record, **selection})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen height-prior and ERA5 layer-selection rule"
    )
    parser.add_argument("--structures", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("physical_audit_paper.json")
    )
    parser.add_argument("--adsb-model", default=None)
    parser.add_argument("--caliop-model", default=None)
    parser.add_argument("--adsb-feature-columns", default=None)
    parser.add_argument("--provenance-output", default=None)
    args = parser.parse_args()

    feature_columns = None
    if args.adsb_feature_columns:
        feature_columns = json.loads(
            Path(args.adsb_feature_columns).read_text(encoding="utf-8")
        )
    config = HeightPriorConfig.from_json(args.config)
    result = run_height_selection(
        pd.read_csv(args.structures, low_memory=False),
        pd.read_csv(args.layers, low_memory=False),
        config,
        args.adsb_model,
        args.caliop_model,
        feature_columns,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)

    provenance_path = (
        Path(args.provenance_output)
        if args.provenance_output
        else destination.with_suffix(".provenance.json")
    )
    provenance = {
        "height_rule": {
            "candidate_window_km": config.candidate_window_km,
            "fallback_pressure_levels_hpa": list(config.fallback_pressure_levels_hpa),
            "rhi_threshold_percent": config.rhi_threshold_percent,
            "distance_penalty_rhi_points_per_km": config.distance_penalty_rhi_points_per_km,
        },
        "artifacts": {
            label: {"path": str(Path(path).resolve()), "sha256": _sha256(path)}
            for label, path in {
                "adsb_height_model": args.adsb_model,
                "caliop_height_model": args.caliop_model,
            }.items()
            if path
        },
        "runtime_versions": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
        },
        "interpretation": (
            "The selected ERA5 geopotential height is a model-assisted constraint; "
            "it is not an independent cloud-top-height observation."
        ),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
