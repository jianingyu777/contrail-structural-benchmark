# Input and output formats

All tabular inputs are UTF-8 CSV files. Identifiers are treated as strings,
missing numeric values use an empty field or `NaN`, angles are degrees and UTC
timestamps are Unix seconds unless noted otherwise.

## Segmentation dataset

Each split contains `image/` and `label/` directories. Files are paired by
identical stem. Images must contain three enhanced 8-bit channels. Labels may
be `0/1` or `0/255`; all non-zero label values are converted to foreground.

The archived corpus contains 19,793 patches: 12,579 in the original training
directory, 3,501 in validation and 3,713 in the unchanged test set. The
training program combines the first two directories and rebuilds stratified
fold 5. It writes every selected filename to `split.json`.

## ADS-B structure table

| Column | Unit/type | Meaning |
|---|---|---|
| `scene_id` | string | Scene identifier shared with the trajectory table |
| `structure_id` | string | Unique connected-structure identifier |
| `date_yyyymmdd` | `YYYYMMDD` | Acquisition date used by the within-date null |
| `acquisition_epoch_utc` | seconds | Acquisition time |
| `skeleton_wkt` | WKT | Complete `LineString` or `MultiLineString` skeleton in a projected metric CRS |
| `component_bearing_deg` | degrees, optional | Axial orientation; principal-axis orientation is calculated from the complete skeleton when absent |
| `coverage_supported` | Boolean, optional | Local archive-coverage flag; defaults to true |

## ADS-B trajectory table

One row represents one trajectory point.

| Column | Unit/type | Meaning |
|---|---|---|
| `scene_id` | string | Scene identifier |
| `candidate_id` | string | Aircraft/trajectory identifier within scene |
| `epoch_utc` | seconds | Point time |
| `x_m`, `y_m` | metres | Coordinates in exactly the same projected CRS as `skeleton_wkt` |
| `altitude_ft` | feet | Pressure or geometric altitude used by the archive screen |

The matcher sorts each candidate by time, filters high-altitude points, builds
one polyline and evaluates minimum polyline distance and axial orientation.
Its candidate output contains all scored trajectories. The audit output
contains the best candidate, runner-up score gap and one of these statuses:

- `geometrically_consistent_unambiguous_candidate`
- `geometrically_consistent_ambiguous_candidate`
- `broad_geometric_candidate`
- `no_geometrically_consistent_candidate`
- `no_high_altitude_adsb_points_in_scene_window`

Absence of a candidate is not a physical negative label when coverage is
incomplete.

## Height-prior structure table

Required: `structure_id`. For layer selection without rerunning the serialized
regressors, also provide `H_adsb_p50_km`, `H_caliop_p50_km` and `gate_w`.
When `gate_w` is absent, it is reconstructed from `width_px`, `length_px` and
`dBT2_med`. When model files are supplied, the code accepts the radiometric and
morphological feature columns required by each frozen pipeline.

The ADS-B-calibrated regressor uses the 14 ordered fields recorded in
`resources/height_lut/adsb_height_feature_cols.json`. Accepted aliases include
`BT*_core_med` for `BT*_c_med`, `BTD23_*_med` for the sign-reversed
`BTD32_*_med`, and `major_axis_px`/`minor_axis_px` for length and width. The
CALIOP-calibrated pipeline additionally consumes acquisition context
(`profile_lon`, `profile_lat`, `dt_min`, scene identifiers and processing
modes). Its thermal differences and the height bin are derived from the same
input row. See `examples/height_model_features.csv` for a non-proprietary
executable schema example.

## Height-layer table

Required: `structure_id`, `level_hPa` and either `z_m` or geopotential `z`.
Provide `RHi` directly or provide `t`, `q` and `level_hPa` so that ice-relative
humidity can be calculated. Optional ERA5 fields include `u`, `v`, `w`, `cc`,
`ciwc` and `clwc`.

## Optical-depth input

Required columns:

- `structure_id`
- `delta_L_TIR1`, `delta_L_TIR2`, `delta_L_TIR3`: calibrated cloudy-minus-
  background radiance anomalies in the same units as the LUT conversion
- `cth_km`: constrained cloud-top height
- `vza_deg`: viewing zenith angle

Optional per-row columns `thickness_km`, `effective_radius_um`,
`solar_zenith_angle_deg` and `atmosphere_index` override configuration
defaults. Output fields include the selected discrete `tau`, residual cost,
model anomaly, actual LUT coordinates and any clipping flag.

## Longwave input

Required: `structure_id`, `tau`, `response_amplitude_W_m2`, and either
`area_m2` or `area_km2`. Output fields are:

- `positive_instantaneous_nighttime_longwave_W_m2`
- `positive_instantaneous_nighttime_longwave_W`
- `positive_instantaneous_nighttime_longwave_MW`
- `response_shape` and `response_coefficient`

This endpoint is instantaneous and nighttime-only; it is not net or lifecycle
forcing.

## Bootstrap input

The input requires date and scene identifiers plus the numeric endpoint(s).
Defaults are `date_yyyymmdd` and `scene_id`. Multiple rows per scene are
allowed: the complete scene cluster is retained whenever that scene is drawn.
Supported statistics are mean, median, ratio of sums and Spearman correlation.

## Sensitivity input

For paired robustness summaries, provide one endpoint per scene and scenario,
with columns `date_yyyymmdd`, `scene_id`, `sensitivity_scenario` and the value
column. Only scenes observed under both the baseline and comparison scenario
enter each paired difference.
