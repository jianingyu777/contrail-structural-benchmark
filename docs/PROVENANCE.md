# Code provenance and evidence boundaries

## Segmentation

The architecture, patch preprocessing and whole-scene inference path were
migrated from the code-only package
`contrail_maxvit_segmentation_open_source_20260624`. The released production
checkpoint confirms encoder, decoder, segmentation-head and centerline-head
parameter names. A release QA load found zero missing and zero unexpected
parameters across 83,948,118 model parameters.

The matching historical training call chain was recovered on 2 September
2026 from `stageA_semantic_training_code_20260902.zip` (SHA-256
`8AFB514EB0D3414546A335D3C17550705B649F7E0D57F42C2D0DC332F2A6EC27`):

```text
main_cv_fp.py
  -> cv_trainer_fp.py::trainer_cv_fp()
  -> models/timm_model.py::Model
  -> models/unet_decoder.py::UnetDecoder
```

The source writes the exact `stageA_semantic_fp_fold*_*.pth` naming family.
Its archived fold summary matches the five reported positive-patch IoUs. A
sanitized snapshot is included under `legacy/stageA_semantic_training`; only
the private default data path was replaced. Original file hashes are retained
in the adjacent `source_manifest.json`.

The plain checkpoint contains no command line, optimizer state, epoch or
package versions. Therefore the most likely launch command and default
arguments are supported by source evidence but command-line overrides cannot
be recovered with certainty. The decoder file also has a filesystem timestamp
later than the checkpoint, although all checkpoint parameter names and tensor
shapes match.

Each training epoch scans thresholds from 0.20 to 0.95. It first maximizes
positive-patch micro-IoU among thresholds with negative-patch false-positive
rate at most 0.15; if none qualifies, it maximizes

```text
positive-patch IoU - lambda_fp * negative-patch false-positive rate.
```

The historical loop increments the negative-patch denominator inside its
threshold loop, yielding a nonstandard rate. This exact behavior remains in
the legacy source. The maintained `contrail-train` implementation corrects the
denominator while preserving the recovered model, data split, augmentation,
loss, EMA, hard-negative curriculum, scheduler and output names. Consequently,
newly retrained checkpoint selection can differ from the historical run.

The reporting program independently selects a threshold by validation Dice
and applies it once to the held-out test set. This reporting threshold does not
replace the frozen 0.5 threshold used for production masks.

The original segmentation working environment recorded PyTorch 2.0.1. The
checkpoint has additionally been loaded strictly and exercised under the
unified release environment (PyTorch 2.5.1, torchvision 0.20.1 and NumPy
2.2.6). The newer release environment is used because the frozen CALIOP
joblib artifact requires NumPy 2 for deserialization.

## ADS-B audit

The paper-level permutation implementation is preserved exactly in logic:

- broad geometry: distance <= 10 km and axial difference <= 45 degrees;
- strict geometry: distance <= 5 km and axial difference <= 20 degrees;
- strict unambiguous: strict geometry and runner-up score gap > 0.25, or no
  finite runner-up;
- 5,000 random-heading rotations over [0, 180) degrees;
- 5,000 within-date reassignments of candidate distance, bearing, and
  ambiguity descriptors.

The matcher operates on complete skeleton and trajectory polylines in a common
projected coordinate system. Its normalized candidate score is exposed in the
output and configuration. The threshold classifications and the two
permutation nulls reproduce the frozen analysis logic. The original matcher
did not preserve a standalone argument dump, so the public runner-up ranking
score is a documented reconstruction. Likewise, the public matcher's +/-30-min
point window is a configurable release default because the manuscript did not
state a numeric timing tolerance. Neither setting should be described as a
verified frozen paper parameter without an archived run record.

Candidate status is physical-consistency evidence, not source attribution.
Absence is not a negative label when archive coverage is incomplete.

## Height and optical depth

The height prior is a morphology-gated blend of two frozen statistical
regressors calibrated separately against ADS-B and CALIOP reference samples.
It proposes
candidate ERA5 layers; it does not directly set a flight height. Ice-
supersaturated candidates are preferred, and the selected layer maximizes RHi
minus 2 percentage points per kilometre from the blended prior. The selected
ERA5 geopotential height is a model-assisted constraint.

The optical-depth value is the discrete node of a full-fill ice-layer LUT that
minimizes the equally weighted three-band radiance-anomaly residual. It is not
independent optical-depth truth. The LUT records libRadtran 2.0.4 in its global
attributes. The primary assumptions are 0.5-km layer thickness and
10-micrometre effective radius. The release code also propagates height,
thickness and effective-radius sensitivity states.

The NetCDF4/HDF5 artifact is opened through a fixed `h5netcdf` backend rather
than xarray's environment-dependent engine discovery. This choice affects file
I/O only; the test suite verifies exact-node retrieval from the frozen LUT.

## Longwave response

The response approximation is

```text
delta_E(tau) = amplitude * (1 - exp(-2.0969536 * tau))
F = area_m2 * max(delta_E, 0).
```

It estimates positive instantaneous nighttime longwave contribution only.

## Statistical inference

The generic bootstrap resamples dates with replacement, then complete scenes
within each selected date. All rows belonging to a sampled scene remain
together. The release reports percentile intervals, finite replicate counts,
cluster counts and the random seed. Sensitivity contrasts are paired on common
date-scene identifiers before applying the same hierarchy.
