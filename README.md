# A 30-m structural benchmark for task-aligned evaluation of geostationary contrail monitoring

## Dataset

The DOI for the 30-m segmentation dataset will be made publicly available
after acceptance of the associated article. 

## 

This repository contains two code paths associated with the 30-m contrail
structural benchmark:

1. MaxViT-Base encoder plus U-Net-like decoder training, validation, held-out
   testing, patch inference and overlapping-window whole-scene inference.
2. Physical and statistical audits covering ADS-B geometry, permutation nulls,
   height-prior layer selection, effective optical depth, instantaneous
   nighttime longwave response, clustered bootstrap and sensitivity analysis.

Figure-generation and manuscript-build code are intentionally excluded.

## Evidence boundary

The segmentation unit is a connected 30-m thermal support, called a
`contrail structure`. It is not necessarily one complete contrail or one
unique flight. An ADS-B candidate establishes geometric consistency, not
source-flight identity. The radiative endpoint is **positive instantaneous
nighttime longwave contribution**. It is not lifecycle energy forcing, net
climate forcing, avoided warming or mitigation benefit.

## Installation

The unified release environment was verified with Python 3.10, PyTorch 2.5.1,
torchvision 0.20.1 and NumPy 2.2.6. NumPy 2 is required to deserialize the
frozen CALIOP-calibrated model. A conda environment is supplied:

```bash
conda env create -f environment.yml
conda activate contrail-structural-benchmark
pip install -e ".[geo,test]"
```

The height regressors were serialized with scikit-learn 1.7.2; the ADS-B-
calibrated model also requires XGBoost 3.1.2. The original segmentation
working environment recorded PyTorch 2.0.1, while the released checkpoint was
strictly compatibility-tested under the unified environment above. Never load
an untrusted joblib or PyTorch file.

## 1. Segmentation

### Data layout

The development data are not bundled with this software. When access has been
authorized, use:

```text
DATA_ROOT/
  train/image/  train/label/
  val/image/    val/label/
  test/image/   test/label/
```

Images and labels are paired by identical filename stem. Images are enhanced
three-channel 8-bit inputs. Labels may contain `0/1` or `0/255`. The recovered
training protocol combines `train` and `val` for five-fold stratified training
with seed 3407. The production checkpoint is from fold 5; `test` remains
untouched until final evaluation.

See the dataset-access notice at the top of this page for availability and
distribution conditions.

### Frozen rules

| Item | Rule |
|---|---|
| Architecture | `maxvit_base_tf_512.in21k_ft_in1k` encoder; decoder channels 384, 192, 96, 64; one-channel mask head; auxiliary centerline head during training |
| Loss | weighted BCE + Tversky for batches containing positives + curriculum clDice + curriculum centerline BCE |
| Curriculum | topology terms start after 20% of training and increase linearly to weights 0.5 and 0.3 |
| Production threshold | 0.5, frozen before whole-scene inference |
| Training threshold | scan 0.20 to 0.95; maximize positive-patch micro-IoU subject to negative-patch FP <= 0.15, otherwise maximize the penalized score |
| Checkpoint score | selected positive-patch micro-IoU minus `lambda_fp` times negative-patch false-positive rate |
| Reporting threshold | maximum validation Dice; negative-patch false-positive rate breaks ties; selected once before held-out test evaluation |

The matching historical launcher and trainer have now been recovered. A
sanitized source snapshot is retained under
[`legacy/stageA_semantic_training`](legacy/stageA_semantic_training), while
`contrail-train` is the maintained implementation for the pinned release
environment. See [docs/TRAINING_SOURCE_AUDIT.md](docs/TRAINING_SOURCE_AUDIT.md)
and [docs/PROVENANCE.md](docs/PROVENANCE.md).

### Train and evaluate

```bash
contrail-train --data-root DATA_ROOT --output-dir outputs/production \
  --config configs/segmentation_paper.json --folds all

contrail-evaluate --data-root DATA_ROOT \
  --checkpoint outputs/production/stageA_semantic_fp_fold5_best_iou.pth \
  --output-json outputs/production/test_metrics.json
```

Use `--folds 5` when only the production fold is required. The maintained
trainer reproduces the recovered architecture, split, augmentation, loss,
EMA, hard-negative curriculum, scheduler, threshold scan and checkpoint file
names. It corrects a denominator-placement defect in the historical
multi-threshold validation loop; the exact historical implementation remains
available in `legacy/` for audit.

The evaluator selects the reporting threshold on validation data and applies
it once to the unchanged held-out test set. It records the checkpoint SHA-256,
threshold and complete confusion counts.

### Patch and whole-scene inference

```bash
contrail-infer-patches --image-dir PATCHES --checkpoint MODEL.pth \
  --output-dir outputs/patches --threshold 0.5 --save-probability

contrail-infer-scenes --manifest scenes.csv --path-column image_path \
  --checkpoint MODEL.pth --output-dir outputs/scenes --threshold 0.5 \
  --input-mode raw-tis --save-probability
```

Whole-scene defaults are 256-pixel tiles, 50% overlap, reflected edge padding
and Hann-weighted probability blending. `raw-tis` reproduces the frozen
three-band enhancement within each tile. Use `enhanced-8bit` only when the
input raster already contains the enhanced model channels.

## 2. Physical and statistical audits

Example CSVs are under `examples/`; complete schemas and units are in
[docs/DATA_FORMATS.md](docs/DATA_FORMATS.md).

### ADS-B geometry and null models

```bash
contrail-adsb-match --structures examples/adsb_structures.csv \
  --tracks examples/adsb_tracks.csv --output outputs/adsb_audit.csv \
  --candidate-output outputs/adsb_all_candidates.csv

contrail-adsb-null --audit outputs/adsb_audit.csv \
  --summary outputs/adsb_null_summary.csv \
  --replicates outputs/adsb_null_replicates.csv.gz
```

The matcher uses the complete skeleton and trajectory polylines in one
projected metric CRS. The broad rule is 10 km/45 degrees; the strict rule is
5 km/20 degrees. Both null models use 5,000 replicates and seed 20260810.
The public matcher defaults to a +/-30-min point window, but that numeric
window was not reported in the manuscript and is therefore explicitly marked
as a release default rather than a frozen paper parameter.

### Height-prior layer selection

```bash
contrail-height-select --structures examples/height_structures.csv \
  --layers examples/height_layers.csv --output outputs/selected_height.csv
```

The command accepts precomputed ADS-B- and CALIOP-calibrated median height
priors. Trusted frozen regressors can instead be supplied with `--adsb-model`
and `--caliop-model`. Candidate ERA5 layers lie within 1.5 km of either prior
or at 300, 250, 225 and 200 hPa. Ice-supersaturated candidates are preferred;
the selected layer maximizes RHi minus 2 percentage points per kilometre from
the blended prior.

The complete serialized-model path can be exercised with the synthetic
feature row supplied in the repository:

```bash
contrail-height-select \
  --structures examples/height_model_features.csv \
  --layers examples/height_layers.csv \
  --adsb-model resources/height_lut/adsb_calibrated_height_model_p50.joblib \
  --caliop-model resources/height_lut/caliop_calibrated_height_model_p50.joblib \
  --adsb-feature-columns resources/height_lut/adsb_height_feature_cols.json \
  --output outputs/selected_height_from_models.csv
```

This command also writes a provenance JSON containing artifact hashes,
runtime package versions and the frozen layer-selection settings.

### Effective optical depth and longwave response

```bash
contrail-retrieve-tau --input examples/optical_depth_input.csv \
  --lut resources/height_lut/contrail_lut_final_v5.nc \
  --output outputs/retrieved_tau.csv \
  --sensitivity-output outputs/retrieved_tau_sensitivities.csv

contrail-longwave --input examples/longwave_input.csv \
  --output outputs/longwave_contribution.csv
```

The optical-depth output is a discrete effective TIR LUT coordinate
conditional on supplied height and ice-layer assumptions. The optional
sensitivity output repeats retrieval at height offsets of +/-0.5 and +/-1 km,
thicknesses of 0.2 and 1.0 km, and effective radii of 5 and 20 micrometres.

The frozen response is

```text
delta_E(tau) = amplitude * (1 - exp(-2.0969536 * tau))
F = area_m2 * max(delta_E, 0)
```

`area_m2` is preferred. `area_km2` is accepted and converted to square metres.

### Confidence intervals and sensitivity summaries

```bash
contrail-bootstrap --input examples/scene_metrics.csv \
  --statistic mean --value fixed_budget_loss \
  --output outputs/fixed_budget_loss_bootstrap.json \
  --replicates-output outputs/fixed_budget_loss_replicates.csv.gz

contrail-sensitivity --input examples/sensitivity_scene_metrics.csv \
  --value fixed_budget_loss --scenario-column sensitivity_scenario \
  --baseline primary --summary outputs/sensitivity_summary.csv \
  --replicates-output outputs/sensitivity_replicates.csv.gz
```

The bootstrap resamples acquisition dates and then complete scene clusters
within each sampled date. Percentile intervals, requested and finite replicate
counts, random seed and cluster counts are written with every result.

## Frozen artifacts

The author-generated 1.1-MB libRadtran LUT and two author-generated
height-prior regressors are in `resources/height_lut/`; their versions and
SHA-256 digests are in the adjacent `artifact_manifest.json`. The 337-MB
segmentation checkpoint is excluded from Git history.
[CHECKPOINTS.md](CHECKPOINTS.md) records its exact digest and release
instructions.

## Tests

```bash
pytest

# Optional integration test for the external production checkpoint
CONTRAIL_CHECKPOINT=/path/to/stageA_semantic_fp_fold5_best_iou.pth \
  pytest tests/test_checkpoint_compatibility.py

# Optional integration test for the provider-controlled fold split
CONTRAIL_DATA_ROOT=/path/to/20251122data \
  pytest tests/test_dataset_split_compatibility.py
```

The test suite uses synthetic geometry and statistics plus the bundled frozen
LUT. Standard tests never require restricted imagery or raw trajectory
archives; the fold-split integration test reads labels in place and copies
nothing.

## Data and licences

The MIT licence covers the software only. Distribution of the original
SDGSAT-1 full-scene imagery is subject to authorization by the data provider
and is not granted by this repository. See
[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) before publishing a release.


