# Six-model Version 3 benchmark (seed 42)

This directory is the authoritative result snapshot for the **current**
Sect. 5 comparison in the final manuscript. It replaces the earlier
four-model, three-seed table as the current benchmark.

## Experiment

- Data: released Version 3 `train/`, `validation/`, `test/` partitions.
- Test partition: 3,122 patches (1,422 positive, 1,700 difficult negative),
  from 120 scene entries and 51 acquisition dates.
- Models: MaxViT-U-Net, U-Net-ResNet34, SegFormer-B0, DeepLabV3+-ResNet50,
  FCN-ResNet50, PSPNet-ResNet50.
- Weights: seed 42 safetensors; SHA-256 in `checkpoint_manifest.csv`.
- Primary metric: mean positive-patch Dice, calculated within each positive
  patch and averaged over the 1,422 positive test patches.

## Table 6: exact thresholds and audit

- `Table6_exact_thresholds.csv`: validation-selected threshold for every model.
  FCN 0.53, U-Net 0.30, DeepLabV3+ 0.24, PSPNet 0.26, SegFormer 0.30,
  MaxViT 0.34 (all used the documented maximum-Dice fallback).
- `Table6_seed42_aggregate_results.csv`: full aggregate metrics, including
  Dice, IoU, precision, recall, negative activation, clDice and centerline
  recall.
- `Table6_full_per_patch_audit_all_models_seed42.csv`: 18,732 rows of
  per-patch metrics for six models across all 3,122 test patches.
- `table6_metrics.csv`: compact transcription used by the manuscript table.
- `checkpoint_manifest.csv`: filenames, byte sizes and SHA-256 hashes of the
  six seed-42 weights. Binary weights are distributed separately.
- `model_summaries/`: validation decisions and held-out test summaries for all
  six models.

## Figure 7

- `figure7/Figure7_Version3_scene_date_disjoint_predictions.png` (also PDF/SVG):
  final qualitative plate used in the manuscript.
- `figure7/Figure7_selected_patch_audit_all_models_seed42.csv`: the five
  selected patches and all six model predictions.
- `figure7/source_data.csv` and `figure7/caption.txt`: selection metadata and
  final caption.

## Table 7: manual reference-mask check

- `table7_manual_masks.csv`: 60-image manual-mask results for the earlier
  archived checkpoint set (47 positive, 13 empty).
- `table7_all60_metrics.csv`: full-precision all-60 metrics with negative
  counts.
- `table7_thresholds.csv`: fixed thresholds used for that check.
- `table7_manual_rois.csv`: 60-row ROI inventory with record IDs, source
  splits, foreground flags and mask SHA-256 values.
- `table7_checkpoint_manifest.csv`: archived checkpoint filenames, byte sizes,
  SHA-256, thresholds and architectures.
- `table7_checkpoint_archive.json`: archive name, size and SHA-256 for the
  separately distributed checkpoint archive.

The 60 reference, valid and ignore masks are in
[`data/table7_manual_masks/`](../../data/table7_manual_masks/). Table 7 uses an
**earlier archived checkpoint set**, not the seed-42 weights in
`checkpoint_manifest.csv`; 43 of the 60 central crops are in its training split,
17 in validation, and none in the held-out test split. It is an in-domain
annotation-reference check and is not directly comparable with Table 6.
See [`docs/TABLE7_REPRODUCTION.md`](../../docs/TABLE7_REPRODUCTION.md) for the
full preparation and evaluation workflow.

## Provenance

- `COMPARISON_AUDIT_20261120.md`: original six-model comparison audit.
- `reproduction_manifest.json`: SHA-256 manifest for the transferred
  reproduction package.
