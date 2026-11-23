# Six-model Version 3 benchmark (seed 42)

This directory is the authoritative result snapshot for the **current**
Sect. 5 comparison in the revised manuscript. It replaces the earlier
four-model, three-seed table as the current benchmark.

## Experiment

- Data: released Version 3 `train/`, `validation/`, `test/` partitions.
- Test partition: 3,122 patches (1,422 positive, 1,700 difficult negative),
  from 120 scene entries and 51 acquisition dates.
- Models: MaxViT-U-Net, U-Net-ResNet34, SegFormer-B0, DeepLabV3+-ResNet50,
  FCN-ResNet50, PSPNet-ResNet50.
- Weights: seed 42 safetensors, SHA-256 in `checkpoint_manifest.csv`.
- Primary metric: mean positive-patch Dice, calculated within each positive
  patch and averaged over the 1,422 positive test patches.

## Files

- `table6_metrics.csv`: current Table 6 values (Dice, positive-patch IoU,
  pixel-pooled IoU, precision, recall, negative-patch activation).
- `table7_manual_masks.csv`: Table 7 values for the separate 20-image
  manually drawn reference-mask check.
- `table7_thresholds.csv`: thresholds used for the Table 7 manual-mask check.
- `checkpoint_manifest.csv`: filenames, byte sizes and SHA-256 hashes of the
  six seed-42 weights. Binary weights are not stored in Git.

## Provenance

The values were transcribed from the project audit record
`recovery_19455/COMPARISON_AUDIT_20261120.md` (2026-11-20). That audit is the
source of record for the exact thresholds and evaluation run. The public
scripts in `code/` provide the model implementations and evaluation path;
the six checkpoint files themselves are distributed separately because of
their size.

Table 7 uses an **earlier archived checkpoint set**, not the seed-42 weights
in `checkpoint_manifest.csv`. It is retained as an annotation-dependent
reference check and is not directly comparable with Table 6.
