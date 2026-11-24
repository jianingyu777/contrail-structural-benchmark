# Table 7 reproduction: 60-image in-domain manual-mask check

Table 7 evaluates the **earlier archived checkpoint set** on 60 hand-annotated
512 x 512 ROIs. It is an in-domain annotation-reference check, not an
unseen-image generalization test: the central 256 x 256 crop of every ROI is a
patch that already occurs in the archived checkpoints' training or validation
pool (43 training, 17 validation, 0 held-out test).

## What is included in this repository

| Path | Contents |
|---|---|
| `data/table7_manual_masks/masks/` | 60 binary reference masks (512 x 512) drawn from scratch by human annotators |
| `data/table7_manual_masks/valid/` | 60 valid-area masks |
| `data/table7_manual_masks/ignore/` | 60 ignore-area masks |
| `results/six_model_seed42/table7_manual_rois.csv` | 60-row inventory: annotation ID, role, source split, scene, dataset record ID, foreground flag, mask SHA-256 |
| `results/six_model_seed42/table7_thresholds.csv` | Fixed thresholds for the six archived checkpoints |
| `results/six_model_seed42/table7_manual_masks.csv` | Final Table 7 values |
| `results/six_model_seed42/table7_all60_metrics.csv` | Full-precision all-60 metrics and negative counts |
| `results/six_model_seed42/table7_checkpoint_manifest.csv` | Archived checkpoint filenames, byte sizes, SHA-256, thresholds, architectures |
| `results/six_model_seed42/table7_checkpoint_archive.json` | Archive name, size and SHA-256 |
| `code/prepare_manual_mask_inputs.py` | Generates the 256 x 256 enhanced inputs from the public uint16 patches |
| `code/evaluate_manual_mask_reference.py` | Runs the six archived checkpoints and writes Dice, IoU, AUPRC and negative activation |

## What is obtained separately

1. **Public dataset.** Download ContrailStruct30 Version 3 from Figshare
   ([10.6084/m9.figshare.33435496](https://doi.org/10.6084/m9.figshare.33435496.v3))
   and extract it so that `train/`, `validation/` and `test/` each contain
   `images_16bit/` and `images_8bit/`. The script reads the three-band uint16
   patch whose record ID is listed in `table7_manual_rois.csv`.
2. **Archived checkpoints.** The binary weights are not stored in Git. The
   archive `contrail_model_benchmark_checkpoints_20260831.zip`
   (727,992,049 bytes; SHA-256
   `88eeae8c53270c22533bbf4f9c26ac2695ba09919f1a04ecde7c1d1e017e8cab`)
   is available from the corresponding authors. Extract the six `.pth` files and
   verify them against `table7_checkpoint_manifest.csv` before use.

## Step 1: build the 256 x 256 enhanced inputs

The 512 x 512 ROIs were rendered as whole images. Table 7 must use the
**per-patch** enhancement instead, applied to the central 256 x 256 crop of the
three-band uint16 source. `prepare_manual_mask_inputs.py` performs exactly that
and centre-crops the reference, valid and ignore masks to the same grid.

```sh
python code/prepare_manual_mask_inputs.py \
  --rois results/six_model_seed42/table7_manual_rois.csv \
  --dataset-root /path/to/ContrailStruct30 \
  --mask-root data/table7_manual_masks \
  --output table7_inputs \
  --verify-released-8bit
```

Expected output: `table7_inputs/{images,masks,valid,ignore}/` with 60 files each.
`--verify-released-8bit` compares each enhanced crop with the released 8-bit
patch; the maximum absolute difference is 1 (rounding only).

## Step 2: evaluate the archived checkpoints

```sh
python code/evaluate_manual_mask_reference.py \
  --weights-dir /path/to/archived_checkpoints \
  --image-dir table7_inputs/images \
  --mask-dir table7_inputs/masks \
  --valid-dir table7_inputs/valid \
  --ignore-dir table7_inputs/ignore \
  --thresholds results/six_model_seed42/table7_thresholds.csv \
  --output table7_recomputed.csv
```

The script uses RGB/255 without ImageNet normalization, the fixed archived
thresholds, and scores only `valid & ~ignore`. It writes Dice, IoU, pixel AUPRC
and empty-mask activation for the six models.

## Expected values

| Model | Threshold | Dice | IoU | AUPRC | Negative activation |
|---|---:|---:|---:|---:|---:|
| MaxViT–U-Net | 0.90 | 0.750 | 0.600 | 0.816 | 3/13 |
| FCN–ResNet50 | 0.85 | 0.713 | 0.555 | 0.799 | 3/13 |
| SegFormer–B0 | 0.75 | 0.701 | 0.540 | 0.754 | 8/13 |
| DeepLabV3+–ResNet50 | 0.75 | 0.669 | 0.503 | 0.723 | 8/13 |
| PSPNet–ResNet50 | 0.85 | 0.669 | 0.502 | 0.759 | 3/13 |
| U-Net–ResNet34 | 0.80 | 0.668 | 0.502 | 0.675 | 8/13 |

## Interpretation notes

- The masks were drawn on the full 512 x 512 rendering and then centre-cropped.
  Their foreground extent can differ from the dataset's own 256 masks; the
  independent redrawing is the point of the check.
- Pixel-pooled IoU is lower than Dice by construction
  (`Dice = 2 IoU / (1 + IoU)`); a few large-mask images dominate the pooled
  values.
- The 43/17/0 overlap means the comparison measures agreement with independent
  human delineation on images seen during model development. It does not
  measure performance on unseen scenes.
