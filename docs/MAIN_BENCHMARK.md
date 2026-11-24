# Six-model Version 3 benchmark (current Sect. 5)

This is the current reproducible application benchmark in the revised
manuscript. It uses **six models**, **seed 42**, and the **Version 3
scene- and date-disjoint test partition** (3,122 patches: 1,422 positive and
1,700 difficult-negative). The earlier four-model, three-seed experiment is
retained separately in [`ARCHIVED_FOUR_MODEL.md`](ARCHIVED_FOUR_MODEL.md).

## Models and inputs

| Model | Implementation | Encoder |
|---|---|---|
| MaxViT–U-Net | `code/models/maxvit_unet.py` | `maxvit_base_tf_512.in21k_ft_in1k` |
| U-Net–ResNet34 | `segmentation-models-pytorch` | `resnet34` |
| SegFormer–B0 | `segmentation-models-pytorch` | `mit_b0` |
| DeepLabV3+–ResNet50 | `segmentation-models-pytorch` | `resnet50` |
| FCN–ResNet50 | `code/models/fcn_resnet50.py` (torchvision) | `resnet50` |
| PSPNet–ResNet50 | `segmentation-models-pytorch` | `resnet50` |

All models read enhanced 8-bit RGB patches at 256 x 256 pixels. FCN–ResNet50
is wrapped so that its released state-dict prefix (`network.*`) and logit
output match the other five implementations. The five ImageNet-initialized
baselines and the ImageNet-pretrained MaxViT encoder are fine-tuned.

## Training

Use the Version 3 release matching `metadata/records.csv`. Run one command per
model; all use seed 42 and the same training budget:

```sh
python code/train_main_benchmark.py --model MAXVIT_BASE_UNET --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
python code/train_main_benchmark.py --model SEGFORMER_B0 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
python code/train_main_benchmark.py --model DEEPLABV3PLUS_RESNET50 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
python code/train_main_benchmark.py --model FCN_RESNET50 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
python code/train_main_benchmark.py --model PSPNET_RESNET50 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root DATA_ROOT
```

The shared objective is BCE-with-logits plus soft Dice, AdamW with learning
rate 1e-4 and weight decay 1e-4, and identical geometric augmentation. The
best-validation-loss checkpoint is retained. Thresholds are selected on the
validation partition with the original rule: maximize positive-patch mean
Dice subject to empty-patch activation no greater than 5%, with the
pre-specified maximum-Dice fallback when the constraint is not met. Test
outputs are written only after the threshold is frozen.

The exact thresholds are in
[`Table6_exact_thresholds.csv`](../results/six_model_seed42/Table6_exact_thresholds.csv):
FCN 0.53, U-Net 0.30, DeepLabV3+ 0.24, PSPNet 0.26, SegFormer 0.30 and
MaxViT 0.34. The full 18,732-row per-patch audit is
[`Table6_full_per_patch_audit_all_models_seed42.csv`](../results/six_model_seed42/Table6_full_per_patch_audit_all_models_seed42.csv),
and the complete aggregate table is
[`Table6_seed42_aggregate_results.csv`](../results/six_model_seed42/Table6_seed42_aggregate_results.csv).
Checkpoint SHA-256 hashes are in
[`checkpoint_manifest.csv`](../results/six_model_seed42/checkpoint_manifest.csv);
binary weights are distributed separately and are not stored in Git.

## Table 6 values

Mean positive-patch Dice is the primary metric. Mean positive-patch IoU is
calculated in the same way; pixel-pooled IoU, precision and recall use pooled
TP/FP/FN counts; negative activation is the percentage of the 1,700 empty
reference patches with any predicted foreground pixel.

| Model | Mean positive-patch Dice | Mean positive-patch IoU | Pixel-pooled IoU | Precision | Recall | Negative activation |
|---|---:|---:|---:|---:|---:|---:|
| MaxViT–U-Net | 0.8202 | 0.7363 | 0.7394 | 0.8417 | 0.8588 | 26.59% |
| U-Net–ResNet34 | 0.8014 | 0.7112 | 0.6790 | 0.7460 | 0.8831 | 45.94% |
| SegFormer–B0 | 0.7851 | 0.6880 | 0.6835 | 0.7613 | 0.8699 | 56.41% |
| DeepLabV3+–ResNet50 | 0.7793 | 0.6818 | 0.6728 | 0.7583 | 0.8565 | 49.24% |
| FCN–ResNet50 | 0.7070 | 0.6202 | 0.6797 | 0.8332 | 0.7867 | 3.94% |
| PSPNet–ResNet50 | 0.6894 | 0.5868 | 0.6652 | 0.7941 | 0.8038 | 5.59% |

## Table 7: manual reference-mask check

Table 7 uses a **separate earlier archived checkpoint set** and 60 manually
drawn 512 x 512 reference masks (47 positive, 13 empty). Each image is
centre-cropped to 256 x 256 and processed with the **same per-patch enhancement
used for the dataset** (not the whole-512 rendering); RGB values are divided by
255, and the archived validation-selected threshold is applied. Dice and IoU are pooled over
evaluated pixels; AUPRC is pixel-level. Reference masks were drawn from scratch
on the full 512 x 512 images and centre-cropped to the same 256 x 256 grid; their
foreground extent can therefore differ from the dataset's own 256 masks. The
central 256 crops overlap the archived checkpoints' training or validation pool
(43 training, 17 validation, none in the held-out test partition), and the
reference masks were drawn independently. This is therefore an in-domain
annotation-reference check, not an unseen-image generalization test, and is not
directly comparable with the seed-42 Table 6 values.

| Model | Threshold | Dice | IoU | AUPRC | Negative activation |
|---|---:|---:|---:|---:|---:|
| MaxViT–U-Net | 0.90 | 0.750 | 0.600 | 0.816 | 3/13 |
| FCN–ResNet50 | 0.85 | 0.713 | 0.555 | 0.799 | 3/13 |
| SegFormer–B0 | 0.75 | 0.701 | 0.540 | 0.754 | 8/13 |
| DeepLabV3+–ResNet50 | 0.75 | 0.669 | 0.503 | 0.723 | 8/13 |
| PSPNet–ResNet50 | 0.85 | 0.669 | 0.502 | 0.759 | 3/13 |
| U-Net–ResNet34 | 0.80 | 0.668 | 0.502 | 0.675 | 8/13 |

The 60 reference masks, valid masks and ignore masks are included under
[`data/table7_manual_masks/`](../data/table7_manual_masks/), with a 60-row
inventory in
[`table7_manual_rois.csv`](../results/six_model_seed42/table7_manual_rois.csv).
The archived checkpoint archive is not stored in Git; its name, size and
SHA-256 are recorded in
[`table7_checkpoint_archive.json`](../results/six_model_seed42/table7_checkpoint_archive.json),
and per-file hashes are in
[`table7_checkpoint_manifest.csv`](../results/six_model_seed42/table7_checkpoint_manifest.csv).

First build the 256 x 256 enhanced inputs from the public uint16 patches
(centre crop plus the dataset's per-patch enhancement), then evaluate:

```sh
python code/prepare_manual_mask_inputs.py \
  --rois results/six_model_seed42/table7_manual_rois.csv \
  --dataset-root PATH/TO/ContrailStruct30 \
  --mask-root data/table7_manual_masks \
  --output table7_inputs --verify-released-8bit

python code/evaluate_manual_mask_reference.py \
  --weights-dir PATH/TO/ARCHIVED_WEIGHTS \
  --image-dir table7_inputs/images \
  --mask-dir table7_inputs/masks \
  --valid-dir table7_inputs/valid \
  --ignore-dir table7_inputs/ignore \
  --thresholds results/six_model_seed42/table7_thresholds.csv \
  --output results/six_model_seed42/table7_recomputed.csv
```

The transcribed reference values are in
[`table7_manual_masks.csv`](../results/six_model_seed42/table7_manual_masks.csv).
Full step-by-step instructions and the checkpoint access note are in
[`TABLE7_REPRODUCTION.md`](TABLE7_REPRODUCTION.md).

## Relationship to the archived experiments

- Four-model, three-seed experiment (previous Sects. 5.1–5.4):
  [`ARCHIVED_FOUR_MODEL.md`](ARCHIVED_FOUR_MODEL.md).
- RGB–uint16 representation comparison (previous Sect. 5.5):
  [`RUNNING.md`](RUNNING.md), retained as an archived experiment.
