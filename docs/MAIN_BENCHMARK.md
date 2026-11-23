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

The exact thresholds and per-patch records for the archived six-model run are
in the project audit record `recovery_19455/COMPARISON_AUDIT_20261120.md`.
The transcribed aggregate values are in
[`results/six_model_seed42/table6_metrics.csv`](../results/six_model_seed42/table6_metrics.csv),
with checkpoint SHA-256 hashes in
[`checkpoint_manifest.csv`](../results/six_model_seed42/checkpoint_manifest.csv).
Binary weights are distributed separately and are not stored in Git.

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

Table 7 uses a **separate earlier archived checkpoint set** and 20 manually
drawn 512 x 512 reference masks (15 positive, five empty). Each image is
centre-cropped to 256 x 256, RGB values are divided by 255, and the archived
validation-selected threshold is applied. Dice and IoU are pooled over
evaluated pixels; AUPRC is pixel-level. This check is not directly comparable
with the seed-42 Table 6 values.

```sh
python code/evaluate_manual_mask_reference.py \
  --weights-dir PATH/TO/ARCHIVED_WEIGHTS \
  --image-dir PATH/TO/manual_mask/images \
  --mask-dir PATH/TO/manual_mask/masks \
  --thresholds results/six_model_seed42/table7_thresholds.csv \
  --output results/six_model_seed42/table7_recomputed.csv
```

The transcribed reference values are in
[`table7_manual_masks.csv`](../results/six_model_seed42/table7_manual_masks.csv).

## Relationship to the archived experiments

- Four-model, three-seed experiment (previous Sects. 5.1–5.4):
  [`ARCHIVED_FOUR_MODEL.md`](ARCHIVED_FOUR_MODEL.md).
- RGB–uint16 representation comparison (previous Sect. 5.5):
  [`RUNNING.md`](RUNNING.md), retained as an archived experiment.
