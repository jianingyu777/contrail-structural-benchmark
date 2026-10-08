# Evaluation protocol (Experiment 2)

This document describes the RGB-uint16 comparison and earlier three-model
reference runs, not the matched four-model main benchmark in Sects. 5.1-5.4.
For that experiment see [MAIN_BENCHMARK.md](MAIN_BENCHMARK.md).

## Reference data

The fixed manifest contains 19,455 records: 13,213 training, 3,120 validation
and 3,122 test patches. The test partition includes 1,422 positives and 1,700
difficult negatives on 51 acquisition dates. All observations from a scene or
date stay within one partition. Both input representations use the same labels.

## Models and optimization

The complete architectures are imported from segmentation-models-pytorch 0.5.0:
U-Net with ResNet34, DeepLabV3+ with ResNet50, and SegFormer with MiT-B0.
Encoders use ImageNet initialization. Defaults are AdamW, learning rate 1e-4,
weight decay 1e-4, one warm-up epoch, cosine decay, at most 30 epochs, patience
8, and physical/effective batch size 32.

The loss combines binary cross-entropy (positive-class weight 5) and soft Dice
(weight 1). Sampling assigns difficult negatives weight 1.2 and positives 1.
Geometric augmentation includes rotation within 180 degrees, shifts within
20%, scale 0.90-1.10, and horizontal/vertical flips. Intensity augmentation
uses brightness/contrast or gamma adjustment with combined probability 0.5.

## Model selection

Checkpoints are selected by validation foreground Dice. The final validation
threshold scan uses 0.05-0.95 in increments of 0.01, plus 0.97 and 0.99.
The selected checkpoint and threshold are frozen before testing. Test masks
are never used to choose a checkpoint or threshold. Seed 3407 defines the
three-model comparison; U-Net representation experiments use seeds 3407-3409.

## Input representations

Enhanced RGB is divided by 255. uint16 inputs are normalized using training-only
channel statistics: lower percentiles [657, 812, 582], upper percentiles
[2112, 2437, 1657], and fill-replacement medians [1228, 1479, 1033]. Values
equal to zero or at least 65000 are fill values. Both inputs then use ImageNet
means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225].

The comparison changes enhancement/normalization while fixing scene content,
labels and splits. It is not an isolated test of storage bit depth.

## Metrics and uncertainty

IoU, Dice, precision and recall are micro-averaged over complete test-set
confusion counts. Mean positive-patch IoU gives each non-empty-reference patch
equal weight. Difficult-negative activation is the fraction of empty-reference
patches with at least one predicted foreground pixel.

Positive foreground strata are at most 1%, greater than 1% through 5%, and
greater than 5%, containing 318, 488 and 616 test patches. The bootstrap draws
whole acquisition-date clusters with replacement, preserving multiplicity and
all patches within each selected date. Percentile 95% intervals use 2,000
resamples. Overall intervals use the run seed; stratified intervals use the
run seed plus 1000, as recorded in the implementation and frozen results.

Bootstrap intervals describe variation across test dates. Three-seed summaries
report between-seed standard deviations; they do not substitute for date-clustered
intervals or establish statistically resolved rankings of architectures.
