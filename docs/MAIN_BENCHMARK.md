# Four-model matched benchmark

This is Experiment 1 (Sects. 5.1-5.4). The RGB-uint16 experiment in
`code/train_benchmarks.py` and `results/summary/` is Experiment 2 (Sect. 5.5)
and uses a different training budget and seeds. Do not combine their scores.

## Models and inputs

All 12 runs read the released enhanced 8-bit RGB patches, scaled by 1/255.
U-Net-ResNet34, DeepLabV3+-ResNet50 and SegFormer-B0 use the public
`segmentation-models-pytorch` implementations and ImageNet-1K encoder
initialization. MaxViT-B-U-Net uses the `timm` MaxViT
`maxvit_base_tf_512.in21k_ft_in1k` encoder, initialized from ImageNet-21K
pretraining followed by ImageNet-1K fine-tuning, and the U-Net-like decoder in
`code/models/maxvit_unet.py`. Its four encoder features are at 1/2, 1/4,
1/8 and 1/16 resolution. The 512 in the weight name does not change the
256 x 256 input. All parameters are fine-tuned.

Use Python 3.10, PyTorch 2.5.1, torchvision 0.20.1, SMP 0.5.0 and timm
1.0.27, as recorded by the archived
[`PROTOCOL_LOCK.json`](../results/main_benchmark/run_configs/PROTOCOL_LOCK.json).
Install
`requirements-main-benchmark.txt` after selecting the appropriate
PyTorch CUDA build. First use of pretrained models may download weights.
The archived SMP runs loaded encoder-only weights from frozen safetensors;
the public rerun resolves the same named upstream ImageNet initializations.
Weight revisions in upstream caches can prevent bit-for-bit reproduction;
archived run configurations and checkpoint hashes are retained in
`results/main_benchmark/run_configs/`.

## Training and evaluation

Use the Figshare release matching `metadata/records.csv`. The public wrapper
adapts that manifest to the archived training loop in
`code/main_benchmark/original_training_core.py`. Run these 12 commands from
the repository root, replacing `DATA_ROOT` with the directory containing
`train/`, `validation/`, and `test/`:

```sh
python code/train_main_benchmark.py --model MAXVIT_BASE_UNET --seed 42 --data-root DATA_ROOT
python code/train_main_benchmark.py --model MAXVIT_BASE_UNET --seed 123 --data-root DATA_ROOT
python code/train_main_benchmark.py --model MAXVIT_BASE_UNET --seed 2025 --data-root DATA_ROOT
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 42 --data-root DATA_ROOT
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 123 --data-root DATA_ROOT
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 2025 --data-root DATA_ROOT
python code/train_main_benchmark.py --model DEEPLABV3PLUS_RESNET50 --seed 42 --data-root DATA_ROOT
python code/train_main_benchmark.py --model DEEPLABV3PLUS_RESNET50 --seed 123 --data-root DATA_ROOT
python code/train_main_benchmark.py --model DEEPLABV3PLUS_RESNET50 --seed 2025 --data-root DATA_ROOT
python code/train_main_benchmark.py --model SEGFORMER_B0 --seed 42 --data-root DATA_ROOT
python code/train_main_benchmark.py --model SEGFORMER_B0 --seed 123 --data-root DATA_ROOT
python code/train_main_benchmark.py --model SEGFORMER_B0 --seed 2025 --data-root DATA_ROOT
```

All runs use AdamW (initial learning rate 2e-4, weight decay 1e-4), BCE plus
soft Dice loss, batch size 12, synchronized horizontal/vertical image-mask
flips, at most 24 epochs and early stopping after six epochs without lower
validation loss. The best-validation-loss checkpoint is used for a threshold
sweep from 0.01 to 0.95 in 0.01 steps. Among thresholds with negative-patch
activation no greater than 5%, the selected threshold maximizes validation
positive-patch mean Dice. None of the 12 archived runs met that constraint,
so all used the pre-specified fallback: maximum validation positive-patch
mean Dice. Per-run decisions are in `results/main_benchmark/run_configs/`.

After each run is complete, evaluate its frozen checkpoint once:

```sh
python code/evaluate_main_benchmark.py --model MAXVIT_BASE_UNET --seed 42 --data-root DATA_ROOT
```

Repeat with each model/seed pair. The evaluator refuses to overwrite a prior
test output. It writes per-patch metrics, a 4096-bin pixel PR curve and a test
summary under `runs/main_benchmark/standard_baselines/MODEL/RENDERED3/seed_SEED/test_evaluation/`.

## Reported statistics

`results/main_benchmark/metrics/per_seed.csv` contains the 12 test results;
`summaries/three_seed_mean_sd.csv` gives mean and sample SD across seeds.
The `bootstrap/` tables contain 2,000 paired cluster resamples by acquisition
date and a scene-cluster sensitivity analysis. Each replicate calculates
metrics for each seed first, then averages seeds. Date-bootstrap intervals
quantify test-sample uncertainty, not variation in random initialization.
The test set has 3,122 patches from 120 scenes and 51 acquisition dates.
Figure 7 uses three-seed summaries and paired date intervals; its PR panel
and Figure 8 use the pre-selected seed 42. Table 6 reports main outcomes and
Table 7 includes parameter counts and between-seed SD.

Binary checkpoint weights and imagery are not in this Git repository. The
training records contain checkpoint SHA256 values for provenance. The
released Figshare dataset remains a separate publication.
