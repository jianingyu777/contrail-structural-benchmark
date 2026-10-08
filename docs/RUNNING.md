# Running Experiment 2

This guide covers the RGB-uint16 comparison and earlier three-model reference
runs. For the matched four-model main benchmark in Sects. 5.1-5.4, use
[MAIN_BENCHMARK.md](MAIN_BENCHMARK.md) and
`requirements-main-benchmark.txt`.

Run commands from the repository root. Replace `/path/to/ContrailStruct30`
with the folder containing the three data partitions. On Windows, quote paths
containing spaces. No model training is needed to inspect metadata or reproduce
statistics from the frozen count tables.

## 1. Data layout

```text
DATA_ROOT/
  train/
    images_8bit/
    images_16bit/
    masks/
  validation/
    images_8bit/
    images_16bit/
    masks/
  test/
    images_8bit/
    images_16bit/
    masks/
```

The repository supplies `metadata/records.csv`. Image paths are relative to
`DATA_ROOT`; join all tables by `record_id`, not by row position. Metadata and
image folders need not share a parent. The training command's `--metadata-root`
is the directory containing `metadata/`, not the metadata directory itself.

```sh
python code/verify_dataset.py --data-root /path/to/ContrailStruct30 --check-files
python code/read_example.py --data-root /path/to/ContrailStruct30
```

`verify_dataset.py` checks the fixed manifest, split counts, grouping, path
conventions and normalization provenance. `--check-files` additionally checks
the existence of all 58,365 component files; it does not claim to rehash them.

## 2. Environments

Training and inference used Python 3.10.17, PyTorch 2.5.1, torchvision 0.20.1,
and segmentation-models-pytorch 0.5.0. Create a separate environment, install
the appropriate PyTorch build from the [official instructions](https://pytorch.org/get-started/previous-versions/), then run:

```sh
python -m pip install -r requirements-ml.txt
python -m pip check
```

The first training run downloads ImageNet encoder weights through the upstream
library if they are not already cached. `--device cpu` is available for small
checks; the default training device is `cuda:0`. Windows users can use
`--workers 0` if worker startup is unsuitable for their environment.

CPU-only image-enhancement and statistical reproduction was separately tested
on Python 3.14.0. In that environment:

```sh
python -m pip install -r requirements-data.txt
python code/verify_enhancement_examples.py
python code/verify_frozen_statistics.py
python -m unittest discover -s tests -v
```

Do not install both requirements files into the same environment. PyTorch is
not required for the CPU-only checks. The CPU tests do not download the dataset
or invoke training.

## 3. Three-model comparison

```sh
python code/train_benchmarks.py train --model unet_resnet34 --representation 8bit --seed 3407 --data-root /path/to/ContrailStruct30 --workspace runs
python code/train_benchmarks.py train --model deeplabv3plus_resnet50 --representation 8bit --seed 3407 --data-root /path/to/ContrailStruct30 --workspace runs
python code/train_benchmarks.py train --model segformer_b0 --representation 8bit --seed 3407 --data-root /path/to/ContrailStruct30 --workspace runs
```

Each command trains, selects a checkpoint on validation data, selects its
validation probability threshold, and evaluates the held-out test partition.
The default budget is at most 30 epochs, patience 8 and batch size 32.
Completed runs are not overwritten unless `--force` is explicitly supplied.
Keep the defaults for comparison with the reference results; batch limits and
shortened runs are diagnostic settings, not paper-reproduction runs.

## 4. Six-run U-Net representation comparison

Run U-Net with seeds 3407, 3408 and 3409 for both representations. The 8-bit
seed-3407 run above is reused, not trained twice.

```sh
python code/train_benchmarks.py train --model unet_resnet34 --representation 8bit --seed 3408 --data-root /path/to/ContrailStruct30 --workspace runs
python code/train_benchmarks.py train --model unet_resnet34 --representation 8bit --seed 3409 --data-root /path/to/ContrailStruct30 --workspace runs
python code/train_benchmarks.py train --model unet_resnet34 --representation uint16 --seed 3407 --data-root /path/to/ContrailStruct30 --workspace runs --uint16-statistics normalization/uint16_train_percentiles.json
python code/train_benchmarks.py train --model unet_resnet34 --representation uint16 --seed 3408 --data-root /path/to/ContrailStruct30 --workspace runs --uint16-statistics normalization/uint16_train_percentiles.json
python code/train_benchmarks.py train --model unet_resnet34 --representation uint16 --seed 3409 --data-root /path/to/ContrailStruct30 --workspace runs --uint16-statistics normalization/uint16_train_percentiles.json
```

The supplied normalization file was estimated from the 13,213 training patches.
To recompute it from that same partition:

```sh
python code/train_benchmarks.py compute-normalization --data-root /path/to/ContrailStruct30 --workspace runs
```

This writes `runs/outputs/normalization/uint16_train_percentiles.json`. Supply
that path explicitly to `--uint16-statistics` when using the recomputed file.

## 5. Test and predict with a checkpoint

Checkpoints are written to
`runs/outputs/REPRESENTATION/MODEL/seed_SEED/best.pth`. Load only trusted files.
The manifest in `results/checkpoint_manifest.csv` records the study checkpoints;
their binary weights are not included in Git.

```sh
python code/evaluate_checkpoint.py --checkpoint runs/outputs/8bit/unet_resnet34/seed_3407/best.pth --data-root /path/to/ContrailStruct30 --output evaluation
python code/predict_patch.py --checkpoint runs/outputs/8bit/unet_resnet34/seed_3407/best.pth --image /path/to/enhanced_patch.png --output prediction
```

Use a new output directory. The patch command writes `mask.png`,
`probability.npy` and `prediction.json`. It reads the input representation and
validation-selected threshold from the checkpoint, uses the same preprocessing
as evaluation, and accepts only a 256 x 256 three-channel patch. For uint16,
provide the corresponding TIFF and the matching normalization file with
`--normalization`; its default is the study's bundled train-only statistics.
The standalone test command accepts the same normalization option.

## 6. Statistics and archived outputs

After training all runs:

```sh
python code/train_benchmarks.py aggregate --workspace runs --replicates 2000
```

Alternatively, reproduce the reported statistics directly from the count
tables already included in this repository:

```sh
python code/verify_frozen_statistics.py
```

The latter verifies 240 overall and stratified metric values and their
associated 2,000-resample intervals across eight runs. It leaves the archived
results unchanged. Repeated optimization can vary with hardware and libraries;
statistical reproduction from fixed counts is a separate check.
