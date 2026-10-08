# ContrailStruct30: a 30 m three-band thermal-infrared dataset for contrail segmentation

## Dataset

**Dataset record: [Figshare](https://doi.org/10.6084/m9.figshare.33435496)**

This repository accompanies the **19,455-patch** ContrailStruct30 dataset.
Every 256 x 256-pixel patch has an enhanced RGB PNG, a corresponding three-band
uint16 TIFF, and a human-corrected binary mask. Download the dataset separately
from Figshare and select the release matching the counts below and the supplied
[record manifest](metadata/records.csv). Cite the version-specific dataset DOI
shown on that release; the link above identifies the dataset record.

| Partition | Patches | Positive | Difficult negative |
|---|---:|---:|---:|
| Training | 13,213 | 6,243 | 6,970 |
| Validation | 3,120 | 1,358 | 1,762 |
| Test | 3,122 | 1,422 | 1,700 |
| Total | 19,455 | 9,023 | 10,432 |

The collection covers 305 dates and 819 processing-level-specific scene
entries, representing 808 source-product identifiers. Source scenes and dates
are disjoint across partitions. Set the data root to the directory directly
containing `train/`, `validation/` and `test/`.

## Code

The repository provides image enhancement, dataset reading, two distinct
segmentation experiments, validation-based model selection, held-out testing,
and date-clustered uncertainty estimates. The three smaller architectures use
[segmentation-models-pytorch 0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch).

| Model | Upstream architecture | Encoder |
|---|---|---|
| U-Net-ResNet34 | `smp.Unet` | `resnet34` |
| DeepLabV3+-ResNet50 | `smp.DeepLabV3Plus` | `resnet50` |
| SegFormer-B0 | `smp.Segformer` | `mit_b0` |
| MaxViT-B-U-Net | `timm` encoder and U-Net-like decoder | `maxvit_base_tf_512.in21k_ft_in1k` |

**Experiment 1: four-model benchmark (Sects. 5.1-5.4).** All four models use
enhanced RGB and seeds 42, 123 and 2025 under the matched training protocol.
See [main benchmark instructions](docs/MAIN_BENCHMARK.md) and
[archived results](results/main_benchmark/). Use
`requirements-main-benchmark.txt` (`timm==1.0.27`, as recorded by the
[archived protocol lock](results/main_benchmark/run_configs/PROTOCOL_LOCK.json)).

**Experiment 2: RGB vs uint16 comparison (Sect. 5.5).** U-Net uses seeds
3407, 3408 and 3409 in each input representation. This is a separate protocol;
its normalization statistics are estimated from the training partition only.
The legacy [protocol](docs/PROTOCOL.md) and [running guide](docs/RUNNING.md)
describe this experiment and use `requirements-ml.txt` (`timm==1.0.15`).

## Quick Start

Use Python 3.10 for training and inference. Install the PyTorch 2.5.1 and
torchvision 0.20.1 builds appropriate for your CPU or CUDA system, followed by:

```sh
python -m pip install -r requirements-main-benchmark.txt
python code/verify_dataset.py --data-root /path/to/ContrailStruct30 --check-files
python code/read_example.py --data-root /path/to/ContrailStruct30
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 42 --data-root /path/to/ContrailStruct30
```

For CPU-only data and statistics checks, a separately tested Python 3.14
environment is specified in `requirements-data.txt`. Installation, all three
model commands, the six-run representation experiment, and inference examples
for Experiment 2 are in [its running guide](docs/RUNNING.md). Use
[MAIN_BENCHMARK.md](docs/MAIN_BENCHMARK.md) for Experiment 1 commands.

## Reference Results

The main four-model results use enhanced RGB and three seeds per model.
Thresholds were selected on validation data and frozen before test evaluation.

| Model | Mean positive-patch Dice (mean +/- SD) | Difficult-negative activation (%) |
|---|---:|---:|
| MaxViT-B-U-Net | 0.8230 +/- 0.0056 | 34.57 |
| U-Net-ResNet34 | 0.7978 +/- 0.0039 | 44.24 |
| DeepLabV3+-ResNet50 | 0.7798 +/- 0.0026 | 53.90 |
| SegFormer-B0 | 0.7903 +/- 0.0047 | 53.45 |

See [per-seed results](results/main_benchmark/metrics/per_seed.csv),
[three-seed summary](results/main_benchmark/summaries/three_seed_mean_sd.csv),
[paired date-clustered 95% intervals](results/main_benchmark/bootstrap/paired_bootstrap_observation_date.csv),
and [RGB-uint16 comparison](results/summary/unet_8bit_uint16_three_seed_summary.csv).
Negative activation is the percentage of empty-reference patches with any
predicted foreground. Test-date intervals and training-seed variation are distinct.

## Reproducibility Materials

- [Image enhancement](enhancement/README.md): patch-local transform, fixed parameters and three real executable examples.
- [Data dictionary](metadata/DATA_DICTIONARY.md): actual fields, types, units and path conventions.
- [Four-model protocol](docs/MAIN_BENCHMARK.md): matched training, validation selection and test evaluation (Experiment 1).
- [Legacy protocol](docs/PROTOCOL.md): RGB-uint16 comparison and older three-model reference (Experiment 2).
- [Contextual reassessment](review/README.md): recorded decisions for 100 patches and source-window metadata.
- [Main benchmark](results/main_benchmark/): 12 archived configurations, validation decisions and statistics.
- [Representation comparison](results/): eight older-protocol runs, including six U-Net representation runs.
- [Checkpoint manifest](results/checkpoint_manifest.csv): hashes and selected thresholds of the study checkpoints. Training generates compatible checkpoints; binary weights are not bundled in Git.

Full dataset imagery and contextual review images belong to the Figshare data
release rather than this repository. Figure-generation and manuscript files
are not part of this software repository.

## Citation and Acknowledgements

Use [CITATION.cff](CITATION.cff) to cite this software, and cite the Figshare
version used for your data. Code snapshot: `essd-20260914`.

We acknowledge the SDGSAT-1 Open Science Program of the International Research
Center of Big Data for Sustainable Development Goals. We thank
[ADSB.lol](https://adsb.lol/) for historical flight trajectories used to guide
candidate search, and the annotators and experts for label construction and
reassessment. This work was supported by the National Natural Science
Foundation of China (grant 62575297).

## Licence

Author-developed code is licensed under [MIT](LICENSE). Enhanced images,
masks and author-generated dataset metadata follow the dataset's CC BY 4.0
terms. SDGSAT-1 source-value imagery follows the applicable source-data terms.
Upstream software retains its own licences. See [NOTICE.md](NOTICE.md).
