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

## Current benchmark (revised Sect. 5)

The current manuscript uses a **six-model, seed-42 comparison** on the Version 3
scene- and date-disjoint test partition. The primary metric is mean
positive-patch Dice over the 1,422 positive test patches.

| Model | Implementation | Encoder |
|---|---|---|
| MaxViT–U-Net | `timm` encoder + U-Net decoder (`code/models/maxvit_unet.py`) | `maxvit_base_tf_512.in21k_ft_in1k` |
| U-Net–ResNet34 | `smp.Unet` | `resnet34` |
| SegFormer–B0 | `smp.Segformer` | `mit_b0` |
| DeepLabV3+–ResNet50 | `smp.DeepLabV3Plus` | `resnet50` |
| FCN–ResNet50 | `torchvision` FCN (`code/models/fcn_resnet50.py`) | `resnet50` |
| PSPNet–ResNet50 | `smp.PSPNet` | `resnet50` |

Reference values:

| Model | Mean positive-patch Dice | Mean positive-patch IoU | Pixel-pooled IoU | Precision | Recall | Negative activation |
|---|---:|---:|---:|---:|---:|---:|
| MaxViT–U-Net | 0.8202 | 0.7363 | 0.7394 | 0.8417 | 0.8588 | 26.59% |
| U-Net–ResNet34 | 0.8014 | 0.7112 | 0.6790 | 0.7460 | 0.8831 | 45.94% |
| SegFormer–B0 | 0.7851 | 0.6880 | 0.6835 | 0.7613 | 0.8699 | 56.41% |
| DeepLabV3+–ResNet50 | 0.7793 | 0.6818 | 0.6728 | 0.7583 | 0.8565 | 49.24% |
| FCN–ResNet50 | 0.7070 | 0.6202 | 0.6797 | 0.8332 | 0.7867 | 3.94% |
| PSPNet–ResNet50 | 0.6894 | 0.5868 | 0.6652 | 0.7941 | 0.8038 | 5.59% |

See [MAIN_BENCHMARK.md](docs/MAIN_BENCHMARK.md) for commands and the Table 7
manual-mask evaluation. The exact validation-selected thresholds, 18,732-row
per-patch audit, aggregate metrics, final Figure 7 and checkpoint SHA-256
hashes are in [results/six_model_seed42/](results/six_model_seed42/). Binary
weights are not bundled in Git.

## Archived experiments

The previous four-model, three-seed experiment and the RGB–uint16
representation comparison are retained for provenance but are **not** the
current Sect. 5 benchmark:

- Four-model, three-seed matched benchmark:
  [docs/ARCHIVED_FOUR_MODEL.md](docs/ARCHIVED_FOUR_MODEL.md)
- RGB–uint16 representation comparison:
  [docs/RUNNING.md](docs/RUNNING.md) and [docs/PROTOCOL.md](docs/PROTOCOL.md)
- Archived four-model results: [results/main_benchmark/](results/main_benchmark/)
  and [results/summary/](results/summary/)

Do not combine scores from the current six-model benchmark with the archived
four-model or representation experiments.

## Quick Start

Use Python 3.10 for training and inference. Install the PyTorch 2.5.1 and
torchvision 0.20.1 builds appropriate for your CPU or CUDA system, followed by:

```sh
python -m pip install -r requirements-main-benchmark.txt
python code/verify_dataset.py --data-root /path/to/ContrailStruct30 --check-files
python code/read_example.py --data-root /path/to/ContrailStruct30
python code/train_main_benchmark.py --model UNET_RESNET34 --seed 42 --epochs 30 --patience 8 --batch-size 32 --data-root /path/to/ContrailStruct30
```

For CPU-only data and statistics checks, a separately tested Python 3.14
environment is specified in `requirements-data.txt`. The archived experiment
commands and inference examples are in [docs/RUNNING.md](docs/RUNNING.md).

## Reproducibility Materials

- [Image enhancement](enhancement/README.md): patch-local transform, fixed parameters and three real executable examples.
- [Data dictionary](metadata/DATA_DICTIONARY.md): actual fields, types, units and path conventions.
- [Current six-model protocol](docs/MAIN_BENCHMARK.md): training, validation selection, test evaluation and Table 7.
- [Archived four-model protocol](docs/ARCHIVED_FOUR_MODEL.md): previous three-seed experiment.
- [Legacy RGB–uint16 protocol](docs/PROTOCOL.md): archived representation comparison.
- [Contextual reassessment](review/README.md): recorded decisions for 100 patches and source-window metadata.
- [Current result files](results/six_model_seed42/): Table 6, Table 7, thresholds and checkpoint hashes.
- [Archived main benchmark](results/main_benchmark/): 12 archived four-model configurations.
- [Checkpoint manifest](results/checkpoint_manifest.csv): hashes and thresholds for the archived four-model/representation checkpoints.

Full dataset imagery and contextual review images belong to the Figshare data
release rather than this repository. Figure-generation and manuscript files
are not part of this software repository.

## Citation and Acknowledgements

Use [CITATION.cff](CITATION.cff) to cite this software, and cite the Figshare
version used for your data. Code snapshot: `essd-20261120`.

We acknowledge the SDGSAT-1 Open Science Program of the International Research
Center of Big Data for Sustainable Development Goals. We thank
[ADSB.lol](https://adsb.lol/) for historical flight trajectories used to guide
candidate search, and the annotators and experts for label construction and
reassessment. This work was supported by the National Natural Science
Foundation of China (grant 62575297).

## Licence

Author-developed code is licensed under [MIT](LICENSE). Enhanced images,
binary masks, corresponding three-band uint16 source-value TIFF patches, and
author-generated metadata are released under the dataset's CC BY 4.0 terms.
Redistribution of the derived 256 x 256 source-value patches was authorized by
the SDGSAT-1 Open Science Program. Original full-scene observations are not
included and remain subject to the applicable source-data terms. Upstream
software retains its own licences. See [NOTICE.md](NOTICE.md).
