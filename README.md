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

The repository provides image enhancement, dataset reading, three segmentation
baselines, validation-based model selection, held-out testing, patch inference,
and date-clustered uncertainty estimates. Models use the complete upstream
architectures in [segmentation-models-pytorch 0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch).

| Model | Upstream architecture | Encoder |
|---|---|---|
| U-Net-ResNet34 | `smp.Unet` | `resnet34` |
| DeepLabV3+-ResNet50 | `smp.DeepLabV3Plus` | `resnet50` |
| SegFormer-B0 | `smp.Segformer` | `mit_b0` |

The U-Net input-representation experiment uses seeds 3407, 3408 and 3409 for
both enhanced 8-bit and normalized uint16 inputs. All runs retain the same
split and training budget. Normalization statistics are estimated from the
training partition only.

## Quick Start

Use Python 3.10 for training and inference. Install the PyTorch 2.5.1 and
torchvision 0.20.1 builds appropriate for your CPU or CUDA system, followed by:

```sh
python -m pip install -r requirements-ml.txt
python code/verify_dataset.py --data-root /path/to/ContrailStruct30 --check-files
python code/read_example.py --data-root /path/to/ContrailStruct30
python code/train_benchmarks.py train --model unet_resnet34 --representation 8bit --seed 3407 --data-root /path/to/ContrailStruct30 --workspace runs
```

For CPU-only data and statistics checks, a separately tested Python 3.14
environment is specified in `requirements-data.txt`. Installation, all three
model commands, the six-run representation experiment, and inference examples
are in [Running the experiments](docs/RUNNING.md).

## Reference Results

The following results use enhanced 8-bit inputs and seed 3407. Probability
thresholds were selected on validation data and frozen before test evaluation.

| Model | Threshold | IoU | Dice | Mean positive-patch IoU | Difficult-negative activation (%) |
|---|---:|---:|---:|---:|---:|
| U-Net-ResNet34 | 0.86 | 0.631 | 0.773 | 0.616 | 35.41 |
| DeepLabV3+-ResNet50 | 0.81 | 0.622 | 0.767 | 0.548 | 17.35 |
| SegFormer-B0 | 0.63 | 0.660 | 0.795 | 0.594 | 20.24 |

See [complete results and 95% intervals](results/summary/three_model_8bit_seed3407.csv),
[foreground-area strata](results/summary/stratified_metrics_with_date_bootstrap.csv),
and [three-seed input comparison](results/summary/unet_8bit_uint16_three_seed_summary.csv).
Intervals use 2,000 acquisition-date-clustered bootstrap resamples. Negative
activation is the percentage of empty-reference patches with any predicted
foreground. Test-date intervals and training-seed variation are distinct.

## Reproducibility Materials

- [Image enhancement](enhancement/README.md): patch-local transform, fixed parameters and three real executable examples.
- [Data dictionary](metadata/DATA_DICTIONARY.md): actual fields, types, units and path conventions.
- [Evaluation protocol](docs/PROTOCOL.md): loss, augmentation, checkpoint and threshold selection, metrics and bootstrap.
- [Contextual reassessment](review/README.md): recorded decisions for 100 patches and source-window metadata.
- [Run records](results/): eight completed runs, training logs, threshold scans and per-patch confusion counts.
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
