"""Prepare 256 x 256 enhanced inputs for the Table 7 manual-mask check.

Table 7 evaluates the archived checkpoints on 60 hand-annotated 512 x 512 ROIs.
The segmentation models were trained on 256 x 256 patches processed by the
per-patch display enhancement, so the evaluation must not feed the whole-512
rendering. This script:

1. reads the central 256 x 256 crop of the public three-band uint16 patch that
   corresponds to each ROI record_id (the ROI centre is the dataset patch);
2. applies the same per-patch enhancement used for the dataset;
3. centre-crops the 512 x 512 reference mask, valid-area mask and ignore mask to
   the same 256 x 256 grid; and
4. writes the flat ``images/``, ``masks/``, ``valid/`` and ``ignore/`` folders
   consumed by ``evaluate_manual_mask_reference.py``.

Run from the repository root:

    python code/prepare_manual_mask_inputs.py \
      --rois results/six_model_seed42/table7_manual_rois.csv \
      --dataset-root /path/to/ContrailStruct30 \
      --mask-root data/table7_manual_masks \
      --output table7_inputs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio

from convert_uint16_to_8bit import convert_array


CENTRE = (128, 128, 384, 384)


def find_uint16_patch(dataset_root: Path, split: str, record_id: str) -> Path:
    folder = dataset_root / split / "images_16bit"
    for extension in (".tif", ".tiff", ".TIF", ".TIFF", ".png"):
        candidate = folder / f"{record_id}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(folder.glob(f"{record_id}.*"))
    if not matches:
        raise FileNotFoundError(f"No uint16 patch for {record_id} under {folder}")
    return matches[0]


def read_uint16_patch(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        if dataset.count != 3:
            raise ValueError(f"{path}: expected 3 bands, found {dataset.count}")
        if any(dtype != "uint16" for dtype in dataset.dtypes):
            raise ValueError(f"{path}: expected uint16 bands, found {dataset.dtypes}")
        data = dataset.read(out_dtype="float32")
    if data.shape[0] != 3 and data.shape[-1] == 3:
        data = data.transpose(2, 0, 1)
    if data.shape[1:] == (512, 512):
        data = data[:, CENTRE[1]:CENTRE[3], CENTRE[0]:CENTRE[2]]
    if data.shape != (3, 256, 256):
        raise ValueError(f"{path}: expected (3, 256, 256) after cropping, found {data.shape}")
    return data


def load_and_crop_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.shape == (512, 512):
        array = array[CENTRE[1]:CENTRE[3], CENTRE[0]:CENTRE[2]]
    if array.shape != (256, 256):
        raise ValueError(f"{path}: expected 256 x 256 after cropping, found {array.shape}")
    return (array > 127).astype(np.uint8) * 255


def load_or_default(path: Path, default: int) -> np.ndarray:
    if path.exists():
        return load_and_crop_binary(path)
    return np.full((256, 256), default, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-released-8bit", action="store_true",
                        help="Compare each enhanced crop with the released 8-bit patch")
    args = parser.parse_args()

    for folder in ("images", "masks", "valid", "ignore"):
        (args.output / folder).mkdir(parents=True, exist_ok=True)

    with args.rois.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    maximum_difference = 0
    positive = 0
    for row in rows:
        annotation_id = row["annotation_id"]
        split = row["original_split"]
        record_id = row["record_id"]

        source = find_uint16_patch(args.dataset_root, split, record_id)
        enhanced = convert_array(read_uint16_patch(source))
        Image.fromarray(enhanced).save(args.output / "images" / f"{annotation_id}.png")

        mask = load_and_crop_binary(args.mask_root / "masks" / f"{annotation_id}_mask.png")
        valid = load_or_default(args.mask_root / "valid" / f"{annotation_id}_valid.png", 255)
        ignore = load_or_default(args.mask_root / "ignore" / f"{annotation_id}_ignore.png", 0)
        Image.fromarray(mask).save(args.output / "masks" / f"{annotation_id}_mask.png")
        Image.fromarray(valid).save(args.output / "valid" / f"{annotation_id}_valid.png")
        Image.fromarray(ignore).save(args.output / "ignore" / f"{annotation_id}_ignore.png")
        if (mask > 0).any():
            positive += 1

        if args.verify_released_8bit:
            released = args.dataset_root / split / "images_8bit" / f"{record_id}.png"
            if released.exists():
                reference = np.asarray(Image.open(released).convert("RGB"))
                difference = int(np.abs(enhanced.astype(np.int16) - reference.astype(np.int16)).max())
                maximum_difference = max(maximum_difference, difference)

    print(f"prepared {len(rows)} inputs ({positive} with foreground)")
    if args.verify_released_8bit:
        print(f"maximum absolute difference versus released 8-bit patches: {maximum_difference}")


if __name__ == "__main__":
    main()
