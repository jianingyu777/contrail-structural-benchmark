"""Evaluate the archived six-model checkpoint set on the 60 manual 512-pixel masks.

This reproduces Table 7. Inputs must be the 256 x 256 enhanced PNGs written by
``prepare_manual_mask_inputs.py``: the central 256 x 256 crop of each 512 x 512
ROI, processed with the same per-patch enhancement used for the dataset.
Reference masks were drawn from scratch on the full 512 x 512 images and
centre-cropped to the same grid. The script applies RGB/255 without ImageNet
normalization, uses the archived validation-selected thresholds, and restricts
scoring to ``valid & ~ignore`` when those masks are supplied.

It writes Dice, IoU, pixel AUPRC and empty-mask activation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


MODEL_FILES = {
    "MAXVIT_BASE_UNET": "contrail_maxvit_unet_fold5_best_iou.pth",
    "FCN_RESNET50": "fcn_resnet50_best.pth",
    "UNET_RESNET34": "unet_resnet34_best.pth",
    "DEEPLABV3PLUS_RESNET50": "deeplabv3plus_resnet50_best.pth",
    "PSPNET_RESNET50": "pspnet_resnet50_best.pth",
    "SEGFORMER_B0": "segformer_b0_best.pth",
}


def average_precision(probability: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(probability.ravel())[::-1]
    labels = truth.ravel()[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall = tp / max(1, int(labels.sum()))
    return float(np.trapz(precision, recall))


def load_checkpoint(model, name: str, weights_dir: Path) -> None:
    checkpoint = torch.load(weights_dir / MODEL_FILES[name], map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if name == "MAXVIT_BASE_UNET":
        state = {
            key.replace("decoder.blocks.", "decoder.", 1) if key.startswith("decoder.blocks.") else key: value
            for key, value in state.items()
            if not key.startswith("centerline_head.")
        }
    elif name == "FCN_RESNET50":
        state = {"network." + key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def load_binary(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    if array.shape == (512, 512):
        array = array[128:384, 128:384]
    return array > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Directory containing the six archived .pth checkpoints")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="256 x 256 enhanced PNGs from prepare_manual_mask_inputs.py")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--valid-dir", type=Path, default=None,
                        help="Optional valid-area masks (default: all pixels valid)")
    parser.add_argument("--ignore-dir", type=Path, default=None,
                        help="Optional ignore masks (default: no ignored pixels)")
    parser.add_argument("--thresholds", type=Path, required=True,
                        help="CSV with columns model,threshold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_main_benchmark import model_factory

    with args.thresholds.open(newline="", encoding="utf-8-sig") as handle:
        thresholds = {row["model"]: float(row["threshold"]) for row in csv.DictReader(handle)}

    ids = sorted(
        path.stem for path in args.image_dir.glob("*.png")
        if not path.stem.endswith(("_mask", "_valid", "_ignore"))
    )
    images, masks, valids, ignores = [], [], [], []
    for name in ids:
        image = Image.open(args.image_dir / f"{name}.png").convert("RGB")
        if image.size == (512, 512):
            image = image.crop((128, 128, 384, 384))
        images.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        masks.append(load_binary(args.mask_dir / f"{name}_mask.png"))
        valids.append(load_binary(args.valid_dir / f"{name}_valid.png") if args.valid_dir else np.ones((256, 256), bool))
        ignores.append(load_binary(args.ignore_dir / f"{name}_ignore.png") if args.ignore_dir else np.zeros((256, 256), bool))
    images = torch.from_numpy(np.stack(images)).to(args.device)
    masks = np.stack(masks)
    evaluated = np.stack(valids) & ~np.stack(ignores)

    rows = []
    for model_name, _ in MODEL_FILES.items():
        model, _ = model_factory(model_name, pretrained=False)
        load_checkpoint(model, model_name, args.weights_dir)
        model.to(args.device).eval()
        threshold = thresholds[model_name]
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(images), 4):
                batch = images[start:start + 4]
                logits = model(batch)
                logits = logits.get("out") if isinstance(logits, dict) else logits
                if logits.shape[-2:] != batch.shape[-2:]:
                    logits = functional.interpolate(logits, size=batch.shape[-2:], mode="bilinear", align_corners=False)
                probabilities.append(torch.sigmoid(logits)[:, 0].float().cpu().numpy())
        probability = np.concatenate(probabilities)

        prediction = probability >= threshold
        truth = masks & evaluated
        predicted = prediction & evaluated
        tp = int(np.logical_and(predicted, truth).sum())
        fp = int(np.logical_and(predicted, ~truth).sum())
        fn = int(np.logical_and(~predicted, truth).sum())
        empty = ~truth.reshape(len(truth), -1).any(axis=1)
        rows.append({
            "model": model_name,
            "threshold": threshold,
            "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": tp / max(1, tp + fp + fn),
            "auprc": average_precision(probability[evaluated], truth[evaluated]),
            "negative_activation": int(predicted[empty].reshape(int(empty.sum()), -1).any(axis=1).sum()),
            "negative_total": int(empty.sum()),
        })
        print(json.dumps(rows[-1]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
