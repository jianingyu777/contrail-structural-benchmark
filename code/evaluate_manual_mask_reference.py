"""Evaluate six segmentation checkpoints against the 20 manual 512-pixel masks.

This reproduces the Table 7 reference-mask check: each 512 x 512 image is
centre-cropped to the 256 x 256 input used by the checkpoints, RGB values are
divided by 255, and each checkpoint uses its fixed validation-selected
threshold. The script writes Dice, IoU, pixel AUPRC and empty-mask activation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file


MODEL_FILES = {
    "MAXVIT_BASE_UNET": "MAXVIT_BASE_UNET_seed42.safetensors",
    "UNET_RESNET34": "UNET_RESNET34_seed42.safetensors",
    "DEEPLABV3PLUS_RESNET50": "DEEPLABV3PLUS_RESNET50_seed42.safetensors",
    "SEGFORMER_B0": "SEGFORMER_B0_seed42.safetensors",
    "FCN_RESNET50": "FCN_RESNET50_seed42.safetensors",
    "PSPNET_RESNET50": "PSPNET_RESNET50_seed42.safetensors",
}


def average_precision(probability, truth):
    order = np.argsort(probability.ravel())[::-1]
    labels = truth.ravel()[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall = tp / max(1, int(labels.sum()))
    return float(np.trapz(precision, recall))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True,
                        help="CSV with columns model,threshold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_main_benchmark import model_factory

    with args.thresholds.open(newline="", encoding="utf-8-sig") as handle:
        thresholds = {r["model"]: float(r["threshold"]) for r in csv.DictReader(handle)}

    ids = sorted(p.stem for p in args.image_dir.glob("EV*.png"))
    images, masks = [], []
    for name in ids:
        image = Image.open(args.image_dir / f"{name}.png").convert("RGB").crop((128, 128, 384, 384))
        mask_path = args.mask_dir / f"{name}_mask.png"
        mask = Image.open(mask_path).convert("L").crop((128, 128, 384, 384))
        images.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        masks.append(np.asarray(mask) > 127)
    images = torch.from_numpy(np.stack(images)).to(args.device)
    masks = np.stack(masks)

    rows = []
    for model_name, filename in MODEL_FILES.items():
        model, _ = model_factory(model_name, pretrained=False)
        model.load_state_dict(load_file(str(args.weights_dir / filename)), strict=True)
        model.to(args.device).eval()
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(images), 4):
                with torch.amp.autocast("cuda", enabled=str(args.device).startswith("cuda")):
                    batch = torch.sigmoid(model(images[start:start + 4]))[:, 0]
                probabilities.append(batch.float().cpu().numpy())
        probability = np.concatenate(probabilities)
        prediction = probability >= thresholds[model_name]
        tp = int(np.logical_and(prediction, masks).sum())
        fp = int(np.logical_and(prediction, ~masks).sum())
        fn = int(np.logical_and(~prediction, masks).sum())
        empty = ~masks.reshape(len(masks), -1).any(axis=1)
        rows.append({
            "model": model_name,
            "threshold": thresholds[model_name],
            "dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "iou": tp / max(1, tp + fp + fn),
            "auprc": average_precision(probability, masks),
            "negative_activation": int(prediction[empty].reshape(int(empty.sum()), -1).any(axis=1).sum()),
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
