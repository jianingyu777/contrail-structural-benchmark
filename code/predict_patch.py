"""Predict one patch using the checkpoint's frozen threshold and preprocessing."""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image
import tifffile
import torch

import train_benchmarks as experiment


ROOT = Path(__file__).resolve().parents[1]


def load_input(path, representation, normalization):
    if representation == "8bit":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError("Expected a 256 x 256 three-channel enhanced uint8 image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if representation == "uint16":
        return experiment.normalize_uint16(tifffile.imread(path), normalization)
    raise ValueError(f"Unsupported input representation: {representation}")


def predict(checkpoint_path, image_path, output, normalization_path, device):
    if output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    threshold = float(checkpoint["selected_threshold"])
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid checkpoint threshold")
    representation = checkpoint["representation"]
    normalization = experiment.load_uint16_statistics(normalization_path) if representation == "uint16" else None
    image = load_input(image_path, representation, normalization)
    tensor = experiment.eval_transform(256)(image=image)["image"].float().unsqueeze(0).to(device)
    model = experiment.build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(experiment.forward_logits(model, tensor))
    probability = probabilities[0, 0].cpu().numpy().astype(np.float32)
    if probability.shape != (256, 256) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability output")
    mask = (probability >= threshold).astype(np.uint8) * 255
    summary = {
        "model": checkpoint["model"], "representation": representation,
        "threshold": threshold, "input_filename": image_path.name,
        "input_sha256": experiment.sha256_file(image_path),
        "checkpoint_sha256": experiment.sha256_file(checkpoint_path),
        "normalization_sha256": experiment.sha256_file(normalization_path) if normalization is not None else None,
        "output_shape": list(probability.shape), "foreground_pixels": int((mask > 0).sum()),
    }
    output.mkdir(parents=True)
    Image.fromarray(mask).save(output / "mask.png")
    np.save(output / "probability.npy", probability, allow_pickle=False)
    (output / "prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, default=ROOT / "normalization/uint16_train_percentiles.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        result = predict(args.checkpoint, args.image, args.output, args.normalization, torch.device(args.device))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Patch prediction failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
