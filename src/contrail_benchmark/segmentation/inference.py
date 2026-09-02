from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import ContrailMaxViTUNet, load_checkpoint, mask_logits
from .preprocessing import pad_reflect, preprocess_raw_tis_tile


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class PatchDataset(Dataset):
    def __init__(self, image_dir: str | Path, size: int) -> None:
        self.paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No patch images found under {image_dir}")
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pad_height = max(0, self.size - image.shape[0])
        pad_width = max(0, self.size - image.shape[1])
        if pad_height or pad_width:
            image = cv2.copyMakeBorder(
                image,
                0,
                pad_height,
                0,
                pad_width,
                borderType=cv2.BORDER_CONSTANT,
                value=0,
            )
        image = cv2.resize(
            image, (self.size, self.size), interpolation=cv2.INTER_LINEAR
        )
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return tensor, path.name


def _device(value: str | None) -> torch.device:
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_model(checkpoint: str | Path, device: torch.device) -> ContrailMaxViTUNet:
    model = ContrailMaxViTUNet(pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device=device, strict=True)
    return model.eval()


def patch_cli() -> None:
    parser = argparse.ArgumentParser(description="Patch-level contrail segmentation")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-probability", action="store_true")
    args = parser.parse_args()

    device = _device(args.device)
    model = _load_model(args.checkpoint, device)
    dataset = PatchDataset(args.image_dir, args.size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for images, names in tqdm(loader, desc="patch inference"):
            images = images.to(device, non_blocking=True)
            probabilities = (
                torch.sigmoid(mask_logits(model(images))).cpu().numpy()[:, 0]
            )
            for probability, name in zip(probabilities, names):
                stem = Path(name).stem
                prediction = probability > args.threshold
                mask_path = output_dir / f"{stem}.png"
                cv2.imwrite(str(mask_path), prediction.astype(np.uint8) * 255)
                probability_path = ""
                if args.save_probability:
                    probability_path = str(output_dir / f"{stem}_probability.npy")
                    np.save(probability_path, probability.astype(np.float32))
                records.append(
                    {
                        "input_name": name,
                        "mask_path": str(mask_path),
                        "probability_path": probability_path,
                        "positive_pixels": int(prediction.sum()),
                        "threshold": args.threshold,
                    }
                )
    with (output_dir / "prediction_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def hann_weight(tile: int, epsilon: float = 1e-3) -> np.ndarray:
    one_dimensional = np.hanning(tile).astype(np.float32)
    return np.clip(np.outer(one_dimensional, one_dimensional), epsilon, None)


def sliding_origins(length: int, tile: int, stride: int) -> list[int]:
    origins = list(range(0, length, stride)) or [0]
    if origins[-1] + tile < length:
        origins.append(length - tile)
    return origins


@torch.no_grad()
def infer_raster(
    model: torch.nn.Module,
    source_path: str | Path,
    tile: int,
    overlap: float,
    batch_size: int,
    input_mode: str,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError as exc:
        raise RuntimeError("Whole-scene inference requires the 'geo' extra") from exc

    stride = max(1, round(tile * (1.0 - overlap)))
    weight = hann_weight(tile)
    with rasterio.open(source_path) as source:
        if source.count < 3:
            raise ValueError(f"{source_path} contains fewer than three bands")
        probability_sum = np.zeros((source.height, source.width), dtype=np.float32)
        weight_sum = np.zeros_like(probability_sum)
        pending: list[torch.Tensor] = []
        locations: list[tuple[int, int, int, int]] = []

        def flush() -> None:
            if not pending:
                return
            batch = torch.stack(pending).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                probability = (
                    torch.sigmoid(mask_logits(model(batch))).float().cpu().numpy()[:, 0]
                )
            for array, (row, column, height, width) in zip(probability, locations):
                probability_sum[row : row + height, column : column + width] += (
                    array[:height, :width] * weight[:height, :width]
                )
                weight_sum[row : row + height, column : column + width] += weight[
                    :height, :width
                ]
            pending.clear()
            locations.clear()

        for row in sliding_origins(source.height, tile, stride):
            for column in sliding_origins(source.width, tile, stride):
                height = min(tile, source.height - row)
                width = min(tile, source.width - column)
                data = source.read(
                    indexes=(1, 2, 3),
                    window=Window(column, row, width, height),
                ).astype(np.float32)
                padded = pad_reflect(data, tile, tile)
                if input_mode == "raw-tis":
                    rgb = preprocess_raw_tis_tile(padded)
                elif input_mode == "enhanced-8bit":
                    rgb = np.moveaxis(np.clip(padded, 0, 255).astype(np.uint8), 0, -1)
                else:
                    raise ValueError(f"Unsupported input mode: {input_mode}")
                pending.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
                locations.append((row, column, height, width))
                if len(pending) >= batch_size:
                    flush()
        flush()
        profile = source.profile.copy()
    return probability_sum / np.maximum(weight_sum, 1e-6), profile


def scene_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Overlapping-window whole-scene inference"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--input-mode", choices=("raw-tis", "enhanced-8bit"), default="raw-tis"
    )
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-probability", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    manifest = pd.read_csv(args.manifest)
    if args.path_column not in manifest:
        raise KeyError(f"Manifest has no '{args.path_column}' column")
    paths = [
        Path(value)
        for value in manifest[args.path_column].dropna().astype(str).unique()
    ]
    device = _device(args.device)
    model = _load_model(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import rasterio

    records: list[dict[str, object]] = []
    for source_path in tqdm(paths, desc="scene inference"):
        probability, profile = infer_raster(
            model,
            source_path,
            args.tile,
            args.overlap,
            args.batch_size,
            args.input_mode,
            device,
        )
        mask = (probability > args.threshold).astype(np.uint8)
        mask_path = output_dir / f"{source_path.stem}_mask.tif"
        profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        with rasterio.open(mask_path, "w", **profile) as destination:
            destination.write(mask, 1)
        probability_path = ""
        if args.save_probability:
            probability_path = str(output_dir / f"{source_path.stem}_probability.npy")
            np.save(probability_path, probability.astype(np.float32))
        records.append(
            {
                "source_path": str(source_path),
                "mask_path": str(mask_path),
                "probability_path": probability_path,
                "threshold": args.threshold,
                "tile": args.tile,
                "overlap": args.overlap,
            }
        )
    pd.DataFrame(records).to_csv(
        output_dir / "scene_prediction_manifest.csv", index=False
    )


if __name__ == "__main__":
    patch_cli()
