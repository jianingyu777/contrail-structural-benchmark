"""Apply the fixed ContrailStruct30 display enhancement to one uint16 TIFF."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image


GAINS = np.array([0.003947, 0.003946, 0.005329], dtype=np.float32)
BIASES = np.array([0.167126, 0.124622, 0.222253], dtype=np.float32)
K1 = np.array([1655.627802, 838.7063188, 543.0579522], dtype=np.float32)
K2 = np.array([1542.760922, 1342.718697, 1232.021436], dtype=np.float32)


def stretch_linear(data: np.ndarray) -> np.ndarray:
    output = np.empty(data.shape, dtype=np.uint8)
    for index in range(3):
        band = data[index].copy()
        zero_mask = band == 0
        nonzero = band[~zero_mask]
        band[zero_mask] = nonzero.mean() if nonzero.size else 0
        low, high = np.percentile(band, (0, 95))
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = (band - low) / (high - low) * 255.0
            np.clip(scaled, 0, 255, out=scaled)
            output[index] = scaled.astype(np.uint8)
    return output


def convert_array(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.shape != (3, 256, 256):
        raise ValueError(f"Expected CHW shape (3, 256, 256), found {data.shape}")
    stretched = stretch_linear(data)
    rgb = cv2.merge([stretched[index] for index in range(3)])
    radiance = data * GAINS[:, None, None] + BIASES[:, None, None]
    temperature = K2[:, None, None] / np.log(K1[:, None, None] / radiance + 1)
    ratio_b2b3 = data[1] / (data[2] + 1e-6)
    ratio_b2b1 = data[1] / (data[0] + 1e-6)
    enhancement_mask = (
        (temperature[1] < np.percentile(temperature[1], 90))
        & (ratio_b2b3 > np.percentile(ratio_b2b3, 40))
        & (ratio_b2b1 < np.percentile(ratio_b2b1, 50))
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[..., 1][enhancement_mask] = np.clip(
        hsv[..., 1][enhancement_mask] * 1.2, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("Destination already exists")
    with rasterio.open(args.source) as dataset:
        if dataset.count != 3 or any(dtype != "uint16" for dtype in dataset.dtypes):
            raise ValueError("Expected a three-band uint16 TIFF")
        data = dataset.read(out_dtype="float32")
    Image.fromarray(convert_array(data)).save(args.destination, format="PNG")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    main()
