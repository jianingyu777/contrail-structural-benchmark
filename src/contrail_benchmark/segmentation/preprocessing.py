from __future__ import annotations

import cv2
import numpy as np


GAINS = np.array([0.003947, 0.003946, 0.005329], dtype=np.float32)
BIASES = np.array([0.167126, 0.124622, 0.222253], dtype=np.float32)
K1 = np.array([1655.627802, 838.7063188, 543.0579522], dtype=np.float32)
K2 = np.array([1542.760922, 1342.718697, 1232.021436], dtype=np.float32)


def stretch_linear(image_chw: np.ndarray) -> np.ndarray:
    """Stretch each channel from its 0th to 95th percentile into uint8."""
    out = np.empty(image_chw.shape, dtype=np.uint8)
    for channel in range(image_chw.shape[0]):
        band = image_chw[channel].astype(np.float32, copy=True)
        zero = band == 0
        nonzero = band[~zero]
        if nonzero.size:
            band[zero] = float(nonzero.mean())
        low, high = np.percentile(band, (0, 95))
        if high <= low:
            scaled = np.zeros_like(band)
        else:
            scaled = (band - low) * (255.0 / (high - low))
        out[channel] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def temperature_from_dn(data_chw: np.ndarray) -> np.ndarray:
    radiance = data_chw * GAINS[:, None, None] + BIASES[:, None, None]
    return (
        K2[:, None, None] / np.log(K1[:, None, None] / (radiance + 1e-6) + 1.0)
    ).astype(np.float32)


def preprocess_raw_tis_tile(data_chw: np.ndarray) -> np.ndarray:
    """Create the enhanced three-channel uint8 segmentation input."""
    if data_chw.shape[0] < 3:
        raise ValueError("At least three TIS bands are required")
    data_chw = data_chw[:3].astype(np.float32, copy=False)
    rgb = np.moveaxis(stretch_linear(data_chw), 0, -1)
    temperature = temperature_from_dn(data_chw)
    cold = temperature[1] < np.percentile(temperature[1], 90)
    ratio_23 = data_chw[1] / (data_chw[2] + 1e-6)
    ratio_21 = data_chw[1] / (data_chw[0] + 1e-6)
    spectral = (ratio_23 > np.percentile(ratio_23, 40)) & (
        ratio_21 < np.percentile(ratio_21, 50)
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    selection = cold & spectral
    saturation[selection] = np.clip(
        saturation[selection].astype(np.float32) * 1.2, 0, 255
    ).astype(np.uint8)
    hsv[..., 1] = saturation
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def pad_reflect(data_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_h = max(0, height - data_chw.shape[1])
    pad_w = max(0, width - data_chw.shape[2])
    if pad_h == 0 and pad_w == 0:
        return data_chw
    return np.stack(
        [
            cv2.copyMakeBorder(
                band,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_REFLECT_101,
            )
            for band in data_chw
        ]
    )
