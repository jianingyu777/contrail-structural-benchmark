from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _constant_fill_arguments() -> dict[str, int]:
    major = int(A.__version__.split(".", maxsplit=1)[0])
    if major >= 2:
        return {"fill": 0, "fill_mask": 0}
    return {"value": 0, "mask_value": 0}


def scan_pairs(split_dir: str | Path) -> list[tuple[Path, Path]]:
    split_dir = Path(split_dir)
    image_dir = split_dir / "image"
    label_dir = split_dir / "label"
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    labels = {
        path.stem: path
        for path in label_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    stems = sorted(images.keys() & labels.keys())
    if not stems:
        raise RuntimeError(f"No image-label pairs found under {split_dir}")
    return [(images[stem], labels[stem]) for stem in stems]


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)


def mask_stratum(mask: np.ndarray) -> str:
    fraction = float(mask.mean())
    if fraction == 0:
        return "negative"
    if fraction <= 0.01:
        return "positive_small"
    if fraction <= 0.05:
        return "positive_medium"
    return "positive_large"


def build_sampler_weights(
    items: list[tuple[Path, Path]], negative_weight: float = 1.2
) -> list[float]:
    """Return the positive/negative patch weights used by StageA training."""
    weights: list[float] = []
    for _, mask_path in items:
        is_positive = bool(read_mask(mask_path).any())
        weights.append(1.0 if is_positive else max(1e-6, float(negative_weight)))
    return weights


def skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """Reproduce the OpenCV morphological skeleton used by the trainer."""
    current = (mask > 0).astype(np.uint8)
    if not current.any():
        return current
    skeleton = np.zeros_like(current)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded
        if cv2.countNonZero(current) == 0:
            break
    return (skeleton > 0).astype(np.uint8)


def make_fold_split(
    data_root: str | Path,
    fold: int = 5,
    folds: int = 5,
    seed: int = 3407,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], dict[str, object]]:
    pool, splits, mapping = make_fold_splits(data_root, folds=folds, seed=seed)
    if not 1 <= fold <= len(splits):
        raise ValueError(f"fold must be in [1, {len(splits)}]")
    train_indices, validation_indices = splits[fold - 1]
    training = [pool[index] for index in train_indices]
    validation = [pool[index] for index in validation_indices]
    record = {
        "pool_size": len(pool),
        "fold": fold,
        "folds": folds,
        "seed": seed,
        "training_size": len(training),
        "validation_size": len(validation),
        "strata_mapping": mapping,
        "training_stems": [image.stem for image, _ in training],
        "validation_stems": [image.stem for image, _ in validation],
    }
    return training, validation, record


def make_fold_splits(
    data_root: str | Path,
    folds: int = 5,
    seed: int = 3407,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[np.ndarray, np.ndarray]],
    dict[str, int],
]:
    """Read the pool once and construct every stratified fold."""
    data_root = Path(data_root)
    pool = scan_pairs(data_root / "train") + scan_pairs(data_root / "val")
    labels = [mask_stratum(read_mask(mask_path)) for _, mask_path in pool]
    mapping = {name: index for index, name in enumerate(sorted(set(labels)))}
    strata = np.asarray([mapping[name] for name in labels])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.arange(len(pool)), strata))
    return pool, splits, mapping


def training_transform(size: int) -> A.Compose:
    fill = _constant_fill_arguments()
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.1,
                rotate_limit=180,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
                **fill,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
                **fill,
            ),
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(0.2, 0.3, p=1.0),
                    A.RandomGamma(gamma_limit=(20, 100), p=1.0),
                ],
                p=0.5,
            ),
            ToTensorV2(),
        ]
    )


def evaluation_transform(size: int) -> A.Compose:
    fill = _constant_fill_arguments()
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
                **fill,
            ),
            A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ]
    )


class ContrailPatchDataset(Dataset):
    def __init__(self, items: list[tuple[Path, Path]], transform: A.Compose) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, mask_path = self.items[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        transformed = self.transform(image=image, mask=read_mask(mask_path))
        image_tensor = transformed["image"].float()
        mask = transformed["mask"]
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)
        mask_np = (np.squeeze(mask_np) > 0.5).astype(np.uint8)
        centerline = skeletonize_binary(mask_np).astype(np.float32)
        return (
            image_tensor,
            torch.from_numpy(mask_np.astype(np.float32))[None],
            torch.from_numpy(centerline)[None],
            image_path.name,
        )
