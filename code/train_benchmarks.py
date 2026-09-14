from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import tifffile
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


WORKSPACE_DEFAULT = Path("runs")
METADATA_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
DATA_ROOT_DEFAULT = Path(os.environ.get("CONTRAILSTRUCT30_DATA_ROOT", "dataset"))
MODELS = ("unet_resnet34", "deeplabv3plus_resnet50", "segformer_b0")
REPRESENTATIONS = ("8bit", "uint16")
DISPLAY_NAMES = {
    "unet_resnet34": "U-Net-ResNet34",
    "deeplabv3plus_resnet50": "DeepLabV3+-ResNet50",
    "segformer_b0": "SegFormer-B0",
}
SOURCES = {
    "unet_resnet34": "segmentation_models_pytorch.Unet",
    "deeplabv3plus_resnet50": "segmentation_models_pytorch.DeepLabV3Plus",
    "segformer_b0": "segmentation_models_pytorch.Segformer",
}
UPSTREAM_URLS = {
    "segmentation_models_pytorch": "https://github.com/qubvel-org/segmentation_models.pytorch",
    "timm": "https://github.com/huggingface/pytorch-image-models",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Protocol:
    seed: int = 3407
    image_size: int = 256
    epochs: int = 30
    patience: int = 8
    physical_batch_size: int = 32
    effective_batch_size: int = 32
    workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    positive_class_weight: float = 5.0
    dice_loss_weight: float = 1.0
    difficult_negative_sampling_weight: float = 1.2


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        transform = getattr(worker_info.dataset, "transform", None)
        if transform is not None and hasattr(transform, "set_random_seed"):
            transform.set_random_seed(worker_seed)


def read_records(metadata_root: Path) -> pd.DataFrame:
    path = metadata_root / "metadata" / "records.csv"
    frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
    required = {
        "record_id",
        "scene_id",
        "acquisition_date",
        "season",
        "split",
        "sample_class",
        "foreground_fraction",
        "image_8bit_path",
        "image_16bit_path",
        "mask_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"records.csv is missing columns: {sorted(missing)}")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("The final manifest must contain train, validation and test")
    if not frame["record_id"].is_unique:
        raise ValueError("record_id is not unique")
    return frame


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def load_uint16_statistics(path: Path) -> dict[str, object]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("estimation_split") != "train":
        raise ValueError("uint16 normalization must be estimated from train only")
    for key in ("p01", "median", "p99"):
        if len(values[key]) != 3:
            raise ValueError(f"Invalid uint16 normalization field: {key}")
    return values


def normalize_uint16(image: np.ndarray, statistics: dict[str, object]) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape != (256, 256, 3) or image.dtype != np.uint16:
        raise ValueError(f"Unexpected uint16 raster: shape={image.shape}, dtype={image.dtype}")
    output = image.astype(np.float32)
    p01 = np.asarray(statistics["p01"], dtype=np.float32)
    median = np.asarray(statistics["median"], dtype=np.float32)
    p99 = np.asarray(statistics["p99"], dtype=np.float32)
    for channel in range(3):
        band = output[..., channel]
        invalid = (band == 0) | (band >= 65000)
        band[invalid] = median[channel]
        np.clip(band, p01[channel], p99[channel], out=band)
        band -= p01[channel]
        band /= max(float(p99[channel] - p01[channel]), 1.0)
    return output


def train_transform(image_size: int, seed: int) -> A.Compose:
    return A.Compose(
        [
            A.Affine(
                scale=(0.90, 1.10),
                translate_percent=(-0.20, 0.20),
                rotate=(-180, 180),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.20, contrast_limit=0.30, p=1.0
                    ),
                    A.RandomGamma(gamma_limit=(40, 120), p=1.0),
                ],
                p=0.5,
            ),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ],
        seed=seed,
    )


def eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                max_pixel_value=1.0,
            ),
            ToTensorV2(),
        ]
    )


class ContrailDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        representation: str,
        transform: A.Compose,
        uint16_statistics: dict[str, object] | None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.data_root = data_root
        self.representation = representation
        self.transform = transform
        self.uint16_statistics = uint16_statistics

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        if self.representation == "8bit":
            image_path = self.data_root / row["image_8bit_path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            image_path = self.data_root / row["image_16bit_path"]
            image = tifffile.imread(image_path)
            if self.uint16_statistics is None:
                raise RuntimeError("Missing train-only uint16 statistics")
            image = normalize_uint16(image, self.uint16_statistics)
        mask = read_mask(self.data_root / row["mask_path"])
        sample = self.transform(image=image, mask=mask)
        image_tensor = sample["image"].float()
        mask_tensor = sample["mask"]
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(mask_tensor)
        metadata = {
            "record_id": row["record_id"],
            "scene_id": row["scene_id"],
            "acquisition_date": str(row["acquisition_date"]),
            "season": row["season"],
            "sample_class": row["sample_class"],
            "foreground_fraction": float(row["foreground_fraction"]),
        }
        return image_tensor, (mask_tensor.float() > 0.5).float().unsqueeze(0), metadata


def build_model(name: str, pretrained: bool) -> nn.Module:
    encoder_weights = "imagenet" if pretrained else None
    if name == "unet_resnet34":
        return smp.Unet(
            encoder_name="resnet34", encoder_weights=encoder_weights,
            in_channels=3, classes=1,
        )
    if name == "deeplabv3plus_resnet50":
        return smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights=encoder_weights,
            in_channels=3, classes=1,
        )
    if name == "segformer_b0":
        return smp.Segformer(
            encoder_name="mit_b0", encoder_weights=encoder_weights,
            in_channels=3, classes=1,
        )
    raise ValueError(f"Unsupported model: {name}")


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, dict):
        output = output["out"]
    if output.shape[-2:] != images.shape[-2:]:
        output = F.interpolate(
            output, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
    return output


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    positive_class_weight: torch.Tensor,
    dice_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, masks, pos_weight=positive_class_weight
    )
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + masks.sum((1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_weight * dice_loss


def threshold_grid() -> list[float]:
    return [round(value, 2) for value in np.arange(0.05, 0.951, 0.01)] + [0.97, 0.99]


def empty_counts() -> dict[str, float]:
    return {
        "tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0,
        "patches": 0.0, "positive_patches": 0.0,
        "difficult_negative_patches": 0.0,
        "difficult_negative_activated": 0.0,
        "positive_patch_iou_sum": 0.0,
    }


def add_counts(
    counts: dict[str, float],
    prediction: np.ndarray,
    reference: np.ndarray,
    difficult_negative: bool,
) -> None:
    tp = float(np.logical_and(prediction, reference).sum())
    fp = float(np.logical_and(prediction, ~reference).sum())
    tn = float(np.logical_and(~prediction, ~reference).sum())
    fn = float(np.logical_and(~prediction, reference).sum())
    counts["tp"] += tp
    counts["fp"] += fp
    counts["tn"] += tn
    counts["fn"] += fn
    counts["patches"] += 1.0
    if reference.any():
        counts["positive_patches"] += 1.0
        counts["positive_patch_iou_sum"] += tp / max(1.0, tp + fp + fn)
    if difficult_negative:
        counts["difficult_negative_patches"] += 1.0
        counts["difficult_negative_activated"] += float(prediction.any())


def finalize_counts(counts: dict[str, float]) -> dict[str, float | int | None]:
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    positive_patches = counts["positive_patches"]
    negative_patches = counts["difficult_negative_patches"]
    return {
        "patches": int(counts["patches"]),
        "positive_patches": int(positive_patches),
        "difficult_negative_patches": int(negative_patches),
        "iou": tp / max(1.0, tp + fp + fn),
        "dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
        "precision": tp / max(1.0, tp + fp),
        "recall": tp / max(1.0, tp + fn),
        "mean_positive_patch_iou": (
            counts["positive_patch_iou_sum"] / positive_patches
            if positive_patches else None
        ),
        "difficult_negative_activation_rate": (
            counts["difficult_negative_activated"] / negative_patches
            if negative_patches else None
        ),
        "tp_pixels": int(tp), "fp_pixels": int(fp),
        "tn_pixels": int(tn), "fn_pixels": int(fn),
    }


def make_loaders(
    records: pd.DataFrame,
    data_root: Path,
    representation: str,
    protocol: Protocol,
    uint16_statistics: dict[str, object] | None,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    dict[str, object],
    torch.Generator,
    torch.Generator,
]:
    frames = {
        split: records.loc[records["split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    }
    summary = {
        split: {
            "records": len(frame),
            "positive": int(frame["sample_class"].eq("positive").sum()),
            "difficult_negative": int(frame["sample_class"].eq("difficult_negative").sum()),
            "scenes": int(frame["scene_id"].nunique()),
            "dates": int(frame["acquisition_date"].nunique()),
        }
        for split, frame in frames.items()
    }
    train_data = ContrailDataset(
        frames["train"], data_root, representation,
        train_transform(protocol.image_size, protocol.seed), uint16_statistics,
    )
    validation_data = ContrailDataset(
        frames["validation"], data_root, representation,
        eval_transform(protocol.image_size), uint16_statistics,
    )
    test_data = ContrailDataset(
        frames["test"], data_root, representation,
        eval_transform(protocol.image_size), uint16_statistics,
    )
    weights = np.where(
        frames["train"]["sample_class"].eq("difficult_negative"),
        protocol.difficult_negative_sampling_weight, 1.0,
    )
    samples_per_epoch = (len(weights) // protocol.effective_batch_size) * protocol.effective_batch_size
    sampler_generator = torch.Generator().manual_seed(protocol.seed)
    sampler = WeightedRandomSampler(
        weights.tolist(), samples_per_epoch, replacement=True, generator=sampler_generator
    )
    loader_generator = torch.Generator().manual_seed(protocol.seed + 100000)
    common = {
        "batch_size": protocol.physical_batch_size,
        "num_workers": protocol.workers,
        "pin_memory": True,
        "persistent_workers": protocol.workers > 0,
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }
    train_loader = DataLoader(train_data, sampler=sampler, drop_last=True, **common)
    validation_loader = DataLoader(validation_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return (
        train_loader,
        validation_loader,
        test_loader,
        summary,
        sampler_generator,
        loader_generator,
    )


@torch.inference_mode()
def select_validation_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, list[dict[str, float]]]:
    model.eval()
    positive_hist = torch.zeros(1001, dtype=torch.int64, device=device)
    negative_hist = torch.zeros(1001, dtype=torch.int64, device=device)
    difficult_negative_maxima: list[np.ndarray] = []
    amp = device.type == "cuda"
    for batch_index, (images, masks, metadata) in enumerate(
        tqdm(loader, desc="Validation threshold", ncols=92, mininterval=5.0)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        references = masks.to(device, non_blocking=True).bool()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            probabilities = torch.sigmoid(forward_logits(model, images))
        bins = torch.clamp((probabilities.float() * 1000.0).floor().long(), 0, 1000)
        positive_hist += torch.bincount(bins[references], minlength=1001)
        negative_hist += torch.bincount(bins[~references], minlength=1001)
        patch_maximum = probabilities.flatten(1).amax(dim=1).float().cpu().numpy()
        classes = list(metadata["sample_class"])
        difficult_negative_maxima.append(
            patch_maximum[np.asarray([value == "difficult_negative" for value in classes])]
        )
    positive_hist_np = positive_hist.cpu().numpy()
    negative_hist_np = negative_hist.cpu().numpy()
    negative_maxima = (
        np.concatenate(difficult_negative_maxima)
        if difficult_negative_maxima else np.empty(0, dtype=np.float32)
    )
    records: list[dict[str, float]] = []
    for threshold in threshold_grid():
        first_bin = int(math.ceil(threshold * 1000.0 - 1e-9))
        tp = float(positive_hist_np[first_bin:].sum())
        fn = float(positive_hist_np[:first_bin].sum())
        fp = float(negative_hist_np[first_bin:].sum())
        records.append({
            "threshold": threshold,
            "dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
            "iou": tp / max(1.0, tp + fp + fn),
            "difficult_negative_activation_rate": (
                float((negative_maxima >= threshold).mean())
                if len(negative_maxima) else 0.0
            ),
        })
    selected = max(
        records,
        key=lambda row: (
            row["dice"], -row["difficult_negative_activation_rate"], row["threshold"]
        ),
    )
    return float(selected["threshold"]), records


@torch.inference_mode()
def evaluate_at_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    output_path: Path | None = None,
    max_batches: int | None = None,
) -> dict[str, dict[str, float | int | None]]:
    model.eval()
    groups = {"all": empty_counts(), "difficult_negative": empty_counts()}
    for size in ("le_1pct", "1_to_5pct", "gt_5pct"):
        groups[f"positive_{size}"] = empty_counts()
    rows: list[dict[str, object]] = []
    amp = device.type == "cuda"
    for batch_index, (images, masks, metadata) in enumerate(
        tqdm(loader, desc="Evaluation", ncols=92, mininterval=5.0)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            probabilities = torch.sigmoid(forward_logits(model, images))
        predictions = (probabilities[:, 0] >= threshold).cpu().numpy()
        references = masks[:, 0].numpy() > 0.5
        for index in range(len(references)):
            sample_class = metadata["sample_class"][index]
            difficult_negative = sample_class == "difficult_negative"
            prediction = predictions[index]
            reference = references[index]
            add_counts(groups["all"], prediction, reference, difficult_negative)
            if difficult_negative:
                add_counts(groups["difficult_negative"], prediction, reference, True)
            if reference.any():
                fraction = float(metadata["foreground_fraction"][index])
                if fraction <= 0.01:
                    group_name = "positive_le_1pct"
                elif fraction <= 0.05:
                    group_name = "positive_1_to_5pct"
                else:
                    group_name = "positive_gt_5pct"
                add_counts(groups[group_name], prediction, reference, False)
            tp = int(np.logical_and(prediction, reference).sum())
            fp = int(np.logical_and(prediction, ~reference).sum())
            tn = int(np.logical_and(~prediction, ~reference).sum())
            fn = int(np.logical_and(~prediction, reference).sum())
            union = tp + fp + fn
            rows.append({
                "record_id": metadata["record_id"][index],
                "scene_id": metadata["scene_id"][index],
                "acquisition_date": metadata["acquisition_date"][index],
                "season": metadata["season"][index],
                "sample_class": sample_class,
                "foreground_fraction": float(metadata["foreground_fraction"][index]),
                "tp_pixels": tp, "fp_pixels": fp, "tn_pixels": tn, "fn_pixels": fn,
                "positive_reference": int(reference.any()),
                "difficult_negative": int(difficult_negative),
                "prediction_nonempty": int(prediction.any()),
                "positive_patch_iou": tp / union if reference.any() and union else np.nan,
            })
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return {name: finalize_counts(values) for name, values in groups.items()}


def output_directory(workspace: Path, representation: str, model: str, seed: int) -> Path:
    return workspace / "outputs" / representation / model / f"seed_{seed}"


def load_normalization(args: argparse.Namespace) -> dict[str, object] | None:
    if args.representation == "8bit":
        return None
    path = Path(args.uint16_statistics)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run the compute-normalization command first"
        )
    return load_uint16_statistics(path)


def train_model(args: argparse.Namespace) -> None:
    protocol = Protocol(
        seed=args.seed, epochs=args.epochs, patience=args.patience,
        physical_batch_size=args.batch_size,
        effective_batch_size=args.effective_batch_size, workers=args.workers,
    )
    if protocol.effective_batch_size % protocol.physical_batch_size:
        raise ValueError("effective batch size must be divisible by physical batch size")
    set_seed(protocol.seed)
    workspace = Path(args.workspace)
    metadata_root = Path(args.metadata_root)
    data_root = Path(args.data_root)
    records = read_records(metadata_root)
    normalization = load_normalization(args)
    (
        train_loader,
        validation_loader,
        test_loader,
        split_summary,
        sampler_generator,
        loader_generator,
    ) = make_loaders(records, data_root, args.representation, protocol, normalization)
    output_dir = output_directory(workspace, args.representation, args.model, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists() and not args.force:
        print(f"Completed result already exists: {output_dir / 'result.json'}")
        return
    write_json(output_dir / "protocol.json", asdict(protocol))
    write_json(output_dir / "split_summary.json", split_summary)
    write_json(output_dir / "implementation_provenance.json", {
        "model": DISPLAY_NAMES[args.model],
        "constructor": SOURCES[args.model],
        "upstream_urls": UPSTREAM_URLS,
        "segmentation_models_pytorch_license": "MIT",
        "timm_license": "Apache-2.0",
    })

    device = torch.device(args.device)
    model = build_model(args.model, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )

    def learning_rate_multiplier(epoch_index: int) -> float:
        if epoch_index < protocol.warmup_epochs:
            return (epoch_index + 1) / max(1, protocol.warmup_epochs)
        progress = (epoch_index - protocol.warmup_epochs) / max(
            1, protocol.epochs - protocol.warmup_epochs
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positive_class_weight = torch.tensor(
        [protocol.positive_class_weight], dtype=torch.float32, device=device
    )
    accumulation_steps = protocol.effective_batch_size // protocol.physical_batch_size
    best_path = output_dir / "best.pth"
    latest_path = output_dir / "latest.pth"
    best_dice = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    log: list[dict[str, object]] = []
    started = time.time()

    if latest_path.exists() and not args.force:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        sampler_generator.set_state(checkpoint["sampler_generator_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        best_dice = float(checkpoint["best_dice"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        log = checkpoint["training_log"]
        print(f"Resuming {output_dir} at epoch {start_epoch}")

    for epoch in range(start_epoch, protocol.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        batches_used = 0
        progress = tqdm(
            train_loader, desc=f"{DISPLAY_NAMES[args.model]} {args.representation} {epoch:02d}",
            ncols=98, mininterval=5.0,
        )
        for batch_index, (images, masks, _) in enumerate(progress):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = forward_logits(model, images)
                raw_loss = segmentation_loss(
                    logits, masks, positive_class_weight, protocol.dice_loss_weight
                )
                loss = raw_loss / accumulation_steps
            scaler.scale(loss).backward()
            batches_used += 1
            if batches_used % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(raw_loss.detach().item()))
        if batches_used % accumulation_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        selected_threshold, threshold_records = select_validation_threshold(
            model, validation_loader, device, args.max_eval_batches
        )
        selected_record = next(
            row for row in threshold_records if row["threshold"] == selected_threshold
        )
        validation_dice = float(selected_record["dice"])
        epoch_record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "selected_threshold": selected_threshold,
            "validation_dice": validation_dice,
            "elapsed_seconds": time.time() - started,
        }
        log.append(epoch_record)
        write_json(output_dir / "training_log.json", log)
        write_json(output_dir / "latest_validation_thresholds.json", threshold_records)
        if validation_dice > best_dice + 1e-5:
            best_dice = validation_dice
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model": args.model, "representation": args.representation,
                "seed": protocol.seed, "state_dict": model.state_dict(),
                "epoch": epoch, "selected_threshold": selected_threshold,
                "validation_dice": validation_dice, "protocol": asdict(protocol),
            }, best_path)
        else:
            stale_epochs += 1
        torch.save({
            "model": args.model, "representation": args.representation,
            "seed": protocol.seed, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch,
            "best_dice": best_dice, "best_epoch": best_epoch,
            "stale_epochs": stale_epochs, "training_log": log,
            "sampler_generator_state": sampler_generator.get_state(),
            "loader_generator_state": loader_generator.get_state(),
            "protocol": asdict(protocol),
        }, latest_path)
        print(json.dumps(epoch_record), flush=True)
        if stale_epochs >= protocol.patience:
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model = build_model(args.model, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    selected_threshold, threshold_records = select_validation_threshold(
        model, validation_loader, device, args.max_eval_batches
    )
    validation_metrics = evaluate_at_threshold(
        model, validation_loader, device, selected_threshold,
        output_dir / "validation_patch_counts.csv", args.max_eval_batches,
    )
    test_metrics = evaluate_at_threshold(
        model, test_loader, device, selected_threshold,
        output_dir / "test_patch_counts.csv", args.max_eval_batches,
    )
    result = {
        "model": args.model, "display_name": DISPLAY_NAMES[args.model],
        "implementation": SOURCES[args.model], "representation": args.representation,
        "seed": protocol.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(checkpoint["epoch"]),
        "selected_threshold": selected_threshold,
        "threshold_selection": "maximum micro-averaged foreground Dice on validation only",
        "split_summary": split_summary,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - started,
        "checkpoint_sha256": sha256_file(best_path),
        "records_manifest_sha256": sha256_file(metadata_root / "metadata" / "records.csv"),
        "data_root": str(data_root.resolve()),
        "cache_verification": (
            str((workspace / "outputs" / "cache_verification.json").resolve())
            if (workspace / "outputs" / "cache_verification.json").exists()
            else None
        ),
        "uint16_statistics_sha256": (
            sha256_file(Path(args.uint16_statistics)) if args.representation == "uint16" else None
        ),
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "segmentation_models_pytorch": smp.__version__,
            "albumentations": A.__version__, "tifffile": tifffile.__version__,
        },
    }
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "validation_thresholds.json", threshold_records)
    print(json.dumps(result, indent=2), flush=True)


def histogram_quantile(histogram: np.ndarray, probability: float) -> int:
    total = int(histogram.sum())
    if total == 0:
        raise ValueError("No valid uint16 pixels")
    rank = int(round(probability * (total - 1)))
    return int(np.searchsorted(np.cumsum(histogram), rank + 1))


def compute_normalization(args: argparse.Namespace) -> None:
    metadata_root = Path(args.metadata_root)
    data_root = Path(args.data_root)
    records = read_records(metadata_root)
    train = records.loc[records["split"].eq("train")].reset_index(drop=True)
    histogram = np.zeros((3, 65536), dtype=np.uint64)
    for row in tqdm(train.itertuples(index=False), total=len(train), desc="uint16 histogram"):
        image = tifffile.imread(data_root / row.image_16bit_path)
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape != (256, 256, 3) or image.dtype != np.uint16:
            raise ValueError(f"Invalid uint16 image: {row.image_16bit_path}")
        encoded = image.astype(np.int64) + np.arange(3, dtype=np.int64) * 65536
        image_counts = np.bincount(
            encoded.ravel(), minlength=3 * 65536
        ).reshape(3, 65536)
        histogram += image_counts.astype(np.uint64, copy=False)
    valid = histogram.copy()
    valid[:, 0] = 0
    valid[:, 65000:] = 0
    output_dir = Path(args.workspace) / "outputs" / "normalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = output_dir / "uint16_train_histogram.npz"
    np.savez_compressed(histogram_path, histogram=histogram)
    statistics = {
        "estimation_split": "train",
        "train_records": len(train),
        "fill_rule": "values equal to 0 or greater than or equal to 65000",
        "fill_replacement": "train-channel median",
        "clipping": "train-channel P1 to P99",
        "scaling": "linear to [0,1] after clipping",
        "quantile_method": "nearest rank from exact uint16 histogram",
        "p01": [histogram_quantile(valid[channel], 0.01) for channel in range(3)],
        "median": [histogram_quantile(valid[channel], 0.50) for channel in range(3)],
        "p99": [histogram_quantile(valid[channel], 0.99) for channel in range(3)],
        "valid_pixel_count": [int(valid[channel].sum()) for channel in range(3)],
        "fill_pixel_count": [
            int(histogram[channel, 0] + histogram[channel, 65000:].sum())
            for channel in range(3)
        ],
        "records_manifest_sha256": sha256_file(metadata_root / "metadata" / "records.csv"),
        "histogram_sha256": sha256_file(histogram_path),
    }
    output_path = output_dir / "uint16_train_percentiles.json"
    write_json(output_path, statistics)
    print(json.dumps(statistics, indent=2))


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    totals = frame[["tp_pixels", "fp_pixels", "tn_pixels", "fn_pixels"]].sum()
    tp, fp, tn, fn = (float(totals[key]) for key in totals.index)
    positive_iou = frame.loc[frame["positive_reference"].eq(1), "positive_patch_iou"]
    difficult = frame.loc[frame["difficult_negative"].eq(1)]
    return {
        "iou": tp / max(1.0, tp + fp + fn),
        "dice": 2.0 * tp / max(1.0, 2.0 * tp + fp + fn),
        "precision": tp / max(1.0, tp + fp),
        "recall": tp / max(1.0, tp + fn),
        "mean_positive_patch_iou": float(positive_iou.mean()) if len(positive_iou) else np.nan,
        "difficult_negative_activation_rate": (
            float(difficult["prediction_nonempty"].mean()) if len(difficult) else np.nan
        ),
    }


def date_cluster_bootstrap(
    frame: pd.DataFrame, seed: int, replicates: int = 2000
) -> dict[str, dict[str, float]]:
    metric_names = (
        "iou",
        "dice",
        "precision",
        "recall",
        "mean_positive_patch_iou",
        "difficult_negative_activation_rate",
    )
    if frame.empty:
        return {}

    sufficient = frame.assign(
        positive_iou_sum=np.where(
            frame["positive_reference"].eq(1), frame["positive_patch_iou"], 0.0
        ),
        positive_count=frame["positive_reference"].eq(1).astype(np.int64),
        difficult_activation_sum=np.where(
            frame["difficult_negative"].eq(1), frame["prediction_nonempty"], 0.0
        ),
        difficult_count=frame["difficult_negative"].eq(1).astype(np.int64),
    ).groupby("acquisition_date", sort=True)[
        [
            "tp_pixels",
            "fp_pixels",
            "tn_pixels",
            "fn_pixels",
            "positive_iou_sum",
            "positive_count",
            "difficult_activation_sum",
            "difficult_count",
        ]
    ].sum()

    cluster_values = sufficient.to_numpy(dtype=np.float64)
    cluster_count = len(cluster_values)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    multiplicities = np.zeros((replicates, cluster_count), dtype=np.int32)
    rows = np.repeat(np.arange(replicates), cluster_count)
    np.add.at(multiplicities, (rows, chosen.ravel()), 1)
    totals = multiplicities @ cluster_values

    tp, fp, _tn, fn, positive_iou_sum, positive_count, difficult_sum, difficult_count = (
        totals[:, index] for index in range(totals.shape[1])
    )
    samples = {
        "iou": tp / np.maximum(1.0, tp + fp + fn),
        "dice": 2.0 * tp / np.maximum(1.0, 2.0 * tp + fp + fn),
        "precision": tp / np.maximum(1.0, tp + fp),
        "recall": tp / np.maximum(1.0, tp + fn),
        "mean_positive_patch_iou": np.divide(
            positive_iou_sum,
            positive_count,
            out=np.full(replicates, np.nan),
            where=positive_count > 0,
        ),
        "difficult_negative_activation_rate": np.divide(
            difficult_sum,
            difficult_count,
            out=np.full(replicates, np.nan),
            where=difficult_count > 0,
        ),
    }
    return {
        name: {
            "lower_95": float(np.quantile(values[np.isfinite(values)], 0.025)),
            "upper_95": float(np.quantile(values[np.isfinite(values)], 0.975)),
        }
        for name, values in samples.items()
        if name in metric_names and np.isfinite(values).any()
    }


def bootstrap_and_aggregate(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace)
    run_rows: list[dict[str, object]] = []
    stratum_rows: list[dict[str, object]] = []
    for result_path in sorted((workspace / "outputs").glob("*/*/seed_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        count_path = result_path.parent / "test_patch_counts.csv"
        frame = pd.read_csv(count_path)
        masks = {
            "all": np.ones(len(frame), dtype=bool),
            "positive_le_1pct": frame["positive_reference"].eq(1) & frame["foreground_fraction"].le(0.01),
            "positive_1_to_5pct": frame["positive_reference"].eq(1) & frame["foreground_fraction"].gt(0.01) & frame["foreground_fraction"].le(0.05),
            "positive_gt_5pct": frame["positive_reference"].eq(1) & frame["foreground_fraction"].gt(0.05),
            "difficult_negative": frame["difficult_negative"].eq(1),
        }
        overall = calculate_metrics(frame)
        intervals = date_cluster_bootstrap(frame, int(result["seed"]), args.replicates)
        row = {
            "model": result["display_name"], "representation": result["representation"],
            "seed": result["seed"], "threshold": result["selected_threshold"],
            "best_epoch": result["best_epoch"], "parameters": result["parameters"], **overall,
        }
        for metric, limits in intervals.items():
            row[f"{metric}_ci95_low"] = limits["lower_95"]
            row[f"{metric}_ci95_high"] = limits["upper_95"]
        run_rows.append(row)
        for stratum, selection in masks.items():
            subset = frame.loc[selection].copy()
            metrics = calculate_metrics(subset)
            ci = date_cluster_bootstrap(
                subset, int(result["seed"]) + 1000, args.replicates
            )
            stratum_row = {
                "model": result["display_name"],
                "representation": result["representation"],
                "seed": result["seed"], "stratum": stratum,
                "records": len(subset), **metrics,
            }
            for metric, limits in ci.items():
                stratum_row[f"{metric}_ci95_low"] = limits["lower_95"]
                stratum_row[f"{metric}_ci95_high"] = limits["upper_95"]
            stratum_rows.append(stratum_row)
    output_dir = workspace / "outputs" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(run_rows)
    strata = pd.DataFrame(stratum_rows)
    runs.to_csv(output_dir / "all_run_metrics_with_date_bootstrap.csv", index=False)
    strata.to_csv(output_dir / "stratified_metrics_with_date_bootstrap.csv", index=False)
    main = runs.loc[
        runs["representation"].eq("8bit") & runs["seed"].eq(3407)
    ].copy()
    main.to_csv(output_dir / "three_model_8bit_seed3407.csv", index=False)
    comparison = runs.loc[runs["model"].eq("U-Net-ResNet34")].copy()
    comparison.to_csv(output_dir / "unet_8bit_uint16_per_seed.csv", index=False)
    summary_columns = [
        "dice", "mean_positive_patch_iou", "difficult_negative_activation_rate"
    ]
    paired_rows: list[dict[str, object]] = []
    for seed, seed_frame in comparison.groupby("seed", sort=True):
        indexed = seed_frame.set_index("representation")
        if not {"8bit", "uint16"}.issubset(indexed.index):
            continue
        paired_row: dict[str, object] = {"seed": seed}
        for metric in summary_columns:
            value_8bit = float(indexed.loc["8bit", metric])
            value_uint16 = float(indexed.loc["uint16", metric])
            paired_row[f"{metric}_8bit"] = value_8bit
            paired_row[f"{metric}_uint16"] = value_uint16
            paired_row[f"{metric}_uint16_minus_8bit"] = value_uint16 - value_8bit
        paired_rows.append(paired_row)
    pd.DataFrame(paired_rows).to_csv(
        output_dir / "unet_uint16_minus_8bit_paired_differences.csv", index=False
    )
    representation_summary = comparison.groupby("representation")[summary_columns].agg(
        ["mean", "std"]
    )
    representation_summary.to_csv(output_dir / "unet_8bit_uint16_three_seed_summary.csv")
    write_json(output_dir / "bootstrap_protocol.json", {
        "cluster": "acquisition_date", "replicates": args.replicates,
        "interval": "percentile 95%", "test_partition_only": True,
    })
    print(main.to_string(index=False))
    print(representation_summary.to_string())


def add_common_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--workspace", default=str(WORKSPACE_DEFAULT))
    parser.add_argument("--metadata-root", default=str(METADATA_ROOT_DEFAULT))
    parser.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    parser.add_argument(
        "--uint16-statistics",
        default=str(WORKSPACE_DEFAULT / "outputs" / "normalization" / "uint16_train_percentiles.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--force", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final ContrailStruct30 segmentation experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    normalization = subparsers.add_parser("compute-normalization")
    normalization.add_argument("--workspace", default=str(WORKSPACE_DEFAULT))
    normalization.add_argument("--metadata-root", default=str(METADATA_ROOT_DEFAULT))
    normalization.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    normalization.set_defaults(func=compute_normalization)
    train = subparsers.add_parser("train")
    add_common_train_arguments(train)
    train.set_defaults(func=train_model)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--workspace", default=str(WORKSPACE_DEFAULT))
    aggregate.add_argument("--replicates", type=int, default=2000)
    aggregate.set_defaults(func=bootstrap_and_aggregate)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)
