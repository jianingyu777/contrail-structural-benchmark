from __future__ import annotations

import argparse
import gc
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import (
    ContrailPatchDataset,
    build_sampler_weights,
    evaluation_transform,
    make_fold_splits,
    training_transform,
)
from .losses import CompositeContrailLoss, LossConfig
from .metrics import ThresholdMetrics, checkpoint_score, choose_training_threshold
from .model import ContrailMaxViTUNet, mask_logits
from .runtime import set_seed, sha256, write_json


class ModelEMA:
    """Exponential moving average used by the recovered StageA trainer."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.ema = deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            incoming = source[key]
            if value.is_floating_point():
                value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
            else:
                value.copy_(incoming)


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    training_epochs: int,
    cycles: float = 0.5,
) -> LambdaLR:
    def multiplier(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(
            max(1, training_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * cycles * 2.0 * progress))
        return max(1e-6, cosine)

    return LambdaLR(optimizer, multiplier)


def _heads(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(output, dict)
        or "mask" not in output
        or "centerline" not in output
    ):
        raise RuntimeError("StageA training requires mask and centerline logits")
    return output["mask"], output["centerline"]


def _make_loader(
    dataset: ContrailPatchDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=sampler is not None,
    )


@torch.no_grad()
def _hard_negative_weights(
    model: torch.nn.Module,
    items: list[tuple[Path, Path]],
    image_size: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    base_negative_weight: float,
    gamma: float,
    maximum_weight: float,
    score_limit: int,
) -> list[float]:
    dataset = ContrailPatchDataset(items, evaluation_transform(image_size))
    loader = _make_loader(dataset, batch_size, workers, device)
    weights = build_sampler_weights(items, base_negative_weight)
    negative_scores: list[tuple[int, float]] = []
    model.eval()
    for batch_index, (images, masks, _, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        probabilities = torch.sigmoid(mask_logits(model(images)))
        for offset in range(probabilities.shape[0]):
            if masks[offset].sum() == 0:
                index = batch_index * batch_size + offset
                negative_scores.append((index, float(probabilities[offset].max())))
        if score_limit > 0 and len(negative_scores) >= score_limit:
            break
    for index, false_positive_score in negative_scores:
        value = base_negative_weight * (1.0 + gamma * false_positive_score)
        weights[index] = float(min(maximum_weight, max(1e-6, value)))
    return weights


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeContrailLoss,
    device: torch.device,
    thresholds: list[float],
    progress: float,
    upsample_x2: bool,
) -> tuple[float, dict[float, dict[str, float]]]:
    model.eval()
    accumulator = ThresholdMetrics(thresholds)
    losses: list[float] = []
    use_amp = device.type == "cuda"
    for images, masks, centerlines, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        centerlines = centerlines.to(device, non_blocking=True)
        model_input = (
            F.interpolate(images, scale_factor=2, mode="bicubic", align_corners=False)
            if upsample_x2
            else images
        )
        mask_target = (
            F.interpolate(masks, scale_factor=2, mode="nearest")
            if upsample_x2
            else masks
        )
        centerline_target = (
            F.interpolate(centerlines, scale_factor=2, mode="nearest")
            if upsample_x2
            else centerlines
        )
        with torch.autocast(device_type=device.type, enabled=use_amp):
            output = model(model_input)
            segmentation_logits, centerline_logits = _heads(output)
            loss, _ = criterion(
                segmentation_logits,
                mask_target,
                centerline_logits,
                centerline_target,
                progress,
            )
        probabilities = torch.sigmoid(segmentation_logits)
        if upsample_x2:
            probabilities = F.interpolate(
                probabilities, size=masks.shape[-2:], mode="nearest"
            )
        accumulator.update(probabilities, masks)
        losses.append(float(loss))
    average_loss = float(np.mean(losses)) if losses else 0.0
    return average_loss, accumulator.results()


def _parse_folds(values: list[str] | None, config: dict, total: int) -> list[int]:
    configured = config.get("training_folds", [config.get("fold", total)])
    if values is None:
        folds = [int(value) for value in configured]
    elif len(values) == 1 and values[0].lower() == "all":
        folds = list(range(1, total + 1))
    else:
        folds = [int(value) for value in values]
    if not folds or any(value < 1 or value > total for value in folds):
        raise ValueError(f"folds must lie in [1, {total}]")
    if len(set(folds)) != len(folds):
        raise ValueError("folds must not contain duplicates")
    return folds


def _split_record(
    pool: list[tuple[Path, Path]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    fold: int,
    total_folds: int,
    seed: int,
) -> dict[str, object]:
    return {
        "pool_size": len(pool),
        "fold": fold,
        "folds": total_folds,
        "seed": seed,
        "training_size": len(train_indices),
        "validation_size": len(validation_indices),
        "training_stems": [pool[int(index)][0].stem for index in train_indices],
        "validation_stems": [pool[int(index)][0].stem for index in validation_indices],
    }


def _train_fold(
    *,
    fold: int,
    pool: list[tuple[Path, Path]],
    split: tuple[np.ndarray, np.ndarray],
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    train_indices, validation_indices = split
    training_items = [pool[int(index)] for index in train_indices]
    validation_items = [pool[int(index)] for index in validation_indices]
    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        fold_dir / "split.json",
        _split_record(
            pool,
            train_indices,
            validation_indices,
            fold,
            int(config["cv_folds"]),
            int(config["seed"]),
        ),
    )

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    workers = int(config["workers"])
    training_data = ContrailPatchDataset(training_items, training_transform(image_size))
    validation_data = ContrailPatchDataset(
        validation_items, evaluation_transform(image_size)
    )
    base_weights = build_sampler_weights(
        training_items, float(config["negative_sample_weight"])
    )
    sampler = WeightedRandomSampler(
        weights=base_weights, num_samples=len(base_weights), replacement=True
    )
    training_loader = _make_loader(
        training_data, batch_size, workers, device, sampler=sampler
    )
    validation_loader = _make_loader(validation_data, batch_size, workers, device)

    model = ContrailMaxViTUNet(
        encoder_name=str(config["encoder_name"]),
        pretrained=bool(config["pretrained_encoder"]),
        decoder_channels=tuple(int(value) for value in config["decoder_channels"]),
        dropout=float(config["dropout"]),
        auxiliary_centerline=True,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=int(config["warmup_epochs"]),
        training_epochs=epochs,
    )
    criterion = CompositeContrailLoss(
        LossConfig(
            positive_class_weight=float(config["positive_class_weight"]),
            background_sample_scale=float(config["background_sample_scale"]),
            tversky_alpha=float(config["tversky_alpha"]),
            tversky_beta=float(config["tversky_beta"]),
            tversky_weight=float(config["tversky_weight"]),
            curriculum_start_fraction=float(config["curriculum_start_fraction"]),
            maximum_cldice_weight=float(config["maximum_cldice_weight"]),
            maximum_centerline_weight=float(config["maximum_centerline_weight"]),
            skeleton_iterations=int(config["cldice_iterations"]),
        )
    )
    use_amp = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, float(config["ema_decay"])) if config["ema"] else None
    model_name = str(config["model_name"])
    best_iou_path = output_dir / f"{model_name}_fp_fold{fold}_best_iou.pth"
    best_loss_path = output_dir / f"{model_name}_fp_fold{fold}_best_loss.pth"
    final_path = output_dir / f"{model_name}_fp_fold{fold}_final.pth"
    thresholds = [float(value) for value in config["validation_thresholds"]]
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_iou = -1.0
    best_threshold = 0.5
    best_negative_fp = 1.0
    best_validation_loss = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        progress = epoch / float(epochs)
        model.train()
        losses: list[float] = []
        last_components: dict[str, float] = {}
        progress_bar = tqdm(
            training_loader,
            desc=f"fold {fold} train {epoch}/{epochs}",
            mininterval=2.0,
        )
        for images, masks, centerlines, _ in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            centerlines = centerlines.to(device, non_blocking=True)
            model_input = (
                F.interpolate(
                    images, scale_factor=2, mode="bicubic", align_corners=False
                )
                if config["upsample_x2"]
                else images
            )
            mask_target = (
                F.interpolate(masks, scale_factor=2, mode="nearest")
                if config["upsample_x2"]
                else masks
            )
            centerline_target = (
                F.interpolate(centerlines, scale_factor=2, mode="nearest")
                if config["upsample_x2"]
                else centerlines
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(model_input)
                segmentation_logits, centerline_logits = _heads(output)
                loss, last_components = criterion(
                    segmentation_logits,
                    mask_target,
                    centerline_logits,
                    centerline_target,
                    progress,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            losses.append(float(loss.detach()))
            progress_bar.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")
        scheduler.step()

        hard_negative_active = bool(config["hard_negative"]) and progress >= float(
            config["hard_negative_start_fraction"]
        )
        evaluation_model = ema.ema if ema is not None else model
        if hard_negative_active and epoch >= int(config["hard_negative_start_epoch"]):
            weights = _hard_negative_weights(
                evaluation_model,
                training_items,
                image_size,
                device,
                batch_size,
                workers,
                float(config["negative_sample_weight"]),
                float(config["hard_negative_gamma"]),
                float(config["hard_negative_maximum_weight"]),
                int(config["hard_negative_score_limit"]),
            )
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(weights), replacement=True
            )
            training_loader = _make_loader(
                training_data, batch_size, workers, device, sampler=sampler
            )

        validation_loss, threshold_results = _validate(
            evaluation_model,
            validation_loader,
            criterion,
            device,
            thresholds,
            progress,
            bool(config["upsample_x2"]),
        )
        selected_threshold, selected = choose_training_threshold(
            threshold_results,
            target_negative_fp=float(config["target_negative_patch_fp"]),
            lambda_fp=float(config["checkpoint_lambda_fp"]),
        )
        score = checkpoint_score(
            selected, lambda_fp=float(config["checkpoint_lambda_fp"])
        )
        state = evaluation_model.state_dict()
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(state, best_loss_path)
        if score > best_score:
            best_score = score
            best_iou = selected["positive_patch_iou_micro"]
            best_threshold = selected_threshold
            best_negative_fp = selected["negative_patch_fp_rate"]
            patience_count = 0
            torch.save(state, best_iou_path)
        else:
            patience_count += 1

        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)) if losses else 0.0,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selected_threshold": selected_threshold,
                "selected_metrics": selected,
                "checkpoint_score": score,
                "hard_negative_active": hard_negative_active,
                "last_batch_loss_components": last_components,
            }
        )
        write_json(fold_dir / "training_log.json", history)
        if patience_count >= int(config["early_stopping_patience"]):
            break

    final_model = ema.ema if ema is not None else model
    torch.save(final_model.state_dict(), final_path)
    return {
        "fold": fold,
        "best_iou_pos": best_iou,
        "best_threshold": best_threshold,
        "best_negative_patch_fp_rate": best_negative_fp,
        "best_validation_loss": best_validation_loss,
        "best_score": best_score,
        "best_iou_path": str(best_iou_path),
        "best_iou_sha256": sha256(best_iou_path),
        "best_loss_path": str(best_loss_path),
        "final_path": str(final_path),
        "epochs_completed": len(history),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the recovered StageA MaxViT-U-Net recipe"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        help="Fold numbers to run, or 'all'; defaults to training_folds in config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_config.json", config)
    seed = int(config["seed"])
    set_seed(seed)
    total_folds = int(config["cv_folds"])
    folds = _parse_folds(args.folds, config, total_folds)
    pool, splits, strata_mapping = make_fold_splits(
        args.data_root, folds=total_folds, seed=seed
    )
    write_json(
        output_dir / "pool_summary.json",
        {
            "pool_size": len(pool),
            "strata_mapping": strata_mapping,
            "folds_requested": folds,
        },
    )
    device = torch.device(args.device)
    summaries: list[dict[str, object]] = []
    started = time.time()
    for fold in folds:
        summaries.append(
            _train_fold(
                fold=fold,
                pool=pool,
                split=splits[fold - 1],
                config=config,
                output_dir=output_dir,
                device=device,
            )
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    by_fold = {int(item["fold"]): item["best_iou_pos"] for item in summaries}
    summary = {
        "source_recipe": "recovered_stageA_semantic_cv_fp",
        "metric_implementation": "corrected one-denominator-per-negative-patch",
        "per_fold_iou": [by_fold.get(fold) for fold in range(1, total_folds + 1)],
        "folds": summaries,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_dir / f"{config['model_name']}_cv_fp_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
