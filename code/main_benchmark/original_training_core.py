#!/usr/bin/env python3
"""Train fair standard segmentation baselines before geographic-test unsealing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GEO_ROOT = Path(os.environ.get("TGRS_GEO_ROOT", str(ROOT / "outputs" / "p1_tgrs_geographic_holdout_v1"))).resolve()
MANIFESTS = GEO_ROOT / "manifests"
PROTOCOL = GEO_ROOT / "protocol"
PROTOCOL_ID = os.environ.get("TGRS_PROTOCOL_ID", "TGRS_GEOGRAPHIC_HOLDOUT_V1_20260819")
MODELS = ("DEEPLABV3PLUS", "SEGFORMER", "DINOV2_LORA")
INPUTS = ("RENDERED3", "BT3", "PHYSICAL7", "EARLY_FUSION10")
SEEDS = (42, 123, 2025)
DEFAULT_BATCH = {"DEEPLABV3PLUS": 24, "SEGFORMER": 16, "DINOV2_LORA": 12}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_rows(role: str) -> list[dict[str, Any]]:
    if role not in {"train", "validation"}:
        raise ValueError("Standard-baseline training is prohibited from opening the sealed test manifest")
    return (
        pd.read_csv(MANIFESTS / f"geographic_{role}.csv", encoding="utf-8-sig", low_memory=False)
        .fillna("")
        .to_dict("records")
    )


def stratified_limit(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    positive = [row for row in rows if str(row.get("positive_negative")) == "positive"]
    negative = [row for row in rows if str(row.get("positive_negative")) == "negative"]
    rng = np.random.default_rng(seed)
    n_positive = min(len(positive), int(round(limit * len(positive) / max(len(rows), 1))))
    n_negative = min(len(negative), limit - n_positive)
    if n_positive + n_negative < limit:
        n_positive = min(len(positive), n_positive + limit - n_positive - n_negative)
    positive_index = rng.choice(len(positive), n_positive, replace=False) if n_positive else []
    negative_index = rng.choice(len(negative), n_negative, replace=False) if n_negative else []
    selected = [positive[int(index)] for index in sorted(positive_index)]
    selected.extend(negative[int(index)] for index in sorted(negative_index))
    return sorted(selected, key=lambda row: (str(row.get("scene_key")), str(row.get("sample_key"))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


class BaselineDataset:
    def __init__(self, rows: list[dict[str, Any]], input_kind: str, train: bool):
        self.rows = rows
        self.input_kind = input_kind
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        rendered = None
        physical = None
        if self.input_kind in {"RENDERED3", "EARLY_FUSION10"}:
            rendered = np.asarray(Image.open(str(row["image8_path"])).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.input_kind in {"BT3", "PHYSICAL7", "EARLY_FUSION10"}:
            with np.load(resolve(str(row["bt_stack_path"])), allow_pickle=False) as archive:
                physical = np.asarray(archive["stack"], dtype=np.float32)[:7]
            physical = np.nan_to_num(physical, nan=0.0, posinf=1.0, neginf=0.0)
            physical = np.clip(physical, 0.0, 1.0)
        if self.input_kind == "RENDERED3":
            array = rendered
        elif self.input_kind == "BT3":
            array = physical[:3]
        elif self.input_kind == "PHYSICAL7":
            array = physical
        else:
            array = np.concatenate((rendered, physical), axis=0)
        if array.shape[1:] != (256, 256):
            raise ValueError(f"Unexpected input shape {array.shape}: {row.get('sample_key')}")
        mask = np.asarray(Image.open(str(row["label_path"])).convert("L"), dtype=np.uint8) > 0
        x = torch.from_numpy(array.astype(np.float32))
        y = torch.from_numpy(mask.astype(np.float32)[None])
        if self.train:
            if random.random() < 0.5:
                x, y = torch.flip(x, dims=(2,)), torch.flip(y, dims=(2,))
            if random.random() < 0.5:
                x, y = torch.flip(x, dims=(1,)), torch.flip(y, dims=(1,))
        return x, y, index


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int, drop_last: bool = False):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)

    def worker_init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=worker_init,
    )


def initialization_files(model_name: str) -> list[Path]:
    import torch

    checkpoint_root = Path(torch.hub.get_dir()) / "checkpoints"
    if model_name == "DEEPLABV3PLUS":
        paths = [checkpoint_root / "resnet34-333f7ec4.pth"]
    elif model_name == "SEGFORMER":
        paths = [checkpoint_root / "mit_b2.pth"]
    else:
        paths = [checkpoint_root / "dinov2_vits14_pretrain.pth"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Frozen initialization is not cached locally: {path}")
    return paths


def build_model(model_name: str, input_channels: int, rank: int, alpha: float):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if model_name in {"DEEPLABV3PLUS", "SEGFORMER"}:
        import segmentation_models_pytorch as smp

        if model_name == "DEEPLABV3PLUS":
            model = smp.DeepLabV3Plus(
                encoder_name="resnet34", encoder_weights="imagenet", in_channels=input_channels, classes=1, activation=None
            )
        else:
            model = smp.Segformer(
                encoder_name="mit_b2", encoder_weights="imagenet", in_channels=input_channels, classes=1, activation=None
            )
        return model, {"lora_rank": None, "lora_alpha": None, "encoder_internal_size": 256}
    module = load_module("tgrs_geo_dinov2_builder", SCRIPTS / "112_train_official_dinov2_and_smp_unet.py")
    model = module.build_official_dinov2_seg(input_channels, True, rank, alpha, 252)
    return model, {"lora_rank": rank, "lora_alpha": alpha, "encoder_internal_size": 252}


def soft_dice_loss(logits, targets, eps: float = 1e-6):
    import torch

    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def validation_loss(model, loader, device) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)
            total += float(loss.item()) * len(inputs)
            count += len(inputs)
    return total / max(count, 1)


def validation_threshold(model, loader, rows: list[dict[str, Any]], device) -> tuple[float, list[dict[str, Any]]]:
    import torch

    thresholds = [round(value, 2) for value in np.arange(0.01, 0.951, 0.01)]
    threshold_tensor = torch.tensor(thresholds, device=device).view(1, -1, 1, 1)
    positive_dice = np.zeros(len(thresholds), dtype=np.float64)
    positive_count = np.zeros(len(thresholds), dtype=np.int64)
    negative_any = np.zeros(len(thresholds), dtype=np.int64)
    negative_count = 0
    model.eval()
    with torch.inference_mode():
        for inputs, targets, indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            truth = targets.to(device, non_blocking=True).bool()[:, 0, None]
            with torch.amp.autocast("cuda", enabled=True):
                probability = torch.sigmoid(model(inputs))[:, 0]
            predicted = probability[:, None] >= threshold_tensor
            tp = (predicted & truth).sum(dim=(2, 3)).cpu().numpy().astype(np.float64)
            fp = (predicted & ~truth).sum(dim=(2, 3)).cpu().numpy().astype(np.float64)
            fn = (~predicted & truth).sum(dim=(2, 3)).cpu().numpy().astype(np.float64)
            dice = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
            any_prediction = predicted.flatten(start_dim=2).any(dim=2).cpu().numpy()
            for batch_index, row_index in enumerate(indices.numpy().tolist()):
                if str(rows[int(row_index)].get("positive_negative")) == "positive":
                    positive_dice += dice[batch_index]
                    positive_count += 1
                else:
                    negative_any += any_prediction[batch_index]
                    negative_count += 1
    positive_macro = positive_dice / np.maximum(positive_count, 1)
    negative_rate = negative_any / max(negative_count, 1)
    sweep = []
    for index, threshold in enumerate(thresholds):
        sweep.append(
            {
                "threshold": threshold,
                "validation_positive_patch_macro_dice": float(positive_macro[index]),
                "validation_negative_patch_any_prediction_rate": float(negative_rate[index]),
                "constraint_neg_any_le_0_05": bool(negative_rate[index] <= 0.05),
                "positive_validation_patches": int(positive_count[index]),
                "negative_validation_patches": int(negative_count),
            }
        )
    eligible = [row for row in sweep if row["constraint_neg_any_le_0_05"]]
    candidates = eligible if eligible else sweep
    selected = max(
        candidates,
        key=lambda row: (
            row["validation_positive_patch_macro_dice"],
            -row["validation_negative_patch_any_prediction_rate"],
            row["threshold"],
        ),
    )
    for row in sweep:
        row["selected"] = bool(row is selected)
        row["selection_fallback_no_threshold_met_fpr_constraint"] = not bool(eligible)
    return float(selected["threshold"]), sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input", choices=INPUTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    import torch.nn.functional as F

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
        raise ValueError(f"Invalid CUDA device: {args.device}")
    input_channels = {"RENDERED3": 3, "BT3": 3, "PHYSICAL7": 7, "EARLY_FUSION10": 10}[args.input]
    batch_size = args.batch_size or DEFAULT_BATCH[args.model]
    eval_batch_size = args.eval_batch_size or batch_size
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        batch_size = min(batch_size, 2)
        eval_batch_size = min(eval_batch_size, 2)
        args.max_train = args.max_train or 32
        args.max_validation = args.max_validation or 24
        args.workers = min(args.workers, 2)
    run_root = GEO_ROOT / ("smoke" if args.smoke else "standard_baselines") / args.model / args.input / f"seed_{args.seed}"
    complete = run_root / "TRAIN_COMPLETE.json"
    if complete.exists():
        raise FileExistsError(f"Completed run will not be overwritten: {complete}")
    run_root.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    train_rows = stratified_limit(read_rows("train"), args.max_train, args.seed + 1)
    validation_rows = stratified_limit(read_rows("validation"), args.max_validation, args.seed + 2)
    train_dataset = BaselineDataset(train_rows, args.input, train=True)
    validation_dataset = BaselineDataset(validation_rows, args.input, train=False)
    validation_loader = make_loader(validation_dataset, eval_batch_size, False, args.workers, args.seed + 13)
    init_paths = initialization_files(args.model)
    model, model_meta = build_model(args.model, input_channels, args.lora_rank, args.lora_alpha)
    model = model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    config = {
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": sha256(PROTOCOL / "PROTOCOL_LOCK.json"),
        "model": args.model,
        "input": args.input,
        "input_channels": input_channels,
        "input_size": 256,
        "seed": args.seed,
        "device": str(device),
        "initialization_files": [
            {"filename": path.name, "sha256": sha256(path)} for path in init_paths
        ],
        "model_meta": model_meta,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "loss": "BCEWithLogits + soft Dice",
        "augmentation": "identical horizontal and vertical flips for every input setting",
        "threshold": "validation positive-patch macro Dice subject to negative-patch any-prediction rate <= 0.05",
        "test_accessed": False,
        "smoke": bool(args.smoke),
    }
    (run_root / "RUN_CONFIG.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    start_epoch = 1
    best_loss = math.inf
    best_epoch = 0
    waiting = 0
    log_rows: list[dict[str, Any]] = []
    last_path = run_root / "last_checkpoint.pth"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        waiting = int(state["waiting"])
        if (run_root / "training_log.csv").exists():
            log_rows = pd.read_csv(run_root / "training_log.csv", encoding="utf-8-sig").to_dict("records")
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_loader(train_dataset, batch_size, True, args.workers, args.seed + 1000 + epoch, drop_last=True)
        model.train()
        total = 0.0
        seen = 0
        for inputs, targets, _ in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.item()) * len(inputs)
            seen += len(inputs)
        validation_value = validation_loss(model, validation_loader, device)
        scheduler.step(validation_value)
        if validation_value < best_loss - 1e-6:
            best_loss = validation_value
            best_epoch = epoch
            waiting = 0
            torch.save(
                {"model_state": model.state_dict(), "run_config": config, "epoch": epoch, "validation_loss": validation_value},
                run_root / "best_checkpoint.pth",
            )
        else:
            waiting += 1
        record = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "validation_loss": validation_value,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "waiting": waiting,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_sec_this_invocation": round(time.time() - started, 2),
        }
        log_rows.append(record)
        write_csv(run_root / "training_log.csv", log_rows)
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "waiting": waiting,
                "run_config": config,
            },
            last_path,
        )
        print(
            f"{args.model}/{args.input} seed={args.seed} epoch={epoch}/{args.epochs} "
            f"train={record['train_loss']:.5f} val={validation_value:.5f} best={best_loss:.5f}",
            flush=True,
        )
        if waiting >= args.patience:
            break

    best = torch.load(run_root / "best_checkpoint.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    threshold, sweep = validation_threshold(model, validation_loader, validation_rows, device)
    write_csv(run_root / "validation_threshold_sweep.csv", sweep)
    selected = next(row for row in sweep if row["selected"])
    decision = {
        "protocol_id": config["protocol_id"],
        "model": args.model,
        "input": args.input,
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "selected_threshold": threshold,
        "selected_validation": selected,
        "checkpoint_sha256": sha256(run_root / "best_checkpoint.pth"),
        "validation_sweep_sha256": sha256(run_root / "validation_threshold_sweep.csv"),
        "test_accessed": False,
    }
    (run_root / "VALIDATION_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (run_root / "TRAIN_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model": args.model,
                "input": args.input,
                "seed": args.seed,
                "validation_decision_sha256": sha256(run_root / "VALIDATION_DECISION.json"),
                "test_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(run_root / "TRAIN_COMPLETE.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
