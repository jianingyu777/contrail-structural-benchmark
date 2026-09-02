from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import packaged_config_path
from .data import (
    ContrailPatchDataset,
    evaluation_transform,
    make_fold_split,
    scan_pairs,
)
from .metrics import choose_reporting_threshold
from .model import ContrailMaxViTUNet, load_checkpoint
from .runtime import evaluate_loader, load_json, set_seed, sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-threshold and held-out test evaluation"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--config", default=packaged_config_path("segmentation_paper.json")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    set_seed(int(config["seed"]))
    _, validation_items, _ = make_fold_split(
        args.data_root,
        fold=int(config["fold"]),
        folds=int(config["cv_folds"]),
        seed=int(config["seed"]),
    )
    test_items = scan_pairs(Path(args.data_root) / "test")
    transform = evaluation_transform(int(config["image_size"]))
    validation_loader = DataLoader(
        ContrailPatchDataset(validation_items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        ContrailPatchDataset(test_items, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device(args.device)
    model = ContrailMaxViTUNet(
        encoder_name=config["encoder_name"],
        pretrained=False,
        decoder_channels=tuple(config["decoder_channels"]),
        auxiliary_centerline=False,
    ).to(device)
    load_message = load_checkpoint(model, args.checkpoint, device=device, strict=True)
    validation_results = evaluate_loader(
        model,
        validation_loader,
        device,
        [float(value) for value in config["reporting_thresholds"]],
    )
    threshold, selected_validation = choose_reporting_threshold(validation_results)
    test_metrics = evaluate_loader(model, test_loader, device, [threshold])[threshold]
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "selected_threshold": threshold,
        "threshold_rule": "maximum validation Dice; negative-patch FP rate breaks ties",
        "validation_metrics": selected_validation,
        "test_metrics": test_metrics,
        "missing_keys": list(load_message.missing_keys),
        "unexpected_keys": list(load_message.unexpected_keys),
    }
    write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
