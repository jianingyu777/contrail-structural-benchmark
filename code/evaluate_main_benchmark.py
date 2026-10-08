"""Evaluate a completed matched run on the fixed held-out test partition."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_benchmark.data import check_partition, load_records, rows_for_split
from main_benchmark.original_statistics import original_metrics, summarize
from train_main_benchmark import ENCODERS, MODELS, load_core, model_factory


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=(42, 123, 2025), type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("runs/main_benchmark"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = args.workspace.resolve() / "standard_baselines" / args.model / "RENDERED3" / f"seed_{args.seed}"
    decision_path = run / "VALIDATION_DECISION.json"
    checkpoint_path = run / "best_checkpoint.pth"
    if not (run / "TRAIN_COMPLETE.json").exists():
        raise FileNotFoundError("Training and validation selection must finish before test evaluation")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if digest(checkpoint_path) != decision["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from frozen validation decision")
    output = run / "test_evaluation"
    if output.exists():
        raise FileExistsError(f"Test output already exists: {output}")
    rows = rows_for_split(load_records(ROOT / "metadata/records.csv"), args.data_root.resolve(), "test")
    check_partition(rows, "test")
    core = load_core()
    core.seed_everything(args.seed)
    model = model_factory(args.model, pretrained=False)[0]
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model = model.to(args.device).eval()
    dataset = core.BaselineDataset(rows, "RENDERED3", train=False)
    loader = core.make_loader(dataset, 12, False, args.workers, args.seed + 547)
    metrics = original_metrics(ROOT / "code/main_benchmark/original_metric_functions.py")
    histogram = metrics.new_probability_histogram()
    records = []
    threshold = float(decision["selected_threshold"])
    with torch.inference_mode():
        for images, targets, indices in loader:
            with torch.amp.autocast("cuda", enabled=str(args.device).startswith("cuda")):
                probabilities = torch.sigmoid(model(images.to(args.device)))[:, 0].float().cpu().numpy()
            truth_batch = targets[:, 0].numpy().astype(bool)
            metrics.update_probability_histogram(histogram, probabilities, truth_batch)
            for offset, index in enumerate(indices.tolist()):
                row = rows[index]
                target = truth_batch[offset]
                prediction = probabilities[offset] >= threshold
                skeleton = skeletonize(target) if target.any() else np.zeros_like(target)
                record = {key: row[key] for key in ("sample_id", "sample_key", "scene_key", "observation_date")}
                record.update(model=args.model, seed=args.seed, threshold=threshold, mask_area_ratio=float(target.mean()))
                record.update(metrics.binary_metrics(prediction, target))
                record.update(metrics.topology_metrics(prediction, target, skeleton))
                record.update(metrics.false_positive_components(prediction, target))
                records.append(record)
    frame = pd.DataFrame(records)
    curve, auprc = metrics.finish_probability_histogram(histogram)
    summary = {"model": args.model, "seed": args.seed, "threshold": threshold, "pixel_auprc_4096bin": auprc, **summarize(frame)}
    output.mkdir(parents=True)
    frame.to_csv(output / "per_sample_metrics.csv", index=False)
    pd.DataFrame(curve).to_csv(output / "pixel_operating_curve_4096bin.csv", index=False)
    (output / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
