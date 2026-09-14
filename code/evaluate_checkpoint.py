"""Evaluate a trusted checkpoint at its frozen validation-selected threshold."""
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import DataLoader
import train_benchmarks as experiment

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--metadata-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--normalization", type=Path, default=Path(__file__).resolve().parents[1]/"normalization/uint16_train_percentiles.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output directory already exists; choose a new directory")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = experiment.build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    frame = experiment.read_records(args.metadata_root)
    frame = frame[frame.split.eq("test")]
    representation = checkpoint["representation"]
    normalization = experiment.load_uint16_statistics(args.normalization) if representation == "uint16" else None
    data = experiment.ContrailDataset(frame, args.data_root, representation, experiment.eval_transform(256), normalization)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    args.output.mkdir(parents=True)
    metrics = experiment.evaluate_at_threshold(model, loader, device, checkpoint["selected_threshold"], args.output/"test_patch_counts.csv")
    (args.output/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
