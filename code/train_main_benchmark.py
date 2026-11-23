"""Train one matched RGB run using the archived training loop."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_benchmark.data import check_partition, load_records, rows_for_split, write_manifest
from models.maxvit_unet import MAXVIT_MODEL_ID, build_model as build_maxvit
from models.fcn_resnet50 import build_model as build_fcn_resnet50

MODELS = ("MAXVIT_BASE_UNET", "UNET_RESNET34", "DEEPLABV3PLUS_RESNET50", "SEGFORMER_B0", "FCN_RESNET50", "PSPNET_RESNET50")
ENCODERS = {"UNET_RESNET34": ("Unet", "resnet34"), "DEEPLABV3PLUS_RESNET50": ("DeepLabV3Plus", "resnet50"), "SEGFORMER_B0": ("Segformer", "mit_b0"), "PSPNET_RESNET50": ("PSPNet", "resnet50")}


def load_core():
    path = ROOT / "code/main_benchmark/original_training_core.py"
    spec = importlib.util.spec_from_file_location("matched_training_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_factory(name, input_channels=3, rank=8, alpha=8.0, pretrained=True):
    if input_channels != 3:
        raise ValueError("The matched benchmark uses RGB only")
    if name == "MAXVIT_BASE_UNET":
        return build_maxvit(pretrained), {"encoder": MAXVIT_MODEL_ID, "decoder": "U-Net-like (384, 192, 96, 64)", "pretraining": "ImageNet-21K then ImageNet-1K"}
    if name == "FCN_RESNET50":
        return build_fcn_resnet50(pretrained), {"implementation": "torchvision.models.segmentation.fcn_resnet50", "encoder": "resnet50", "pretraining": "ImageNet-1K"}
    import segmentation_models_pytorch as smp
    architecture, encoder = ENCODERS[name]
    model = getattr(smp, architecture)(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1, activation=None)
    return model, {"implementation": f"segmentation_models_pytorch.{architecture}", "encoder": encoder, "pretraining": "ImageNet-1K"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", choices=(42, 123, 2025), type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("runs/main_benchmark"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    records = load_records(ROOT / "metadata/records.csv")
    core = load_core()
    core.ROOT = args.data_root.resolve()
    core.GEO_ROOT = workspace
    core.MANIFESTS = workspace / "manifests"
    core.PROTOCOL = workspace / "protocol"
    core.PROTOCOL_ID = "P2_20261120_SIX_MODEL_SEED42_V1"
    core.MODELS = MODELS
    core.INPUTS = ("RENDERED3",)
    core.DEFAULT_BATCH = {name: args.batch_size for name in MODELS}
    core.build_model = model_factory
    core.initialization_files = lambda name: []
    for split in ("train", "validation"):
        path = core.MANIFESTS / f"geographic_{split}.csv"
        rows = rows_for_split(records, core.ROOT, split)
        check_partition(rows, split)
        if not path.exists():
            write_manifest(rows, path)
        elif load_records(path) != rows:
            raise ValueError(f"Existing {split} manifest differs from the released records")
    core.PROTOCOL.mkdir(parents=True, exist_ok=True)
    lock = core.PROTOCOL / "PROTOCOL_LOCK.json"
    if not lock.exists():
        lock.write_text(json.dumps({"protocol_id": core.PROTOCOL_ID, "data_manifest": "metadata/records.csv", "note": "Public rerun; initial weights resolved by upstream libraries"}, indent=2) + "\n", encoding="utf-8")
    sys.argv = [sys.argv[0], "--model", args.model, "--input", "RENDERED3", "--seed", str(args.seed), "--device", args.device, "--epochs", str(args.epochs), "--patience", str(args.patience), "--batch-size", str(args.batch_size), "--eval-batch-size", str(args.eval_batch_size), "--workers", str(args.workers), "--lr", "0.0002", "--weight-decay", "0.0001"]
    if args.smoke:
        sys.argv.append("--smoke")
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
