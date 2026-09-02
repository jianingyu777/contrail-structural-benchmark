from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_recovered_training_source_matches_manifest_except_sanitized_args() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "legacy" / "stageA_semantic_training"
    manifest = json.loads((source / "source_manifest.json").read_text("utf-8"))
    for record in manifest["unmodified_source_files"]:
        if record["path"] == "args.py":
            continue
        path = source / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_historical_source_uses_production_checkpoint_naming() -> None:
    root = Path(__file__).resolve().parents[1]
    trainer = (
        root / "legacy" / "stageA_semantic_training" / "cv_trainer_fp.py"
    ).read_text("utf-8")
    assert 'f"{args.model_name}_fp_fold{fold}_best_iou.pth"' in trainer
    assert 'f"{args.model_name}_cv_fp_summary.json"' in trainer
