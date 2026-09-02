from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("segmentation_models_pytorch")

from contrail_benchmark.segmentation.model import ContrailMaxViTUNet, load_checkpoint  # noqa: E402


CHECKPOINT = os.environ.get("CONTRAIL_CHECKPOINT")
EXPECTED_SHA256 = "4FEC1A57D650CA3D10C0B4343BDF2A1295A5895A112E2FED589B9A2ACB9CBD51"
EXPECTED_PARAMETERS = 83_948_118


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@pytest.mark.skipif(
    not CHECKPOINT or not Path(CHECKPOINT).exists(),
    reason="set CONTRAIL_CHECKPOINT to run the 337-MB production-checkpoint test",
)
def test_production_checkpoint_matches_release_architecture() -> None:
    checkpoint_path = Path(CHECKPOINT)
    assert _sha256(checkpoint_path) == EXPECTED_SHA256
    model = ContrailMaxViTUNet(pretrained=False, auxiliary_centerline=True)
    message = load_checkpoint(model, checkpoint_path, device="cpu", strict=True)
    assert message.missing_keys == []
    assert message.unexpected_keys == []
    assert model.centerline_head is not None
    assert (
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_PARAMETERS
    )
