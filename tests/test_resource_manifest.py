from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_frozen_resource_manifest_matches_files() -> None:
    resource_dir = Path(__file__).resolve().parents[1] / "resources" / "height_lut"
    manifest = json.loads(
        (resource_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["artifacts"]:
        path = resource_dir / record["path"]
        assert path.is_file()
        assert path.stat().st_size == int(record["bytes"])
        assert _sha256(path) == str(record["sha256"]).upper()
