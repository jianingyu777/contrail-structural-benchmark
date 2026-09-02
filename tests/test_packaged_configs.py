from __future__ import annotations

import json
from pathlib import Path

from contrail_benchmark.config import packaged_config_path


def test_packaged_defaults_match_repository_copies() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename in ("segmentation_paper.json", "physical_audit_paper.json"):
        packaged = json.loads(
            Path(packaged_config_path(filename)).read_text(encoding="utf-8")
        )
        repository = json.loads(
            (root / "configs" / filename).read_text(encoding="utf-8")
        )
        assert packaged == repository
