from __future__ import annotations

from pathlib import Path


def packaged_config_path(filename: str) -> str:
    """Return an installed, package-relative configuration path."""
    path = Path(__file__).resolve().parent / "configs" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Packaged configuration is missing: {path}")
    return str(path)
