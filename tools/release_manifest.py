from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "outputs",
}
EXCLUDED_SUFFIXES = {
    ".docx",
    ".grib",
    ".grib2",
    ".pdf",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".tmp",
    ".zip",
}
EXCLUDED_PATH_PREFIXES = {
    "checkpoints/",
    "data/private/",
    "data/raw/",
}
ROOT_MANIFEST = "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files(root: Path) -> list[Path]:
    def included(path: Path) -> bool:
        relative = path.relative_to(root)
        relative_posix = relative.as_posix()
        directory_parts = relative.parts[:-1]
        return (
            path.is_file()
            and relative_posix != ROOT_MANIFEST
            and not any(
                part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
                for part in directory_parts
            )
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
            and not any(
                relative_posix.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES
            )
        )

    return sorted(
        (path for path in root.rglob("*") if included(path)),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def build_manifest(root: Path) -> dict:
    records = []
    for path in release_files(root):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "1.0",
        "repository_version": "0.2.0",
        "hash_algorithm": "SHA-256",
        "files": records,
    }


def verify_manifest(root: Path, manifest: dict) -> list[str]:
    expected = {record["path"]: record for record in manifest.get("files", [])}
    current = {path.relative_to(root).as_posix(): path for path in release_files(root)}
    errors: list[str] = []
    for missing in sorted(set(expected) - set(current)):
        errors.append(f"missing file: {missing}")
    for unexpected in sorted(set(current) - set(expected)):
        errors.append(f"unlisted file: {unexpected}")
    for relative_path in sorted(set(expected) & set(current)):
        path = current[relative_path]
        if path.stat().st_size != int(expected[relative_path]["bytes"]):
            errors.append(f"size mismatch: {relative_path}")
        if sha256(path).upper() != str(expected[relative_path]["sha256"]).upper():
            errors.append(f"hash mismatch: {relative_path}")
    return errors


def build_archive(root: Path, destination: Path) -> int:
    manifest_path = root / ROOT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError("Build the release manifest before the archive")
    files = release_files(root) + [manifest_path]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            archive.write(
                path,
                arcname=f"{root.name}/{path.relative_to(root).as_posix()}",
            )
    return len(files)


def verify_archive(root: Path, destination: Path, manifest: dict) -> list[str]:
    prefix = f"{root.name}/"
    expected = {f"{prefix}{record['path']}": record for record in manifest["files"]}
    manifest_path = root / ROOT_MANIFEST
    expected[f"{prefix}{ROOT_MANIFEST}"] = {
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256(manifest_path),
    }
    errors: list[str] = []
    with zipfile.ZipFile(destination) as archive:
        members = {
            item.filename: item for item in archive.infolist() if not item.is_dir()
        }
        for missing in sorted(set(expected) - set(members)):
            errors.append(f"archive missing file: {missing}")
        for unexpected in sorted(set(members) - set(expected)):
            errors.append(f"archive has unlisted file: {unexpected}")
        for name in sorted(set(expected) & set(members)):
            payload = archive.read(name)
            if len(payload) != int(expected[name]["bytes"]):
                errors.append(f"archive size mismatch: {name}")
            digest = hashlib.sha256(payload).hexdigest()
            if digest.upper() != str(expected[name]["sha256"]).upper():
                errors.append(f"archive hash mismatch: {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the release manifest and archive"
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default=ROOT_MANIFEST)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--archive", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = root / destination
    if args.verify:
        manifest = json.loads(destination.read_text(encoding="utf-8"))
        errors = verify_manifest(root, manifest)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Verified {len(manifest['files'])} release files")
        if args.archive:
            archive_path = Path(args.archive).resolve()
            archive_errors = verify_archive(root, archive_path, manifest)
            if archive_errors:
                raise SystemExit("\n".join(archive_errors))
            print(f"Verified release archive: {archive_path}")
        return
    manifest = build_manifest(root)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(manifest['files'])} release files in {destination}")
    if args.archive:
        archive_path = Path(args.archive).resolve()
        archived = build_archive(root, archive_path)
        print(f"Archived {archived} files in {archive_path}")


if __name__ == "__main__":
    main()
