"""Validate the fixed release manifest without training or changing any files."""
from pathlib import Path, PurePosixPath
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "train": (13213, 6243, 6970, 580, 203),
    "validation": (3120, 1358, 1762, 119, 51),
    "test": (3122, 1422, 1700, 120, 51),
}
COMPONENTS = {
    "image_8bit_path": ("images_8bit", ".png"),
    "image_16bit_path": ("images_16bit", ".tif"),
    "mask_path": ("masks", ".png"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def component_path(value, split, folder, record_id, suffix):
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in value,
            f"Invalid relative component path: {value}")
    require(path.parts == (split, folder, record_id + suffix),
            f"Component path does not match its record: {value}")
    return path


def verify(metadata_root=ROOT, data_root=None, check_files=False):
    metadata_root = Path(metadata_root)
    path = metadata_root / "metadata/records.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 19455, "Expected the 19,455-record release")
    require(len({r["record_id"] for r in rows}) == len(rows), "Duplicate record IDs")
    require({r["split"] for r in rows} == set(EXPECTED), "Unexpected split labels")
    for field in ("scene_id", "acquisition_date"):
        groups = defaultdict(set)
        for row in rows:
            groups[row[field]].add(row["split"])
        require(all(len(splits) == 1 for splits in groups.values()), f"Split overlap: {field}")
    counts = {}
    for split, expected in EXPECTED.items():
        part = [r for r in rows if r["split"] == split]
        classes = Counter(r["sample_class"] for r in part)
        actual = (len(part), classes["positive"], classes["difficult_negative"],
                  len({r["scene_id"] for r in part}), len({r["acquisition_date"] for r in part}))
        require(actual == expected, f"Unexpected split composition for {split}: {actual}")
        counts[split] = dict(zip(("records", "positive", "difficult_negative", "scenes", "dates"), actual))
    require(len({r["source_product_id"] for r in rows}) == 808, "Unexpected source-product count")
    require(len({r["image_8bit_pixel_sha256"] for r in rows}) == len(rows), "Duplicate decoded RGB content")
    if check_files:
        require(data_root is not None, "--check-files requires --data-root")
        data_root = Path(data_root)
        available = {}
        # One directory scan avoids thousands of round trips on network disks.
        for split in EXPECTED:
            for folder, _ in COMPONENTS.values():
                with os.scandir(data_root / split / folder) as entries:
                    available[split, folder] = {entry.name for entry in entries if entry.is_file()}
    files_checked = 0
    for row in rows:
        foreground = int(row["foreground_pixels"])
        require(0 <= foreground <= 256 * 256, "Invalid foreground area")
        require((foreground > 0) == (row["sample_class"] == "positive"), "Class/foreground mismatch")
        require(abs(float(row["foreground_fraction"]) - foreground / (256 * 256)) < 1e-12,
                "Foreground fraction mismatch")
        for field, (folder, suffix) in COMPONENTS.items():
            relative = component_path(row[field], row["split"], folder, row["record_id"], suffix)
            if check_files:
                require(relative.name in available[row["split"], folder], f"Missing component: {relative}")
                files_checked += 1
    normalization = json.loads((metadata_root / "normalization/uint16_train_percentiles.json").read_text())
    require(normalization["estimation_split"] == "train", "Normalization is not train-only")
    require(normalization["train_records"] == 13213, "Unexpected normalization sample count")
    require(normalization["records_manifest_sha256"] == digest, "Normalization/manifest hash mismatch")
    for low, middle, high in zip(normalization["p01"], normalization["median"], normalization["p99"]):
        require(low <= middle < high, "Invalid normalization quantiles")
    return {"status": "PASS", "records": len(rows), "splits": counts,
            "records_manifest_sha256": digest, "component_files_checked": files_checked,
            "image_pixels_rechecked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.metadata_root, args.data_root, args.check_files), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Dataset check failed: {error}\n")


if __name__ == "__main__":
    main()
