"""Map the released record manifest to the original benchmark row contract."""

import csv
from pathlib import Path

CLASS_MAP = {"positive": "positive", "difficult_negative": "negative"}


def rows_for_split(records, data_root, split):
    rows = []
    for record in records:
        if record["split"] != split:
            continue
        rows.append({
            "sample_id": record["record_id"],
            "sample_key": Path(record["image_8bit_path"]).name,
            "scene_key": record["scene_id"],
            "observation_date": record["acquisition_date"],
            "positive_negative": CLASS_MAP[record["sample_class"]],
            "image8_path": str(data_root / record["image_8bit_path"]),
            "label_path": str(data_root / record["mask_path"]),
        })
    return rows


def check_partition(rows, split):
    expected = {"train": (13213, 6243, 6970), "validation": (3120, 1358, 1762), "test": (3122, 1422, 1700)}[split]
    actual = (len(rows), sum(row["positive_negative"] == "positive" for row in rows), sum(row["positive_negative"] == "negative" for row in rows))
    if actual != expected or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Unexpected {split} manifest composition: {actual}")


def load_records(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
