"""Lightweight release checks; no training, network access or full dataset."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from verify_dataset import component_path, verify


class ReleaseTests(unittest.TestCase):
    def test_fixed_manifest(self):
        result = verify(ROOT)
        self.assertEqual(result["records"], 19455)
        self.assertEqual(result["splits"]["test"]["dates"], 51)

    def test_component_path_rejects_traversal_and_wrong_split(self):
        for value in ("../train/images_8bit/a.png", "/train/images_8bit/a.png",
                      "test/images_8bit/a.png", "train/images_8bit/b.png"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                component_path(value, "train", "images_8bit", "a", ".png")

    def test_checkpoint_manifests(self):
        with (ROOT / "results/checkpoint_manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        manifest_hash = hashlib.sha256((ROOT / "metadata/records.csv").read_bytes()).hexdigest()
        for row in rows:
            self.assertEqual(row["records_manifest_sha256"], manifest_hash)
            directory = ROOT / "results" / row["representation"] / row["model"] / ("seed_" + row["seed"])
            result = json.loads((directory / "result.json").read_text())
            self.assertAlmostEqual(float(row["threshold"]), result["selected_threshold"])
            self.assertEqual(int(row["best_epoch"]), result["best_epoch"])

    def test_review_counts(self):
        for name, expected in (("all_100_decisions.csv", 100), ("focus_23_cases.csv", 23),
                               ("source_windows_100.csv", 100)):
            with (ROOT / "review" / name).open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), expected, name)

    def test_local_markdown_links(self):
        for path in ROOT.rglob("*.md"):
            if ".git" in path.parts:
                continue
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" in link or link.startswith("#"):
                    continue
                self.assertTrue((path.parent / link.split("#")[0]).exists(), (path, link))


if __name__ == "__main__":
    unittest.main()
