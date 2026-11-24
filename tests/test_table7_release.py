"""Validate the Table 7 60-image release materials without model dependencies."""

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "six_model_seed42"
MASKS = ROOT / "data" / "table7_manual_masks"


class Table7ReleaseTest(unittest.TestCase):
    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def test_roi_inventory(self):
        rows = self.read_csv(RESULTS / "table7_manual_rois.csv")
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(int(row["evaluation_positive"]) for row in rows), 47)
        self.assertEqual(len({row["annotation_id"] for row in rows}), 60)
        for row in rows:
            self.assertEqual(len(row["mask_sha256"]), 64)
            self.assertEqual(row["central_offset_row"], "128")
            self.assertEqual(row["central_offset_col"], "128")

    def test_masks_present(self):
        for folder, suffix in (("masks", "_mask.png"), ("valid", "_valid.png"), ("ignore", "_ignore.png")):
            files = list((MASKS / folder).glob(f"*{suffix}"))
            self.assertEqual(len(files), 60, folder)

    def test_checkpoint_manifest(self):
        rows = self.read_csv(RESULTS / "table7_checkpoint_manifest.csv")
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertEqual(row["source_archive"], "contrail_model_benchmark_checkpoints_20260831.zip")
        archive = json.loads((RESULTS / "table7_checkpoint_archive.json").read_text())
        self.assertEqual(len(archive["sha256"]), 64)
        self.assertGreater(archive["bytes"], 0)

    def test_table7_values(self):
        rows = {row["model"]: row for row in self.read_csv(RESULTS / "table7_manual_masks.csv")}
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(float(rows["MAXVIT_BASE_UNET"]["dice"]), 0.750, places=3)
        self.assertAlmostEqual(float(rows["MAXVIT_BASE_UNET"]["iou"]), 0.600, places=3)
        self.assertAlmostEqual(float(rows["MAXVIT_BASE_UNET"]["auprc"]), 0.816, places=3)
        self.assertEqual(rows["MAXVIT_BASE_UNET"]["negative_total"], "13")

    def test_all60_metrics(self):
        rows = self.read_csv(RESULTS / "table7_all60_metrics.csv")
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["negative_total"], "13")


if __name__ == "__main__":
    unittest.main()
