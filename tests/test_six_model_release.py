"""Validate the current six-model result snapshot without model dependencies."""

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "six_model_seed42"


class SixModelReleaseTest(unittest.TestCase):
    def test_table6_rows_and_ranking(self):
        with (RESULTS / "table6_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        scores = {row["model"]: float(row["mean_positive_patch_dice"]) for row in rows}
        thresholds = {row["model"]: float(row["threshold"]) for row in rows}
        self.assertEqual(max(scores, key=scores.get), "MAXVIT_BASE_UNET")
        self.assertAlmostEqual(scores["MAXVIT_BASE_UNET"], 0.8202, places=4)
        self.assertAlmostEqual(thresholds["MAXVIT_BASE_UNET"], 0.34, places=2)
        self.assertAlmostEqual(thresholds["FCN_RESNET50"], 0.53, places=2)

    def test_table7_and_hashes(self):
        with (RESULTS / "table7_manual_masks.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        with (RESULTS / "checkpoint_manifest.csv").open(newline="", encoding="utf-8") as handle:
            checkpoints = list(csv.DictReader(handle))
        self.assertEqual(len(checkpoints), 6)
        for row in checkpoints:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertEqual(row["seed"], "42")
        with (RESULTS / "Table6_full_per_patch_audit_all_models_seed42.csv").open(newline="", encoding="utf-8") as handle:
            audit = list(csv.DictReader(handle))
        self.assertEqual(len(audit), 6 * 3122)
        self.assertTrue((RESULTS / "Figure7_Version3_scene_date_disjoint_predictions.png").exists() or (RESULTS / "figure7" / "Figure7_Version3_scene_date_disjoint_predictions.png").exists())


if __name__ == "__main__":
    unittest.main()
