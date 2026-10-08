"""Validate the archived matched-benchmark release without model dependencies."""

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MainBenchmarkReleaseTest(unittest.TestCase):
    def test_twelve_frozen_decisions_match_metrics(self):
        with (ROOT / "results/main_benchmark/metrics/per_seed.csv").open(newline="", encoding="utf-8-sig") as handle:
            results = list(csv.DictReader(handle))
        self.assertEqual(len(results), 12)
        for row in results:
            run = ROOT / "results/main_benchmark/run_configs" / row["model"] / f"seed_{row['seed']}"
            config = json.loads((run / "RUN_CONFIG.json").read_text(encoding="utf-8"))
            decision = json.loads((run / "VALIDATION_DECISION.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model"], row["model"])
            self.assertEqual(config["seed"], int(row["seed"]))
            self.assertAlmostEqual(decision["selected_threshold"], float(row["threshold"]))
            self.assertTrue(decision["selected_validation"]["selection_fallback_no_threshold_met_fpr_constraint"])
            self.assertEqual(config["train_rows"], 13213)
            self.assertEqual(config["validation_rows"], 3120)

    def test_three_seed_means(self):
        with (ROOT / "results/main_benchmark/metrics/per_seed.csv").open(newline="", encoding="utf-8-sig") as handle:
            per_seed = list(csv.DictReader(handle))
        with (ROOT / "results/main_benchmark/summaries/three_seed_mean_sd.csv").open(newline="", encoding="utf-8-sig") as handle:
            summaries = list(csv.DictReader(handle))
        self.assertEqual(len(summaries), 4)
        for row in summaries:
            scores = [float(run["positive_patch_macro_dice"]) for run in per_seed if run["model"] == row["model"]]
            self.assertEqual(len(scores), 3)
            self.assertAlmostEqual(sum(scores) / 3, float(row["positive_patch_macro_dice_mean"]))


if __name__ == "__main__":
    unittest.main()
