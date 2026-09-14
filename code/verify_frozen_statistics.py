"""Reproduce published metrics and intervals from unchanged per-patch counts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from statistics_core import calculate_metrics, date_cluster_bootstrap

def main():
    root = Path(__file__).resolve().parents[1] / "results"
    expected = pd.read_csv(root / "summary/all_run_metrics_with_date_bootstrap.csv")
    stratified = pd.read_csv(root / "summary/stratified_metrics_with_date_bootstrap.csv")
    comparisons = 0
    for path in sorted(root.glob("*/*/seed_*/result.json")):
        info = json.loads(path.read_text())
        frame = pd.read_csv(path.parent / "test_patch_counts.csv")
        selections = {"all": np.ones(len(frame), dtype=bool),
            "positive_le_1pct": frame.positive_reference.eq(1) & frame.foreground_fraction.le(.01),
            "positive_1_to_5pct": frame.positive_reference.eq(1) & frame.foreground_fraction.gt(.01) & frame.foreground_fraction.le(.05),
            "positive_gt_5pct": frame.positive_reference.eq(1) & frame.foreground_fraction.gt(.05),
            "difficult_negative": frame.difficult_negative.eq(1)}
        for key, selection in selections.items():
            sub = frame.loc[selection]
            table = expected if key == "all" else stratified[stratified.stratum.eq(key)]
            reference = table[(table.model == info["display_name"]) & (table.representation == info["representation"]) & (table.seed == info["seed"])].iloc[0]
            seed = info["seed"] if key == "all" else info["seed"] + 1000
            values = calculate_metrics(sub)
            intervals = date_cluster_bootstrap(sub, seed, 2000)
            for metric, value in values.items():
                assert np.isclose(value, reference[metric], equal_nan=True), (path, key, metric)
                if metric in intervals:
                    for output, suffix in [("lower_95", "ci95_low"), ("upper_95", "ci95_high")]:
                        assert np.isclose(intervals[metric][output], reference[f"{metric}_{suffix}"], atol=1e-10), (path, key, metric, output)
                comparisons += 1
        print(f"Verified {info['model']} / {info['representation']} / {info['seed']}", flush=True)
    print(f"PASS: {comparisons} metric comparisons and associated 2,000-resample intervals")

if __name__ == "__main__":
    main()
