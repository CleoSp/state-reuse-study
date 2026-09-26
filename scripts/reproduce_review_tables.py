"""Recompute the frozen primary table from the bounded review supplement.

Only NumPy is required. The inputs are explicitly derived episode aggregates,
not raw node actions. This does not reproduce the full raw-action audit.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from state_repair.eval.statistics import paired_contrast, paired_cost_ratio  # noqa: E402


def main() -> None:
    with gzip.open(ROOT / "review_data/primary_episodes.json.gz", "rt") as f:
        data = json.load(f)
    if data["synthetic"]:
        raise ValueError("Synthetic primary inputs")
    summary = json.loads((ROOT / "reports/confirmatory/summary.json").read_text())
    primary_suites = {"maze12-on-maze12", "circuit32-on-circuit32", "circuit32-on-circuit48"}
    for expected in summary["decisions_batch64"].values():
        suite = expected["suite"]
        if suite not in primary_suites:
            continue
        rows = [r for r in data["episodes"] if r["suite"] == suite]
        left = [r for r in rows if r["policy"] == "answer_only"]
        right = [r for r in rows if r["policy"] == "restart"]
        acc = paired_contrast(left, right)
        cost = paired_cost_ratio(left, right)
        checks = {"accuracy_delta": acc["delta"], "accuracy_root_ci": acc["root_ci"],
                  "accuracy_crossed_ci": acc["crossed_ci"], "cost_ratio": cost["ratio"],
                  "cost_root_ci": cost["root_ci"], "cost_crossed_ci": cost["crossed_ci"]}
        import numpy as np
        for key, value in checks.items():
            if not np.allclose(value, expected[key], rtol=0, atol=1e-12):
                raise ValueError(f"Primary mismatch: {suite} {key}")
        print(f"PASS {suite}: delta={100*acc['delta']:+.6f} pp, cost ratio={cost['ratio']:.8f}; intervals match to 1e-12")


if __name__ == "__main__":
    main()
