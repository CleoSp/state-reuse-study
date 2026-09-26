"""Summarize saved K8/K16 frame records for the manuscript's exploratory case study.

No model evaluation or fitting occurs. Each source job is checked against its
existing seal; the output preserves source hashes and root counts by frame.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from state_repair.execution.driver import verify  # noqa: E402
from state_repair.execution.records import raw_step_hash, step_rows  # noqa: E402
from state_repair.eval.jobs import saved_records  # noqa: E402

JOBS = [
    "test-circuit32-on-circuit32-answer_only-s137-h32",
    "test-circuit32-on-circuit32-spatial_gate-s137-h32",
    "test-circuit32-on-circuit32-restart-s137-h32",
    "test-maze16-on-maze16-spatial_gate-s71-h32",
]


def main() -> None:
    start = time.perf_counter()
    summary = json.loads((ROOT / "reports/confirmatory/summary.json").read_text())
    outputs = []
    for name in JOBS:
        source = ROOT / "runs/confirmatory_v1" / name
        job = json.loads((source / "job.json").read_text())
        verify(source, job)
        counts = defaultdict(lambda: [0, 0])
        seen = set()
        for row in saved_records(source):
            if row["K"] not in (8, 16):
                continue
            if row["synthetic"]:
                raise ValueError("Synthetic diagnostic input")
            key = (row["K"], row["frame"], row["root_id"])
            if key in seen:
                raise ValueError(f"Duplicate root/frame: {key}")
            seen.add(key)
            counts[row["K"], row["frame"]][0] += int(row["exact_correct"])
            counts[row["K"], row["frame"]][1] += 1
        curves = []
        for k in (8, 16):
            frames = []
            for f in range(33):
                correct, n = counts[k, f]
                if n != 256:
                    raise ValueError(f"Missing roots for {name}, K{k}, frame{f}: {n}")
                frames.append({"frame": f, "correct": correct, "roots": n, "accuracy": correct / n})
            mean = sum(r["accuracy"] for r in frames[1:]) / 32
            saved = next(r for r in summary["curves"][job["suite"] + "|h32"]
                         if r["policy"] == job["arm"] and r["K"] == k)
            if abs(mean - saved["per_seed_accuracy"][str(job["seed"] )]) > 1e-12:
                raise ValueError("Raw frame means differ from saved report")
            curves.append({"K": k, "post_accuracy": mean, "frames": frames})
        outputs.append({"job": name, "suite": job["suite"], "policy": job["arm"], "seed": job["seed"],
                        "raw_records_sha256": raw_step_hash(source), "curves": curves})
        print(name, [(c["K"], c["frames"][0]["correct"], c["post_accuracy"]) for c in curves], flush=True)
    data = {"synthetic": False, "exploratory": True, "selection": "cases selected after inspecting the full budget curves",
            "generator": "scripts/extract_budget_diagnostic.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "elapsed_seconds": time.perf_counter() - start, "gpu_seconds": 0, "jobs": outputs}
    out = ROOT / "reports/paper/budget_diagnostic.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out.name}; {data['elapsed_seconds']:.2f}s CPU wall time")


if __name__ == "__main__":
    main()
