"""Compare the real reboot's synthetic result with a fresh uninterrupted run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from state_repair.execution.driver import verify
from state_repair.execution.durable import atomic_json, read_json
from state_repair.execution.training import Shutdown, SyntheticTrainer, load_resume, run_training
from state_repair.provenance import file_hash


def equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def check(root: Path) -> dict:
    matrix = read_json(root / "matrix.json")
    if len(matrix["jobs"]) != 1:
        raise ValueError("reboot acceptance requires one synthetic job")
    job = matrix["jobs"][0]
    if not job["synthetic"] or job["device"] != "cpu" or job["kind"] != "synthetic_training":
        raise ValueError("this checker is synthetic CPU only")
    resumed_path = root / job["id"]
    checked = verify(resumed_path, job)
    if not checked["resume_events"]:
        raise ValueError("no actual resume event to check")
    baseline = root / "uninterrupted-equivalence"
    baseline.mkdir(exist_ok=False)
    original_boot = os.environ.get("PAPER1_BOOT_ID")
    try:
        os.environ["PAPER1_BOOT_ID"] = "synthetic-uninterrupted-before"
        trainer = SyntheticTrainer(job)


        os.environ["PAPER1_BOOT_ID"] = "synthetic-uninterrupted-after"
        with Shutdown() as shutdown:
            summary = run_training(trainer, job, baseline, 120, shutdown,
                                   checkpoint_steps=job["checkpoint_steps"],
                                   checkpoint_seconds=job["checkpoint_seconds"])
        atomic_json(baseline / "summary.json", summary)
    finally:
        if original_boot is None:
            os.environ.pop("PAPER1_BOOT_ID", None)
        else:
            os.environ["PAPER1_BOOT_ID"] = original_boot
    resumed, fresh = load_resume(resumed_path / "checkpoint.pt", job), load_resume(baseline / "resume.pt", job)
    fields = ("model", "optimizer", "rng", "schedule", "remaining_batch_order", "step", "forward_calls")
    comparisons = {field: equal(resumed[field], fresh[field]) for field in fields}
    comparisons["accumulated_loss"] = resumed["trainer"]["sum_loss"] == fresh["trainer"]["sum_loss"]
    comparisons["step_log_bytes"] = (resumed_path / "steps.jsonl").read_bytes() == (baseline / "steps.jsonl").read_bytes()
    result = {"verified": all(comparisons.values()), "synthetic": True, "comparisons": comparisons,
              "resumed_checkpoint_sha256": file_hash(resumed_path / "checkpoint.pt"),
              "uninterrupted_checkpoint_sha256": file_hash(baseline / "resume.pt"),
              "uninterrupted_summary": summary,
              "note": "Different boot identities and elapsed times are provenance, not model outputs."}
    atomic_json(root / "reboot-equivalence-check.json", result)
    if not result["verified"]:
        raise ValueError("reboot result differs from uninterrupted synthetic training")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/confirmatory_restart_boot_v2"))
    args = parser.parse_args()
    print(json.dumps(check(args.output)), flush=True)
