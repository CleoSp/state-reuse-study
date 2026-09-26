"""Regenerate a planning estimate from synthetic measured capacity profiles."""
from __future__ import annotations

import json
import argparse
import math
from pathlib import Path
import statistics

from state_repair.accounting.gpu_job import JOURNAL, remaining_seconds
from state_repair.provenance import file_hash


def project(version: str = "v2") -> dict:
    if version not in ("v1", "v2"):
        raise ValueError("unknown capacity profile revision")
    training_path = Path(f"runs/adapter_pilot_capacity_{version}/profile.json")
    evaluation_path = Path(f"runs/pilot_evaluation_capacity_{version}/profile.json")
    training = json.loads(training_path.read_text())["rows"]
    evaluation = json.loads(evaluation_path.read_text())["rows"]
    budgets = (1, 2, 4, 8)
    learned = ("spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only")
    principal = ("restart", "carry", *learned)
    controls = ("local_reset_1", "local_reset_2", "local_reset_3", "random_reset", "noisy_carry", "shuffled_gate")
    arms = (*principal, *controls)
    expected_training = {(t, a, k) for t in ("frozen", "joint") for a in principal for k in budgets}
    actual_training = [(r["track"], r["arm"], r["K"]) for r in training]
    expected_evaluation = {(a, k, rep, frame) for a in arms for k in budgets for rep in range(2) for frame in range(4)}
    actual_evaluation = [(r["arm"], r["K"], r["repetition"], r["frame"]) for r in evaluation]
    if (set(actual_training) != expected_training or len(actual_training) != len(expected_training)
            or set(actual_evaluation) != expected_evaluation or len(actual_evaluation) != len(expected_evaluation)):
        raise ValueError("incomplete/duplicate capacity profile coverage")
    if not all(r["synthetic"] is True and len(r["step_seconds"]) == 2
               and all(math.isfinite(v) and v > 0 for v in r["step_seconds"]) for r in training):
        raise ValueError("invalid synthetic training timings")
    if not all(r["synthetic"] is True and r["batch_size"] == 64 and r["state_budget"] == r["K"]
               and all(math.isfinite(r[n]) and r[n] > 0 for n in ("deployment_batch_seconds", "offline_score_batch_seconds")) for r in evaluation):
        raise ValueError("invalid synthetic evaluation timings")
    nseeds, ngrid, steps, batches = 3, 2, 256, 4
    training_rows = []
    for track, names in (("frozen", learned), ("joint", principal)):
        for arm in names:
            seconds = statistics.mean(statistics.median(r["step_seconds"]) for r in training if r["track"] == track and r["arm"] == arm)
            training_rows.append({"track": track, "arm": arm, "step_seconds": seconds,
                                  "runs": nseeds*ngrid, "projected_seconds": seconds*steps*nseeds*ngrid})
    costs = {}
    for arm in arms:
        for k in budgets:
            selected = [r for r in evaluation if r["arm"] == arm and r["K"] == k]
            costs[arm, k] = {"initial": statistics.median(r["deployment_batch_seconds"]+r["offline_score_batch_seconds"] for r in selected if r["frame"] == 0),
                             "edit": statistics.median(r["deployment_batch_seconds"]+r["offline_score_batch_seconds"] for r in selected if r["frame"] > 0)}
    tuning_s = sum(batches*nseeds*ngrid*sum(costs[a,k]["initial"]+4*costs[a,k]["edit"] for k in budgets)
                   for names in (learned, principal) for a in names)
    tuning_s += batches*nseeds*sum(costs[a,k]["initial"]+4*costs[a,k]["edit"] for a in ("restart", "carry") for k in budgets)
    streams_s = batches*nseeds*2*sum(costs[a,k]["initial"]+32*costs[a,k]["edit"] for a in arms for k in budgets)
    joint_interventions = ("restart", "carry", "spatial_gate", *controls)


    interventions_s = 8*nseeds*sum(costs[a,k]["edit"] for names in (arms, joint_interventions) for a in names for k in budgets)
    training_s = sum(r["projected_seconds"] for r in training_rows)
    setup_allowance_s = 600.
    subtotal = training_s+tuning_s+streams_s+interventions_s+setup_allowance_s
    total = subtotal*1.25
    events = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    return {"projection": True, "capacity_revision": version, "empirical_method_results": False, "timing_sources_synthetic": True,
            "source_hashes": {str(p): file_hash(p) for p in (training_path, evaluation_path)},
            "planning_inputs": {"optimizer_steps": steps, "seeds": nseeds, "grid_points": ngrid,
                                "training_runs": sum(r["runs"] for r in training_rows), "validation_roots": 256,
                                "stream_edits": 32, "short_stream_from_prefix": 4, "budgets": budgets},
            "training_rows": training_rows, "training_seconds": training_s,
            "tuning_four_edit_seconds": tuning_s, "selected_32_edit_seconds": streams_s,
            "frozen_intervention_seconds": interventions_s, "setup_allowance_seconds": setup_allowance_s,
            "subtotal_seconds": subtotal, "contingency_fraction": .25, "total_projected_seconds": total,
            "total_projected_hours": total/3600, "remaining_authorized_seconds": remaining_seconds(events),
            "projected_extra_seconds": max(0, total-remaining_seconds(events)),
            "peak_training_allocated_bytes_from_all_trials": max(r["peak_allocated_bytes"] for r in training),
            "peak_training_reserved_bytes_from_all_trials": max(r["peak_reserved_bytes"] for r in training)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    result = project(args.version)
    path = Path(f"runs/prompt05-pilot-cost-projection-{args.version}.json")
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "training_rows"}, indent=2))
