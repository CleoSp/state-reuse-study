"""Synchronized local capacity timing on synthetic observed streams."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from time import monotonic, perf_counter

import torch

from state_repair.execution.budget import Budget
from state_repair.execution.datasets import generate_roots, identities, collate, frames
from state_repair.execution.durable import DriverLock, atomic_json, append_jsonl
from state_repair.execution.jobs import environment, model_for, adapter_for
from state_repair.eval.metrics import score
from state_repair.eval.timing import measured_frame
from state_repair.models.policy import FixedBudgetPolicy


def main():
    root = Path("runs/confirmatory_v1")
    with DriverLock(root / "driver.lock"), torch.no_grad():
        budget = Budget(root)
        budget.reconcile()
        for family, size in (("maze", 12), ("maze", 16), ("maze", 20), ("circuit", 32), ("circuit", 48), ("circuit", 64), ("circuit", 96)):
            name = f"capacity-evaluation-{family}{size}"
            out = root / name
            if (out / "profile.json").exists():
                continue
            out.mkdir()
            job = {"id": name, "synthetic": True, "device": "cuda", "family": family, "size": size,
                "width": 64 if family == "maze" else 128, "heads": 2 if family == "maze" else 4,
                "inner_cycles": 2, "context_width": 16, "seed": 73, "seconds": 900, "estimated_peak_bytes": 4*2**30}
            receipt = budget.begin(job, out)
            start = monotonic()
            try:
                environment(job)
                spec = {"family": family, "size": size, "inputs": 8 if size in (32, 48) else 16,
                    "depth_shift": size in (48, 96), "data_seed": 73017, "identities": {"val": identities(name, "val", 64)}}
                roots = generate_roots(spec, "val", synthetic=True)
                stream = frames(roots, 2, 54019)
                solver = model_for(job).eval()
                rows = []
                for b in (1, 64):
                    observations = [collate(f[:b], None if i == 0 else stream[i-1][:b])[0] for i, f in enumerate(stream)]
                    for arm in ("restart", "spatial_gate"):
                        adapter = adapter_for(arm, job).to("cuda").eval()
                        for k in (1, 2, 4, 8, 16):
                            for rep in range(3):
                                policy = FixedBudgetPolicy(solver, adapter, k)
                                for f, obs in enumerate(observations):
                                    if monotonic()-start >= 880:
                                        raise TimeoutError("capacity job cap reached")
                                    result, actions, times = measured_frame(policy, obs, "cuda")
                                    began = perf_counter()
                                    for e, action in zip(stream[f][:b], actions):
                                        score(e, action)
                                    row = {"synthetic": True, "family": family, "size": size, "batch_size": b,
                                        "arm": arm, "K": k, "repetition": rep, "frame": f, "milliseconds": times,
                                        "offline_score_seconds": perf_counter()-began, "state_budget": result.state.budget}
                                    rows.append(row)
                                    append_jsonl(out / "steps.jsonl", row)
                                policy.reset()
                            print(json.dumps({"class": name, "batch_size": b, "arm": arm, "K": k}), flush=True)
                        del adapter, policy, result
                atomic_json(out / "profile.json", {"synthetic": True, "rows": rows, "config": job,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "timing_scope": "synchronized stages plus separately recorded offline CPU scoring"})
                budget.finish(receipt, monotonic()-start, aborted=False)
                del solver
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException as exc:
                budget.finish(receipt, monotonic()-start, aborted=True, error=str(exc))
                raise


if __name__ == "__main__":
    main()
