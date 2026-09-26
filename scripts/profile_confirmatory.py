"""Profile synthetic capacity and record shared resource accounting."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from state_repair.execution.budget import Budget
from state_repair.execution.datasets import identities, prepare_development
from state_repair.execution.durable import DriverLock, append_jsonl, atomic_json
from state_repair.execution.jobs import build
from state_repair.provenance import file_hash


def main():
    root = Path("runs/confirmatory_v1")
    root.mkdir(exist_ok=True)
    with DriverLock(root / "driver.lock"):
        budget = Budget(root)
        budget.reconcile()
        for family, size, recipe in (("maze", 12, "static"), ("maze", 16, "static"),
                ("maze", 16, "stream"), ("maze", 16, "one_edit"),
                ("circuit", 32, "static"), ("circuit", 64, "static"), ("circuit", 64, "stream"),
                ("maze", 12, "stream"), ("maze", 12, "one_edit"), ("circuit", 32, "stream")):
            name = f"capacity-{family}{size}-{recipe}"
            out = root / name
            if out.exists():
                if (out / "profile.json").exists():
                    continue
                raise ValueError("preserve failed capacity directory and version retry explicitly")
            out.mkdir()
            b = 128 if family == "circuit" and recipe == "static" else 64
            spec = {"family": family, "size": size, "inputs": 8 if size == 32 else 16,
                "data_seed": 73017, "identities": {s: identities(name, s, b if s == "train" else 2) for s in ("train", "val")}}
            path = out / "synthetic-data.json"
            prepare_development(spec, path, synthetic=True)
            job = {"id": name, "kind": "research_training", "synthetic": True, "device": "cuda",
                "family": family, "size": size, "recipe": recipe, "arm": "spatial_gate", "width": 64 if family == "maze" else 128,
                "heads": 2 if family == "maze" else 4, "inner_cycles": 2, "context_width": 16,
                "batch_size": b, "microbatch_size": 16 if size == 16 and recipe == "static" else 32 if size == 16 and recipe == "one_edit" else b,
                "dataset": str(path), "seed": 73, "edit_seed": 54017, "steps": 1, "budgets": [1],
                "learning_rate": .001, "adapter_learning_rate": .001, "final_learning_rate": .001,
                "weight_decay": 0., "gradient_clip": 1., "seconds": 900, "estimated_peak_bytes": int(9.8*2**30)}
            atomic_json(out / "job.json", job)
            receipt = budget.begin(job, out)
            started = time.monotonic()
            try:
                trainer = build(job, root)
                setup_seconds = time.monotonic()-started
                rows = []
                budgets = [1, 2, 4, 8, 16] if recipe == "static" else [1, 2, 4, 8] if family == "maze" else [1, 2]
                for k in budgets:
                    samples = []
                    for repetition in range(3):
                        if time.monotonic()-started >= 880:
                            raise TimeoutError("bounded capacity cap")
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        row = trainer.step(0, 0, k)
                        torch.cuda.synchronize()
                        samples.append(time.perf_counter()-begin)
                        append_jsonl(out / "steps.jsonl", {"synthetic": True, "K": k, "repetition": repetition,
                            "step_seconds": samples[-1], **row})
                    rows.append({"K": k, "step_seconds": samples,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                    print(json.dumps({"class": name, **rows[-1]}), flush=True)
                atomic_json(out / "profile.json", {"synthetic": True, "rows": rows, "config": job,
                    "setup_seconds": setup_seconds,
                    "script_sha256": file_hash(__file__), "dtype": "float32", "device": torch.cuda.get_device_name(0)})
                budget.finish(receipt, time.monotonic()-started, aborted=False)
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            except BaseException as exc:
                budget.finish(receipt, time.monotonic()-started, aborted=True, error=str(exc))
                raise


if __name__ == "__main__":
    main()
