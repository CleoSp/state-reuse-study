"""Measure validation-selected operating points on fixed stream-training weights."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from time import monotonic

import torch

import run_adapter_pilot as maze
import run_circuit_stream as circuit
from state_repair.execution.budget import Budget
from state_repair.execution.durable import DriverLock, atomic_json, append_jsonl
from state_repair.eval.timing import measured_frame
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.provenance import file_hash


def main():
    root = Path("runs/confirmatory_v1")
    with DriverLock(root / "driver.lock"), torch.no_grad():
        budget = Budget(root)
        budget.reconcile()
        for family in ("maze", "circuit"):
            name = "development-cost-" + family
            out = root / name
            if (out / "profile.json").exists():
                continue
            out.mkdir()
            job = {"id": name, "device": "cuda", "synthetic": False, "seconds": 900,
                "estimated_peak_bytes": 2*2**30, "purpose": "validation-only cost selection, not confirmatory test"}
            receipt = budget.begin(job, out)
            started = monotonic()
            try:
                torch.set_num_threads(2)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.cuda.set_per_process_memory_fraction(10*2**30/torch.cuda.get_device_properties(0).total_memory)
                torch.cuda.reset_peak_memory_stats()
                source = Path("runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2")
                config = maze.read(source / "prepared/config.json")
                payload = maze.read(source / "prepared/dataset.json")
                selected = maze.read(source / "selection.json")

                payload["streams"] = [f[:4] for f in payload["streams"]]
                tiny = {**config, "batch_size": 1}
                if family == "maze":
                    by_frame = maze.frame_batches(payload, 32, tiny)
                    data = [[frame[i] for frame in by_frame] for i in range(len(by_frame[0]))]
                else:
                    data = circuit.batches(payload["streams"], tiny, 32)
                rows = []
                for arm in ("restart", "answer_only"):
                    for seed in config["seeds"]:
                        if family == "maze":
                            model, adapter, identity = maze.load_selected(source, config, selected, {}, "joint", arm, seed, "cuda")
                        else:
                            model, adapter, identity = circuit.load_selected(source, config, arm, seed, "cuda")
                        model.eval(); adapter.eval()
                        for k in (1, 2, 4, 8):
                            for repetition in range(2):
                                for stream in data:
                                    policy = FixedBudgetPolicy(model, adapter, k)
                                    policy(stream[0][1].to("cuda"))
                                    policy.reset()
                                    for examples, obs, *_ in stream:
                                        if monotonic()-started > 885:
                                            raise TimeoutError("bounded development cost cap")
                                        result, actions, times = measured_frame(policy, obs, "cuda")
                                        row = {"family": family, "policy": arm, "seed": seed, "K": k,
                                            "repetition": repetition, "root_id": examples[0].root_id,
                                            "frame": examples[0].frame_index, "milliseconds": times,
                                            "synthetic": False, "split": "val", "checkpoint_sha256": identity["checkpoint_sha256"],
                                            "state_budget": result.state.budget, "batch_size": 1}
                                        rows.append(row)
                                        append_jsonl(out / "steps.jsonl", row)
                                    policy.reset()
                            print(json.dumps({"development_cost": family, "policy": arm, "seed": seed, "K": k}), flush=True)
                        del model, adapter, result, policy
                        gc.collect()
                        torch.cuda.empty_cache()
                atomic_json(out / "profile.json", {"synthetic": False, "validation_only": True, "rows": rows,
                    "source_selection_sha256": file_hash(source / "selection.json"), "dtype": "float32",
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                budget.finish(receipt, monotonic()-started, aborted=False)
            except BaseException as exc:
                budget.finish(receipt, monotonic()-started, aborted=True, error=str(exc))
                raise


if __name__ == "__main__":
    main()
