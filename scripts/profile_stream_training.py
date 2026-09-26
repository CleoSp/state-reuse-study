"""Synthetic five-frame capacity check; no empirical results or saved fitted model."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import time

import torch

from run_adapter_pilot import adapter_for, gpu_job, new_model, read, seal, stream_training_batches, write
from state_repair.data.maze import MazeExample, generate_maze
from state_repair.train.pilot import PRINCIPAL, schedule, streams
from state_repair.train.stream_step import stream_backward


def main() -> None:
    config = read("configs/adapter_stream_v2.json")
    out = Path(config["capacity_run"])
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(73)
    roots = [MazeExample(generate_maze(12, 12, i), f"synthetic-{i}", 0, "train", True) for i in range(64)]
    frames = streams(roots, 4, config["edit_seed"])
    payload = {"train": [asdict(e) for e in frames[0]], "views": [[asdict(e) for e in f] for f in frames[1:]]}
    batch = stream_training_batches(payload, config)[0]
    from state_repair.models.recursive import RecursiveSolver
    weights = RecursiveSolver(width=64, heads=2, inner_cycles=2).state_dict()
    rows = []
    with gpu_job(config, out, 300, "prompt05b_stream_capacity", synthetic=True) as job:
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        for arm in PRINCIPAL:
            model = new_model(config, weights, "joint", "cpu")
            adapter = adapter_for(arm, config)
            if arm == "spatial_gate":
                reference, ref_adapter = copy.deepcopy(model), copy.deepcopy(adapter)
                cpu_records = stream_backward(reference, ref_adapter, [o for o,t in batch], [t for o,t in batch], 1, track="joint")
            model, adapter = model.cuda(), adapter.cuda()
            gpu_batch = [(o.cuda() if hasattr(o, "cuda") else o.to("cuda"), t.to("cuda")) for o,t in batch]
            params = [*model.parameters(), *adapter.parameters()]
            optimizer = torch.optim.AdamW(params, lr=0., weight_decay=0.)
            for k in config["budgets"]:
                samples = []
                for repeat in range(2):
                    job.check_limit()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    records = stream_backward(model, adapter, [o for o,t in gpu_batch], [t for o,t in gpu_batch], k, track="joint")
                    if arm == "spatial_gate" and k == 1 and repeat == 0:
                        torch.testing.assert_close(torch.tensor([r.loss for r in records]), torch.tensor([r.loss for r in cpu_records]), atol=2e-5, rtol=2e-4)
                        for a, b in zip([*model.parameters(), *adapter.parameters()], [*reference.parameters(), *ref_adapter.parameters()]):
                            if a.grad is not None:
                                torch.testing.assert_close(a.grad.cpu(), b.grad, atol=3e-5, rtol=5e-3)
                    grad = sum(float(p.grad.square().sum()) for p in adapter.parameters() if p.grad is not None)**.5
                    if list(adapter.parameters()) and not grad > 0:
                        raise ValueError("missing adapter gradient")
                    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                    optimizer.step()
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter()-started)
                rows.append({"arm": arm, "K": k, "step_seconds": samples, "synthetic": True,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                write(out / "steps.json", rows)
            del optimizer, params, model, adapter, gpu_batch
            torch.cuda.empty_cache()
        measured = {(r["arm"], r["K"]): max(r["step_seconds"]) for r in rows}
        training = sum(measured[arm, k] for seed in config["seeds"] for arm in PRINCIPAL for grid in range(2)
                       for _, k in schedule(seed, 16, config))


        components = {"maze_training_s": training, "maze_tuning_s": 900,
            "maze_streams_s": 3600, "maze_interventions_s": 600,
            "circuit_replication_s": 7200, "diagnostics_and_setup_s": 2400}
        projected = sum(components.values())*1.25
        write(out / "profile.json", {"synthetic": True, "rows": rows, "cpu_cuda_stream_gradient_parity": True,
            "components": components, "projected_seconds_with_25_percent_contingency": projected,
            "fits_12_hours": projected+300 < 12*3600, "batch_size": 64, "frames_per_step": 5})
    seal(out)


if __name__ == "__main__":
    main()
