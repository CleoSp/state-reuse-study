"""Synthetic CPU/CUDA solver parity and synchronized timing; no empirical fit.

Predeclared allocation estimate: below 3 GiB including CUDA runtime, batch 8,
width 64, K <= 16. Uses the existing CUDA venv; no package/driver changes.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from state_repair.accounting.resources import memory_snapshot
from state_repair.data.maze import MazeExample, collate, generate_maze
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import source_provenance
from state_repair.train.losses import maze_valid_set_loss


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no pass recorded")
    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    examples = [MazeExample(generate_maze(8, 8, i), f"synthetic-{i}", 0, "train", True) for i in range(8)]
    obs, target, _ = collate(examples)
    cpu = RecursiveSolver(attention_mode="dense")
    gpu = RecursiveSolver(attention_mode="dense").to("cuda")
    gpu.load_state_dict(cpu.state_dict())
    cuda_obs, cuda_target = obs.to("cuda"), target.to("cuda")
    rows = []
    for k in (1, 16):
        cpu.zero_grad(set_to_none=True)
        gpu.zero_grad(set_to_none=True)
        expected = cpu(obs, k).prediction.logits
        actual = gpu(cuda_obs, k).prediction.logits
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
        cpu_loss = maze_valid_set_loss(expected, target, obs.valid_nodes)
        gpu_loss = maze_valid_set_loss(actual, cuda_target, cuda_obs.valid_nodes)
        cpu_loss.backward()
        gpu_loss.backward()
        max_gradient_error = 0.
        for p, q in zip(cpu.parameters(), gpu.parameters()):
            torch.testing.assert_close(q.grad.cpu(), p.grad, atol=5e-4, rtol=5e-3)
            max_gradient_error = max(max_gradient_error, (q.grad.cpu() - p.grad).abs().max().item())
        timings = {}
        for name, model, x, y in (("cpu", cpu, obs, target), ("cuda", gpu, cuda_obs, cuda_target)):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
            samples = []
            for _ in range(3):
                if name == "cuda": torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                result = model(x, k)
                loss = maze_valid_set_loss(result.prediction.logits, y, x.valid_nodes)
                loss.backward()
                optimizer.step()
                if name == "cuda": torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            timings[name] = samples
        rows.append({"K": k, "synthetic": True, "max_gradient_error": max_gradient_error,
                     "step_seconds": timings, "loss": cpu_loss.item()})
    output = {"synthetic": True, "purpose": "compatibility_and_training_step_profile", "device": torch.cuda.get_device_name(),
        "torch": torch.__version__, "source": source_provenance(), "batch_size": 8, "width": 64,
        "rows": rows, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "host_memory": memory_snapshot(),
        "limits": "Precollated device-resident synthetic batches; includes solver validation synchronizations; three samples, no learning evidence."}
    path = Path("runs/static_generalization_v1/cuda-profile.json")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
