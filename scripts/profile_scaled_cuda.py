"""Synthetic capacity/compatibility profile; never an empirical result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.maze import MazeExample, collate, generate_maze
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import maze_valid_set_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(64, 128), default=128)
    args = parser.parse_args()
    heads = args.width // 32
    output = Path(f"runs/scaled_cuda_profile_w{args.width}")
    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    with GPUJob(output, 300, "prompt05_cuda_capacity", synthetic=True) as job:
        examples = [MazeExample(generate_maze(12, 12, i), f"synthetic-{i}", 0, "train", True) for i in range(64)]
        obs, target, _ = collate(examples, "cuda")
        model = RecursiveSolver(width=args.width, heads=heads, inner_cycles=2).cuda()
        cpu = RecursiveSolver(width=args.width, heads=heads, inner_cycles=2)
        cpu.load_state_dict(model.state_dict())
        cpu_obs, cpu_target, _ = collate(examples[:1])


        expected = cpu(cpu_obs, 1).prediction.logits
        actual = model(obs, 1).prediction.logits
        torch.testing.assert_close(actual[:1].cpu(), expected, atol=3e-4, rtol=3e-4)
        maze_valid_set_loss(expected, cpu_target, cpu_obs.valid_nodes).backward()
        maze_valid_set_loss(actual[:1], cpu_target.to("cuda"), cpu_obs.valid_nodes.cuda()).backward()
        for a, b in zip(cpu.parameters(), model.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=8e-4, rtol=8e-3)
        del cpu, actual, expected
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
        rows = []
        for k in (1, 2, 4, 8, 16):
            samples = []
            for repetition in range(3):
                job.check_limit()
                torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                output_model = model(obs, k)
                loss = maze_valid_set_loss(output_model.prediction.logits, target, obs.valid_nodes)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite synthetic loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                samples.append(time.perf_counter()-started)
                del output_model, loss
            row = {"K": k, "step_seconds": samples, "gradient_norm": norm.item(),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "synthetic": True}
            rows.append(row)
            (output / "steps.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
        result = {"synthetic": True, "batch_size": 64, "width": args.width, "heads": heads,
            "inner_cycles": 2, "dtype": "float32", "device": "cuda", "cpu_cuda_K1_gradient_parity": True,
            "rows": rows, "source": source_provenance(), "script_sha256": file_hash(__file__)}
        (output / "profile.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
