"""Conditional synthetic circuit capacity test, before any circuit learning."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from train_scaled_maze import write
from train_static_circuit import require_maze_gate
from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.circuit import CircuitExample, collate, generate_circuit
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import circuit_value_loss
from state_repair.types import Domain


def main() -> None:
    require_maze_gate()
    config = json.loads(Path("configs/static_circuit_v1.json").read_text())
    out = Path(config["capacity_run"])
    torch.set_num_threads(config["threads"])
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    with GPUJob(out, 300, "prompt05_circuit_capacity", synthetic=True) as job:
        write(out / "config.json", {**config, "synthetic": True})
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        examples = [CircuitExample(generate_circuit(config["nodes"], i, config["inputs"]),
                    f"synthetic-{i}", 0, "train", tuple(range(config["nodes"])), True) for i in range(config["batch_size"])]
        obs, target, _ = collate(examples, "cuda")
        cpu_obs, cpu_target, _ = collate(examples[:1])
        kwargs = dict(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"], domain=Domain.CIRCUIT)
        model, cpu = RecursiveSolver(**kwargs).cuda(), RecursiveSolver(**kwargs)
        cpu.load_state_dict(model.state_dict())
        expected, actual = cpu(cpu_obs, 1).prediction.logits, model(obs, 1).prediction.logits
        torch.testing.assert_close(actual[:1].cpu(), expected, atol=3e-4, rtol=3e-4)
        circuit_value_loss(expected, cpu_target).backward()
        circuit_value_loss(actual[:1], cpu_target.to("cuda")).backward()
        for a, b in zip(cpu.parameters(), model.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), atol=8e-4, rtol=8e-3)
        del cpu, expected, actual
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
        rows = []
        for k in config["budgets"]:
            samples = []
            for _ in range(3):
                job.check_limit()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = model(obs, k)
                loss = circuit_value_loss(result.prediction.logits, target)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                samples.append(time.perf_counter()-started)
                del result, loss
            row = {"K": k, "step_seconds": samples, "gradient_norm": norm.item(), "synthetic": True,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            rows.append(row)
            write(out / "steps.json", rows)
            print(json.dumps(row), flush=True)
        write(out / "profile.json", {**{name: config[name] for name in ("batch_size", "model_width", "heads", "inner_cycles")},
            "synthetic": True, "cpu_cuda_K1_gradient_parity": True, "dtype": "float32", "device": "cuda", "rows": rows,
            "source": source_provenance(), "script_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
