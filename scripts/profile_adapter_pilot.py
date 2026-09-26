"""Synthetic timing/memory/gradient checks for the proposed one-edit pilot."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.maze import MazeExample, collate, generate_maze, possible_edges
from state_repair.models.adapters import make_adapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash
from state_repair.train.adapter_step import FrozenPrior, one_edit_loss
from state_repair.types import PredictionBatch
from train_scaled_maze import write


def main() -> None:
    out = Path("runs/adapter_pilot_capacity_v2")
    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    with GPUJob(out, 300, "prompt05_adapter_capacity", synthetic=True) as job:
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        mazes = [generate_maze(12, 12, i) for i in range(64)]
        old_cpu, old_target_cpu, _ = collate([MazeExample(m, f"synthetic-{i}", 0, "train", True) for i, m in enumerate(mazes)])
        edges = possible_edges(12, 12)
        new_cpu, new_target_cpu, _ = collate([MazeExample(m.toggle(*edges[i]), f"synthetic-{i}", 1, "train", True) for i, m in enumerate(mazes)])
        old, new = old_cpu.to("cuda"), new_cpu.to("cuda")
        old_target, new_target = old_target_cpu.to("cuda"), new_target_cpu.to("cuda")
        reference = RecursiveSolver(width=64, heads=2, inner_cycles=2).cuda().requires_grad_(False)
        initial_weights = {n: p.detach().cpu().clone() for n, p in reference.state_dict().items()}
        priors = {}
        with torch.no_grad():
            for k in (1, 2, 4, 8):
                initial = reference(old, k)
                priors[k] = FrozenPrior(old, initial.state.detach(), PredictionBatch(initial.prediction.logits.detach().clone()),
                                         "synthetic-checkpoint", "restart", k)
        del reference, initial
        rows = []
        for track in ("frozen", "joint"):
            for arm in ("restart", "carry", "spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only"):
                model = RecursiveSolver(width=64, heads=2, inner_cycles=2).cuda().requires_grad_(track == "joint")
                model.load_state_dict(initial_weights)
                kwargs = {} if arm in ("restart", "carry") else {"width": 64}
                adapter = make_adapter(arm, **kwargs).cuda()
                parameters = [p for p in [*model.parameters(), *adapter.parameters()] if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=0., weight_decay=0.) if parameters else None
                for k in (1, 2, 4, 8):
                    samples, gradients = [], []
                    for _ in range(2):
                        job.check_limit()
                        if optimizer is not None:
                            optimizer.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        started = time.perf_counter()


                        current_obs, current_target = new_cpu.to("cuda"), new_target_cpu.to("cuda")
                        result = one_edit_loss(model, adapter, old, current_obs, old_target, current_target, k, track=track,
                                               cache=priors[k] if track == "frozen" else None,
                                               checkpoint_sha256="synthetic-checkpoint" if track == "frozen" else None)
                        if optimizer is not None:
                            result.loss.backward()
                            torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                            gradient = sum(float(p.grad.square().sum()) for p in adapter.parameters() if p.grad is not None)**.5
                            if any(p.requires_grad for p in adapter.parameters()) and not gradient > 0:
                                raise RuntimeError("adapter gradient missing during capacity check")
                            gradients.append(gradient)
                            optimizer.step()
                        torch.cuda.synchronize()
                        samples.append(time.perf_counter()-started)
                        del result, current_obs, current_target
                    row = {"track": track, "arm": arm, "K": k, "step_seconds": samples,
                           "adapter_gradient_norms_after_clip": gradients,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "synthetic": True}
                    rows.append(row)
                    write(out / "steps.json", rows)
                    print(json.dumps(row), flush=True)
                del optimizer, parameters, adapter, model
        write(out / "profile.json", {"synthetic": True, "batch_size": 64, "width": 64, "heads": 2,
               "inner_cycles": 2, "dtype": "float32", "rows": rows, "script_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
