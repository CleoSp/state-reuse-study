"""Profile synthetic end-to-end batch timing for pilot validation."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.maze import MazeExample, collate, generate_maze, possible_edges
from state_repair.models.adapters import make_adapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy
from state_repair.provenance import file_hash
from train_scaled_maze import write


def main() -> None:
    out = Path("runs/pilot_evaluation_capacity_v2")
    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    with GPUJob(out, 300, "prompt05_pilot_evaluation_capacity", synthetic=True) as job:
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        roots = [generate_maze(12, 12, i) for i in range(64)]
        edges = possible_edges(12, 12)
        examples, observations = [], []
        for frame in range(4):
            if frame:
                roots = [m.toggle(*edges[(i+frame)%len(edges)]) for i, m in enumerate(roots)]
            batch = [MazeExample(m, f"synthetic-{i}", frame, "val", True) for i, m in enumerate(roots)]
            examples.append(batch)
            observations.append(collate(batch)[0])
        model = RecursiveSolver(width=64, heads=2, inner_cycles=2).cuda().eval()
        arms = ("restart", "carry", "spatial_gate", "global_gate", "gru_adapter", "residual_adapter",
                "answer_only", "local_reset_1", "local_reset_2", "local_reset_3", "random_reset", "noisy_carry", "shuffled_gate")
        rows = []
        with torch.no_grad():
            for arm in arms:
                if arm.startswith("local_reset_"):
                    adapter = make_adapter("local_reset", radius=int(arm[-1]))
                elif arm == "random_reset":
                    adapter = make_adapter(arm, reset_rate=.5)
                elif arm == "noisy_carry":
                    adapter = make_adapter(arm, noise_scale=.01)
                elif arm == "shuffled_gate":
                    adapter = make_adapter(arm, spatial_gate=make_adapter("spatial_gate", width=64))
                else:
                    adapter = make_adapter(arm, **({} if arm in ("restart", "carry") else {"width": 64}))
                adapter = adapter.cuda().eval()
                for k in (1, 2, 4, 8):
                    for repetition in range(2):
                        policy = FixedBudgetPolicy(model, adapter, k)
                        for frame, cpu_obs in enumerate(observations):
                            job.check_limit()
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            obs = cpu_obs.to("cuda")
                            result = policy(obs)
                            predictions = result.prediction.logits.argmax(-1).cpu().tolist()
                            torch.cuda.synchronize()
                            deployment_s = time.perf_counter()-started
                            score_start = time.perf_counter()
                            scores = [score_policy(e.maze, p) for e, p in zip(examples[frame], predictions)]
                            offline_score_s = time.perf_counter()-score_start
                            rows.append({"arm": arm, "K": k, "repetition": repetition, "frame": frame,
                                         "deployment_batch_seconds": deployment_s, "offline_score_batch_seconds": offline_score_s,
                                         "batch_size": 64, "scored_roots": len(scores), "state_budget": result.state.budget,
                                         "block_calls": result.block_calls, "synthetic": True})
                            del result, obs
                    write(out / "steps.json", rows)
                    print(json.dumps({"arm": arm, "K": k, "complete": True, "synthetic": True}), flush=True)
        write(out / "profile.json", {"synthetic": True, "batch_size": 64, "dtype": "float32", "device": "cuda",
               "initial_frames_and_three_edits": True, "rows": rows, "script_sha256": file_hash(__file__)})


if __name__ == "__main__":
    main()
