"""CPU checkpoint replay and independent oracle spot checks, not a new result set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from evaluate_frozen_crossover import read_maze
from state_repair.accounting.resources import memory_snapshot
from state_repair.data.maze import observation
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import independent_distances, score_policy, solve_maze
from state_repair.provenance import file_hash


def replay(directory: Path, seed: int) -> dict:
    started = time.perf_counter()
    arm = directory / f"seed-{seed}"
    config = json.loads((arm / "config.json").read_text())
    torch.set_num_threads(2)
    checkpoint = torch.load(arm / "checkpoint.pt", weights_only=True, map_location="cpu")
    model = RecursiveSolver(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
                            attention_mode=config["attention_mode"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = json.loads((directory / "dataset.json").read_text())
    recorded = {(r["suite"], r["root_id"], r["K"]): r for r in
                (json.loads(line) for line in (arm / "validation.jsonl").read_text().splitlines())}
    results = []


    with torch.no_grad():
        for suite in ("ordinary", "rooms", "size16"):
            for raw in dataset[suite][:2]:
                maze = read_maze(raw["maze"])
                if independent_distances(maze) != solve_maze(maze)[0]:
                    raise ValueError("independent Floyd-Warshall oracle disagrees with BFS")
                obs = observation(maze, raw["root_id"], 0)
                for k in config["budgets"]:
                    prediction = model(obs, k).prediction.logits[0].argmax(-1).tolist()
                    expected = recorded[suite, raw["root_id"], k]
                    metrics = score_policy(maze, prediction)
                    results.append({"suite": suite, "root_id": raw["root_id"], "K": k,
                        "actions_identical": prediction == expected["actions"],
                        "metrics_identical": all(expected[name] == value for name, value in metrics.items()),
                        "independent_oracle_agrees": True})
    return {"purpose": "checkpoint_and_oracle_verification_only", "selection": "first two listed roots per suite",
            "synthetic": False, "seed": seed, "device": "cpu", "batch_size": 1,
            "torch": str(torch.__version__), "checkpoint_sha256": file_hash(arm / "checkpoint.pt"),
            "source_predictions_sha256": file_hash(arm / "validation.jsonl"),
            "verified": all(r["actions_identical"] and r["metrics_identical"] for r in results),
            "checks": results, "wall_s": time.perf_counter()-started, "memory": memory_snapshot()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/scaled_maze_v1"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    result = replay(args.run, args.seed)
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit("CPU replay differs from CUDA records; inspect before accepting equivalence")
