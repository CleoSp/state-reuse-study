"""Verify local raw generalization artifacts and print metrics; no report files."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from state_repair.data.maze import Maze, MazeExample
from state_repair.data.splits import canonical_hash, reject_cross_split_duplicates
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.provenance import file_hash


def check(directory: Path) -> dict:
    dataset_path = directory / "dataset.json"
    data = json.loads(dataset_path.read_text())
    examples = {}
    for split, records in data.items():
        assert split in ("train", "val")
        for raw in records:
            raw["maze"]["edges"] = tuple(tuple(edge) for edge in raw["maze"]["edges"])
            example = MazeExample(**{**raw, "maze": Maze(**raw["maze"])})
            assert example.split == split and example.synthetic is False and example.frame_index == 0
            assert example.root_id not in examples
            examples[example.root_id] = example
    reject_cross_split_duplicates(examples.values())
    arms = []
    for path in sorted(directory.glob("*/summary.json")):
        arm = path.parent
        summary = json.loads(path.read_text())
        for name, digest in summary["artifact_hashes"].items():
            assert file_hash(arm / name) == digest, name
        config = json.loads((arm / "config.json").read_text())
        provenance = json.loads((arm / "provenance.json").read_text())
        for name, digest in provenance["source_files"].items():
            if name == "pyproject.toml":
                continue
            actual = hashlib.sha256((arm / "source" / name).read_text(encoding="utf-8").encode()).hexdigest()
            assert actual == digest, name
        assert file_hash(arm / "source/scripts/train_generalization.py") == provenance["script_sha256"]


        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(arm / "checkpoint.pt", map_location="cpu", weights_only=True)
        assert checkpoint["dataset_sha256"] == file_hash(dataset_path)
        assert checkpoint["config"] == config and checkpoint["provenance"] == provenance
        steps = [json.loads(line) for line in (arm / "steps.jsonl").read_text().splitlines()]
        assert len(steps) == checkpoint["steps"] == summary["optimizer_steps"]
        assert [r["step"] for r in steps] == list(range(1, len(steps) + 1))
        assert sum(r["block_calls"] for r in steps) == summary["training_batched_F_calls"]
        counts = Counter(r["K"] for r in steps)
        assert dict(counts) == {int(k): n for k, n in summary["sampled_K_counts"].items()}
        rows = [json.loads(line) for line in (arm / "validation.jsonl").read_text().splitlines()]
        seen = set()
        for row in rows:
            identity = (row["root_id"], row["K"])
            assert identity not in seen
            seen.add(identity)
            example = examples[row["root_id"]]
            assert example.split == row["split"] == "val" and row["synthetic"] is False
            assert row["input_sha256"] == canonical_hash(example.maze)
            assert row["checkpoint_sha256"] == summary["artifact_hashes"]["checkpoint.pt"]
            assert row["config_sha256"] == summary["artifact_hashes"]["config.json"]
            for key, value in score_policy(example.maze, row["actions"]).items():
                assert row[key] == value
        expected = {(e.root_id, k) for e in examples.values() if e.split == "val" for k in config["budgets"]}
        assert seen == expected
        metrics = []
        for original in summary["validation"]:
            selected = [r for r in rows if r["K"] == original["K"]]
            assert sum(r["route_correct"] for r in selected) == original["routes"]
            assert sum(r["valid_action_accuracy"] for r in selected) / len(selected) == original["action_accuracy"]
            reachable = [r for r in selected if solve_maze(examples[r["root_id"]].maze)[0][examples[r["root_id"]].maze.start] > 0]
            unreachable = [r for r in selected if solve_maze(examples[r["root_id"]].maze)[0][examples[r["root_id"]].maze.start] < 0]
            metrics.append({**original, "reachable_nontrivial_roots": len(reachable),
                "reachable_nontrivial_routes": sum(r["route_correct"] for r in reachable),
                "unreachable_roots": len(unreachable), "unreachable_correct": sum(r["route_correct"] for r in unreachable),
                "reasons": dict(Counter(r["reason"] for r in selected))})
        counts_by_k = [m["routes"] for m in metrics]
        assert [m["K"] for m in metrics] == config["budgets"]
        nondecreasing = all(a <= b for a, b in zip(counts_by_k, counts_by_k[1:]))
        improvement = counts_by_k[-1] > counts_by_k[0]
        competent = metrics[-1]["route_accuracy"] >= config["minimum_validation_route_accuracy"]
        assert summary["gate"] == {"nondecreasing": nondecreasing,
            "strict_endpoint_improvement": improvement, "competent": competent,
            "minimum_route_accuracy": config["minimum_validation_route_accuracy"],
            "prompt04_allowed": nondecreasing and improvement and competent}
        arms.append({"arm": arm.name, "validation": metrics, "gate": summary["gate"],
                     "steps": summary["optimizer_steps"], "wall_s": summary["wall_s"], "memory": summary["memory"]})
    if not arms:
        raise ValueError("no completed generalization arms to verify")
    return {"verified": True, "train_roots": len(data["train"]), "val_roots": len(data["val"]),
            "dataset_sha256": file_hash(dataset_path), "arms": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/static_generalization_v1"))
    parser.add_argument("--frozen", action="store_true", help="verify frozen-state artifacts")
    parser.add_argument("--scaled", action="store_true", help="verify scaled-maze artifacts")
    parser.add_argument("--circuit", action="store_true", help="verify static-circuit artifacts")
    parser.add_argument("--pilot", action="store_true", help="verify the complete adapter pilot")
    parser.add_argument("--replay", action="store_true", help="replay all frozen-state inference on CPU")
    args = parser.parse_args()
    if sum((args.frozen, args.scaled, args.circuit, args.pilot)) > 1:
        parser.error("choose one run type")
    if args.pilot:
        if args.replay:
            parser.error("--replay currently supports frozen runs only")
        from check_adapter_pilot import check_pilot
        result = check_pilot(args.run)
    elif args.circuit:
        if args.replay:
            parser.error("--replay currently supports frozen runs only")
        from check_static_circuit import check_circuit
        result = check_circuit(args.run)
    elif args.scaled:
        if args.replay:
            parser.error("--replay currently supports frozen runs only")
        from check_scaled_maze import check_scaled
        result = check_scaled(args.run)
    elif args.frozen:
        from check_frozen_crossover import check_frozen
        result = check_frozen(args.run, replay=args.replay)
    else:
        if args.replay:
            parser.error("--replay requires --frozen")
        result = check(args.run)
    print(json.dumps(result, indent=2))
