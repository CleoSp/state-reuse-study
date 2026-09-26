"""Validate saved maze runs by rebuilding data and scoring every prediction."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import sqlite3

import torch

from check_frozen_crossover import require
from train_scaled_maze import gate, make_data, metrics
from state_repair.accounting.gpu_job import JOURNAL, remaining_seconds
from state_repair.data.splits import canonical_hash
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.provenance import file_hash


def check_scaled(directory: Path) -> dict:
    paths = sorted(directory.glob("seed-*/summary.json"))
    require(bool(paths), "no completed scaled-maze runs")
    dataset = json.loads((directory / "dataset.json").read_text())
    outputs, shared = [], None
    for path in paths:
        arm = path.parent
        summary = json.loads(path.read_text())
        require(summary["synthetic"] is False, "synthetic run excluded from empirical mode")
        names = {"config.json", "checkpoint.pt", "provenance.json", "steps.jsonl", "validation.jsonl", "preregistered-status.md"}
        require(set(summary["artifact_hashes"]) == names, "incomplete artifact manifest")
        for name, digest in summary["artifact_hashes"].items():
            require(file_hash(arm / name) == digest, f"artifact hash mismatch: {name}")
        config = json.loads((arm / "config.json").read_text())
        seed = config["seed"]
        require(seed in config["seeds"] and seed == summary["seed"] and arm.name == f"seed-{seed}", "seed mismatch")
        common = {k: v for k, v in config.items() if k != "seed"}
        if shared is None:
            shared = common
            data = make_data(common)
            from dataclasses import asdict
            require(json.loads(json.dumps({s: [asdict(e) for e in es] for s, es in data.items()})) == dataset,
                    "dataset reconstruction mismatch")
        require(common == shared, "configuration changed between seeds")
        require(file_hash(directory / "dataset.json") == summary["dataset_sha256"], "dataset hash mismatch")
        provenance = json.loads((arm / "provenance.json").read_text())
        for name, digest in provenance["snapshot_hashes"].items():
            require(file_hash(arm / "source" / name) == digest, f"source snapshot mismatch: {name}")
        require(file_hash(arm / "source/scripts/train_scaled_maze.py") == provenance["script_sha256"], "runner snapshot mismatch")
        checkpoint = torch.load(arm / "checkpoint.pt", weights_only=True, map_location="cpu")
        require(checkpoint["config"] == config and checkpoint["provenance"] == provenance, "checkpoint config/source mismatch")
        require(checkpoint["dataset_sha256"] == summary["dataset_sha256"], "checkpoint dataset mismatch")
        model = RecursiveSolver(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
                                attention_mode=config["attention_mode"])
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite checkpoint")
        require(sum(p.numel() for p in model.parameters()) == summary["parameter_count"], "parameter count mismatch")
        steps = [json.loads(line) for line in (arm / "steps.jsonl").read_text().splitlines()]
        require(len(steps) == checkpoint["steps"] == summary["optimizer_steps"], "optimizer step mismatch")
        require(bool(steps), "no optimization records")
        require(all(int(state["step"].item()) == len(steps) for state in checkpoint["optimizer"]["state"].values()),
                "optimizer state step mismatch")
        require(summary["final_loss"] == steps[-1]["loss"], "final loss mismatch")
        rng, order = random.Random(seed), []
        nbatches = config["train_roots"]//config["batch_size"]
        factor = config["inner_cycles"]+1
        for i, row in enumerate(steps, 1):
            if not order:
                order = list(range(nbatches))
                rng.shuffle(order)
            index, k = order.pop(), rng.choice(config["budgets"])
            require(row["step"] == i and row["batch_index"] == index and row["K"] == k, "seeded batch/K replay mismatch")
            require(row["seed"] == seed and row["synthetic"] is False and row["batch_size"] == config["batch_size"], "step lineage mismatch")
            require(row["block_calls"] == k*factor, "step recurrence accounting mismatch")
            require(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0,
                    "invalid training loss/gradient")
            require(row["lr"] == (config["learning_rate"] if i <= config["steps"]*.75 else config["final_learning_rate"]),
                    "LR schedule mismatch")
        require(rng.getstate() == checkpoint["python_rng"] and order == checkpoint["batch_order"], "resume RNG/order mismatch")
        require(dict(Counter(r["K"] for r in steps)) == {int(k): v for k, v in summary["sampled_K_counts"].items()}, "K coverage mismatch")
        require(dict(Counter(r["batch_index"] for r in steps)) == {int(k): v for k, v in summary["batch_counts"].items()}, "batch coverage mismatch")
        require(set(r["batch_index"] for r in steps) == set(range(nbatches)), "not every training batch used")
        calls = sum(r["block_calls"] for r in steps)
        require(summary["training_batched_F_calls"] == calls and summary["training_example_F_calls"] == calls*config["batch_size"]
                and summary["training_transformer_layer_executions"] == 2*calls, "training call totals mismatch")
        examples = {(s, e.root_id): e for s, es in data.items() if s != "train" for e in es}
        rows = [json.loads(line) for line in (arm / "validation.jsonl").read_text().splitlines()]
        expected = {(*key, k) for key in examples for k in config["budgets"]}
        seen = set()
        for row in rows:
            identity = row["suite"], row["root_id"], row["K"]
            require(identity in expected and identity not in seen, "unexpected/duplicate validation identity")
            seen.add(identity)
            example = examples[row["suite"], row["root_id"]]
            require(row["record_kind"] == "static_development" and row["policy"] == "restart" and row["split"] == "val"
                    and row["seed"] == seed and row["synthetic"] is False, "validation lineage mismatch")
            require(row["input_sha256"] == canonical_hash(example.maze), "validation input mismatch")
            require(row["checkpoint_sha256"] == summary["artifact_hashes"]["checkpoint.pt"]
                    and row["config_sha256"] == summary["artifact_hashes"]["config.json"], "validation checkpoint/config mismatch")
            require(row["state_budget"] == row["K"] and row["block_calls_per_example"] == row["K"]*factor, "validation budget mismatch")
            require(row["unreachable"] == (solve_maze(example.maze)[0][example.maze.start] < 0)
                    and row["start_equals_goal"] == (example.maze.start == example.maze.goal), "denominator mismatch")
            for name, value in score_policy(example.maze, row["actions"]).items():
                require(row[name] == value, f"validation score mismatch: {name}")
        require(seen == expected, "incomplete validation coverage")
        aggregate = metrics(rows)
        require(aggregate == summary["validation"], "regenerated aggregate mismatch")
        require(gate(aggregate, len(steps), config) == summary["gate"], "regenerated gate mismatch")
        require(summary["validation_example_F_calls"] == sum(r["block_calls_per_example"] for r in rows), "evaluation calls mismatch")
        resources = json.loads((arm / "resources.json").read_text())
        require(resources["complete"] and resources["error"] is None and resources["synthetic"] is False, "incomplete empirical job")
        require(0 < resources["wall_s"] <= config["job_seconds"] and resources["reservation_seconds"] == config["job_seconds"],
                "job time limit mismatch")
        require(resources["peak_reserved_gpu_bytes"] <= 6*2**30 and resources["device"] == "cuda" and resources["dtype"] == "float32",
                "device/memory limit mismatch")
        with sqlite3.connect("file:runs/accounting.sqlite?mode=ro", uri=True) as conn:
            events = conn.execute("SELECT kind,cents,category FROM events WHERE job_id=? ORDER BY seq", (resources["job_id"],)).fetchall()
        require(events == [("reserve", 0, "prompt05_scaled_static_maze"), ("actual", 0, "prompt05_scaled_static_maze")],
                "ledger reservation/reconciliation mismatch")
        time_events = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
        actual = [r for r in time_events if r["job_id"] == resources["job_id"] and r["kind"] == "actual"]
        require(len(actual) == 1 and actual[0]["seconds"] == resources["wall_s"], "GPU time journal mismatch")
        outputs.append({"seed": seed, "gate": summary["gate"], "validation": aggregate,
                        "resources": resources, "summary_sha256": file_hash(path)})
    seeds = [r["seed"] for r in outputs]
    require(len(set(seeds)) == len(seeds), "duplicate training seed")
    completed = set(seeds) == set(shared["seeds"])
    passing = [r["seed"] for r in outputs if r["gate"]["backbone_eligible"]]
    return {"verified": True, "arms": outputs, "all_seeds_complete": completed, "passing_seeds": passing,
            "all_seeds_pass": completed and len(passing) == len(shared["seeds"]),
            "decision": "await_remaining_seeds" if not completed else "continue_step3" if len(passing) >= 2 else "stop_for_replan",
            "remaining_gpu_seconds": remaining_seconds(time_events)}
