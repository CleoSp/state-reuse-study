"""Rebuild circuit data, replay sampling and score all saved predictions."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sqlite3

import torch

from check_frozen_crossover import require
from train_static_circuit import gate, input_hash, make_data, metrics, score
from state_repair.accounting.gpu_job import JOURNAL, remaining_seconds
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash
from state_repair.types import Domain


def check_circuit(directory: Path) -> dict:
    paths = sorted(directory.glob("seed-*/summary.json"))
    require(bool(paths), "no completed static-circuit runs")
    dataset = json.loads((directory / "dataset.json").read_text())
    outputs, shared = [], None
    for path in paths:
        arm = path.parent
        summary = json.loads(path.read_text())
        require(summary["synthetic"] is False, "synthetic run excluded from empirical circuit mode")
        names = {"config.json", "checkpoint.pt", "provenance.json", "steps.jsonl", "validation.jsonl", "preregistered-status.md"}
        require(set(summary["artifact_hashes"]) == names, "incomplete artifact manifest")
        for name, digest in summary["artifact_hashes"].items():
            require(file_hash(arm / name) == digest, f"artifact hash mismatch: {name}")
        config = json.loads((arm / "config.json").read_text())
        seed = config["seed"]
        require(seed in config["seeds"] and seed == summary["seed"] and arm.name == f"seed-{seed}", "circuit seed mismatch")
        common = {k: v for k, v in config.items() if k != "seed"}
        if shared is None:
            shared = common
            data = make_data(common)
            require(json.loads(json.dumps({s: [asdict(e) for e in es] for s, es in data.items()})) == dataset,
                    "circuit dataset reconstruction mismatch")
        require(common == shared, "circuit configuration changed between seeds")
        require(file_hash(directory / "dataset.json") == summary["dataset_sha256"], "dataset hash mismatch")
        provenance = json.loads((arm / "provenance.json").read_text())
        if config.get("preload_training_batches", False):
            diagnostic = Path(config["transfer_diagnostic"])
            require(file_hash(diagnostic / "profile.json") == provenance["transfer_profile_sha256"]
                    and file_hash(diagnostic / "resources.json") == provenance["transfer_resources_sha256"],
                    "transfer diagnostic provenance mismatch")
        capacity_path = Path(config["capacity_run"])
        require(file_hash(capacity_path / "profile.json") == provenance["capacity_profile_sha256"]
                and file_hash(capacity_path / "resources.json") == provenance["capacity_resources_sha256"],
                "capacity provenance mismatch")
        capacity = json.loads((capacity_path / "profile.json").read_text())
        capacity_resources = json.loads((capacity_path / "resources.json").read_text())
        require(capacity_resources["complete"] and capacity_resources["error"] is None
                and capacity["cpu_cuda_K1_gradient_parity"]
                and [r["K"] for r in capacity["rows"]] == config["budgets"], "incomplete capacity validation")
        require(all(capacity[name] == config[name] for name in ("batch_size", "model_width", "heads", "inner_cycles")),
                "capacity configuration mismatch")
        for name, digest in provenance["snapshot_hashes"].items():
            require(file_hash(arm / "source" / name) == digest, f"source snapshot mismatch: {name}")
        require(file_hash(arm / "source/scripts/train_static_circuit.py") == provenance["script_sha256"], "runner snapshot mismatch")
        checkpoint = torch.load(arm / "checkpoint.pt", weights_only=True, map_location="cpu")
        require(checkpoint["config"] == config and checkpoint["provenance"] == provenance, "checkpoint config/source mismatch")
        require(checkpoint["dataset_sha256"] == summary["dataset_sha256"], "checkpoint dataset mismatch")
        model = RecursiveSolver(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
                                attention_mode=config["attention_mode"], domain=Domain.CIRCUIT)
        model.load_state_dict(checkpoint["model"], strict=True)
        require(all(torch.isfinite(p).all() for p in model.parameters()), "nonfinite checkpoint")
        require(sum(p.numel() for p in model.parameters()) == summary["parameter_count"], "parameter count mismatch")
        steps = [json.loads(line) for line in (arm / "steps.jsonl").read_text().splitlines()]
        require(bool(steps) and len(steps) == checkpoint["steps"] == summary["optimizer_steps"], "optimizer step mismatch")
        require(all(int(state["step"].item()) == len(steps) for state in checkpoint["optimizer"]["state"].values()),
                "optimizer state step mismatch")
        require(summary["final_loss"] == steps[-1]["loss"], "final loss mismatch")
        rng, order = random.Random(seed), []
        nbatches, factor = config["train_roots"]//config["batch_size"], config["inner_cycles"]+1
        for i, row in enumerate(steps, 1):
            if not order:
                order = list(range(nbatches))
                rng.shuffle(order)
            index, k = order.pop(), rng.choice(config["budgets"])
            require(row["step"] == i and row["batch_index"] == index and row["K"] == k, "seeded circuit batch/K mismatch")
            require(row["seed"] == seed and row["synthetic"] is False and row["batch_size"] == config["batch_size"], "step lineage mismatch")
            require(row["block_calls"] == k*factor, "step recurrence mismatch")
            require(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) and row["gradient_norm"] >= 0,
                    "invalid loss/gradient")
            require(row["lr"] == (config["learning_rate"] if i <= config["steps"]*.75 else config["final_learning_rate"]), "LR mismatch")
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
            require(identity in expected and identity not in seen, "unexpected/duplicate circuit validation identity")
            seen.add(identity)
            example = examples[row["suite"], row["root_id"]]
            require(row["record_kind"] == "static_development" and row["domain"] == "circuit" and row["policy"] == "restart"
                    and row["split"] == "val" and row["seed"] == seed and row["synthetic"] is False, "validation lineage mismatch")
            require(row["input_sha256"] == input_hash(example), "validation input/order mismatch")
            require(row["checkpoint_sha256"] == summary["artifact_hashes"]["checkpoint.pt"]
                    and row["config_sha256"] == summary["artifact_hashes"]["config.json"], "validation checkpoint/config mismatch")
            require(row["state_budget"] == row["K"] and row["block_calls_per_example"] == row["K"]*factor, "validation budget mismatch")
            for name, value in score(example, row["predictions"]).items():
                require(row[name] == value, f"circuit score/denominator mismatch: {name}")
        require(seen == expected, "incomplete circuit validation coverage")
        aggregate = metrics(rows, config)

        require(json.loads(json.dumps(aggregate)) == summary["validation"], "regenerated aggregate mismatch")
        require(gate(aggregate, len(steps), config) == summary["gate"], "regenerated gate mismatch")
        require(summary["validation_example_F_calls"] == sum(r["block_calls_per_example"] for r in rows), "validation call total mismatch")
        resources = json.loads((arm / "resources.json").read_text())
        require(resources["complete"] and resources["error"] is None and resources["synthetic"] is False, "incomplete empirical circuit job")
        require(0 < resources["wall_s"] <= config["job_seconds"] and resources["reservation_seconds"] == config["job_seconds"], "job time mismatch")
        require(resources["peak_reserved_gpu_bytes"] <= 6*2**30 and resources["device"] == "cuda" and resources["dtype"] == "float32",
                "device/memory limit mismatch")
        with sqlite3.connect("file:runs/accounting.sqlite?mode=ro", uri=True) as conn:
            events = conn.execute("SELECT kind,cents,category FROM events WHERE job_id=? ORDER BY seq", (resources["job_id"],)).fetchall()
        require(events == [("reserve", 0, "prompt05_static_circuit"), ("actual", 0, "prompt05_static_circuit")], "ledger mismatch")
        time_events = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
        actual = [r for r in time_events if r["job_id"] == resources["job_id"] and r["kind"] == "actual"]
        require(len(actual) == 1 and actual[0]["seconds"] == resources["wall_s"], "GPU time journal mismatch")
        outputs.append({"seed": seed, "gate": summary["gate"], "validation": aggregate,
                        "resources": resources, "summary_sha256": file_hash(path)})
    seeds = [r["seed"] for r in outputs]
    require(len(seeds) == len(set(seeds)), "duplicate training seed")
    completed = set(seeds) == set(shared["seeds"])
    passing = [r["seed"] for r in outputs if r["gate"]["backbone_eligible"]]
    eligible = completed and len(passing) == len(shared["seeds"])
    return {"verified": True, "arms": outputs, "all_seeds_complete": completed,
            "passing_seeds": passing, "family_eligible": eligible,
            "decision": "await_remaining_seeds" if not completed else "include_circuits" if eligible else "exclude_circuits_continue_maze",
            "remaining_gpu_seconds": remaining_seconds(time_events)}
