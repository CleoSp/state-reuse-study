"""Reconstruct selections, pairing, scores and gate; optionally replay inference."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import torch

from evaluate_frozen_crossover import load_model, read_maze, read_roots, select_edits, state_hash
from state_repair.data.maze import MazeExample, observation
from state_repair.data.splits import canonical_hash, reject_cross_split_duplicates
from state_repair.eval.crossover import summarize
from state_repair.eval.frozen import frozen_prediction
from state_repair.oracles.maze import score_policy
from state_repair.provenance import file_hash


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_frozen(directory: Path, *, replay: bool = False) -> dict:
    summary = json.loads((directory / "summary.json").read_text())
    required = {"config.json", "dataset.json", "checkpoint.pt", "provenance.json", "selection.json",
                "prior_states.pt", "initial.json", "predictions.jsonl", "preregistered-status.md"}
    require(set(summary["artifact_hashes"]) == required, "incomplete artifact manifest")
    for name, digest in summary["artifact_hashes"].items():
        require(file_hash(directory / name) == digest, f"artifact hash mismatch: {name}")
    config = json.loads((directory / "config.json").read_text())
    require(config["synthetic"] is False, "empirical run must not be synthetic")
    require(config["checkpoint_sha256"] == file_hash(directory / "checkpoint.pt"), "wrong preregistered checkpoint")
    require(config["dataset_sha256"] == file_hash(directory / "dataset.json"), "wrong preregistered dataset")
    provenance = json.loads((directory / "provenance.json").read_text())
    for name, digest in provenance["snapshot_hashes"].items():
        require(file_hash(directory / "source" / name) == digest, f"source snapshot mismatch: {name}")
    require(file_hash(directory / "source/scripts/evaluate_frozen_crossover.py") == provenance["script_sha256"],
            "runner snapshot mismatch")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    model, checkpoint = load_model(directory / "checkpoint.pt")
    require(checkpoint["dataset_sha256"] == config["dataset_sha256"], "checkpoint dataset lineage mismatch")
    require(summary["parameter_count"] == sum(p.numel() for p in model.parameters()), "parameter count mismatch")
    data = read_roots(directory / "dataset.json")
    require(len(data["train"]) == 512 and len(data["val"]) == summary["roots"] == 128, "wrong source root counts")
    selections, audit = select_edits(data["val"], config)
    serialized = json.loads(json.dumps({"edits": selections, "challenge_audit": audit}))
    require(serialized == json.loads((directory / "selection.json").read_text()), "selection reconstruction mismatch")
    require(audit == summary["challenge_audit"], "challenge audit mismatch")
    roots = {r.root_id: r for r in data["val"]}
    branches = {(s["suite"], s["root_id"], s["branch"]): s for s in selections}
    reject_cross_split_duplicates([*data["train"], *data["val"], *[
        MazeExample(read_maze(s["maze"]), s["root_id"], 1, "val", False) for s in selections]])
    states = torch.load(directory / "prior_states.pt", weights_only=True, map_location="cpu")
    require(set(states) == set(roots), "cached state root coverage mismatch")
    initial = json.loads((directory / "initial.json").read_text())
    initial_by_root = {r["root_id"]: r for r in initial}
    require(len(initial) == len(initial_by_root) == len(roots) and set(initial_by_root) == set(roots),
            "initial solve root coverage mismatch")
    factor = model.inner_cycles + 1
    with torch.no_grad():
        for root_id, root in roots.items():
            old = observation(root.maze, root_id, 0)
            state = replace(model.fresh_state(old), **states[root_id])
            model._check_state(old, state)
            row = initial_by_root[root_id]
            require(state.budget == row["source_K"] == config["source_K"], "initial budget mismatch")
            require(state_hash(state) == row["state_sha256"], "initial state hash mismatch")
            require(row["input_sha256"] == canonical_hash(root.maze), "initial input mismatch")
            require(row["source_policy"] == "restart" and row["split"] == "val" and row["synthetic"] is False,
                    "initial lineage mismatch")
            require(row["checkpoint_sha256"] == config["checkpoint_sha256"], "initial checkpoint mismatch")
            require(row["block_calls"] == factor*config["source_K"], "initial block count mismatch")
            for name, value in score_policy(root.maze, row["actions"]).items():
                require(row[name] == value, f"initial score mismatch: {name}")
            decoded = model.decode(old, state).logits[0].argmax(-1).tolist()
            require(decoded == row["actions"], "initial state/prediction mismatch")
            if replay:
                rerun = model(old, config["source_K"])
                require(state_hash(rerun.state) == state_hash(state), "initial state replay mismatch")
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        expected = {(*key, k, p) for key in branches for k in config["budgets"] for p in config["policies"]}
        seen = set()
        for row in rows:
            key = row["suite"], row["root_id"], row["branch"]
            identity = *key, row["K"], row["policy"]
            require(identity in expected and identity not in seen, "unexpected/duplicate prediction identity")
            seen.add(identity)
            selection = branches[key]
            for name, value in selection.items():
                if name != "maze":
                    require(row[name] == value, f"prediction selection mismatch: {name}")
            require(row["privileged"] is False, "privileged prediction in primary gate")
            require(row["config_sha256"] == file_hash(directory / "config.json"), "prediction config mismatch")
            require(row["checkpoint_sha256"] == config["checkpoint_sha256"], "prediction checkpoint mismatch")
            require(row["source_K"] == config["source_K"] and row["source_policy"] == "restart", "source policy/budget mismatch")
            require(row["prior_state_sha256"] == initial_by_root[row["root_id"]]["state_sha256"], "prior cache mismatch")
            require(row["state_budget"] == row["K"] and row["block_calls"] == row["K"]*factor, "update budget/calls mismatch")
            maze = read_maze(selection["maze"])
            for name, value in score_policy(maze, row["actions"]).items():
                require(row[name] == value, f"prediction score mismatch: {name}")
            if replay:
                root = roots[row["root_id"]]
                old = observation(root.maze, root.root_id, 0)
                new = observation(maze, root.root_id, 1)
                prior = replace(model.fresh_state(old), **states[root.root_id])
                rerun = frozen_prediction(model, old, new, prior, row["policy"], row["K"])
                require(rerun.prediction.logits[0].argmax(-1).tolist() == row["actions"], "prediction replay mismatch")
    require(seen == expected and len(rows) == summary["prediction_records"], "incomplete prediction coverage")
    calculated = summarize(rows, config)
    for name, value in calculated.items():
        require(summary[name] == value, f"regenerated summary mismatch: {name}")
    require(summary["initial_block_calls"] == sum(r["block_calls"] for r in initial), "initial total mismatch")
    require(summary["update_block_calls"] == sum(r["block_calls"] for r in rows), "update total mismatch")
    require(summary["transformer_layer_executions"] == 2*(summary["initial_block_calls"]+summary["update_block_calls"]),
            "layer execution total mismatch")
    resources = json.loads((directory / "resources.json").read_text())
    require(resources["complete"] and resources["error"] is None, "run did not complete")
    require(0 < resources["wall_s"] <= config["job_seconds"], "run exceeded time limit")
    require(resources["gpu_hours"] == 0 and resources["device"] == "cpu" and resources["dtype"] == "float32",
            "resource protocol mismatch")
    with sqlite3.connect("file:runs/accounting.sqlite?mode=ro", uri=True) as conn:
        events = conn.execute("SELECT kind,cents,category FROM events WHERE job_id=? ORDER BY seq",
                              (resources["job_id"],)).fetchall()
    require(events == [("reserve", 0, "prompt05_cpu_frozen_crossover"), ("actual", 0, "prompt05_cpu_frozen_crossover")],
            "missing/mismatched ledger reservation or reconciliation")
    return {"verified": True, "inference_replayed": replay, "prediction_records": len(rows),
            "roots": len(roots), "gate": calculated["gate"], "challenge_audit": audit,
            "resources": resources, "summary_sha256": file_hash(directory / "summary.json")}
