"""Evaluate frozen-state crossover on validation data without training."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time

import torch

from state_repair.accounting.ledger import AccountingLedger
from state_repair.accounting.resources import memory_snapshot
from state_repair.data.challenges import sample_challenges
from state_repair.data.maze import Maze, MazeExample, observation, possible_edges
from state_repair.data.splits import canonical_hash, reject_cross_split_duplicates
from state_repair.eval.crossover import impact_stratum, summarize
from state_repair.eval.frozen import frozen_prediction
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import impact_metadata, score_policy
from state_repair.provenance import file_hash, source_provenance


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_maze(raw: dict) -> Maze:
    return Maze(**{**raw, "edges": tuple(tuple(edge) for edge in raw["edges"])})


def read_roots(path: Path) -> dict[str, list[MazeExample]]:
    data = json.loads(path.read_text())
    if set(data) != {"train", "val"}:
        raise ValueError("source must contain train/validation only")
    roots = {split: [MazeExample(**{**r, "maze": read_maze(r["maze"])}) for r in records]
             for split, records in data.items()}
    for split, examples in roots.items():
        if any(e.split != split or e.synthetic or e.frame_index != 0 for e in examples):
            raise ValueError("invalid empirical base-root lineage")
    flat = [e for examples in roots.values() for e in examples]
    if len({e.root_id for e in flat}) != len(flat):
        raise ValueError("duplicate root IDs")
    reject_cross_split_duplicates(flat)
    return roots


def select_edits(roots: list[MazeExample], config: dict) -> tuple[list[dict], dict]:
    selections = []

    def add(root: MazeExample, new: Maze, suite: str, branch: str) -> None:
        changed = set(root.maze.edges) ^ set(new.edges)
        if len(changed) != 1:
            raise ValueError("gate requires exactly one undirected passage toggle")
        fraction = sum(impact_metadata(root.maze, new)["action_set_changed"]) / new.n
        selections.append({"suite": suite, "root_id": root.root_id, "branch": branch,
            "split": root.split, "synthetic": root.synthetic, "maze": asdict(new),
            "edit_type": "removal" if len(new.edges) < len(root.maze.edges) else "addition",
            "action_set_changed_fraction": fraction, "stratum": impact_stratum(fraction, config),
            "input_sha256": canonical_hash(new), "old_input_sha256": canonical_hash(root.maze)})

    for root in roots:
        seed = int.from_bytes(hashlib.sha256(f'{config["ordinary_edit_seed"]}:{root.root_id}'.encode()).digest()[:8], "big")
        edge = random.Random(seed).choice(possible_edges(root.maze.height, root.maze.width))
        add(root, root.maze.toggle(*edge), "ordinary", "uniform_toggle")
    sample = sample_challenges(roots, pairs_per_type=config["challenge_pairs_per_type"],
        seed=config["challenge_seed"], low_max=config["low_max"], high_min=config["high_min"])
    for pair in sample.pairs:
        add(pair.root, pair.low, "challenge", "low")
        add(pair.root, pair.high, "challenge", "high")
    return selections, sample.audit


def state_hash(state) -> str:
    digest = hashlib.sha256()
    for value in (state.a, state.z, state.node_ids, state.valid_nodes):
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps([state.episode_ids, state.frame_indices, state.budget]).encode())
    return digest.hexdigest()


def load_model(path: Path) -> tuple[RecursiveSolver, dict]:
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = RecursiveSolver(**checkpoint["config"]["model_kwargs"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["synthetic"] is not False or config["source_K"] != 8 or config["budgets"] != [1, 2, 4, 8]:
        raise ValueError("not the declared empirical frozen-crossover protocol")
    if not 0 < config["job_seconds"] <= 1200:
        raise ValueError("CPU gate cap must be <=1200 seconds")
    source = Path(config["source_run"])
    checkpoint_path = source / config["checkpoint_arm"] / "checkpoint.pt"
    for path, name in ((checkpoint_path, "checkpoint_sha256"), (source / "dataset.json", "dataset_sha256")):
        if file_hash(path) != config[name]:
            raise ValueError(f"preregistered {name} mismatch")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    ledger = AccountingLedger("runs/accounting.sqlite")
    job = f"{output.name}:{time.time_ns()}"
    ledger.reserve(job, 0, "prompt05_cpu_frozen_crossover")
    started = time.perf_counter()
    complete, error, result = False, None, None

    def deadline() -> None:
        if time.perf_counter() - started > config["job_seconds"]:
            raise TimeoutError("CPU frozen comparison exceeded preregistered cap")

    try:
        shutil.copyfile(config_path, output / "config.json")
        shutil.copyfile(source / "dataset.json", output / "dataset.json")
        shutil.copyfile(checkpoint_path, output / "checkpoint.pt")
        shutil.copyfile("REPRODUCING.md", output / "preregistered-status.md")
        provenance = {**source_provenance(), "script_sha256": file_hash(__file__),
            "python": platform.python_version(), "torch": str(torch.__version__),
            "device": "cpu", "dtype": "float32", "batch_size": 1, "threads": config["threads"],
            "deterministic_algorithms": True, "host_memory": memory_snapshot()}
        for path in [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), Path("pyproject.toml")]:
            target = output / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        provenance["snapshot_hashes"] = {p.relative_to(output / "source").as_posix(): file_hash(p)
                                          for p in sorted((output / "source").rglob("*")) if p.is_file()}
        write(output / "provenance.json", provenance)
        data = read_roots(output / "dataset.json")
        if len(data["train"]) != 512 or len(data["val"]) != 128:
            raise ValueError("gate requires the existing 512/128 split")
        selections, audit = select_edits(data["val"], config)
        write(output / "selection.json", {"edits": selections, "challenge_audit": audit})
        model, checkpoint = load_model(output / "checkpoint.pt")
        if checkpoint["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("checkpoint data provenance mismatch")
        roots = {e.root_id: e for e in data["val"]}
        states, initial_rows, rows = {}, [], []
        initial_calls = update_calls = 0
        with torch.no_grad():
            for root in roots.values():
                deadline()
                old = observation(root.maze, root.root_id, 0)
                initial = model(old, config["source_K"])
                states[root.root_id] = asdict(initial.state.detach())
                actions = initial.prediction.logits[0].argmax(-1).tolist()
                initial_rows.append({"root_id": root.root_id, "source_K": config["source_K"],
                    "source_policy": "restart", "synthetic": False, "split": "val",
                    "input_sha256": canonical_hash(root.maze), "checkpoint_sha256": config["checkpoint_sha256"],
                    "state_sha256": state_hash(initial.state), "block_calls": initial.block_calls,
                    "actions": actions, **score_policy(root.maze, actions)})
                initial_calls += initial.block_calls
            torch.save(states, output / "prior_states.pt")
            write(output / "initial.json", initial_rows)
            initial_by_root = {r["root_id"]: r for r in initial_rows}
            with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for i, selection in enumerate(selections):
                    root = roots[selection["root_id"]]
                    old = observation(root.maze, root.root_id, 0)
                    new_maze = read_maze(selection["maze"])
                    new = observation(new_maze, root.root_id, 1)
                    prior = replace(model.fresh_state(old), **states[root.root_id])
                    for k in config["budgets"]:
                        for policy in config["policies"]:
                            deadline()
                            prediction = frozen_prediction(model, old, new, prior, policy, k)
                            actions = prediction.prediction.logits[0].argmax(-1).tolist()
                            row = {**{name: value for name, value in selection.items() if name != "maze"},
                                "record_kind": "frozen_state_intervention", "privileged": False,
                                "K": k, "policy": policy, "source_K": config["source_K"], "source_policy": "restart",
                                "prior_state_sha256": initial_by_root[root.root_id]["state_sha256"],
                                "checkpoint_sha256": config["checkpoint_sha256"],
                                "config_sha256": file_hash(output / "config.json"),
                                "block_calls": prediction.block_calls, "state_budget": prediction.state.budget,
                                "actions": actions, **score_policy(new_maze, actions)}
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            rows.append(row)
                            update_calls += prediction.block_calls
                    if state_hash(prior) != initial_by_root[root.root_id]["state_sha256"]:
                        raise ValueError("intervention mutated shared prior state")
                    if (i + 1) % 32 == 0:
                        print(json.dumps({"edits_completed": i+1, "edits_total": len(selections),
                                          "wall_s": time.perf_counter()-started}), flush=True)
        deadline()
        result = {**summarize(rows, config), "protocol_version": config["protocol_version"],
            "roots": len(roots), "prediction_records": len(rows), "challenge_audit": audit,
            "initial_block_calls": initial_calls, "update_block_calls": update_calls,
            "transformer_layer_executions": 2*(initial_calls+update_calls),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "artifact_hashes": {name: file_hash(output / name) for name in
                ("config.json", "dataset.json", "checkpoint.pt", "provenance.json", "selection.json",
                 "prior_states.pt", "initial.json", "predictions.jsonl", "preregistered-status.md")}}
        write(output / "summary.json", result)
        complete = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        resources = {"job_id": job, "complete": complete, "error": error,
            "wall_s": time.perf_counter()-started, "memory": memory_snapshot(),
            "device": "cpu", "dtype": "float32", "gpu_hours": 0,
            "peak_allocated_gpu_bytes": 0, "peak_reserved_gpu_bytes": 0,
            "external_cost_usd": 0, "electricity_cost_usd": None,
            "ledger": ledger.reconcile(job, 0)}
        write(output / "resources.json", resources)
        print(json.dumps({"resources": resources, "gate": None if result is None else result["gate"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_crossover_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/frozen_crossover_v1"))
    args = parser.parse_args()
    run(args.config, args.output)
