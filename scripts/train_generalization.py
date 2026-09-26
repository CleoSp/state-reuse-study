"""Train a static solver with sampled budgets and paired validation."""
from __future__ import annotations

import argparse
from collections import Counter
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
from state_repair.data.maze import MazeExample, collate, generate_maze
from state_repair.data.splits import assign_splits, canonical_hash, reject_cross_split_duplicates
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import maze_valid_set_loss


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def make_roots(config: dict) -> dict[str, list[MazeExample]]:
    """Split first. Generate only train/val; 25% get an observable middle cut.

    Cut assignment is independent of solution difficulty and is identical in
    proportion across splits. No rejection based on reachability or impact.
    """
    assignments = assign_splits(768, config["data_seed"], 2 / 3, 1 / 6)
    output: dict[str, list[MazeExample]] = {"train": [], "val": []}
    for root, split in sorted(assignments.items()):
        if split == "test":
            continue
        seed = int.from_bytes(hashlib.sha256(f'{config["data_seed"]}:{root}'.encode()).digest()[:8], "big")
        maze = generate_maze(config["height"], config["width"], seed)
        if len(output[split]) % 4 == 0:
            cut = maze.width // 2
            maze = replace(maze, edges=tuple((u, v) for u, v in maze.edges
                if (u % maze.width < cut) == (v % maze.width < cut)))
        output[split].append(MazeExample(maze, f"generalization-v1-{root}", 0, split, False))
    reject_cross_split_duplicates(output["train"] + output["val"])
    assert len(output["train"]) == config["train_roots"] >= 512
    assert len(output["val"]) == config["val_roots"]
    return output


def gate(rows: list[dict], minimum_accuracy: float) -> dict:
    """No tolerance for declines; flat curves alone do not establish refinement."""
    counts = [row["routes"] for row in rows]
    monotonic = all(a <= b for a, b in zip(counts, counts[1:]))
    improvement = counts[-1] > counts[0]
    competent = rows[-1]["route_accuracy"] >= minimum_accuracy
    return {"nondecreasing": monotonic, "strict_endpoint_improvement": improvement,
            "minimum_route_accuracy": minimum_accuracy, "competent": competent,
            "prompt04_allowed": monotonic and improvement and competent}


def evaluate(model: RecursiveSolver, examples: list[MazeExample], budgets: list[int],
             path: Path, deadline: float, checkpoint_hash: str, config_hash: str) -> list[dict]:
    """Fresh fixed-budget states; batched evaluation is not a latency benchmark."""
    model.eval()
    rows = []
    with torch.no_grad(), path.open("x", encoding="utf-8") as handle:
        for k in budgets:
            current = []
            for offset in range(0, len(examples), 8):
                if time.perf_counter() > deadline:
                    raise TimeoutError("job cap reached during validation; no pass may be declared")
                batch = examples[offset:offset + 8]
                obs, target, _ = collate(batch)
                result = model(obs, k)
                logits = result.prediction.logits
                for example, values in zip(batch, logits):
                    actions = values.argmax(-1).tolist()
                    record = {"root_id": example.root_id, "split": example.split, "K": k,
                        "synthetic": False, "record_kind": "static_development", "policy": "restart",
                        "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash,
                        "input_sha256": canonical_hash(example.maze), "actions": actions,
                        "block_calls_per_example": result.block_calls,
                        **score_policy(example.maze, actions)}
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    current.append(record)
            rows.append({"K": k, "roots": len(current), "routes": sum(r["route_correct"] for r in current),
                "route_accuracy": sum(r["route_correct"] for r in current) / len(current),
                "action_accuracy": sum(r["valid_action_accuracy"] for r in current) / len(current),
                "nontrivial_routes": sum(r["route_correct"] for r, e in zip(current, examples) if e.maze.start != e.maze.goal),
                "nontrivial_roots": sum(e.maze.start != e.maze.goal for e in examples)})
    return rows


def run_arm(mode: str, config: dict, data: dict, out: Path, deadline: float) -> dict:
    out.mkdir()
    started = time.perf_counter()
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    model_kwargs = dict(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"])
    if mode != "dense":
        model_kwargs["attention_mode"] = mode
    model = RecursiveSolver(**model_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    arm_config = {**config, "attention_mode": mode, "model_kwargs": model_kwargs,
                  "selection": "last completed step; no validation checkpoint selection"}
    write(out / "config.json", arm_config)
    provenance = {**source_provenance(), "script_sha256": file_hash(__file__),
                  "python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"}
    write(out / "provenance.json", provenance)

    for source in [*Path("src").rglob("*.py"), Path(__file__)]:
        destination = out / "source" / source.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    train = data["train"]
    batches = [collate(train[i:i + config["batch_size"]])[:2] for i in range(0, len(train), config["batch_size"])]
    order: list[int] = []
    counts: Counter = Counter()
    f_calls = 0
    train_start = time.perf_counter()
    with (out / "steps.jsonl").open("x", encoding="utf-8") as handle:
        for step in range(1, config["steps"] + 1):
            elapsed = time.perf_counter() - train_start
            if elapsed >= config["training_seconds_per_arm"] or time.perf_counter() > deadline - 65:
                break
            if not order:
                order = list(range(len(batches)))
                rng.shuffle(order)
            index = order.pop()
            obs, target = batches[index]
            k = rng.choice(config["budgets"])
            fraction = max((step - 1) / config["steps"], elapsed / config["training_seconds_per_arm"])
            lr = config["learning_rate"] if fraction < .75 else config["final_learning_rate"]
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            result = model(obs, k)
            loss = maze_valid_set_loss(result.prediction.logits, target, obs.valid_nodes)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            counts[k] += 1
            f_calls += result.block_calls
            row = {"step": step, "K": k, "batch_index": index, "loss": loss.item(),
                   "lr": lr, "gradient_norm": norm.item(), "elapsed_s": time.perf_counter() - train_start,
                   "block_calls": result.block_calls, "synthetic": False}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if step % 100 == 0:
                print(json.dumps({"arm": mode, **row}), flush=True)
    steps = sum(counts.values())
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "python_rng": rng.getstate(), "batch_order": order, "steps": steps, "config": arm_config,
                "provenance": provenance, "dataset_sha256": file_hash(out.parent / "dataset.json")}, out / "checkpoint.pt")
    checkpoint_hash = file_hash(out / "checkpoint.pt")
    rows = evaluate(model, data["val"], config["budgets"], out / "validation.jsonl", deadline,
                    checkpoint_hash, file_hash(out / "config.json"))
    summary = {"attention_mode": mode, "optimizer_steps": steps, "sampled_K_counts": dict(counts),
        "training_batched_F_calls": f_calls, "training_transformer_layer_executions": f_calls * 2,
        "training_example_F_calls": f_calls * config["batch_size"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "validation": rows, "gate": gate(rows, config["minimum_validation_route_accuracy"]),
        "wall_s": time.perf_counter() - started, "memory": memory_snapshot(),
        "artifact_hashes": {name: file_hash(out / name) for name in
            ("checkpoint.pt", "config.json", "provenance.json", "validation.jsonl", "steps.jsonl")}}
    write(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/static_generalization_v1.json")
    parser.add_argument("--output", default="runs/static_generalization_v1")
    parser.add_argument("--arm", choices=["dense", "masked_neighbor"], default="dense")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not 0 < config["training_seconds_per_arm"] < config["job_seconds"] <= 1100:
        raise ValueError("this CPU protocol requires training_seconds_per_arm < job_seconds <= 1100")
    if config["budgets"] != [1, 2, 4, 8, 16]:
        raise ValueError("the depth gate requires K=1,2,4,8,16")
    torch.set_num_threads(config["threads"])
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    if (out / args.arm).exists():
        raise FileExistsError("arm already exists; preserve it and use a reviewed new protocol/output")
    if args.arm == "masked_neighbor":
        baseline = json.loads((out / "dense/summary.json").read_text())
        if baseline["gate"]["nondecreasing"]:
            raise ValueError("masked follow-up requires a failed dense depth gate")
    started = time.perf_counter()
    prior = sum(json.loads(p.read_text())["wall_s"] for p in out.glob("*-resources.json"))
    remaining = config["job_seconds"] - prior
    if remaining < 100:
        raise ValueError("cumulative CPU smoke allowance exhausted")
    data = make_roots(config)
    payload = {split: [asdict(e) for e in examples] for split, examples in data.items()}
    if (out / "dataset.json").exists():
        if json.loads((out / "dataset.json").read_text()) != json.loads(json.dumps(payload)):
            raise ValueError("dataset changed between arms")
    else:
        write(out / "dataset.json", payload)
    ledger = AccountingLedger("runs/accounting.sqlite")
    job = f"{out.name}:{args.arm}:{time.time_ns()}"
    ledger.reserve(job, 0, "local_cpu_static_generalization")
    try:
        run_arm(args.arm, config, data, out / args.arm, started + remaining)
    finally:
        write(out / f"{args.arm}-resources.json", {"wall_s": time.perf_counter() - started,
            "external_cost_usd": 0, "electricity_cost_usd": None, "ledger": ledger.reconcile(job, 0)})


if __name__ == "__main__":
    main()
