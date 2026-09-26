"""Train static maze solvers across three seeds using validation data."""
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

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.maze import MazeExample, collate, generate_maze
from state_repair.data.rooms import generate_rooms
from state_repair.data.splits import assign_splits, canonical_hash, reject_cross_split_duplicates
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import maze_valid_set_loss


def write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def make_data(config: dict) -> dict[str, list[MazeExample]]:
    count = config["train_roots"] + 2*config["val_roots"]
    assignments = assign_splits(count, config["data_seed"], config["train_roots"]/count, config["val_roots"]/count)
    data: dict[str, list[MazeExample]] = {key: [] for key in ("train", "ordinary", "rooms", "size16")}

    def create(root_id: str, split: str, suite: str) -> None:
        seed = int.from_bytes(hashlib.sha256(f'{config["data_seed"]}:{root_id}'.encode()).digest()[:8], "big")
        if suite == "rooms":
            maze = generate_rooms(config["height"], config["width"], seed, config["room_size"])
        elif suite == "size16":
            maze = generate_maze(config["size_shift"], config["size_shift"], seed)
        else:
            maze = generate_maze(config["height"], config["width"], seed)
        if len(data[suite]) % 4 == 0:
            cut = maze.width//2
            maze = replace(maze, edges=tuple((u, v) for u, v in maze.edges
                                            if (u % maze.width < cut) == (v % maze.width < cut)))
        data[suite].append(MazeExample(maze, root_id, 0, split, config["synthetic"]))

    for root, split in sorted(assignments.items()):
        if split != "test":
            create(f'scaled-maze-v1-{root}', split, "train" if split == "train" else "ordinary")



    for suite in ("rooms", "size16"):
        for i in range(config["val_roots"]):
            create(f"scaled-maze-v1-{suite}-{i:06d}", "val", suite)
    reject_cross_split_duplicates(e for examples in data.values() for e in examples)
    if len(data["train"]) != config["train_roots"] or any(len(data[s]) != config["val_roots"] for s in data if s != "train"):
        raise ValueError("split count mismatch")
    return data


def metrics(rows: list[dict]) -> list[dict]:
    result = []
    for suite in ("ordinary", "rooms", "size16"):
        for k in (1, 2, 4, 8, 16):
            selected = [r for r in rows if r["suite"] == suite and r["K"] == k]
            if not selected:
                raise ValueError("incomplete validation suite/budget")
            result.append({"suite": suite, "K": k, "roots": len(selected),
                "routes": sum(r["route_correct"] for r in selected),
                "route_accuracy": sum(r["route_correct"] for r in selected)/len(selected),
                "all_node_accuracy": sum(r["all_node_correct"] for r in selected)/len(selected),
                "valid_action_accuracy": sum(r["valid_action_accuracy"] for r in selected)/len(selected),
                "unreachable_roots": sum(r["unreachable"] for r in selected),
                "unreachable_correct": sum(r["unreachable"] and r["route_correct"] for r in selected),
                "trivial_roots": sum(r["start_equals_goal"] for r in selected),
                "reachable_nontrivial_roots": sum(not r["unreachable"] and not r["start_equals_goal"] for r in selected),
                "reachable_nontrivial_correct": sum(not r["unreachable"] and not r["start_equals_goal"] and r["route_correct"] for r in selected),
                "reasons": dict(Counter(r["reason"] for r in selected))})
    return result


def gate(rows: list[dict], steps: int, config: dict) -> dict:
    ordinary = [r for r in rows if r["suite"] == "ordinary"]
    if [r["K"] for r in ordinary] != config["budgets"]:
        raise ValueError("gate needs all declared budgets in order")
    monotonic = all(a["route_accuracy"] <= b["route_accuracy"] for a, b in zip(ordinary, ordinary[1:]))
    competent = ordinary[-1]["route_accuracy"] >= config["minimum_accuracy"]
    complete = steps == config["steps"]
    return {"nondecreasing": monotonic, "competent": competent, "training_complete": complete,
            "minimum_accuracy": config["minimum_accuracy"], "backbone_eligible": monotonic and competent and complete}


def run(config_path: Path, output_root: Path, seed: int) -> dict:
    config = json.loads(config_path.read_text())
    if seed not in config["seeds"] or config["batch_size"] < 64 or config["synthetic"] is not False:
        raise ValueError("seed/batch/empirical protocol mismatch")
    crossover = json.loads(Path("runs/frozen_crossover_v1/summary.json").read_text())
    check = json.loads(Path("runs/prompt05-frozen-check.json").read_text(encoding="utf-8-sig"))
    if not crossover["gate"]["crossover"] or not check["verified"] or not check["inference_replayed"]:
        raise ValueError("verified crossover required before scaling")
    if check["summary_sha256"] != file_hash("runs/frozen_crossover_v1/summary.json"):
        raise ValueError("crossover evidence changed since verification")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"seed-{seed}"
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    rng = random.Random(seed)
    with GPUJob(out, config["job_seconds"], "prompt05_scaled_static_maze", synthetic=False) as job:
        write(out / "config.json", {**config, "seed": seed})
        shutil.copyfile("REPRODUCING.md", out / "preregistered-status.md")
        provenance = {**source_provenance(), "script_sha256": file_hash(__file__),
            "python": platform.python_version(), "torch": str(torch.__version__), "device": "cuda",
            "dtype": "float32", "tf32": False, "source_gate_sha256": file_hash("runs/frozen_crossover_v1/summary.json"),
            "source_gate_check_sha256": file_hash("runs/prompt05-frozen-check.json")}
        for path in [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), Path("pyproject.toml")]:
            target = out / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        provenance["snapshot_hashes"] = {p.relative_to(out / "source").as_posix(): file_hash(p)
                                          for p in (out / "source").rglob("*") if p.is_file()}
        write(out / "provenance.json", provenance)
        data = make_data(config)
        payload = {suite: [asdict(e) for e in examples] for suite, examples in data.items()}
        data_path = output_root / "dataset.json"
        if data_path.exists():
            if json.loads(data_path.read_text()) != json.loads(json.dumps(payload)):
                raise ValueError("data changed across training seeds")
        else:
            write(data_path, payload)
        model = RecursiveSolver(width=config["model_width"], heads=config["heads"],
                                inner_cycles=config["inner_cycles"], attention_mode=config["attention_mode"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        batches = [collate(data["train"][i:i+config["batch_size"]])[:2]
                   for i in range(0, len(data["train"]), config["batch_size"])]
        if any(obs.valid_nodes.shape[0] < 64 for obs, target in batches):
            raise ValueError("training would use a batch below 64")
        order, counts, batch_counts = [], Counter(), Counter()
        calls = 0
        train_start = time.perf_counter()
        with (out / "steps.jsonl").open("x", encoding="utf-8") as handle:
            for step in range(1, config["steps"]+1):
                job.check_limit()
                if time.perf_counter()-train_start >= config["training_seconds"]:
                    break
                if not order:
                    order = list(range(len(batches)))
                    rng.shuffle(order)
                index, k = order.pop(), rng.choice(config["budgets"])
                obs, target = (value.to("cuda") for value in batches[index])
                lr = config["learning_rate"] if step <= config["steps"]*.75 else config["final_learning_rate"]
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                result = model(obs, k)
                loss = maze_valid_set_loss(result.prediction.logits, target, obs.valid_nodes)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite empirical loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                counts[k] += 1
                batch_counts[index] += 1
                calls += result.block_calls
                row = {"step": step, "seed": seed, "K": k, "batch_index": index, "batch_size": obs.valid_nodes.shape[0],
                    "loss": loss.item(), "gradient_norm": norm.item(), "lr": lr,
                    "block_calls": result.block_calls, "synthetic": False,
                    "elapsed_s": time.perf_counter()-train_start}
                handle.write(json.dumps(row, allow_nan=False)+"\n")
                if step % 100 == 0:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                del result, loss, obs, target
        steps = sum(counts.values())
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": rng.getstate(), "batch_order": order, "steps": steps,
            "config": {**config, "seed": seed}, "provenance": provenance,
            "dataset_sha256": file_hash(data_path)}, out / "checkpoint.pt")
        model.eval()
        checkpoint_hash, config_hash = file_hash(out / "checkpoint.pt"), file_hash(out / "config.json")
        rows = []
        with torch.no_grad(), (out / "validation.jsonl").open("x", encoding="utf-8") as handle:
            for suite in ("ordinary", "rooms", "size16"):
                for k in config["budgets"]:
                    for offset in range(0, len(data[suite]), config["batch_size"]):
                        job.check_limit()
                        batch = data[suite][offset:offset+config["batch_size"]]
                        obs, _, _ = collate(batch, "cuda")
                        result = model(obs, k)
                        actions_batch = result.prediction.logits.argmax(-1).cpu().tolist()
                        for example, actions in zip(batch, actions_batch):
                            distance = solve_maze(example.maze)[0][example.maze.start]
                            row = {"root_id": example.root_id, "suite": suite, "split": "val", "seed": seed,
                                "K": k, "synthetic": False, "record_kind": "static_development", "policy": "restart",
                                "checkpoint_sha256": checkpoint_hash,
                                "config_sha256": config_hash, "input_sha256": canonical_hash(example.maze),
                                "block_calls_per_example": result.block_calls, "state_budget": result.state.budget,
                                "unreachable": distance < 0, "start_equals_goal": example.maze.start == example.maze.goal,
                                "actions": actions, **score_policy(example.maze, actions)}
                            handle.write(json.dumps(row, allow_nan=False)+"\n")
                            rows.append(row)
                        del result, obs
                    print(json.dumps({"seed": seed, "evaluation_suite": suite, "K": k,
                        "routes": sum(r["route_correct"] for r in rows if r["suite"] == suite and r["K"] == k)}), flush=True)
        aggregate = metrics(rows)
        summary = {"protocol_version": config["protocol_version"], "seed": seed, "synthetic": False,
            "optimizer_steps": steps, "final_loss": row.get("loss") if "loss" in row else None,
            "sampled_K_counts": dict(counts), "batch_counts": dict(batch_counts),
            "training_batched_F_calls": calls, "training_example_F_calls": calls*config["batch_size"],
            "training_transformer_layer_executions": calls*2,
            "validation_example_F_calls": sum(r["block_calls_per_example"] for r in rows),
            "parameter_count": sum(p.numel() for p in model.parameters()), "validation": aggregate,
            "gate": gate(aggregate, steps, config), "dataset_sha256": file_hash(data_path),
            "artifact_hashes": {name: file_hash(out / name) for name in
                ("config.json", "checkpoint.pt", "provenance.json", "steps.jsonl", "validation.jsonl", "preregistered-status.md")}}

        summary["final_loss"] = json.loads((out / "steps.jsonl").read_text().splitlines()[-1])["loss"]
        write(out / "summary.json", summary)
        print(json.dumps({"seed": seed, "gate": summary["gate"], "validation": aggregate}), flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/scaled_maze_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/scaled_maze_v1"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.seed)
