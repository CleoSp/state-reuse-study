"""Train static circuit solvers from the configured validation protocol."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import time

import torch

from train_scaled_maze import write
from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.circuit import Circuit, CircuitExample, Operator, collate, generate_circuit, reject_duplicate_roots
from state_repair.data.deep_circuits import generate_deep_circuit
from state_repair.data.splits import assign_splits
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.circuit import evaluate
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.losses import circuit_value_loss
from state_repair.types import Domain


def input_hash(example: CircuitExample) -> str:
    return hashlib.sha256(json.dumps({"circuit": asdict(example.circuit), "node_order": example.node_order},
                                    sort_keys=True).encode()).hexdigest()


def graph_depth(circuit: Circuit) -> int:
    """Evaluator-only structural descriptor; never passed to the model."""
    depths = {}
    def depth(u: int) -> int:
        if u not in depths:
            depths[u] = 0 if not circuit.parents[u] else 1+max(depth(p) for p in circuit.parents[u])
        return depths[u]
    return max(depth(u) for u in range(circuit.n))


def make_data(config: dict) -> dict[str, list[CircuitExample]]:
    count = config["train_roots"]+2*config["val_roots"]
    assignments = assign_splits(count, config["data_seed"], config["train_roots"]/count, config["val_roots"]/count)
    data: dict[str, list[CircuitExample]] = {s: [] for s in ("train", "ordinary", "depth48")}
    def create(root_id: str, split: str, suite: str) -> None:
        seed = int.from_bytes(hashlib.sha256(f'{config["data_seed"]}:{root_id}'.encode()).digest()[:8], "big")
        circuit = (generate_deep_circuit(config["depth_shift_nodes"], seed, config["inputs"])
                   if suite == "depth48" else generate_circuit(config["nodes"], seed, config["inputs"]))
        order = list(range(circuit.n))
        random.Random(seed ^ 0xFACE).shuffle(order)
        data[suite].append(CircuitExample(circuit, root_id, 0, split, tuple(order), config["synthetic"]))
    for root, split in sorted(assignments.items()):
        if split != "test":
            create(f"static-circuit-v1-{root}", split, "train" if split == "train" else "ordinary")
    for i in range(config["val_roots"]):
        create(f"static-circuit-v1-depth48-{i:06d}", "val", "depth48")
    reject_duplicate_roots([e for es in data.values() for e in es])
    if len(data["train"]) != config["train_roots"] or any(len(data[s]) != config["val_roots"] for s in ("ordinary", "depth48")):
        raise ValueError("circuit split count mismatch")
    return data


def score(example: CircuitExample, predictions: list[int]) -> dict:
    if len(predictions) != example.circuit.n or any(type(v) is not int or v not in (0, 1) for v in predictions):
        raise ValueError("circuit predictions require N binary labels in presentation order")
    answer = evaluate(example.circuit)
    correct = [predictions[pos] == answer[u] for pos, u in enumerate(example.node_order)
               if example.circuit.operators[u] != Operator.INPUT]
    return {"exact_correct": all(correct), "node_accuracy": sum(correct)/len(correct),
            "scored_nodes": len(correct), "graph_depth": graph_depth(example.circuit)}


def metrics(rows: list[dict], config: dict) -> list[dict]:
    result = []
    for suite in ("ordinary", "depth48"):
        for k in config["budgets"]:
            selected = [r for r in rows if r["suite"] == suite and r["K"] == k]
            if not selected:
                raise ValueError("incomplete circuit evaluation")
            result.append({"suite": suite, "K": k, "roots": len(selected),
                "exact_correct": sum(r["exact_correct"] for r in selected),
                "exact_accuracy": sum(r["exact_correct"] for r in selected)/len(selected),
                "node_accuracy": sum(r["node_accuracy"] for r in selected)/len(selected),
                "depth_counts": dict(Counter(r["graph_depth"] for r in selected))})
    return result


def gate(aggregate: list[dict], steps: int, config: dict) -> dict:
    rows = [r for r in aggregate if r["suite"] == "ordinary"]
    if [r["K"] for r in rows] != config["budgets"]:
        raise ValueError("circuit gate requires complete budget curve")
    nondecreasing = all(a["exact_accuracy"] <= b["exact_accuracy"] for a, b in zip(rows, rows[1:]))
    competent = rows[-1]["exact_accuracy"] >= config["minimum_accuracy"]
    complete = steps == config["steps"]
    return {"nondecreasing": nondecreasing, "competent": competent, "training_complete": complete,
            "minimum_accuracy": config["minimum_accuracy"], "backbone_eligible": nondecreasing and competent and complete}


def require_maze_gate() -> dict:
    path = Path("runs/prompt05-scaled-check.json")
    checked = json.loads(path.read_text(encoding="utf-8-sig"))
    if not checked["verified"] or not checked["all_seeds_complete"] or checked["decision"] != "continue_step3":
        raise ValueError("complete, verified maze evaluation with >=2 passing backbones required")
    for arm in checked["arms"]:
        if arm["summary_sha256"] != file_hash(Path("runs/scaled_maze_v1") / f'seed-{arm["seed"]}' / "summary.json"):
            raise ValueError("step-2 evidence changed since checking")
    return checked


def run(config_path: Path, output_root: Path, seed: int) -> dict:
    require_maze_gate()
    config = json.loads(config_path.read_text())
    if seed not in config["seeds"] or config["batch_size"] < 64 or config["synthetic"] is not False:
        raise ValueError("circuit seed/batch/empirical protocol mismatch")
    capacity_path = Path(config["capacity_run"])
    capacity = json.loads((capacity_path / "profile.json").read_text())
    capacity_resources = json.loads((capacity_path / "resources.json").read_text())
    if (not capacity_resources["complete"] or capacity_resources["error"] is not None
            or not capacity["cpu_cuda_K1_gradient_parity"]
            or [row["K"] for row in capacity["rows"]] != config["budgets"]):
        raise ValueError("circuit training requires complete capacity and parity checks")
    transfer_provenance = {}
    if config.get("preload_training_batches", False):
        diagnostic = Path(config["transfer_diagnostic"])
        transfer = json.loads((diagnostic / "profile.json").read_text())
        transfer_resources = json.loads((diagnostic / "resources.json").read_text())
        if (not transfer_resources["complete"] or not transfer["prediction_and_parameter_tolerance_parity"]
                or not transfer["predicted_labels_identical"] or not transfer["cached_inputs_unchanged"]):
            raise ValueError("preloaded circuit training requires validated transfer diagnostic")
        transfer_provenance = {"transfer_profile_sha256": file_hash(diagnostic / "profile.json"),
                               "transfer_resources_sha256": file_hash(diagnostic / "resources.json")}
    for name in ("batch_size", "model_width", "heads", "inner_cycles"):
        if capacity[name] != config[name]:
            raise ValueError("circuit configuration differs from passed capacity profile")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / f"seed-{seed}"
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = random.Random(seed)
    with GPUJob(out, config["job_seconds"], "prompt05_static_circuit", synthetic=False) as job:
        write(out / "config.json", {**config, "seed": seed})
        shutil.copyfile("REPRODUCING.md", out / "preregistered-status.md")
        provenance = {**source_provenance(), "script_sha256": file_hash(__file__),
            "python": platform.python_version(), "torch": str(torch.__version__), "device": "cuda", "dtype": "float32",
            "tf32": False, "maze_gate_check_sha256": file_hash("runs/prompt05-scaled-check.json"),
            "capacity_profile_sha256": file_hash(capacity_path / "profile.json"),
            "capacity_resources_sha256": file_hash(capacity_path / "resources.json"), **transfer_provenance}
        for path in [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), Path("pyproject.toml")]:
            target = out / "source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        provenance["snapshot_hashes"] = {p.relative_to(out / "source").as_posix(): file_hash(p)
                                          for p in (out / "source").rglob("*") if p.is_file()}
        write(out / "provenance.json", provenance)
        data = make_data(config)
        payload = {s: [asdict(e) for e in es] for s, es in data.items()}
        data_path = output_root / "dataset.json"
        if data_path.exists():
            if json.loads(data_path.read_text()) != json.loads(json.dumps(payload)):
                raise ValueError("circuit data changed across seeds")
        else:
            write(data_path, payload)
        model = RecursiveSolver(width=config["model_width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
                                attention_mode=config["attention_mode"], domain=Domain.CIRCUIT).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        batches = [collate(data["train"][i:i+config["batch_size"]])[:2]
                   for i in range(0, len(data["train"]), config["batch_size"])]
        if any(obs.valid_nodes.shape[0] < 64 for obs, target in batches):
            raise ValueError("circuit training would use a batch below 64")
        if config.get("preload_training_batches", False):
            batches = [(obs.to("cuda"), target.to("cuda")) for obs, target in batches]
            for obs, target in batches:
                target.check_observation(obs)
        order, counts, batch_counts, calls = [], Counter(), Counter(), 0
        train_start = time.perf_counter()
        last_loss = None
        with (out / "steps.jsonl").open("x", encoding="utf-8") as handle:
            for step in range(1, config["steps"]+1):
                job.check_limit()
                if time.perf_counter()-train_start >= config["training_seconds"]:
                    break
                if not order:
                    order = list(range(len(batches)))
                    rng.shuffle(order)
                index, k = order.pop(), rng.choice(config["budgets"])
                obs, target = (batches[index] if config.get("preload_training_batches", False)
                               else tuple(value.to("cuda") for value in batches[index]))
                target.check_observation(obs)
                lr = config["learning_rate"] if step <= config["steps"]*.75 else config["final_learning_rate"]
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                result = model(obs, k)
                loss = circuit_value_loss(result.prediction.logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite circuit loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                counts[k] += 1
                batch_counts[index] += 1
                calls += result.block_calls
                last_loss = loss.item()
                row = {"step": step, "seed": seed, "K": k, "batch_index": index, "batch_size": obs.valid_nodes.shape[0],
                    "loss": last_loss, "gradient_norm": norm.item(), "lr": lr, "block_calls": result.block_calls,
                    "synthetic": False, "elapsed_s": time.perf_counter()-train_start}
                handle.write(json.dumps(row, allow_nan=False)+"\n")
                if step % 100 == 0:
                    handle.flush()
                    print(json.dumps(row), flush=True)
                del result, loss, obs, target
        steps = sum(counts.values())
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": rng.getstate(), "batch_order": order,
            "steps": steps, "config": {**config, "seed": seed}, "provenance": provenance,
            "dataset_sha256": file_hash(data_path)}, out / "checkpoint.pt")
        model.eval()
        checkpoint_hash, config_hash = file_hash(out / "checkpoint.pt"), file_hash(out / "config.json")
        rows = []
        with torch.no_grad(), (out / "validation.jsonl").open("x", encoding="utf-8") as handle:
            for suite in ("ordinary", "depth48"):
                for k in config["budgets"]:
                    for offset in range(0, len(data[suite]), config["batch_size"]):
                        job.check_limit()
                        batch = data[suite][offset:offset+config["batch_size"]]
                        obs, _, _ = collate(batch, "cuda")
                        result = model(obs, k)
                        for example, predictions in zip(batch, result.prediction.logits.argmax(-1).cpu().tolist()):
                            row = {"root_id": example.root_id, "suite": suite, "split": "val", "seed": seed, "K": k,
                                "synthetic": False, "record_kind": "static_development", "domain": "circuit", "policy": "restart",
                                "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash, "input_sha256": input_hash(example),
                                "block_calls_per_example": result.block_calls, "state_budget": result.state.budget,
                                "predictions": predictions, **score(example, predictions)}
                            handle.write(json.dumps(row, allow_nan=False)+"\n")
                            rows.append(row)
                        del result, obs
                    print(json.dumps({"seed": seed, "evaluation_suite": suite, "K": k,
                        "exact_correct": sum(r["exact_correct"] for r in rows if r["suite"] == suite and r["K"] == k)}), flush=True)
        aggregate = metrics(rows, config)
        summary = {"protocol_version": config["protocol_version"], "seed": seed, "synthetic": False,
            "optimizer_steps": steps, "final_loss": last_loss, "sampled_K_counts": dict(counts), "batch_counts": dict(batch_counts),
            "training_batched_F_calls": calls, "training_example_F_calls": calls*config["batch_size"],
            "training_transformer_layer_executions": 2*calls,
            "validation_example_F_calls": sum(r["block_calls_per_example"] for r in rows),
            "parameter_count": sum(p.numel() for p in model.parameters()), "validation": aggregate,
            "gate": gate(aggregate, steps, config), "dataset_sha256": file_hash(data_path),
            "artifact_hashes": {name: file_hash(out / name) for name in
                ("config.json", "checkpoint.pt", "provenance.json", "steps.jsonl", "validation.jsonl", "preregistered-status.md")}}
        write(out / "summary.json", summary)
        print(json.dumps({"seed": seed, "gate": summary["gate"], "validation": aggregate}), flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/static_circuit_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/static_circuit_v1"))
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.config, args.output, args.seed)
