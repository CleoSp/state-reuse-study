"""Joint circuit stream replication using the shared version-detached trainer."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gc
import gzip
from pathlib import Path
import shutil
import time

import torch

from run_adapter_pilot import gpu_job, json_rows, read, seal, verify_seal, write
from train_static_circuit import input_hash, score
from state_repair.data.circuit import Circuit, CircuitExample, Operator, collate, generate_episode, reject_duplicate_roots
from state_repair.models.adapters import make_adapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.pilot import child_seed, schedule
from state_repair.train.stream_step import stream_backward
from state_repair.types import Domain


def decode(row: dict) -> CircuitExample:
    c = row["circuit"]
    return CircuitExample(Circuit(tuple(Operator(o) for o in c["operators"]),
        tuple(tuple(p) for p in c["parents"]), tuple(c["input_bits"])),
        row["root_id"], row["frame_index"], row["split"], tuple(row["node_order"]), row["synthetic"])


def stream_frames(roots: list[CircuitExample], edits: int, seed: int) -> list[list[CircuitExample]]:
    episodes = [[replace(e, node_order=root.node_order) for e in generate_episode(root.circuit,
        root.root_id, root.split, edits, child_seed(seed, root.root_id), synthetic=root.synthetic)] for root in roots]
    return [[episode[f] for episode in episodes] for f in range(edits+1)]


def make_payload(source: dict, config: dict) -> dict:
    train, val = [decode(e) for e in source["train"]], [decode(e) for e in source["ordinary"]]
    if any(e.split != "train" or e.frame_index for e in train) or any(e.split != "val" or e.frame_index for e in val):
        raise ValueError("base roots must be split before stream generation")
    if any(e.synthetic != config["synthetic"] for e in train+val):
        raise ValueError("synthetic flag mismatch")
    reject_duplicate_roots(train+val)
    return {"train_frames": [[asdict(e) for e in f] for f in stream_frames(train, 4, config["edit_seed"])],
        "streams": [[asdict(e) for e in f] for f in stream_frames(val, 32, config["stream_seed"])],
        "ordinary": [[asdict(e) for e in f] for f in stream_frames(val, 1, config["intervention_seed"])]}


def batches(raw_frames: list[list[dict]], config: dict, edits: int) -> list:
    frames = [[decode(e) for e in f] for f in raw_frames[:edits+1]]
    b = config["batch_size"]
    if len(frames) != edits+1 or len(frames[0]) % b:
        raise ValueError("require full declared frame and batch coverage")
    result = []
    for index in range(0, len(frames[0]), b):
        stream = []
        for f, frame in enumerate(frames):
            examples = frame[index:index+b]
            previous = None if f == 0 else [e.circuit for e in frames[f-1][index:index+b]]
            obs, target, _ = collate(examples, previous=previous)
            stream.append((examples, obs, target))
        result.append(stream)
    return result


def new_model(config: dict, weights: dict, device: str, *, training: bool = True) -> RecursiveSolver:
    model = RecursiveSolver(width=config["width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
        domain=Domain.CIRCUIT, attention_mode="masked_neighbor").to(device)
    model.load_state_dict(weights)
    return model.requires_grad_(training)


def adapter_for(arm: str, config: dict):
    kwargs = {} if arm in ("restart", "carry") else {"width": config["width"], "domain": Domain.CIRCUIT}
    if arm == "spatial_gate":
        kwargs["context_width"] = config["context_width"]
    return make_adapter(arm, **kwargs)


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["K"], row["root_id"]].append(row)
    episodes = []
    for (k, root), frames in sorted(grouped.items()):
        frames.sort(key=lambda r: r["frame"])
        if [r["frame"] for r in frames] != list(range(len(frames))):
            raise ValueError("missing/duplicate circuit stream frame")
        row = {"K": k, "root_id": root, "initial_correct": frames[0]["exact_correct"]}
        for end in (4, 32):
            if len(frames) > end:
                row[f"post{end}"] = sum(r["exact_correct"] for r in frames[1:end+1])/end
                row[f"node{end}"] = sum(r["node_accuracy"] for r in frames[1:end+1])/end
                row[f"whole{end}"] = all(r["exact_correct"] for r in frames[:end+1])
        episodes.append(row)
    per_frame = []
    for k, f in sorted({(r["K"], r["frame"]) for r in rows}):
        chosen = [r for r in rows if r["K"] == k and r["frame"] == f]
        per_frame.append({"K": k, "frame": f, "roots": len(chosen),
            "exact_accuracy": sum(r["exact_correct"] for r in chosen)/len(chosen),
            "node_accuracy": sum(r["node_accuracy"] for r in chosen)/len(chosen)})
    return {"episodes": episodes, "per_frame": per_frame}


def evaluate_streams(model, adapter, data: list, config: dict, out: Path, identity: dict, job, device: str) -> dict:
    rows = []
    model.eval(); adapter.eval()
    with torch.no_grad(), gzip.open(out, "xt", encoding="utf-8") as handle:
        for k in config["budgets"]:
            torch.manual_seed(config["evaluation_seed"]+identity["seed"]*100+k)
            for stream in data:
                policy = FixedBudgetPolicy(model, adapter, k)
                for examples, obs, _ in stream:
                    job.check_limit()
                    result = policy(obs.to(device))
                    actions = result.prediction.logits.argmax(-1).cpu().tolist()
                    for e, a in zip(examples, actions):
                        a = a[:e.circuit.n]
                        row = {**identity, "K": k, "frame": e.frame_index, "root_id": e.root_id,
                            "synthetic": e.synthetic, "split": e.split, "record_kind": "fixed_budget_stream",
                            "state_budget": result.state.budget, "block_calls_per_example": result.block_calls,
                            "input_sha256": input_hash(e), "actions": a, **score(e, a)}
                        handle.write(__import__("json").dumps(row, allow_nan=False)+"\n")
                        rows.append(row)
                policy.reset()
    return summarize(rows)


def prepare(root: Path, config_path: Path) -> None:
    config = read(config_path)
    if config["synthetic"] or config["batch_size"] < 64:
        raise ValueError("empirical circuit protocol required")
    if file_hash(config["source_checkpoint"]) != config["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    if not read("runs/static_circuit_v12/seed-29/summary.json")["gate"]["backbone_eligible"]:
        raise ValueError("passing circuit source required")
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    write(out / "config.json", config)
    shutil.copyfile("REPRODUCING.md", out / "preregistered-status.md")
    shutil.copyfile("REPRODUCING.md", out / "approved-plan.md")
    for p in [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py")]:
        target = out / "source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write(out / "dataset.json", make_payload(read(config["source_dataset"]), config))
    write(out / "provenance.json", {**source_provenance(), "source_dataset_sha256": file_hash(config["source_dataset"]),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"],
        "source_snapshot_hashes": {p.relative_to(out / "source").as_posix(): file_hash(p)
            for p in (out / "source").rglob("*.py")}})
    seal(out)


def train_one(root: Path, config: dict, train: list, tune: list, weights: dict, arm: str, seed: int, grid: int,
              device: str = "cuda") -> None:
    out = root / "training" / f"joint-{arm}-seed{seed}-grid{grid}"
    if out.exists():
        verify_seal(out)
        return
    out.parent.mkdir(exist_ok=True)
    with gpu_job(config, out, config["job_seconds"], "prompt05b_circuit_training_tuning", synthetic=config["synthetic"]) as job:
        torch.manual_seed(seed)
        model, adapter = new_model(config, weights, device), adapter_for(arm, config).to(device)
        groups = [{"params": list(model.parameters()), "lr": config["backbone_lrs"][grid]}]
        if list(adapter.parameters()):
            groups.append({"params": list(adapter.parameters()), "lr": config["adapter_lrs"][grid]})
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        params = [p for group in groups for p in group["params"]]
        identity = {"track": "joint", "arm": arm, "seed": seed, "grid": grid}
        write(out / "config.json", {**config, **identity})
        rows = []
        started = time.perf_counter()
        with (out / "steps.jsonl").open("x", encoding="utf-8") as handle:
            for step, (index, k) in enumerate(schedule(seed, len(train), config), 1):
                job.check_limit()
                optimizer.zero_grad(set_to_none=True)
                data = [(o.to(device), t.to(device)) for _, o, t in train[index]]
                frames = stream_backward(model, adapter, [o for o,t in data], [t for o,t in data], k, track="joint")
                def norm(module):
                    return sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
                ag, bg = norm(adapter), norm(model)
                if list(adapter.parameters()) and not ag > 0:
                    raise ValueError("missing adapter gradient")
                total = torch.nn.utils.clip_grad_norm_(params, config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                losses = [f.loss for f in frames]
                row = {**identity, "step": step, "batch_index": index, "K": k, "frame_losses": losses,
                    "frame_block_calls": [f.block_calls for f in frames], "loss": sum(losses)/len(losses),
                    "frames_per_step": len(frames), "adapter_gradient_norm": ag, "backbone_gradient_norm": bg,
                    "gradient_norm": float(total), "batch_size": config["batch_size"],
                    "elapsed_s": time.perf_counter()-started, "synthetic": config["synthetic"]}
                handle.write(__import__("json").dumps(row, allow_nan=False)+"\n")
                rows.append(row)
                del data, frames
        training_seconds = time.perf_counter()-started
        torch.save({"model": model.state_dict(), "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "schedule": schedule(seed, len(train), config), "steps": len(rows), "config": {**config, **identity},
            "dataset_sha256": file_hash(root / "prepared/dataset.json"),
            "prepared_manifest_sha256": file_hash(root / "prepared/manifest.json")}, out / "checkpoint.pt")
        identity.update(checkpoint_sha256=file_hash(out / "checkpoint.pt"), config_sha256=file_hash(out / "config.json"))
        summary = evaluate_streams(model, adapter, tune, config, out / "tuning.jsonl.gz", identity, job, device)
        write(out / "summary.json", {**identity, "synthetic": config["synthetic"], "optimizer_steps": len(rows),
            "training_seconds": training_seconds, "validation": summary,
            "training_example_F_calls": sum(sum(r["frame_block_calls"]) for r in rows)*config["batch_size"],
            "parameter_count": {"backbone": sum(p.numel() for p in model.parameters()), "adapter": sum(p.numel() for p in adapter.parameters())}})
    seal(out)
    del optimizer, params, groups, model, adapter
    gc.collect(); torch.cuda.empty_cache()
    print(out, flush=True)


def select(root: Path, config: dict, persist: bool = True) -> dict:
    result = {}
    for arm in config["arms"]:
        candidates = []
        for grid in range(2):
            vals = []
            for seed in config["seeds"]:
                out = root / "training" / f"joint-{arm}-seed{seed}-grid{grid}"
                verify_seal(out)
                summary = read(out / "summary.json")
                if summary["optimizer_steps"] != config["steps"]:
                    raise ValueError("incomplete optimization")
                vals.extend(r["post4"] for r in summary["validation"]["episodes"])
            candidates.append({"grid": grid, "score": sum(vals)/len(vals)})
        result[arm] = {"grid": max(candidates, key=lambda r: (r["score"], -r["grid"]))["grid"], "candidates": candidates}
    if persist:
        if (root / "selection.json").exists() and read(root / "selection.json") != result:
            raise ValueError("selection changed")
        write(root / "selection.json", result)
    return result


def load_selected(root: Path, config: dict, arm: str, seed: int, device: str):
    grid = read(root / "selection.json")[arm]["grid"]
    path = root / "training" / f"joint-{arm}-seed{seed}-grid{grid}" / "checkpoint.pt"
    cp = torch.load(path, map_location="cpu", weights_only=False)
    model, adapter = new_model(config, cp["model"], device, training=False), adapter_for(arm, config).to(device)
    adapter.load_state_dict(cp["adapter"])
    return model, adapter, {"track": "joint", "arm": arm, "seed": seed, "grid": grid,
        "checkpoint_sha256": file_hash(path), "config_sha256": file_hash(root / "prepared/config.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "train", "select", "streams"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root, config_path = Path("runs/circuit_stream_v2"), Path("configs/circuit_stream_v2.json")
    config = read(config_path)
    torch.set_num_threads(config["threads"]); torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == "prepare":
        prepare(root, config_path)
        return
    verify_seal(root / "prepared")
    if read(root / "prepared/config.json") != config:
        raise ValueError("configuration changed after preparation")
    for path, digest in read(root / "prepared/provenance.json")["source_snapshot_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError(f"source changed: {path}")
    if args.stage == "select":
        select(root, config)
        return
    if args.seed not in config["seeds"]:
        parser.error("declared seed required")
    payload = read(root / "prepared/dataset.json")
    if args.stage == "train":
        if not read(Path(config["capacity_run"]) / "profile.json")["passed"]:
            raise ValueError("circuit capacity required")
        train, tune = batches(payload["train_frames"], config, 4), batches(payload["streams"], config, 4)
        weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
        for arm in config["arms"]:
            for grid in range(2):
                train_one(root, config, train, tune, weights, arm, args.seed, grid)
    else:
        data = batches(payload["streams"], config, 32)
        for arm in config["arms"]:
            out = root / "streams" / f"joint-{arm}-seed{args.seed}"
            if out.exists():
                verify_seal(out)
                continue
            out.parent.mkdir(exist_ok=True)
            with gpu_job(config, out, config["job_seconds"], "prompt05b_circuit_32_edit_validation", synthetic=False) as job:
                model, adapter, identity = load_selected(root, config, arm, args.seed, "cuda")
                summary = evaluate_streams(model, adapter, data, config, out / "predictions.jsonl.gz", identity, job, "cuda")
                write(out / "summary.json", {**identity, "synthetic": False, "validation": summary})
                del model, adapter
            seal(out)


if __name__ == "__main__":
    main()
