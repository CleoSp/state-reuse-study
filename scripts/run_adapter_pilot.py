"""Run resumable validation-only pilot experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gc
import gzip
import json
from pathlib import Path
import shutil
import time
from typing import Callable

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.accounting.resources import memory_snapshot
from state_repair.data.challenges import sample_challenges
from state_repair.data.maze import MazeExample, collate
from state_repair.data.splits import canonical_hash
from state_repair.eval.crossover import impact_stratum
from state_repair.eval.frozen import frozen_adapter_prediction
from state_repair.models.adapters import StateAdapter, make_adapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import impact_metadata, score_policy, solve_maze
from state_repair.provenance import file_hash, source_provenance
from state_repair.train.adapter_step import FrozenPrior, one_edit_loss
from state_repair.train.pilot import AUXILIARY, LEARNED, PRINCIPAL, decode_example, make_payload, schedule
from state_repair.types import PredictionBatch
from state_repair.train.stream_step import stream_backward


def gpu_job(config: dict, out: Path, seconds: float, purpose: str, *, synthetic: bool):
    kwargs = {"authorization": config["gpu_authorization"]} if "gpu_authorization" in config else {}
    return GPUJob(out, seconds, purpose, synthetic=synthetic, **kwargs)


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def seal(out: Path) -> None:
    write(out / "manifest.json", {p.relative_to(out).as_posix(): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"})


def verify_seal(out: Path) -> None:
    hashes = read(out / "manifest.json")
    if not hashes or set(hashes) != {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}:
        raise ValueError(f"incomplete artifact manifest: {out}")
    for name, digest in hashes.items():
        if file_hash(out / name) != digest:
            raise ValueError(f"artifact hash mismatch: {out / name}")


def new_model(config: dict, weights: dict, track: str, device: str) -> RecursiveSolver:
    model = RecursiveSolver(width=config["width"], heads=config["heads"], inner_cycles=config["inner_cycles"],
                            attention_mode="masked_neighbor").to(device)
    model.load_state_dict(weights)
    return model.requires_grad_(track == "joint")


def adapter_for(arm: str, config: dict, *, spatial: StateAdapter | None = None, reset_rate: float = .5) -> StateAdapter:
    if arm.startswith("local_reset_"):
        return make_adapter("local_reset", radius=int(arm[-1]))
    if arm == "random_reset":
        return make_adapter(arm, reset_rate=reset_rate)
    if arm == "noisy_carry":
        return make_adapter(arm, noise_scale=config["noise_scale"])
    if arm == "shuffled_gate":
        return make_adapter(arm, spatial_gate=spatial)
    kwargs = {} if arm in ("restart", "carry") else {"width": config["width"]}
    if arm in LEARNED and arm != "answer_only":
        kwargs["context_width"] = config["context_width"]
    return make_adapter(arm, **kwargs)


def prepare(root: Path, config_path: Path) -> None:
    config = read(config_path)
    if config["synthetic"] or config["batch_size"] < 64:
        raise ValueError("empirical pilot protocol required")
    if file_hash(config["source_checkpoint"]) != config["source_checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    scaled = read("runs/prompt05-scaled-check.json")
    if not scaled["verified"] or not read("runs/frozen_crossover_v1/summary.json")["gate"]["crossover"]:
        raise ValueError("pilot needs verified static and crossover gates")
    if config.get("training_protocol") == "stream":
        if not read("runs/prompt05-pilot-check.json")["complete"]:
            raise ValueError("stream training needs completed pilot evidence")
        profile = read(Path(config["capacity_run"]) / "profile.json")
        if not profile["cpu_cuda_stream_gradient_parity"] or not profile["fits_12_hours"]:
            raise ValueError("stream protocol needs passing capacity and full-matrix projection")
    out = root / "prepared"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(out / "config.json", config)
    shutil.copyfile("REPRODUCING.md", out / "preregistered-status.md")
    shutil.copyfile("REPRODUCING.md", out / "approved-plan.md")
    for path in [*Path("src").rglob("*.py"), *Path("scripts").glob("*.py"), Path("pyproject.toml")]:
        target = out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    payload = make_payload(read(config["source_dataset"]), config)
    val = [decode_example(e) for e in payload["streams"][0]]
    challenge = sample_challenges(val, pairs_per_type=config["pairs_per_type"], seed=config["challenge_seed"],
                                  low_max=config["low_max"], high_min=config["high_min"])
    payload["challenge"] = [{"root": asdict(p.root), "edit_type": p.edit_type,
        "low": asdict(replace(p.root, maze=p.low, frame_index=1)),
        "high": asdict(replace(p.root, maze=p.high, frame_index=1))} for p in challenge.pairs]
    write(out / "dataset.json", payload)
    write(out / "challenge_audit.json", challenge.audit)
    write(out / "provenance.json", {**source_provenance(), "source_dataset_sha256": file_hash(config["source_dataset"]),
        "source_checkpoint_sha256": config["source_checkpoint_sha256"], "static_check_sha256": file_hash("runs/prompt05-scaled-check.json"),
        "crossover_sha256": file_hash("runs/frozen_crossover_v1/summary.json"), "device": "cpu", "torch": str(torch.__version__),
        "source_snapshot_hashes": {p.relative_to(out / "source").as_posix(): file_hash(p)
                                    for p in (out / "source").rglob("*") if p.is_file()}})
    write(out / "resources.json", {"wall_s": time.perf_counter()-started, "device": "cpu", "memory": memory_snapshot(),
                                   "synthetic": False, "external_cost_usd": 0})
    seal(out)
    print(json.dumps({"prepared": str(out), "challenge": challenge.audit}), flush=True)


def stream_training_batches(payload: dict, config: dict):
    b = config["batch_size"]
    frames = [[decode_example(e) for e in frame] for frame in [payload["train"], *payload["views"]]]
    if len(frames) != config["training_edits"] + 1 or len(frames[0]) % b:
        raise ValueError("stream training needs complete batches and the declared frame count")
    return [[collate(frame[i:i+b])[:2] for frame in frames] for i in range(0, len(frames[0]), b)]


def training_batches(payload: dict, config: dict):
    roots = [decode_example(e) for e in payload["train"]]
    b = config["batch_size"]
    old = [collate(roots[i:i+b])[:2] for i in range(0, len(roots), b)]
    new = []
    for view in payload["views"]:
        examples = [decode_example(e) for e in view]
        new.extend(collate(examples[i:i+b])[:2] for i in range(0, len(examples), b))
    if any(o.valid_nodes.shape[0] != b for o, t in [*old, *new]):
        raise ValueError("partial training batches are not authorized")
    return old, new


def build_cache(root: Path, config: dict, payload: dict, device: str = "cuda") -> None:
    out = root / "cache"
    with gpu_job(config, out, 900, "prompt05_pilot_shared_frozen_priors", synthetic=False) as job:
        weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
        model = new_model(config, weights, "frozen", device).eval()
        old, _ = training_batches(payload, config)
        manifest = []
        with torch.no_grad():
            for i, (cpu_obs, _) in enumerate(old):
                obs = cpu_obs.to(device)
                for k in config["budgets"]:
                    job.check_limit()
                    result = model(obs, k)
                    path = out / f"batch-{i}-K{k}.pt"
                    cache = FrozenPrior(cpu_obs, result.state.detach().to("cpu"),
                        PredictionBatch(result.prediction.logits.detach().cpu()), config["source_checkpoint_sha256"], "restart", k)
                    torch.save(cache, path)
                    manifest.append({"file": path.name, "sha256": file_hash(path), "batch": i, "K": k,
                        "root_ids": list(cpu_obs.episode_ids), "source_policy": "restart", "source_checkpoint_sha256": cache.checkpoint_sha256,
                        "input_sha256": [canonical_hash(decode_example(e).maze) for e in payload["train"][i*config["batch_size"]:(i+1)*config["batch_size"]]],
                        "block_calls_per_example": result.block_calls})
        write(out / "cache_index.json", {"entries": manifest, "dataset_sha256": file_hash(root / "prepared/dataset.json"),
            "synthetic": False, "construction_example_F_calls": sum(r["block_calls_per_example"]*len(r["root_ids"]) for r in manifest)})
    seal(out)


def frame_batches(payload: dict, edits: int, config: dict):
    examples = [[decode_example(e) for e in frame] for frame in payload["streams"][:edits+1]]
    b = config["batch_size"]
    return [[(frame[i:i+b], collate(frame[i:i+b])[0]) for i in range(0, len(frame), b)] for frame in examples]


def evaluation_summary(rows: list[dict]) -> dict:
    """Compact episode-level results; nodes/edits are never independent samples."""
    episodes = defaultdict(list)
    for r in rows:
        episodes[r["K"], r["root_id"]].append(r)
    episode_rows = []
    for (k, root), frames in sorted(episodes.items()):
        frames.sort(key=lambda x: x["frame"])
        if [r["frame"] for r in frames] != list(range(len(frames))):
            raise ValueError("duplicate or missing stream frame")
        record = {"K": k, "root_id": root, "initial_correct": frames[0]["route_correct"]}
        for end in (4, 32):
            if len(frames) > end:
                selected = frames[1:end+1]
                record[f"post{end}"] = sum(r["route_correct"] for r in selected)/end
                record[f"whole{end}"] = all(r["route_correct"] for r in frames[:end+1])
        episode_rows.append(record)
    metrics = []
    for k in sorted({r["K"] for r in rows}):
        for start, end in ((0, 0), (1, 4), (1, 32)):
            chosen = [r for r in rows if r["K"] == k and start <= r["frame"] <= end]
            roots = {r["root_id"] for r in chosen}
            if not chosen or len(chosen) != len(roots)*(end-start+1):
                continue
            metrics.append({"K": k, "frames": [start, end], "roots": len(roots), "predictions": len(chosen),
                **{m: sum(r[m] for r in chosen)/len(chosen) for m in ("route_correct", "all_node_correct", "valid_action_accuracy")},
                "unreachable_count": sum(r["unreachable"] for r in chosen),
                "unreachable_correct": sum(r["unreachable"] and r["route_correct"] for r in chosen),
                "trivial_count": sum(r["start_equals_goal"] for r in chosen),
                "reachable_nontrivial_count": sum(not r["unreachable"] and not r["start_equals_goal"] for r in chosen),
                "reachable_nontrivial_correct": sum(not r["unreachable"] and not r["start_equals_goal"] and r["route_correct"] for r in chosen)})
    retain = [r["mean_retention"] for r in rows if r["frame"] and r["mean_retention"] is not None]
    return {"episodes": episode_rows, "metrics": metrics,
            "mean_reset_rate": 1-sum(retain)/len(retain) if retain else None}


def evaluate_streams(model: RecursiveSolver, adapter: StateAdapter, batches: list, config: dict,
                     path: Path, identity: dict, check_limit: Callable[[], None], device: str) -> dict:
    rows = []
    model.eval()
    adapter.eval()
    with torch.no_grad(), gzip.open(path, "xt", encoding="utf-8") as handle, path.with_name("timing.jsonl").open("x", encoding="utf-8") as times:
        for k in config["budgets"]:

            torch.manual_seed(config["evaluation_seed"] + identity["seed"]*100 + k)
            for batch in range(len(batches[0])):
                policy = FixedBudgetPolicy(model, adapter, k)
                for frame in range(len(batches)):
                    check_limit()
                    examples, cpu_obs = batches[frame][batch]
                    if device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = policy(cpu_obs.to(device))
                    actions = result.prediction.logits.argmax(-1).cpu().tolist()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    deployment_s = time.perf_counter()-started
                    times.write(json.dumps({**identity, "K": k, "frame": frame, "batch": batch,
                        "batch_size": len(examples), "deployment_batch_seconds": deployment_s,
                        "stages_ms": result.milliseconds, "operations": result.operations,
                        "synthetic": config["synthetic"], "mode": "throughput_development", "includes_offline_oracle": False})+"\n")
                    retention = None
                    if result.adapter.retain_a is not None:
                        retention = ((result.adapter.retain_a+result.adapter.retain_z)/2).flatten(1).mean(1).cpu().tolist()
                    for j, (e, prediction) in enumerate(zip(examples, actions)):
                        prediction = prediction[:e.maze.n]
                        distance = solve_maze(e.maze)[0][e.maze.start]
                        row = {**identity, "root_id": e.root_id, "frame": frame, "K": k, "split": e.split,
                            "synthetic": e.synthetic, "record_kind": "fixed_budget_stream", "state_budget": result.state.budget,
                            "block_calls_per_example": result.block_calls, "input_sha256": canonical_hash(e.maze),
                            "actions": prediction, **score_policy(e.maze, prediction), "unreachable": distance < 0,
                            "start_equals_goal": e.maze.start == e.maze.goal,
                            "mean_retention": None if retention is None else retention[j]}
                        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                        rows.append({key: value for key, value in row.items() if key not in ("actions", "input_sha256")})
                    del result
                del policy
    return evaluation_summary(rows)


def train_one(root: Path, config: dict, payload: dict, old: list, new: list, tuning: list,
              caches: dict, weights: dict, track: str, arm: str, seed: int, grid: int, device: str = "cuda") -> None:
    out = root / "training" / f"{track}-{arm}-seed{seed}-grid{grid}"
    if out.exists():
        verify_seal(out)
        return
    out.parent.mkdir(exist_ok=True)
    with gpu_job(config, out, config["job_seconds"], "prompt05_adapter_training_and_tuning", synthetic=config["synthetic"]) as job:
        torch.manual_seed(seed)
        model = new_model(config, weights, track, device)
        adapter = adapter_for(arm, config).to(device)
        groups = []
        if track == "joint":
            groups.append({"params": list(model.parameters()), "lr": config["backbone_lrs"][grid]})
        if list(adapter.parameters()):
            groups.append({"params": list(adapter.parameters()), "lr": config["adapter_lrs"][grid]})
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        parameters = [p for group in groups for p in group["params"]]
        identity = {"track": track, "arm": arm, "seed": seed, "grid": grid}
        write(out / "config.json", {**config, **identity})
        rows = []
        started = time.perf_counter()
        with (out / "steps.jsonl").open("x", encoding="utf-8") as handle:
            for step, (index, k) in enumerate(schedule(seed, len(new), config), 1):
                job.check_limit()
                base = index % len(old)
                optimizer.zero_grad(set_to_none=True)
                if config.get("training_protocol") == "stream":
                    batch = [(o.to(device), t.to(device)) for o, t in new[index]]
                    frames = stream_backward(model, adapter, [o for o, _ in batch], [t for _, t in batch], k, track=track)
                    losses = [r.loss for r in frames]
                    values = {"loss": sum(losses)/len(losses), "initial_loss": losses[0],
                        "post_edit_loss": sum(losses[1:])/(len(losses)-1),
                        "initial_block_calls_executed": frames[0].block_calls,
                        "post_edit_block_calls": sum(r.block_calls for r in frames[1:]),
                        "frame_losses": losses, "frame_block_calls": [r.block_calls for r in frames],
                        "frames_per_step": len(frames), "state_source": "own_policy_same_K_detached_each_version"}
                    del batch, frames
                else:
                    previous, old_target = (value.to(device) for value in old[base])
                    current, target = (value.to(device) for value in new[index])
                    cache = None
                    if track == "frozen":
                        source = caches[base, k]
                        cache = FrozenPrior(previous, source.state.to(device), PredictionBatch(source.prediction.logits.to(device)),
                            source.checkpoint_sha256, source.source_policy, source.budget)


                        cache = replace(cache, observation=source.observation.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    result = one_edit_loss(model, adapter, previous, current, old_target, target, k,
                        track=track, cache=cache, checkpoint_sha256=config["source_checkpoint_sha256"])
                    result.loss.backward()
                    values = {"loss": result.loss.item(), "initial_loss": result.initial_loss.item(),
                        "post_edit_loss": result.post_edit_loss.item(),
                        "initial_block_calls_executed": result.initial_block_calls_executed,
                        "post_edit_block_calls": result.post_edit_block_calls}
                    del result, previous, current, target, old_target, cache
                def norm(module):
                    values = [p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.
                ag, bg = norm(adapter), norm(model)
                if list(adapter.parameters()) and not ag > 0:
                    raise ValueError("missing adapter gradient")
                total_norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                row = {**identity, "step": step, "batch_index": index, "base_batch": base, "K": k,
                    **values,
                    "adapter_gradient_norm": ag, "backbone_gradient_norm": bg, "gradient_norm": total_norm.item(),
                    "batch_size": config["batch_size"],
                    "elapsed_s": time.perf_counter()-started, "synthetic": config["synthetic"]}
                handle.write(json.dumps(row, allow_nan=False)+"\n")
                rows.append(row)
        train_seconds = time.perf_counter()-started
        torch.save({"model": model.state_dict(), "adapter": adapter.state_dict(), "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "schedule": schedule(seed, len(new), config), "steps": len(rows), "config": {**config, **identity},
            "dataset_sha256": file_hash(root / "prepared/dataset.json"), "prepared_manifest_sha256": file_hash(root / "prepared/manifest.json"),
            "cache_index_sha256": file_hash(root / "cache/cache_index.json") if track == "frozen" else None}, out / "checkpoint.pt")
        identity["checkpoint_sha256"] = file_hash(out / "checkpoint.pt")
        identity["config_sha256"] = file_hash(out / "config.json")
        summary = evaluate_streams(model, adapter, tuning, config, out / "tuning.jsonl.gz", identity, job.check_limit, device)
        write(out / "summary.json", {**identity, "synthetic": config["synthetic"], "optimizer_steps": len(rows), "training_seconds": train_seconds,
            "parameter_count": {"backbone": sum(p.numel() for p in model.parameters()), "adapter": sum(p.numel() for p in adapter.parameters())},
            "final_loss": rows[-1]["loss"], "first32_post_edit_loss": sum(r["post_edit_loss"] for r in rows[:32])/min(32, len(rows)),
            "last32_post_edit_loss": sum(r["post_edit_loss"] for r in rows[-32:])/min(32, len(rows)),
            "training_example_F_calls": sum(r["initial_block_calls_executed"]+r["post_edit_block_calls"] for r in rows)*config["batch_size"],
            "minimum_adapter_gradient": min(r["adapter_gradient_norm"] for r in rows), "validation": summary})
        del optimizer, parameters, groups, model, adapter
    seal(out)
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"completed": str(out), "training_seconds": train_seconds}), flush=True)


def select(root: Path, config: dict, *, persist: bool = True) -> dict:
    result = {}
    for track in config.get("tracks", ("frozen", "joint")):
        for arm in (LEARNED if track == "frozen" else PRINCIPAL):
            candidates = []
            for grid in range(len(config["adapter_lrs"])):
                summaries = []
                for seed in config["seeds"]:
                    out = root / "training" / f"{track}-{arm}-seed{seed}-grid{grid}"
                    verify_seal(out)
                    summary = read(out / "summary.json")
                    if summary["optimizer_steps"] != config["steps"]:
                        raise ValueError("incomplete pilot training")
                    summaries.append(summary)
                episodes = [e["post4"] for s in summaries for e in s["validation"]["episodes"]]
                candidates.append({"grid": grid, "score": sum(episodes)/len(episodes),
                    "run_summary_hashes": [file_hash(root / "training" / f"{track}-{arm}-seed{seed}-grid{grid}" / "summary.json") for seed in config["seeds"]]})
            chosen = max(candidates, key=lambda r: (r["score"], -r["grid"]))["grid"]
            result[f"{track}-{arm}"] = {"grid": chosen, "candidates": candidates}
    destination = root / "selection.json"
    if destination.exists() and read(destination) != result:
        raise ValueError("selection changed; do not overwrite")
    if persist:
        write(destination, result)
    return result


def load_selected(root: Path, config: dict, selection: dict, weights: dict, track: str, arm: str, seed: int, device: str):
    source_arm = "spatial_gate" if arm in AUXILIARY else arm
    if track == "frozen" and source_arm in ("restart", "carry"):
        model = new_model(config, weights, "frozen", device)
        adapter = adapter_for(arm, config).to(device)
        return model, adapter, {"checkpoint_sha256": config["source_checkpoint_sha256"], "grid": None, "reset_rate": None}
    grid = selection[f"{track}-{source_arm}"]["grid"]
    out = root / "training" / f"{track}-{source_arm}-seed{seed}-grid{grid}"
    checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = new_model(config, checkpoint["model"], "frozen", device)
    trained = adapter_for(source_arm, config)
    trained.load_state_dict(checkpoint["adapter"])
    rate = read(out / "summary.json")["validation"]["mean_reset_rate"]
    adapter = adapter_for(arm, config, spatial=trained, reset_rate=rate) if arm in AUXILIARY else trained
    return model, adapter.to(device), {"checkpoint_sha256": file_hash(out / "checkpoint.pt"), "grid": grid,
        "reset_rate": rate if arm == "random_reset" else None,
        "backbone_owner": source_arm, "auxiliary_initializer_control": arm in AUXILIARY}


def selected_streams(root: Path, config: dict, payload: dict, selection: dict, weights: dict, track: str, seed: int) -> None:
    batches = frame_batches(payload, config["stream_edits"], config)
    for arm in (*PRINCIPAL, *AUXILIARY):

        if track == "frozen" and arm in ("restart", "carry") and seed != config["seeds"][0]:
            continue
        out = root / "streams" / f"{track}-{arm}-seed{seed}"
        if out.exists():
            verify_seal(out)
            continue
        out.parent.mkdir(exist_ok=True)
        with gpu_job(config, out, config["job_seconds"], "prompt05_selected_32_edit_validation", synthetic=False) as job:
            model, adapter, identity = load_selected(root, config, selection, weights, track, arm, seed, "cuda")
            identity.update(track=track, arm=arm, seed=seed, config_sha256=file_hash(root / "prepared/config.json"))
            summary = evaluate_streams(model, adapter, batches, config, out / "predictions.jsonl.gz", identity, job.check_limit, "cuda")
            write(out / "summary.json", {**identity, "synthetic": False, "validation": summary})
            del model, adapter
        seal(out)
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"completed": str(out)}), flush=True)


def intervention_entries(payload: dict, config: dict) -> list[dict]:
    roots = {e["root_id"]: e for e in payload["streams"][0]}
    entries = [{"old": roots[e["root_id"]], "new": e, "suite": "ordinary", "branch": "uniform"}
               for e in payload["ordinary"]]
    for pair in payload["challenge"]:
        for branch in ("low", "high"):
            entries.append({"old": pair["root"], "new": pair[branch], "suite": "challenge", "branch": branch})
    for row in entries:
        old, new = decode_example(row["old"]), decode_example(row["new"])
        fraction = sum(impact_metadata(old.maze, new.maze)["action_set_changed"])/old.maze.n
        row.update(impact_fraction=fraction, stratum=impact_stratum(fraction, config),
                   edit_type="addition" if len(new.maze.edges) > len(old.maze.edges) else "removal")
    return entries


def run_interventions(root: Path, config: dict, payload: dict, selection: dict, weights: dict, track: str, seed: int) -> None:
    out = root / "interventions" / f"{track}-seed{seed}"
    if out.exists():
        verify_seal(out)
        return
    out.parent.mkdir(exist_ok=True)
    with gpu_job(config, out, config["job_seconds"], "prompt05_frozen_initializer_interventions", synthetic=False) as job:
        spatial_model, spatial_adapter, owner = load_selected(root, config, selection, weights, track, "spatial_gate", seed, "cuda")
        base = new_model(config, weights, "frozen", "cuda") if track == "frozen" else spatial_model
        base.eval()
        entries = intervention_entries(payload, config)
        prepared = []
        b = config["batch_size"]
        with torch.no_grad():
            for offset in range(0, len(entries), b):
                subset = entries[offset:offset+b]


                padded = subset + [subset[-1]]*(b-len(subset))
                old = collate([decode_example(e["old"]) for e in padded])[0]
                new = collate([decode_example(e["new"]) for e in padded])[0]
                job.check_limit()
                initial = base(old.to("cuda"), config["source_K"])
                prior_path = out / f"prior-batch{offset//b}.pt"
                source = FrozenPrior(old, initial.state.detach().to("cpu"), PredictionBatch(initial.prediction.logits.cpu()),
                    config["source_checkpoint_sha256"] if track == "frozen" else owner["checkpoint_sha256"], "restart", config["source_K"])
                torch.save(source, prior_path)
                prepared.append((subset, old, new, source, prior_path.name, file_hash(prior_path)))
            arms = (*PRINCIPAL, *AUXILIARY) if track == "frozen" else ("spatial_gate", "restart", "carry", *AUXILIARY)
            for arm in arms:
                if track == "frozen":
                    disposable, adapter, identity = load_selected(root, config, selection, weights, track, arm, seed, "cuda")
                    if any(not torch.equal(value, base.state_dict()[name]) for name, value in disposable.state_dict().items()):
                        raise ValueError("frozen adapter solver weights differ")
                    del disposable
                else:
                    rate = read(root / "training" / f"joint-spatial_gate-seed{seed}-grid{owner['grid']}" / "summary.json")["validation"]["mean_reset_rate"]
                    adapter = spatial_adapter if arm == "spatial_gate" else adapter_for(arm, config, spatial=spatial_adapter, reset_rate=rate).cuda()
                    identity = {**owner, "backbone_owner": "spatial_gate", "auxiliary_initializer_control": arm != "spatial_gate"}
                adapter.eval()
                identity.update(track=track, arm=arm, seed=seed, config_sha256=file_hash(root / "prepared/config.json"))
                with gzip.open(out / f"{arm}.jsonl.gz", "xt", encoding="utf-8") as handle:
                    for k in config["budgets"]:
                        torch.manual_seed(config["evaluation_seed"]+seed*100+k)
                        for subset, cpu_old, cpu_new, source, prior_file, prior_sha in prepared:
                            job.check_limit()
                            old, new = cpu_old.to("cuda"), cpu_new.to("cuda")
                            prior = source.state.to("cuda")
                            prediction = PredictionBatch(source.prediction.logits.cuda()) if adapter.requires_previous_prediction else None
                            result = frozen_adapter_prediction(base, adapter, old, new, prior, k, previous_prediction=prediction)
                            actions = result.solver.prediction.logits.argmax(-1).cpu().tolist()
                            retention = None
                            if config.get("training_protocol") == "stream" and result.adapter.retain_a is not None:
                                retention = ((result.adapter.retain_a + result.adapter.retain_z)/2).squeeze(-1).cpu().tolist()
                            for i, entry in enumerate(subset):
                                e = decode_example(entry["new"])
                                distance = solve_maze(e.maze)[0][e.maze.start]
                                row = {**identity, **{n: v for n, v in entry.items() if n not in ("old", "new")},
                                    "root_id": e.root_id, "split": "val", "synthetic": False, "privileged": False,
                                    "record_kind": "frozen_state_intervention", "K": k, "source_K": config["source_K"],
                                    "state_budget": result.solver.state.budget, "block_calls_per_example": result.solver.block_calls,
                                    "source_policy": "restart", "prior_file": prior_file, "prior_file_sha256": prior_sha,
                                    "prior_index": i, "source_checkpoint_sha256": source.checkpoint_sha256,
                                    "input_sha256": canonical_hash(e.maze), "actions": actions[i], **score_policy(e.maze, actions[i]),
                                    "unreachable": distance < 0, "start_equals_goal": e.maze.start == e.maze.goal}
                                if config.get("training_protocol") == "stream":
                                    row["node_retention"] = None if retention is None else retention[i]
                                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
                            del result, old, new, prior, prediction
                del adapter
        write(out / "summary.json", {"track": track, "seed": seed, "arms": list(arms), "entries": len(entries),
            "synthetic": False, "source_K": config["source_K"], "padded_batch_size": b,
            "source_example_F_calls_executed": len(prepared)*b*config["source_K"]*(config["inner_cycles"]+1),
            "post_edit_example_F_calls_executed": len(arms)*len(prepared)*b*sum(config["budgets"])*(config["inner_cycles"]+1)})
    seal(out)
    print(json.dumps({"completed": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "cache", "train", "select", "streams", "interventions"))
    parser.add_argument("--root", type=Path, default=Path("runs/adapter_pilot_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/adapter_pilot_v1.json"))
    parser.add_argument("--track", choices=("frozen", "joint"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = read(args.config)
    if config.get("training_protocol") == "stream":
        if args.stage == "cache" or args.track == "frozen":
            parser.error("stream v2 is joint-only and cannot use frozen caches")
        if config["training_edits"] != 4 or config["training_views"] != 4 or config["tracks"] != ["joint"]:
            raise ValueError("stream v2 requires four edits and joint-only training")
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == "prepare":
        prepare(args.root, args.config)
        return
    verify_seal(args.root / "prepared")
    if read(args.root / "prepared/config.json") != config:
        raise ValueError("configuration changed after data freeze")
    for name, digest in read(args.root / "prepared/provenance.json")["source_snapshot_hashes"].items():
        if file_hash(name) != digest:
            raise ValueError(f"executable source changed after preparation: {name}")
    payload = read(args.root / "prepared/dataset.json")
    if args.stage == "cache":
        build_cache(args.root, config, payload)
        return
    if args.stage == "select":
        print(json.dumps(select(args.root, config)), flush=True)
        return
    if args.track is None or args.seed not in config["seeds"]:
        parser.error("this stage requires --track and a declared --seed")
    weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
    if args.stage == "train":
        if config.get("training_protocol") == "stream":
            new = stream_training_batches(payload, config)
            old = [batch[0] for batch in new]
        else:
            old, new = training_batches(payload, config)
        tuning = frame_batches(payload, config["tuning_edits"], config)
        caches = {}
        if args.track == "frozen":
            verify_seal(args.root / "cache")
            for row in read(args.root / "cache/cache_index.json")["entries"]:
                value = torch.load(args.root / "cache" / row["file"], weights_only=False)
                value.check(old[row["batch"]][0], config["source_checkpoint_sha256"], row["K"])
                caches[row["batch"], row["K"]] = value
        for arm in (LEARNED if args.track == "frozen" else PRINCIPAL):
            for grid in range(2):
                train_one(args.root, config, payload, old, new, tuning, caches, weights, args.track, arm, args.seed, grid)
    elif args.stage == "streams":
        selected_streams(args.root, config, payload, read(args.root / "selection.json"), weights, args.track, args.seed)
    elif args.stage == "interventions":
        run_interventions(args.root, config, payload, read(args.root / "selection.json"), weights, args.track, args.seed)


if __name__ == "__main__":
    main()
