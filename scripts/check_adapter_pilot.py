"""Independent raw-action scoring, coverage, training provenance and pilot decision."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch

from state_repair.data.splits import canonical_hash
from state_repair.data.maze import Maze
from state_repair.oracles.maze import score_policy, solve_maze
from state_repair.provenance import file_hash
from state_repair.train.pilot import AUXILIARY, LEARNED, PRINCIPAL, decode_example, make_payload, schedule
from run_adapter_pilot import (adapter_for, evaluation_summary, intervention_entries, json_rows,
                               read, select, verify_seal, write)




input_hash = lru_cache(maxsize=20000)(canonical_hash)


@lru_cache(maxsize=20000)
def start_distance(maze: Maze) -> int:
    return solve_maze(maze)[0][maze.start]


@lru_cache(maxsize=65536)
def exact_score(maze: Maze, actions: tuple[int, ...]) -> tuple:
    return tuple(score_policy(maze, actions).items())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_resource(out: Path) -> None:
    resource = read(out / "resources.json")
    require(resource["complete"] and not resource["synthetic"], f"incomplete/synthetic job: {out}")
    require(0 < resource["wall_s"] <= resource["reservation_seconds"], "GPU job exceeded reservation")
    events = list(json_rows(Path(resource.get("gpu_time_journal", "runs/prompt05_gpu_time.jsonl"))))
    actual = [r for r in events if r["job_id"] == resource["job_id"] and r["kind"] == "actual"]
    require(len(actual) == 1 and actual[0]["seconds"] == resource["wall_s"], "resource journal mismatch")


def check_stream_file(path: Path, config: dict, examples: dict, expected_frames: int,
                      identity: dict) -> tuple[dict, int]:
    seen, compact = set(), []
    for row in json_rows(path):
        key = row["K"], row["frame"], row["root_id"]
        require(key not in seen, "duplicate stream prediction")
        seen.add(key)
        require(row["synthetic"] is False and row["split"] == "val" and row["record_kind"] == "fixed_budget_stream", "invalid empirical record")
        require(all(row.get(k) == v for k, v in identity.items()), "run identity mismatch")
        require(row["state_budget"] == row["K"] and row["block_calls_per_example"] == row["K"]*(config["inner_cycles"]+1), "wrong stream budget/cost")
        e = examples[row["frame"], row["root_id"]]
        require(row["input_sha256"] == input_hash(e.maze), "stream input hash mismatch")
        require(all(row[k] == v for k, v in exact_score(e.maze, tuple(row["actions"]))), "raw stream prediction score mismatch")
        distance = start_distance(e.maze)
        require(row["unreachable"] == (distance < 0) and row["start_equals_goal"] == (e.maze.start == e.maze.goal), "denominator mismatch")
        retain = row["mean_retention"]
        require(retain is None or math.isfinite(retain) and 0 <= retain <= 1, "invalid retention diagnostic")
        compact.append({k: v for k, v in row.items() if k not in ("actions", "input_sha256")})
    roots = {root for frame, root in examples if frame == 0}
    expected = {(k, f, root) for k in config["budgets"] for f in range(expected_frames+1) for root in roots}
    require(seen == expected, "missing/extra stream prediction")
    return evaluation_summary(compact), len(seen)


def stop_screen(curves: dict, config: dict) -> dict:
    """Pointwise exploratory root bootstrap; average seeds within each root.

    Training seeds are retained as separate curves in source summaries. This
    interval measures example uncertainty conditional on these seeds only.
    """
    comparisons, dominated = [], []
    for track in config.get("tracks", ("frozen", "joint")):
        roots = sorted(curves[track, "spatial_gate", config["seeds"][0], config["budgets"][0]])
        draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(roots),
            size=(config["bootstrap_repetitions"], len(roots)))
        def values(arm, k):
            seeds = [config["seeds"][0]] if track == "frozen" and arm in ("restart", "carry") else config["seeds"]
            return np.array([[curves[track, arm, seed, k][r] for r in roots] for seed in seeds]).mean(0)
        for k in config["budgets"]:
            spatial = values("spatial_gate", k)
            beaten = False
            for baseline in ("restart", "carry"):
                for baseline_k in config["budgets"]:
                    if baseline_k > k:
                        continue
                    delta = values(baseline, baseline_k)-spatial
                    interval = np.quantile(delta[draws].mean(1), [.025, .975]).tolist()
                    comparisons.append({"track": track, "spatial_K": k, "baseline": baseline, "baseline_K": baseline_k,
                        "baseline_minus_spatial": float(delta.mean()), "ci": interval, "roots": len(roots)})
                    beaten |= interval[0] > 0
            dominated.append({"track": track, "K": k, "dominated": beaten})
    stop = all(row["dominated"] for row in dominated)
    return {"decision": "stop" if stop else "proceed_to_cost_evaluation", "scientific_stop": stop,
        "criterion": "every spatial K beaten by restart/carry at same/lower K in both tracks; positive lower 95% paired-root CI",
        "metric": "32-edit mean post-edit route accuracy", "interval_scope": "exploratory pointwise test-root variability conditional on training seeds; validation only",
        "dominance": dominated, "comparisons": comparisons}


def stream_gates(curves: dict, config: dict) -> dict:
    """Paired roots retain full sequences and all seeds; pointwise validation CIs."""
    roots = sorted(curves["joint", "restart", config["seeds"][0], config["budgets"][0]])
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(roots),
        size=(config["bootstrap_repetitions"], len(roots)))
    contrasts = []
    for arm, baseline in [(a, "answer_only") for a in PRINCIPAL if a not in ("restart", "answer_only")] + [("answer_only", "restart")]:
        for k in config["budgets"]:
            seed_deltas = []
            for seed in config["seeds"]:
                a, b = curves["joint", arm, seed, k], curves["joint", baseline, seed, k]
                require(sorted(a) == sorted(b) == roots, "unpaired gate roots")
                seed_deltas.append(np.array([a[r]-b[r] for r in roots]))
            delta = np.mean(seed_deltas, axis=0)
            contrasts.append({"arm": arm, "baseline": baseline, "K": k, "delta": float(delta.mean()),
                "ci": np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                "per_seed_delta": [float(d.mean()) for d in seed_deltas], "roots": len(roots)})
    candidates = sorted({r["arm"] for r in contrasts if r["baseline"] == "answer_only" and r["ci"][0] > 0})
    output = [r for r in contrasts if r["arm"] == "answer_only" and r["K"] in (1, 2, 4)]
    return {"G2_latent_reuse_reopened": bool(candidates), "candidates": candidates,
        "G3_output_reuse_all_K124": all(r["ci"][0] > 0 for r in output), "contrasts": contrasts,
        "G1": "action-dynamics diagnostic reported separately",
        "interval_scope": "pointwise exploratory paired-root variability conditional on three training seeds"}


def check_pilot(root: Path, *, complete: bool = True) -> dict:
    torch.set_num_threads(2)
    verify_seal(root / "prepared")
    config, payload = read(root / "prepared/config.json"), read(root / "prepared/dataset.json")
    require(config["synthetic"] is False, "synthetic pilot excluded")
    require(file_hash(config["source_checkpoint"]) == config["source_checkpoint_sha256"], "backbone hash mismatch")
    require(file_hash(config["source_dataset"]) == read(root / "prepared/provenance.json")["source_dataset_sha256"], "source roots changed")
    regenerated = json.loads(json.dumps(make_payload(read(config["source_dataset"]), config)))
    require(all(payload[k] == v for k, v in regenerated.items()), "data generation replay mismatch")
    examples = {(e["frame_index"], e["root_id"]): decode_example(e) for frame in payload["streams"] for e in frame}
    source = torch.load(config["source_checkpoint"], weights_only=False, map_location="cpu")["model"]
    if config.get("training_protocol") != "stream":
        verify_seal(root / "cache")
        check_resource(root / "cache")
        cache_index = read(root / "cache/cache_index.json")
        require(cache_index["dataset_sha256"] == file_hash(root / "prepared/dataset.json"), "cache dataset mismatch")
        require({(r["batch"], r["K"]) for r in cache_index["entries"]} ==
                {(i, k) for i in range(len(payload["train"])//config["batch_size"]) for k in config["budgets"]}, "cache coverage mismatch")
        for row in cache_index["entries"]:
            require(file_hash(root / "cache" / row["file"]) == row["sha256"], "cache hash mismatch")
            cache = torch.load(root / "cache" / row["file"], weights_only=False, map_location="cpu")
            cache.check(cache.observation, config["source_checkpoint_sha256"], row["K"])
            require(list(cache.observation.episode_ids) == row["root_ids"], "cache lineage mismatch")
    counts = {"training_runs": 0, "tuning_predictions": 0, "stream_predictions": 0, "intervention_predictions": 0}
    summaries = []
    expected_names = {f"{track}-{arm}-seed{seed}-grid{grid}" for track in config.get("tracks", ("frozen", "joint"))
        for arm in (LEARNED if track == "frozen" else PRINCIPAL) for seed in config["seeds"] for grid in range(2)}
    paths = sorted((root / "training").glob("*"))
    require(bool(paths), "no completed pilot runs")
    require(all(p.name in expected_names for p in paths), "unexpected pilot arm")
    if complete:
        require({p.name for p in paths} == expected_names, "incomplete training matrix")
    for out in paths:
        verify_seal(out)
        check_resource(out)
        summary, run_config = read(out / "summary.json"), read(out / "config.json")
        track, arm, seed, grid = (summary[k] for k in ("track", "arm", "seed", "grid"))
        require(run_config == {**config, "track": track, "arm": arm, "seed": seed, "grid": grid}, "run configuration differs")
        steps = list(json_rows(out / "steps.jsonl"))
        require(len(steps) == summary["optimizer_steps"] == config["steps"], "incomplete optimization")
        expected_schedule = schedule(seed, len(payload["train"])*(1 if config.get("training_protocol") == "stream" else config["training_views"])//config["batch_size"], config)
        require([(r["batch_index"], r["K"]) for r in steps] == expected_schedule, "unpaired training schedule")
        for i, row in enumerate(steps, 1):
            require(row["step"] == i and row["synthetic"] is False, "step identity mismatch")
            require(all(math.isfinite(row[k]) for k in ("loss", "initial_loss", "post_edit_loss", "adapter_gradient_norm", "backbone_gradient_norm", "gradient_norm")), "nonfinite training")
            if config.get("training_protocol") == "stream":
                n = config["training_edits"] + 1
                require(row["frames_per_step"] == n and len(row["frame_losses"]) == n, "wrong training stream length")
                require(all(math.isfinite(v) for v in row["frame_losses"]), "nonfinite frame loss")
                require(math.isclose(row["loss"], sum(row["frame_losses"])/n, abs_tol=1e-7), "unequal frame weights")
                require(row["initial_loss"] == row["frame_losses"][0], "initial frame loss mismatch")
                require(math.isclose(row["post_edit_loss"], sum(row["frame_losses"][1:])/(n-1)), "post-edit mean mismatch")
                require(row["frame_block_calls"] == [row["K"]*(config["inner_cycles"]+1)]*n, "wrong per-frame budget")
                require(row["state_source"] == "own_policy_same_K_detached_each_version", "wrong training prior contract")
            else:
                require(math.isclose(row["loss"], (row["initial_loss"]+row["post_edit_loss"])/2, abs_tol=1e-6), "wrong objective")
            require(row["adapter_gradient_norm"] > 0 if arm in LEARNED else row["adapter_gradient_norm"] == 0, "broken adapter gradients")
            require(row["backbone_gradient_norm"] > 0 if track == "joint" else row["backbone_gradient_norm"] == 0, "backbone gradient contract")
            require(row["post_edit_block_calls"] == row["K"]*(config["inner_cycles"]+1)*(config["training_edits"] if config.get("training_protocol") == "stream" else 1), "training block count")
            require(row["initial_block_calls_executed"] == (row["K"]*(config["inner_cycles"]+1) if track == "joint" else 0), "cached/joint source mismatch")
        cp = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        require(cp["steps"] == config["steps"] and cp["schedule"] == expected_schedule and cp["config"] == run_config, "checkpoint training provenance")
        require(cp["dataset_sha256"] == file_hash(root / "prepared/dataset.json") and
                cp["prepared_manifest_sha256"] == file_hash(root / "prepared/manifest.json"), "checkpoint dataset/source binding")
        require(all(torch.isfinite(v).all().item() for key in ("model", "adapter") for v in cp[key].values()), "nonfinite checkpoint")
        require(all(int(v["step"]) == config["steps"] for v in cp["optimizer"]["state"].values()), "optimizer step count mismatch")
        expected_lrs = ([config["backbone_lrs"][grid]] if track == "joint" else []) + ([config["adapter_lrs"][grid]] if arm in LEARNED else [])
        require([g["lr"] for g in cp["optimizer"]["param_groups"]] == expected_lrs, "optimizer grid mismatch")
        require(all(g["weight_decay"] == config["weight_decay"] for g in cp["optimizer"]["param_groups"]), "optimizer decay mismatch")
        if track == "frozen":
            require(all(torch.equal(cp["model"][name], value) for name, value in source.items()), "frozen backbone changed")
            require(cp["cache_index_sha256"] == file_hash(root / "cache/cache_index.json"), "cache provenance mismatch")
        else:
            require(cp["cache_index_sha256"] is None, "joint training used frozen cache")
        identity = {"track": track, "arm": arm, "seed": seed, "grid": grid,
                    "checkpoint_sha256": file_hash(out / "checkpoint.pt"), "config_sha256": file_hash(out / "config.json")}
        calculated, n = check_stream_file(out / "tuning.jsonl.gz", config, examples, config["tuning_edits"], identity)
        require(calculated == summary["validation"], "tuning summary mismatch")
        counts["training_runs"] += 1
        counts["tuning_predictions"] += n
        summaries.append({**identity, "first32_loss": summary["first32_post_edit_loss"], "last32_loss": summary["last32_post_edit_loss"]})
        print(json.dumps({"checked": out.name}), flush=True)
    if not complete:
        return {"verified": True, "complete": False, **counts, "learning_curves": summaries}
    selection = select(root, config, persist=False)
    require(read(root / "selection.json") == selection, "selection mismatch")
    curves = {}
    for track in config.get("tracks", ("frozen", "joint")):
        for seed in config["seeds"]:
            for arm in (*PRINCIPAL, *AUXILIARY):
                if track == "frozen" and arm in ("restart", "carry") and seed != config["seeds"][0]:
                    continue
                out = root / "streams" / f"{track}-{arm}-seed{seed}"
                verify_seal(out)
                check_resource(out)
                summary = read(out / "summary.json")
                source_arm = "spatial_gate" if arm in AUXILIARY else arm
                grid = None if track == "frozen" and arm in ("restart", "carry") else selection[f"{track}-{source_arm}"]["grid"]
                cp_path = Path(config["source_checkpoint"]) if grid is None else root / "training" / f"{track}-{source_arm}-seed{seed}-grid{grid}" / "checkpoint.pt"
                identity = {"track": track, "arm": arm, "seed": seed, "grid": grid,
                    "checkpoint_sha256": file_hash(cp_path), "config_sha256": file_hash(root / "prepared/config.json")}
                calculated, n = check_stream_file(out / "predictions.jsonl.gz", config, examples, config["stream_edits"], identity)
                require(calculated == summary["validation"], "stream summary mismatch")
                counts["stream_predictions"] += n
                for k in config["budgets"]:
                    curves[track, arm, seed, k] = {e["root_id"]: e["post32"] for e in calculated["episodes"] if e["K"] == k}
                print(json.dumps({"checked": out.name}), flush=True)
            out = root / "interventions" / f"{track}-seed{seed}"
            verify_seal(out)
            check_resource(out)
            entries = {(e["suite"], e["branch"], e["new"]["root_id"]): e for e in intervention_entries(payload, config)}
            arms = (*PRINCIPAL, *AUXILIARY) if track == "frozen" else ("spatial_gate", "restart", "carry", *AUXILIARY)
            expected = {(suite, branch, rid, k) for suite, branch, rid in entries for k in config["budgets"]}
            shared = {}
            prior_hashes = {p.name: file_hash(p) for p in out.glob("prior-*.pt")}
            for arm in arms:
                seen = set()
                for row in json_rows(out / f"{arm}.jsonl.gz"):
                    key = row["suite"], row["branch"], row["root_id"], row["K"]
                    require(key not in seen, "duplicate intervention prediction")
                    seen.add(key)
                    require(row["record_kind"] == "frozen_state_intervention" and row["source_K"] == config["source_K"] and row["state_budget"] == row["K"] and row["source_policy"] == "restart", "invalid intervention source/budget")
                    require(row["synthetic"] is False and row["privileged"] is False and row["split"] == "val", "invalid intervention flags")
                    entry = entries[key[:3]]
                    e = decode_example(entry["new"])
                    require(row["stratum"] == entry["stratum"] and row["impact_fraction"] == entry["impact_fraction"], "stratum mismatch")
                    require(row["input_sha256"] == input_hash(e.maze), "intervention input mismatch")
                    require(all(row[k] == v for k, v in exact_score(e.maze, tuple(row["actions"]))), "intervention score mismatch")
                    prior = row["prior_file_sha256"], row["prior_index"], row["source_checkpoint_sha256"]
                    require(shared.setdefault(key, prior) == prior, "counterfactual priors differ across arms")
                    require(prior_hashes[row["prior_file"]] == prior[0], "prior state file changed")
                require(seen == expected, "incomplete intervention coverage")
                counts["intervention_predictions"] += len(seen)
    return {"verified": True, "complete": True, **counts, "learning_curves": summaries,
            "gate": stream_gates(curves, config) if config.get("training_protocol") == "stream" else stop_screen(curves, config), "selection_sha256": file_hash(root / "selection.json"),
            "inference_replayed": False, "checker_sha256": file_hash(__file__),
            "validation_scope": "all raw actions rescored; hashes, coverage, optimization and pairing checked"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/adapter_pilot_v1"))
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_pilot(args.run, complete=not args.partial)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("learning_curves", "gate")}), flush=True)
