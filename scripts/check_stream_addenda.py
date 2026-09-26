"""Rescore every addendum action; verify schedules, matching and frozen selection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from check_adapter_pilot import check_resource, check_stream_file, require
from check_circuit_stream import check_actions
from run_adapter_pilot import json_rows, read, verify_seal, write
from run_stream_addenda import ROOTS, depth_payload
from run_circuit_stream import decode
from state_repair.provenance import file_hash
from state_repair.train.pilot import decode_example, schedule


def paired(a: list[dict], b: list[dict], config: dict, metric: str = "post32") -> dict:
    """Full episode means paired by seed and root; seeds conditional, not resampled."""
    roots = sorted(e["root_id"] for e in a if e["seed"] == config["seeds"][0])
    require(len(roots) == len(set(roots)) == 256, "wrong paired root coverage")
    values = []
    for seed in config["seeds"]:
        left = {e["root_id"]: e[metric] for e in a if e["seed"] == seed}
        right = {e["root_id"]: e[metric] for e in b if e["seed"] == seed}
        require(sorted(left) == sorted(right) == roots, "unpaired addendum roots")
        values.append([left[r]-right[r] for r in roots])
    means = np.mean(values, axis=0)
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(roots),
        size=(config["bootstrap_repetitions"], len(roots)))
    return {"delta": float(means.mean()), "ci": np.quantile(means[draws].mean(1), [.025,.975]).tolist(),
        "per_seed_delta": np.mean(values, axis=1).tolist(), "roots": len(roots)}


def check_training(root: Path, config: dict, payload: dict, selection: dict, examples: dict) -> tuple[list, int]:
    matches, count = [], 0
    expected = {f"joint-{a}-seed{s}-grid{selection[f'joint-{a}']['grid']}" for a in config["arms"] for s in config["seeds"]}
    require({p.name for p in (root / "training").iterdir()} == expected, "wrong longer training matrix")
    for out in sorted((root / "training").iterdir()):
        verify_seal(out); check_resource(out)
        summary, cfg = read(out / "summary.json"), read(out / "config.json")
        ident = {k: summary[k] for k in ("track", "arm", "seed", "grid")}
        require(cfg == {**config, **ident} and ident["track"] == "joint", "longer config mismatch")
        plan = schedule(ident["seed"], len(payload["train"])*config["training_views"]//config["batch_size"], config)
        rows = list(json_rows(out / "steps.jsonl"))
        require(len(rows) == config["steps"] == summary["optimizer_steps"] and [(r["batch_index"], r["K"]) for r in rows] == plan, "longer schedule mismatch")
        for i, r in enumerate(rows, 1):
            require(r["step"] == i and r["synthetic"] is False and r["batch_size"] == config["batch_size"], "invalid step")
            require(all(r[k] == v for k,v in ident.items()), "step identity mismatch")
            require(all(math.isfinite(r[k]) for k in ("loss", "initial_loss", "post_edit_loss", "gradient_norm", "adapter_gradient_norm", "backbone_gradient_norm")), "nonfinite training")
            require(math.isclose(r["loss"], (r["initial_loss"]+r["post_edit_loss"])/2, abs_tol=1e-6), "one-edit objective changed")
            require(r["initial_block_calls_executed"] == r["post_edit_block_calls"] == r["K"]*(config["inner_cycles"]+1), "one-edit call count changed")
            require(r["backbone_gradient_norm"] > 0 and (r["adapter_gradient_norm"] > 0 if ident["arm"] in ("spatial_gate", "answer_only") else r["adapter_gradient_norm"] == 0), "missing gradients")
        calls = sum(r["initial_block_calls_executed"]+r["post_edit_block_calls"] for r in rows)*config["batch_size"]
        target_path = Path(config["matched_stream_source"]) / "training" / f"joint-{ident['arm']}-seed{ident['seed']}-grid{ident['grid']}" / "summary.json"
        target = read(target_path)["training_example_F_calls"]
        require(calls == summary["training_example_F_calls"] and abs(calls/target-1) <= config["matching_tolerance"], "compute not matched within tolerance")
        cp = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        require(cp["steps"] == config["steps"] and cp["config"] == cfg and cp["schedule"] == plan, "checkpoint schedule mismatch")
        require(cp["dataset_sha256"] == file_hash(root / "prepared/dataset.json") and cp["prepared_manifest_sha256"] == file_hash(root / "prepared/manifest.json") and cp["cache_index_sha256"] is None, "checkpoint lineage mismatch")
        require(all(torch.isfinite(v).all() for key in ("model", "adapter") for v in cp[key].values()), "nonfinite checkpoint")
        require(all(int(v["step"]) == config["steps"] for v in cp["optimizer"]["state"].values()), "optimizer incomplete")
        lrs = [config["backbone_lrs"][ident["grid"]]] + ([config["adapter_lrs"][ident["grid"]]] if ident["arm"] in ("spatial_gate", "answer_only") else [])
        require([g["lr"] for g in cp["optimizer"]["param_groups"]] == lrs and all(g["weight_decay"] == config["weight_decay"] for g in cp["optimizer"]["param_groups"]), "optimizer recipe changed")
        ident.update(checkpoint_sha256=file_hash(out / "checkpoint.pt"), config_sha256=file_hash(out / "config.json"))
        calculated, n = check_stream_file(out / "tuning.jsonl.gz", config, examples, 4, ident)
        require(calculated == summary["validation"], "longer tuning mismatch")
        count += n
        matches.append({**ident, "steps": config["steps"], "example_F_calls": calls, "stream_example_F_calls": target,
            "relative_difference": calls/target-1, "training_seconds": summary["training_seconds"],
            "first32_loss": summary["first32_post_edit_loss"], "last32_loss": summary["last32_post_edit_loss"]})
        print(f"verified training {out.name}", flush=True)
    return matches, count


def check(kind: str) -> dict:
    root = ROOTS[kind]
    verify_seal(root / "prepared")
    config, payload = read(root / "prepared/config.json"), read(root / "prepared/dataset.json")
    provenance, selection = read(root / "prepared/provenance.json"), read(root / "prepared/selection.json")
    require(file_hash(config["source_dataset"]) == provenance["source_dataset_sha256"] and file_hash(config["source_checkpoint"]) == config["source_checkpoint_sha256"], "source changed")
    require(file_hash(provenance["selection_source"]) == provenance["selection_source_sha256"], "selection source changed")
    original = read(provenance["selection_source"])
    require(selection == ({f"joint-{a}": original[f"joint-{a}"] for a in config["arms"]} if kind == "longer" else original), "selection not frozen")
    if kind == "longer":
        require(payload == read("runs/adapter_pilot_v1/prepared/dataset.json"), "one-edit roots/views changed")
        require(payload["streams"] == read(Path(config["matched_stream_source"]) / "prepared/dataset.json")["streams"], "matched-compute comparison changed a validation stream")
        original_config = read("runs/adapter_pilot_v1/prepared/config.json")
        require(all(config[k] == v for k,v in original_config.items() if k not in ("steps", "protocol_version", "job_seconds")), "one-edit recipe changed beyond step count")
    else:
        require(payload == json.loads(json.dumps(depth_payload(config))), "depth generation replay mismatch")
    decoder = decode_example if kind == "longer" else decode
    examples = {(e["frame_index"], e["root_id"]): decoder(e) for frame in payload["streams"] for e in frame}
    matches, tuning = check_training(root, config, payload, selection, examples) if kind == "longer" else ([], 0)
    expected = {f"joint-{a}-seed{s}" + ("" if kind == "longer" else f"-grid{g}")
        for a in config["arms"] for s in config["seeds"] for g in ([0] if kind == "longer" else config["checkpoint_grids"])}
    require({p.name for p in (root / "streams").iterdir()} == expected, "wrong stream matrix")
    curves, frames, predictions = [], [], 0
    for out in sorted((root / "streams").iterdir()):
        verify_seal(out); check_resource(out)
        summary = read(out / "summary.json")
        ident = {k: summary[k] for k in ("track", "arm", "seed", "grid")}
        cp_root = root if kind == "longer" else Path(config["source_run"])
        cp = cp_root / "training" / f"joint-{ident['arm']}-seed{ident['seed']}-grid{ident['grid']}" / "checkpoint.pt"
        require(ident["grid"] == selection[f"joint-{ident['arm']}"]["grid"] if kind == "longer" else ident["grid"] in config["checkpoint_grids"], "wrong grid")
        ident.update(checkpoint_sha256=file_hash(cp), config_sha256=file_hash(root / "prepared/config.json"))
        calculated, n = (check_stream_file(out / "predictions.jsonl.gz", config, examples, 32, ident) if kind == "longer" else
            check_actions(out / "predictions.jsonl.gz", examples, config, 32, ident))
        require(calculated == summary["validation"], "addendum stream summary mismatch")
        predictions += n
        primary = kind == "longer" or ident["grid"] == selection[ident["arm"]]["grid"]
        curves.extend({**ident, **e, "primary": primary} for e in calculated["episodes"])
        if kind == "depth":
            frames.extend({**ident, **r, "primary": primary} for r in calculated["per_frame"])
        print(f"verified streams {out.name}", flush=True)
    comparisons = []
    for arm in config["arms"] if kind == "longer" else ("answer_only", "spatial_gate"):
        for k in config["budgets"]:
            a = [r for r in curves if r["arm"] == arm and r["K"] == k and r["primary"]]
            if kind == "longer":
                b = []
                for seed in config["seeds"]:
                    old = read(Path(config["matched_stream_source"]) / "streams" / f"joint-spatial_gate-seed{seed}" / "summary.json")
                    b.extend({**e, "seed": seed} for e in old["validation"]["episodes"] if e["K"] == k)
                baseline = "stream_spatial_gate"
            else:
                b = [r for r in curves if r["arm"] == "restart" and r["K"] == k and r["primary"]]
                baseline = "restart"
            comparisons.append({"arm": arm, "baseline": baseline, "K": k, **paired(a,b,config)})
    gate = (all(r["ci"][0] <= 0 <= r["ci"][1] for r in comparisons if r["arm"] == "spatial_gate" and r["K"] in (1,4)) if kind == "longer" else
        any(r["ci"][0] > 0 for r in comparisons if r["arm"] == "answer_only"))
    exploratory = []
    if kind == "depth":
        for k in config["budgets"]:
            a = [r for r in curves if r["arm"] == "spatial_gate" and r["K"] == k and r["primary"]]
            b = [r for r in curves if r["arm"] == "answer_only" and r["K"] == k and r["primary"]]
            exploratory.append({"arm":"spatial_gate", "baseline":"answer_only", "K":k,
                "exploratory":True, "declared_before_run":False, "reopens_G2":False, **paired(a,b,config)})
    return {"verified": True, "complete": True, "kind": kind, "training_matches": matches,
        "tuning_predictions": tuning, "stream_predictions": predictions, "curves": curves, "per_frame": frames,
        "contrasts": comparisons, "exploratory_contrasts":exploratory, "gate": gate,
        "gate_name": "compute_compatible_operational_screen" if kind == "longer" else "answer_only_positive_at_any_K_under_shift",
        "gate_limitation": "CI overlap is not equivalence or causal attribution; pointwise exploratory validation, conditional on three seeds",
        "checker_sha256": file_hash(__file__), "config_sha256": file_hash(root / "prepared/config.json"),
        "inference_replayed": False, "validation_scope": "raw actions rescored; full coverage, schedules, matching and frozen selection checked"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("longer", "depth"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    write(ROOTS[args.kind] / "check.json", check(args.kind))
