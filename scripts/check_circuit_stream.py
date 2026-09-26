"""Verify circuit actions, full paired coverage, training and checkpoint lineage."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from check_adapter_pilot import check_resource, require
from run_adapter_pilot import json_rows, read, verify_seal, write
from run_circuit_stream import decode, make_payload, select, summarize
from train_static_circuit import input_hash, score
from state_repair.provenance import file_hash
from state_repair.train.pilot import schedule


def check_actions(path: Path, examples: dict, config: dict, edits: int, identity: dict) -> tuple[dict, int]:
    seen, rows = set(), []
    for row in json_rows(path):
        key = row["K"], row["frame"], row["root_id"]
        require(key not in seen, "duplicate circuit action record")
        seen.add(key)
        require(row["synthetic"] is False and row["split"] == "val", "synthetic/non-validation record")
        require(all(row[k] == v for k,v in identity.items()), "circuit identity mismatch")
        require(row["state_budget"] == row["K"] and row["block_calls_per_example"] == row["K"]*(config["inner_cycles"]+1), "wrong circuit K")
        require(row["record_kind"] == "fixed_budget_stream", "not an own-state stream")
        e = examples[row["frame"], row["root_id"]]
        require(row["input_sha256"] == input_hash(e), "circuit observation hash mismatch")
        require(all(row[k] == v for k,v in score(e, row["actions"]).items()), "circuit raw score mismatch")
        rows.append(row)
    roots = {r for f,r in examples if f == 0}
    require(seen == {(k,f,r) for k in config["budgets"] for f in range(edits+1) for r in roots}, "circuit coverage mismatch")
    return summarize(rows), len(rows)


def contrasts(curves: dict, config: dict) -> list[dict]:
    roots = sorted(curves["restart", config["seeds"][0], config["budgets"][0]])
    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0,len(roots),
        size=(config["bootstrap_repetitions"],len(roots)))
    result = []
    for arm in ("answer_only", "carry"):
        for k in config["budgets"]:
            values = []
            for seed in config["seeds"]:
                a,b = curves[arm,seed,k], curves["restart",seed,k]
                require(sorted(a) == sorted(b) == roots, "unpaired circuit roots")
                values.append([a[r]-b[r] for r in roots])
            means = np.mean(values,axis=0)
            result.append({"arm": arm, "baseline": "restart", "K": k, "delta": float(means.mean()),
                "ci": np.quantile(means[draws].mean(1), [.025,.975]).tolist(),
                "per_seed_delta": np.mean(values,axis=1).tolist(), "roots": len(roots)})
    return result


def check(root: Path, *, complete: bool = True) -> dict:
    torch.set_num_threads(2)
    verify_seal(root / "prepared")
    config, payload = read(root / "prepared/config.json"), read(root / "prepared/dataset.json")
    require(config["synthetic"] is False, "empirical protocol required")
    require(file_hash(config["source_checkpoint"]) == config["source_checkpoint_sha256"], "source checkpoint changed")
    require(file_hash(config["source_dataset"]) == read(root / "prepared/provenance.json")["source_dataset_sha256"], "source dataset changed")
    regenerated = json.loads(json.dumps(make_payload(read(config["source_dataset"]),config)))
    require(payload == regenerated, "circuit generation replay mismatch")
    examples = {(e["frame_index"],e["root_id"]): decode(e) for f in payload["streams"] for e in f}
    expected = {f"joint-{arm}-seed{seed}-grid{grid}" for arm in config["arms"] for seed in config["seeds"] for grid in range(2)}
    paths = sorted((root / "training").glob("*"))
    require(bool(paths) and all(p.name in expected for p in paths), "missing/unexpected circuit training")
    if complete:
        require({p.name for p in paths} == expected, "incomplete circuit matrix")
    counts = {"training_runs": 0, "tuning_predictions": 0, "stream_predictions": 0}
    learning = []
    for out in paths:
        verify_seal(out); check_resource(out)
        cfg, summary = read(out / "config.json"), read(out / "summary.json")
        identity = {k: summary[k] for k in ("track","arm","seed","grid")}
        require(cfg == {**config,**identity}, "circuit run config mismatch")
        steps = list(json_rows(out / "steps.jsonl"))
        plan = schedule(identity["seed"],len(payload["train_frames"][0])//config["batch_size"],config)
        require(len(steps) == config["steps"] == summary["optimizer_steps"], "incomplete circuit optimizer")
        require([(r["batch_index"],r["K"]) for r in steps] == plan, "unpaired circuit training schedule")
        for i,r in enumerate(steps,1):
            require(r["step"] == i and r["synthetic"] is False and r["batch_size"] == config["batch_size"], "training identity mismatch")
            require(r["frames_per_step"] == len(r["frame_losses"]) == 5 and r["frame_block_calls"] == [r["K"]*(config["inner_cycles"]+1)]*5, "circuit stream budget mismatch")
            require(math.isclose(r["loss"],sum(r["frame_losses"])/5) and all(math.isfinite(v) for v in r["frame_losses"]), "circuit objective mismatch")
            require(math.isfinite(r["gradient_norm"]) and r["backbone_gradient_norm"] > 0, "invalid circuit gradients")
            require(r["adapter_gradient_norm"] > 0 if identity["arm"] not in ("restart","carry") else r["adapter_gradient_norm"] == 0, "broken circuit adapter gradient")
        require(summary["training_example_F_calls"] == sum(sum(r["frame_block_calls"]) for r in steps)*config["batch_size"], "circuit cost mismatch")
        cp = torch.load(out / "checkpoint.pt",map_location="cpu",weights_only=False)
        require(cp["steps"] == config["steps"] and cp["schedule"] == plan and cp["config"] == cfg, "circuit checkpoint schedule mismatch")
        require(cp["dataset_sha256"] == file_hash(root / "prepared/dataset.json") and cp["prepared_manifest_sha256"] == file_hash(root / "prepared/manifest.json"), "circuit checkpoint provenance mismatch")
        require(all(torch.isfinite(v).all() for key in ("model","adapter") for v in cp[key].values()), "nonfinite circuit checkpoint")
        require(all(int(v["step"]) == config["steps"] for v in cp["optimizer"]["state"].values()), "circuit optimizer state mismatch")
        lrs = [config["backbone_lrs"][identity["grid"]]] + ([] if identity["arm"] in ("restart","carry") else [config["adapter_lrs"][identity["grid"]]])
        require([g["lr"] for g in cp["optimizer"]["param_groups"]] == lrs and all(g["weight_decay"] == config["weight_decay"] for g in cp["optimizer"]["param_groups"]), "circuit LR mismatch")
        identity.update(checkpoint_sha256=file_hash(out / "checkpoint.pt"),config_sha256=file_hash(out / "config.json"))
        calculated,n = check_actions(out / "tuning.jsonl.gz",examples,config,4,identity)
        require(calculated == summary["validation"], "circuit tuning summary mismatch")
        counts["training_runs"] += 1; counts["tuning_predictions"] += n
        learning.append({**identity,"first32_loss":np.mean([r["loss"] for r in steps[:32]]),"last32_loss":np.mean([r["loss"] for r in steps[-32:]])})
        print(f"checked {out.name}",flush=True)
    if not complete:
        return {"verified":True,"complete":False,**counts,"learning_curves":learning}
    selection = select(root,config,persist=False)
    require(read(root / "selection.json") == selection,"circuit selection mismatch")
    curves = {}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            out = root / "streams" / f"joint-{arm}-seed{seed}"
            verify_seal(out); check_resource(out)
            cp = root / "training" / f"joint-{arm}-seed{seed}-grid{selection[arm]['grid']}" / "checkpoint.pt"
            identity = {"track":"joint","arm":arm,"seed":seed,"grid":selection[arm]["grid"],
                "checkpoint_sha256":file_hash(cp),"config_sha256":file_hash(root / "prepared/config.json")}
            calculated,n = check_actions(out / "predictions.jsonl.gz",examples,config,32,identity)
            require(calculated == read(out / "summary.json")["validation"],"circuit stream summary mismatch")
            counts["stream_predictions"] += n
            for k in config["budgets"]:
                curves[arm,seed,k] = {e["root_id"]:e["post32"] for e in calculated["episodes"] if e["K"] == k}
    return {"verified":True,"complete":True,**counts,"learning_curves":learning,"contrasts":contrasts(curves,config),
        "checker_sha256":file_hash(__file__),"inference_replayed":False,
        "validation_scope":"raw actions rescored; all coverage, schedules, manifests and checkpoint bindings verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,default=Path("runs/circuit_stream_v2"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    write(args.output,check(args.run,complete=not args.partial))
