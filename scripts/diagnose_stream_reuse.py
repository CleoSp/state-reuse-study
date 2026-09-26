"""Evaluation-only lesions and carried-state dynamics; no parameter updates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import torch

import run_adapter_pilot as maze
import run_circuit_stream as circuit
from state_repair.eval.frozen import frozen_adapter_prediction
from state_repair.models.adapters import make_adapter
from state_repair.train.adapter_step import FrozenPrior
from state_repair.types import PredictionBatch
from state_repair.provenance import file_hash


def lesion_hook(kind: str):
    if kind not in ("uniform", "shuffled_nodes"):
        raise ValueError("unknown evaluation-time lesion")
    def hook(module, args, kwargs):
        prior = kwargs["previous_prediction"]
        if prior is None:


            return args, kwargs
        logits = prior.logits.detach().clone()
        valid = args[0].new.valid_nodes
        if kind == "uniform":
            logits.zero_()
        else:
            for b in range(logits.shape[0]):
                indices = valid[b].nonzero().flatten()
                perm = indices[torch.randperm(len(indices), device=logits.device)]
                logits[b, indices] = prior.logits[b, perm]
        return args, {**kwargs, "previous_prediction": PredictionBatch(logits)}
    return hook


def run_lesions(family: str, version: str, seed: int) -> None:
    root = Path("runs/adapter_pilot_v1" if family == "maze" and version == "pilot" else
                "runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2")
    config = maze.read(root / "prepared/config.json")
    payload = maze.read(root / "prepared/dataset.json")
    job_config = {**config, "gpu_authorization": "prompt05b"}
    destination = Path("runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2") / "diagnostics"
    for kind in ("uniform", "shuffled_nodes"):
        out = destination / f"{version}-answer_only-{kind}-seed{seed}"
        if out.exists():
            maze.verify_seal(out)
            continue
        out.parent.mkdir(exist_ok=True)
        with maze.gpu_job(job_config, out, 1200, "prompt05b_answer_content_lesion", synthetic=False) as job:
            (out / "diagnostic_script.py").write_bytes(Path(__file__).read_bytes())
            maze.write(out / "config.json", config)
            if family == "maze":
                model, adapter, identity = maze.load_selected(root, config, maze.read(root / "selection.json"), {}, "joint", "answer_only", seed, "cuda")
                data = maze.frame_batches(payload, 32, config)
            else:
                model, adapter, identity = circuit.load_selected(root, config, "answer_only", seed, "cuda")
                data = circuit.batches(payload["streams"], config, 32)
            identity.update(track="joint", arm="answer_only", seed=seed, config_sha256=file_hash(root / "prepared/config.json"),
                evaluation_time_lesion=kind, deployable_policy=False, source_version=version,
                diagnostic_script_sha256=file_hash(__file__))
            hook = adapter.register_forward_pre_hook(lesion_hook(kind), with_kwargs=True)
            try:
                summary = (maze.evaluate_streams(model, adapter, data, config, out / "predictions.jsonl.gz", identity, job.check_limit, "cuda")
                    if family == "maze" else circuit.evaluate_streams(model, adapter, data, config, out / "predictions.jsonl.gz", identity, job, "cuda"))
                maze.write(out / "summary.json", {**identity, "validation": summary})
            finally:
                hook.remove()
            del model, adapter
        maze.seal(out)


def dynamics(family: str, version: str, seed: int, owner: str, *, extended_k: bool = False) -> None:
    if extended_k and family != "circuit":
        raise ValueError("K18 supplement is circuit-only; maze already uses K1/K8")
    root = Path("runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2")
    config = maze.read(root / "prepared/config.json")
    payload = maze.read(root / "prepared/dataset.json")
    suffix = "-K18" if extended_k else ""
    out = root / "diagnostics" / f"{version}-{owner}-dynamics{suffix}-seed{seed}"
    if out.exists():
        maze.verify_seal(out)
        return
    out.parent.mkdir(exist_ok=True)
    with maze.gpu_job(config, out, 600, "prompt05b_carried_state_dynamics", synthetic=False) as job:
        (out / "diagnostic_script.py").write_bytes(Path(__file__).read_bytes())
        maze.write(out / "config.json", config)
        if version == "pilot":
            cp_hash = config["source_checkpoint_sha256"]
            weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
            model = (maze.new_model(config, weights, "frozen", "cuda") if family == "maze" else
                     circuit.new_model(config, weights, "cuda", training=False))
            trained = None
        else:
            if family == "maze":
                model, trained, identity = maze.load_selected(root, config, maze.read(root / "selection.json"), {}, "joint", owner, seed, "cuda")
            else:
                model, trained, identity = circuit.load_selected(root, config, owner, seed, "cuda")
            cp_hash = identity["checkpoint_sha256"]
        model.eval()
        if family == "maze":
            entries = maze.intervention_entries(payload, config)
            prepared = []
            b = config["batch_size"]
            for offset in range(0,len(entries),b):
                subset = entries[offset:offset+b]
                padded = subset+[subset[-1]]*(b-len(subset))
                old = maze.collate([maze.decode_example(e["old"]) for e in padded])[0]
                new = maze.collate([maze.decode_example(e["new"]) for e in padded])[0]
                prepared.append((subset,old,new))
            budgets = (1,8)
        else:
            prepared = []
            for stream in circuit.batches(payload["ordinary"],config,1):
                old_e,old,_ = stream[0]; new_e,new,_ = stream[1]
                subset = []
                for a,b in zip(old_e,new_e):
                    from state_repair.oracles.circuit import evaluate
                    va,vb = evaluate(a.circuit),evaluate(b.circuit)
                    nodes = [u for u,op in enumerate(a.circuit.operators) if op != circuit.Operator.INPUT]
                    fraction = sum(va[u] != vb[u] for u in nodes)/len(nodes)
                    subset.append({"old":a,"new":b,"suite":"ordinary","branch":"uniform", "impact_fraction":fraction,
                        "stratum":"low" if fraction <= config["low_max"] else "high" if fraction >= config["high_min"] else "middle"})
                prepared.append((subset,old,new))
            budgets = (1,8) if extended_k else (1,2)
        records = []
        with torch.no_grad(), gzip.open(out / "actions.jsonl.gz","xt",encoding="utf-8") as handle:
            for batch_index,(subset,cpu_old,cpu_new) in enumerate(prepared):
                job.check_limit()
                old,new = cpu_old.to("cuda"),cpu_new.to("cuda")
                initial = model(old,config["source_K"])
                prior = FrozenPrior(cpu_old,initial.state.detach().to("cpu"),PredictionBatch(initial.prediction.logits.cpu()),cp_hash,"restart",config["source_K"])
                prior_path = out / f"prior-batch{batch_index}.pt"
                torch.save(prior,prior_path)
                results = [frozen_adapter_prediction(model,make_adapter("carry"),old,new,initial.state,k) for k in budgets]
                acts = [r.solver.prediction.logits.argmax(-1).cpu().tolist() for r in results]
                retention = None
                if trained is not None and owner == "spatial_gate":
                    retained = frozen_adapter_prediction(model,trained,old,new,initial.state,1)
                    retention = ((retained.adapter.retain_a+retained.adapter.retain_z)/2).squeeze(-1).cpu().tolist()
                    del retained
                for i,entry in enumerate(subset):
                    e = maze.decode_example(entry["new"]) if family == "maze" else entry["new"]
                    n = e.maze.n if family == "maze" else e.circuit.n
                    a,b = acts[0][i][:n],acts[1][i][:n]
                    scored = list(range(n)) if family == "maze" else [j for j,u in enumerate(e.node_order) if e.circuit.operators[u] != circuit.Operator.INPUT]
                    scores = [maze.score_policy(e.maze,x) if family == "maze" else circuit.score(e,x) for x in (a,b)]
                    row = {"root_id":e.root_id,"suite":entry["suite"],"branch":entry["branch"],"stratum":entry["stratum"],
                        "impact_fraction":entry["impact_fraction"],"source_K":config["source_K"],"budgets":budgets,
                        "source_checkpoint_sha256":cp_hash,"prior_file":prior_path.name,"prior_sha256":file_hash(prior_path),"prior_index":i,
                        "actions_low":a,"actions_high":b,"scores":scores,"changed_actions":sum(x!=y for x,y in zip(a,b)),"nodes":n,
                        "scored_nodes":len(scored),"changed_scored_actions":sum(a[j] != b[j] for j in scored),
                        "input_sha256":maze.canonical_hash(e.maze) if family == "maze" else circuit.input_hash(e),
                        "node_retention":None if retention is None else retention[i][:n],
                        "synthetic":False,"split":"val","record_kind":"frozen_state_intervention","deployable_policy":False,
                        "owner":owner,"source_version":version,"seed":seed}
                    handle.write(json.dumps(row,allow_nan=False)+"\n"); records.append(row)
                del initial,results,prior,old,new
        summary = []
        for stratum in ("low","middle","high"):
            selected = [r for r in records if r["stratum"] == stratum]
            nodes = sum(r["nodes"] for r in selected)
            changed = sum(r["changed_actions"] for r in selected)
            scored_nodes = sum(r["scored_nodes"] for r in selected)
            changed_scored = sum(r["changed_scored_actions"] for r in selected)
            metric = "route_correct" if family == "maze" else "exact_correct"
            summary.append({"stratum":stratum,"branches":len(selected),"nodes":nodes,"changed_actions":changed,
                "fraction":changed/nodes if nodes else None,
                "scored_nodes":scored_nodes,"changed_scored_actions":changed_scored,
                "scored_fraction":changed_scored/scored_nodes if scored_nodes else None,
                "accuracy_low":sum(r["scores"][0][metric] for r in selected)/len(selected) if selected else None,
                "accuracy_high":sum(r["scores"][1][metric] for r in selected)/len(selected) if selected else None})
        maze.write(out / "summary.json",{"source_version":version,"owner":owner,"seed":seed,"budgets":budgets,
            "strata":summary,"source_checkpoint_sha256":cp_hash,"script_sha256":file_hash(__file__),"synthetic":False})
        del model,trained
    maze.seal(out)


def circuit_reference() -> None:
    """Static circuit checkpoint restarted at each frame of the same v2 stream."""
    root = Path("runs/circuit_stream_v2")
    config = maze.read(root / "prepared/config.json")
    out = root / "diagnostics" / "pilot-restart-reference"
    if out.exists():
        maze.verify_seal(out)
        return
    out.parent.mkdir(exist_ok=True)
    with maze.gpu_job(config,out,900,"prompt05b_circuit_static_drift_reference",synthetic=False) as job:
        (out / "diagnostic_script.py").write_bytes(Path(__file__).read_bytes())
        maze.write(out / "config.json",config)
        cp = torch.load(config["source_checkpoint"],map_location="cpu",weights_only=False)
        model = circuit.new_model(config,cp["model"],"cuda",training=False)
        data = circuit.batches(maze.read(root / "prepared/dataset.json")["streams"],config,32)
        identity = {"track":"static","arm":"restart","seed":29,"grid":None,
            "checkpoint_sha256":config["source_checkpoint_sha256"],"config_sha256":file_hash(root / "prepared/config.json"),
            "source_version":"pilot","distribution_drift_reference":True}
        summary = circuit.evaluate_streams(model,make_adapter("restart"),data,config,out / "predictions.jsonl.gz",identity,job,"cuda")
        maze.write(out / "summary.json",{**identity,"validation":summary})
    maze.seal(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=("dynamics","lesions","reference"))
    parser.add_argument("--family",choices=("maze","circuit"),required=True)
    parser.add_argument("--version",choices=("pilot","stream"),required=True)
    parser.add_argument("--seed",type=int,default=29)
    parser.add_argument("--owner",choices=("carry","spatial_gate"),default="carry")
    parser.add_argument("--extended-k",action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    if args.extended_k and args.stage != "dynamics":
        raise ValueError("extended K applies only to the declared dynamics supplement")
    if args.stage == "reference":
        if args.family != "circuit" or args.version != "pilot":
            raise ValueError("only the missing pilot circuit reference needs a new sweep")
        circuit_reference()
    elif args.stage == "lesions":
        if args.family == "circuit" and args.version == "pilot":
            raise ValueError("no pilot circuit answer-only checkpoint exists")
        run_lesions(args.family,args.version,args.seed)
    else:
        dynamics(args.family,args.version,args.seed,args.owner,extended_k=args.extended_k)


if __name__ == "__main__":
    main()
