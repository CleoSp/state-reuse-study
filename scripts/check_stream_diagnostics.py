"""Rescore diagnostic records and derive descriptive retention histograms."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_adapter_pilot as maze
import run_circuit_stream as circuit
from check_adapter_pilot import check_resource, check_stream_file, require
from check_circuit_stream import check_actions
from state_repair.provenance import file_hash


def groups(entry: dict, family: str) -> tuple[set[int],set[int]]:
    """Edited nodes versus nodes beyond two hops in current observed wiring."""
    if family == "maze":
        old,new = maze.decode_example(entry["old"]).maze,maze.decode_example(entry["new"]).maze
        edited = {u for edge in set(old.edges)^set(new.edges) for u in edge}
        n,edges = new.n,new.edges
        order = tuple(range(n))
    else:
        old,new = entry["old"].circuit,entry["new"].circuit
        edited = {u for u in range(new.n) if old.operators[u] != new.operators[u] or old.input_bits[u] != new.input_bits[u]}
        n,edges = new.n,[(u,p) for u,ps in enumerate(new.parents) for p in ps]
        order = entry["new"].node_order
    reached = set(edited)
    for _ in range(2):
        reached |= {v for u,v in [*edges,*[(v,u) for u,v in edges]] if u in reached}
    return {i for i,u in enumerate(order) if u in edited},{i for i,u in enumerate(order) if u not in reached}


def check(family: str, *, complete: bool = True, extended_k: bool = False) -> dict:
    require(not extended_k or family == "circuit" and not complete, "extended K is a separate circuit dynamics-only check")
    root = Path("runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2")
    config = maze.read(root / "prepared/config.json")
    payload = maze.read(root / "prepared/dataset.json")
    if family == "maze":
        entries = {(e["suite"],e["branch"],e["new"]["root_id"]):e for e in maze.intervention_entries(payload,config)}
    else:
        entries = {}
        from state_repair.oracles.circuit import evaluate
        for a,b in zip(payload["ordinary"][0],payload["ordinary"][1]):
            old,new = circuit.decode(a),circuit.decode(b)
            va,vb = evaluate(old.circuit),evaluate(new.circuit)
            nodes = [u for u,op in enumerate(old.circuit.operators) if op != circuit.Operator.INPUT]
            fraction = sum(va[u] != vb[u] for u in nodes)/len(nodes)
            entries["ordinary","uniform",new.root_id] = {"old":old,"new":new,"impact_fraction":fraction,
                "stratum":"low" if fraction <= config["low_max"] else "high" if fraction >= config["high_min"] else "middle"}
    required = [f"pilot-carry-dynamics-seed{config['seeds'][0]}"]
    required += [f"stream-{arm}-dynamics-seed{seed}" for arm in ("carry","spatial_gate") for seed in config["seeds"]]
    if extended_k:
        required = [name.replace("-dynamics-", "-dynamics-K18-") for name in required]
    dynamics, hist_values = [],defaultdict(list)
    histogram_root_means = []
    for name in required:
        out = root / "diagnostics" / name
        maze.verify_seal(out); check_resource(out)
        summary = maze.read(out / "summary.json")
        version,owner,seed = summary["source_version"],summary["owner"],summary["seed"]
        if version == "pilot":
            cp_hash = config["source_checkpoint_sha256"]
        else:
            selection = maze.read(root / "selection.json")
            grid = selection[f"joint-{owner}" if family == "maze" else owner]["grid"]
            cp_hash = file_hash(root / "training" / f"joint-{owner}-seed{seed}-grid{grid}" / "checkpoint.pt")
        require(summary["source_checkpoint_sha256"] == cp_hash,"dynamics checkpoint mismatch")
        expected_budgets = [1,8] if family == "maze" or extended_k else [1,2]
        require(summary["budgets"] == expected_budgets,"dynamics budget mismatch")
        require(summary["script_sha256"] == file_hash(out / "diagnostic_script.py"),"diagnostic source mismatch")
        seen,rows = set(),[]
        for row in maze.json_rows(out / "actions.jsonl.gz"):
            key = row["suite"],row["branch"],row["root_id"]
            require(key not in seen,"duplicate dynamics branch")
            seen.add(key)
            entry = entries[key]
            e = maze.decode_example(entry["new"]) if family == "maze" else entry["new"]
            n = e.maze.n if family == "maze" else e.circuit.n
            require(row["synthetic"] is False and row["split"] == "val" and row["record_kind"] == "frozen_state_intervention","invalid diagnostic flags")
            require(row["source_checkpoint_sha256"] == cp_hash and row["budgets"] == expected_budgets and row["source_K"] == config["source_K"],"dynamics source/budget mismatch")
            require(row["prior_sha256"] == file_hash(out / row["prior_file"]),"dynamics prior changed")
            require(row["stratum"] == entry["stratum"] and row["impact_fraction"] == entry["impact_fraction"],"diagnostic strata changed")
            require(row["input_sha256"] == (maze.canonical_hash(e.maze) if family == "maze" else circuit.input_hash(e)),"diagnostic input mismatch")
            actions = [row["actions_low"],row["actions_high"]]
            scores = [maze.score_policy(e.maze,a) if family == "maze" else circuit.score(e,a) for a in actions]
            require(row["scores"] == scores and row["nodes"] == n and row["changed_actions"] == sum(a!=b for a,b in zip(*actions)),"dynamics raw action mismatch")
            scored = list(range(n)) if family == "maze" else [j for j,u in enumerate(e.node_order) if e.circuit.operators[u] != circuit.Operator.INPUT]
            require(row["scored_nodes"] == len(scored) and row["changed_scored_actions"] == sum(actions[0][j] != actions[1][j] for j in scored),"scored-node dynamics mismatch")
            if owner == "spatial_gate":
                retention = row["node_retention"]
                require(len(retention) == n and all(np.isfinite(v) and 0 <= v <= 1 for v in retention),"invalid retention")
                edited,distant = groups(entry,family)
                for group,indices in (("all",set(range(n))),("edited",edited),("distant",distant)):
                    vals = [retention[i] for i in indices]
                    hist_values[row["stratum"],group].extend(vals)
                    if vals:
                        histogram_root_means.append({"seed":seed,"root_id":row["root_id"],"suite":row["suite"],"branch":row["branch"],
                            "stratum":row["stratum"],"group":group,"mean":float(np.mean(vals)),"nodes":len(vals)})
            rows.append(row)
        require(seen == set(entries),"incomplete dynamics coverage")
        regenerated = []
        metric = "route_correct" if family == "maze" else "exact_correct"
        for stratum in ("low","middle","high"):
            chosen = [r for r in rows if r["stratum"] == stratum]
            nodes,changed = sum(r["nodes"] for r in chosen),sum(r["changed_actions"] for r in chosen)
            scored_nodes,changed_scored = sum(r["scored_nodes"] for r in chosen),sum(r["changed_scored_actions"] for r in chosen)
            regenerated.append({"stratum":stratum,"branches":len(chosen),"nodes":nodes,"changed_actions":changed,
                "fraction":changed/nodes if nodes else None,
                "scored_nodes":scored_nodes,"changed_scored_actions":changed_scored,
                "scored_fraction":changed_scored/scored_nodes if scored_nodes else None,
                "accuracy_low":sum(r["scores"][0][metric] for r in chosen)/len(chosen) if chosen else None,
                "accuracy_high":sum(r["scores"][1][metric] for r in chosen)/len(chosen) if chosen else None})
        require(regenerated == summary["strata"],"dynamics summary mismatch")
        dynamics.append(summary)
    if not complete:
        return {"verified":True,"complete":False,"scope":"G1 carried-state dynamics only", "extended_k":extended_k,
            "family":family,"dynamics":dynamics,"checker_sha256":file_hash(__file__),"inference_replayed":False}
    lesions = []
    for version in (("pilot","stream") if family == "maze" else ("stream",)):
        source_root = Path("runs/adapter_pilot_v1") if version == "pilot" else root
        source_config = maze.read(source_root / "prepared/config.json")
        source_data = maze.read(source_root / "prepared/dataset.json")
        decoder = maze.decode_example if family == "maze" else circuit.decode
        examples = {(e["frame_index"],e["root_id"]):decoder(e) for frame in source_data["streams"] for e in frame}
        for seed in config["seeds"]:
            grid = maze.read(source_root / "selection.json")["joint-answer_only" if family == "maze" else "answer_only"]["grid"]
            cp = source_root / "training" / f"joint-answer_only-seed{seed}-grid{grid}" / "checkpoint.pt"
            for kind in ("uniform","shuffled_nodes"):
                out = root / "diagnostics" / f"{version}-answer_only-{kind}-seed{seed}"
                maze.verify_seal(out); check_resource(out)
                identity = {"track":"joint","arm":"answer_only","seed":seed,"grid":grid,
                    "checkpoint_sha256":file_hash(cp),"config_sha256":file_hash(source_root / "prepared/config.json"),
                    "evaluation_time_lesion":kind,"deployable_policy":False,"source_version":version,
                    "diagnostic_script_sha256":file_hash(out / "diagnostic_script.py")}
                checker = check_stream_file if family == "maze" else check_actions
                calculated,n = checker(out / "predictions.jsonl.gz",source_config,examples,32,identity) if family == "maze" else checker(out / "predictions.jsonl.gz",examples,source_config,32,identity)
                require(calculated == maze.read(out / "summary.json")["validation"],"lesion summary mismatch")
                lesions.append({"source_version":version,"kind":kind,"seed":seed,"predictions":n,"episodes":calculated["episodes"]})
    bins = np.linspace(0,1,21)
    histograms = []
    for stratum in ("low","middle","high"):
        for group in ("all","edited","distant"):
            vals = hist_values[stratum,group]
            histograms.append({"stratum":stratum,"group":group,"bins":bins.tolist(),"counts":np.histogram(vals,bins)[0].tolist(),
                "mean":float(np.mean(vals)) if vals else None,"std":float(np.std(vals)) if vals else None,"nodes":len(vals)})
    if family == "circuit":
        out = root / "diagnostics/pilot-restart-reference"
        maze.verify_seal(out); check_resource(out)
        examples = {(e["frame_index"],e["root_id"]):circuit.decode(e) for frame in payload["streams"] for e in frame}
        identity = {"track":"static","arm":"restart","seed":29,"grid":None,
            "checkpoint_sha256":config["source_checkpoint_sha256"],"config_sha256":file_hash(root / "prepared/config.json"),
            "source_version":"pilot","distribution_drift_reference":True}
        calculated,_ = check_actions(out / "predictions.jsonl.gz",examples,config,32,identity)
        require(calculated == maze.read(out / "summary.json")["validation"],"drift reference mismatch")
    return {"verified":True,"complete":True,"family":family,"dynamics":dynamics,"lesions":lesions,"retention_histograms":histograms,
        "retention_branch_means":histogram_root_means,"retention_scope":"descriptive pooled nodes, not independent replications or causal evidence; distant means beyond two hops in current observed undirected adjacency",
        "checker_sha256":file_hash(__file__),"inference_replayed":False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family",choices=("maze","circuit"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dynamics-only",action="store_true")
    parser.add_argument("--extended-k",action="store_true")
    args = parser.parse_args()
    maze.write(args.output,check(args.family,complete=not args.dynamics_only,extended_k=args.extended_k))
