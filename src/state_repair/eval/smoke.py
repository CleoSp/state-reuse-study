"""Static CPU evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from state_repair.data.maze import MazeExample, observation
from state_repair.data.serialization import load_split
from state_repair.models.recursive import RecursiveSolver
from state_repair.oracles.maze import score_policy
from state_repair.provenance import file_hash, source_provenance


def evaluate_examples(model: RecursiveSolver, examples: Sequence[MazeExample],
                      budgets: Sequence[int], batch_size: int = 1) -> list[dict]:
    """Each static root/K starts fresh; score only after prediction is fixed.

    Timings include observable tensor construction and state bookkeeping, with
    offline oracle scoring excluded. This is batch-one latency, not throughput.
    No warmup is hidden; callers must log any warmup they choose to perform.
    """
    if batch_size!=1 or model.initial_a.device.type!="cpu":
        raise ValueError("smoke evaluation currently supports CPU batch_size=1 only")
    if not budgets or len(set(budgets))!=len(budgets) or any(type(k)!=int or k<0 for k in budgets):
        raise ValueError("budgets must be unique nonnegative integers")
    if any(e.frame_index!=0 for e in examples):
        raise ValueError("static evaluation accepts frame zero only")
    if any(e.split not in ("train","val") for e in examples):
        raise ValueError("smoke evaluation is restricted to train/val")
    was_training=model.training
    model.eval()
    records=[]
    try:
        with torch.no_grad():
            for example in examples:
                for budget in budgets:
                    times=[time.perf_counter()]
                    obs=observation(example.maze,example.root_id,0)
                    times.append(time.perf_counter())
                    encoded=model.encode(obs)
                    times.append(time.perf_counter())
                    state=model.fresh_state(obs)
                    times.append(time.perf_counter())
                    for _ in range(budget):
                        state=model.step(encoded,state)
                    times.append(time.perf_counter())
                    prediction=model.decode(obs,state)
                    times.append(time.perf_counter())
                    actions=prediction.logits[0].argmax(dim=-1).tolist()
                    times.append(time.perf_counter())
                    measured={name:1000*(times[i+1]-times[i]) for i,name in enumerate(
                        ("input_ms","encoder_ms","initialization_ms","core_ms","decode_ms","copy_ms"))}
                    calls=(model.inner_cycles+1)*budget
                    records.append({"schema_version":1,"synthetic":example.synthetic,
                        "record_kind":"static_development","root_id":example.root_id,"frame_index":0,
                        "split":example.split,"K":budget,"policy":"restart","batch_size":1,
                        "outer_cycles":budget,"block_calls":calls,"transformer_layer_executions":2*calls,
                        "input_sha256":hashlib.sha256(json.dumps({"shape":[example.maze.height,example.maze.width],
                            "edges":example.maze.edges,"start":example.maze.start,"goal":example.maze.goal},sort_keys=True).encode()).hexdigest(),
                        "prediction_sha256":hashlib.sha256(json.dumps(actions).encode()).hexdigest(),
                        "state_bytes":sum(t.numel()*t.element_size() for t in (state.a,state.z,state.node_ids,state.valid_nodes)),
                        "total_inference_ms":1000*(times[-1]-times[0]),**measured,**score_policy(example.maze,actions)})
    finally:
        model.train(was_training)
    return records


def aggregate(records: Sequence[dict], *, empirical: bool = True) -> list[dict]:
    if not records:
        raise ValueError("no episode records")
    if empirical and any(r.get("synthetic") is not False for r in records):
        raise ValueError("empirical report rejects synthetic or unmarked fixtures")
    groups={}
    seen=set()
    for row in records:
        key=(row.get("stage","unspecified"),row["split"],row["K"])
        identity=(*key,row["root_id"])
        if identity in seen or row["frame_index"]!=0:
            raise ValueError("static aggregate requires one frame per root/K/stage")
        seen.add(identity)
        groups.setdefault(key,[]).append(row)
    output=[]
    for (stage,split,budget),rows in sorted(groups.items()):
        output.append({"stage":stage,"split":split,"K":budget,"episodes":len(rows),
            **{name:sum(float(r[name]) for r in rows)/len(rows) for name in
               ("route_correct","all_node_correct","valid_action_accuracy","total_inference_ms")}})
    return output


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"required raw records missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_run(run: Path) -> tuple[RecursiveSolver,dict,dict]:
    required=("config.json","summary.json","provenance.json","checkpoint.pt","pre_metrics.jsonl","post_metrics.jsonl","steps.jsonl")
    for name in required:
        if not (run/name).is_file():
            raise FileNotFoundError(f"required artifact missing: {run/name}")
    config=json.loads((run/"config.json").read_text(encoding="utf-8"))
    summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
    if summary.get("status")!="completed":
        raise ValueError("run did not complete; inspect failure summary before evaluation")
    for name in ("pre_metrics.jsonl","post_metrics.jsonl","steps.jsonl","provenance.json"):
        if summary.get("artifact_hashes",{}).get(name)!=file_hash(run/name):
            raise ValueError(f"raw artifact hash mismatch or missing: {name}")
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if file_hash(run/"checkpoint.pt")!=summary["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    checkpoint=torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=True)
    if checkpoint.get("schema_version")!=1 or checkpoint.get("mode")!="static":
        raise ValueError("unsupported checkpoint schema/mode")
    if checkpoint["config_sha256"]!=digest or checkpoint["model_config"]!=config["model"]:
        raise ValueError("checkpoint/config mismatch")
    provenance=json.loads((run/"provenance.json").read_text(encoding="utf-8"))
    if provenance!=checkpoint["source_provenance"]:
        raise ValueError("checkpoint/provenance mismatch")
    if summary.get("config_sha256")!=digest or summary.get("split_hashes")!=checkpoint["split_hashes"] or summary.get("optimizer_steps")!=checkpoint["optimizer_steps"]:
        raise ValueError("summary/checkpoint provenance mismatch")
    if checkpoint["source_provenance"]["source_tree_sha256"]!=source_provenance()["source_tree_sha256"]:
        raise ValueError("checkpoint source differs from current code; use original source before evaluating")
    model_cfg=config["model"]
    model=RecursiveSolver(width=model_cfg["width"],heads=model_cfg["heads"],inner_cycles=model_cfg["inner_cycles"],
                          attention_mode="dense")
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    torch.set_num_threads(config["training"]["threads"])
    return model,config,checkpoint


def evaluate_run(run: Path) -> dict:
    model,config,checkpoint=_load_run(run)
    dataset=Path(config["output_dir"])/"dataset"
    manifest=json.loads((dataset/"manifest.json").read_text(encoding="utf-8"))
    for split in ("train","val"):
        if manifest["splits"][split]["sha256"]!=checkpoint["split_hashes"][split]:
            raise ValueError("checkpoint/dataset split hash mismatch")
    destination=run/"evaluation.jsonl"
    if destination.exists():
        raise FileExistsError("evaluation already exists; inspect existing records before another run")
    examples=[e for e in load_split(dataset,"val") if e.frame_index==0]
    records=evaluate_examples(model,examples,config["training"]["eval_budgets"])
    for row in records:
        row.update(stage="reloaded_validation",checkpoint_sha256=file_hash(run/"checkpoint.pt"),seed=checkpoint["seed"])
    with destination.open("x",encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    evidence={"schema_version":1,"evaluation_sha256":file_hash(destination),"checkpoint_sha256":file_hash(run/"checkpoint.pt")}
    (run/"evaluation_manifest.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    return {"evaluation_file":str(destination),"aggregates":aggregate(records)}


def report_run(run: Path) -> dict:
    raise RuntimeError("Report creation is frozen pending a verified static generalization result; see REPRODUCING.md")


def _historical_report_run(run: Path) -> dict:
    """Build a static evaluation report from saved records."""
    _model,config,checkpoint=_load_run(run)
    records=_read_jsonl(run/"pre_metrics.jsonl")+_read_jsonl(run/"post_metrics.jsonl")
    steps=_read_jsonl(run/"steps.jsonl")
    if len(steps)!=checkpoint["optimizer_steps"]:
        raise ValueError("optimizer-step log count mismatch")
    output={"status":"development_only","run":str(run),"seed":config["seed"],
        "aggregates":aggregate(records),"limits":"One seed, static train/validation only. No repair or generalization claim; accelerators not tested."}
    if (run/"evaluation.jsonl").exists():
        evidence=json.loads((run/"evaluation_manifest.json").read_text(encoding="utf-8"))
        if file_hash(run/"evaluation.jsonl")!=evidence["evaluation_sha256"] or file_hash(run/"checkpoint.pt")!=evidence["checkpoint_sha256"]:
            raise ValueError("evaluation artifact hash mismatch")
        output["reloaded_validation"]=aggregate(_read_jsonl(run/"evaluation.jsonl"))
    return output
