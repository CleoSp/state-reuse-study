"""Rescore raw actions and check every recorded own-budget state transition."""
from __future__ import annotations

from pathlib import Path

from state_repair.execution.datasets import decode, frames
from state_repair.execution.durable import digest_json, read_json
from state_repair.provenance import file_hash
from .jobs import saved_records
from .metrics import episodes, input_hash, score, validate_record
from .timing import validate_timing
from state_repair.execution.records import step_rows, stored_steps


def check_job(output: Path, job: dict) -> dict:
    if job.get("mode") in ("mechanism", "dynamics"):
        return check_mechanism(output, job)
    roots = [decode(e) for e in read_json(Path(job["dataset"]))[job["split"]]][:job.get("root_limit")]
    by_root = {e.root_id: e for e in roots}
    cache = {}
    dataset_hash = file_hash(Path(job["dataset"]))
    checkpoint_hashes = {}
    from .jobs import resolve_source
    expected_source = resolve_source(job, output.parent) if job["kind"] == "evaluation" else None
    expected_keys = {(e.root_id, frame, budget, rep) for e in roots for frame in range(job["edits"]+1)
                     for budget in (job["budgets"] if job["kind"] == "evaluation" else [0]) for rep in range(job.get("repetitions", 1))}
    actual_keys = set()
    rows = []
    for row in saved_records(output):
        validate_record(row, empirical=not job["synthetic"])
        root = row["root_id"]
        key = root, row["frame"], row["K"], row.get("repetition", 0)
        if key not in expected_keys or key in actual_keys or row["seed"] != job["seed"]:
            raise ValueError("unexpected or duplicate record coverage")
        actual_keys.add(key)
        if root not in by_root:
            raise ValueError("unexpected root")
        if root not in cache:
            cache[root] = [f[0] for f in frames([by_root[root]], job["edits"], job["stream_seed"])]
        example = cache[root][row["frame"]]
        if row["input_sha256"] != input_hash(example) or row["split"] != example.split:
            raise ValueError("input provenance mismatch")
        if any(row[name] != value for name, value in score(example, row["actions"]).items()):
            raise ValueError("raw-action rescore mismatch")
        if row["record_kind"] == "fixed_budget_stream":
            if row["config_sha256"] != digest_json(job) or row["dataset_sha256"] != dataset_hash:
                raise ValueError("record configuration/dataset mismatch")
            name = row["checkpoint_job"]
            if name != expected_source:
                raise ValueError("checkpoint is not the frozen selected source")
            if name not in checkpoint_hashes:
                checkpoint_hashes[name] = file_hash(output.parent / name / "checkpoint.pt")
            if row["checkpoint_sha256"] != checkpoint_hashes[name]:
                raise ValueError("checkpoint provenance mismatch")
        validate_timing(row["milliseconds"])
        rows.append(row)
    expected = len(roots)*(job["edits"]+1)*(len(job["budgets"]) if job["kind"] == "evaluation" else 1)*job.get("repetitions", 1)
    if len(rows) != expected:
        raise ValueError("missing raw records")
    if job["edits"]:
        aggregated = episodes(rows, job["edits"], empirical=not job["synthetic"])
    else:
        aggregated = []
    return {"verified": True, "raw_actions_rescored": len(rows), "episodes": aggregated,
        "records_sha256": file_hash(stored_steps(output)), "synthetic": job["synthetic"]}


def check_mechanism(output: Path, job: dict) -> dict:
    from dataclasses import asdict
    value = read_json(Path(job["dataset"]))
    entries = value.get("interventions")
    if job.get("mode") == "dynamics":
        entries = [{"old": asdict(decode(r)), "new": asdict(frames([decode(r)], 4, job["stream_seed"])[4][0]), "branch": "carried-frame4"}
                   for r in value[job["split"]]][:job.get("root_limit")]
    if entries is None:
        roots = [decode(e) for e in value[job["split"]]][:job.get("root_limit")]
        entries = [{"old": asdict(r), "new": asdict(frames([r], 1, job["stream_seed"])[1][0]), "branch": "ordinary"} for r in roots]
    examples = {(entry["old"]["root_id"], entry["branch"]): decode(entry["new"]) for entry in entries}
    priors, keys = {}, set()
    policies = {"spatial_gate", "global_gate", "answer_only", "restart", "carry", "random_reset", "local_reset_1",
                "local_reset_2", "local_reset_3", "noisy_carry", "shuffled_gate", "impact_mask", "answer_uniform", "answer_shuffled_nodes"}
    policies = set(job.get("intervention_policies", policies))
    if job.get("mode") == "dynamics":
        policies = {"carry_refinement"}
    from .jobs import resolve_source
    checkpoint = file_hash(output.parent / resolve_source(job, output.parent) / "checkpoint.pt")
    data_hash = file_hash(Path(job["dataset"]))
    import json
    import hashlib
    import base64
    import zlib
    saved_priors = {}
    for unit in step_rows(output):
        digest = hashlib.sha256()
        for name in ("a", "z", "logits"):
            value = unit["saved_prior"][name]
            raw = zlib.decompress(base64.b64decode(value["float32_zlib_base64"]))
            import math
            if len(raw) != 4*math.prod(value["shape"]):
                raise ValueError("saved prior tensor shape mismatch")
            digest.update(raw)
        if digest.hexdigest() != unit["prior_state_sha256"]:
            raise ValueError("saved prior hash mismatch")
        saved_priors[unit["root_id"], unit["branch"]] = digest.hexdigest()
    for row in saved_records(output):
        validate_record(row, empirical=not job["synthetic"])
        if row["checkpoint_sha256"] != checkpoint or row["source_K"] != job["source_K"] or row["dataset_sha256"] != data_hash or row["config_sha256"] != digest_json(job):
            raise ValueError("frozen intervention provenance mismatch")
        group = row["root_id"], row["branch"]
        key = (*group, row["K"], row["policy"])
        if key in keys or group not in examples:
            raise ValueError("duplicate or unexpected intervention")
        keys.add(key)
        if saved_priors[group] != row["prior_state_sha256"] or priors.setdefault(row["root_id"], row["prior_state_sha256"]) != row["prior_state_sha256"]:
            raise ValueError("interventions do not share a frozen prior")
        example = examples[group]
        if row["input_sha256"] != input_hash(example) or any(row[name] != v for name, v in score(example, row["actions"]).items()):
            raise ValueError("intervention raw-action rescore mismatch")
    if keys != {(*group, k, p) for group in examples for k in job["budgets"] for p in policies}:
        raise ValueError("incomplete intervention matrix")
    return {"verified": True, "raw_actions_rescored": len(keys), "episodes": [],
        "records_sha256": file_hash(stored_steps(output)), "synthetic": job["synthetic"],
        "record_kinds": ["frozen_state_intervention", "privileged_diagnostic"]}
