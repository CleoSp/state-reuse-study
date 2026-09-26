"""Frozen-prior interventions under identical solver weights; diagnostics only."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from state_repair.execution.datasets import collate, decode, frames
from state_repair.execution.durable import digest_json, read_json
from state_repair.execution.jobs import adapter_for
from state_repair.models.adapters import make_adapter
from state_repair.types import PredictionBatch, ObservedEdit, TransitionInput
from .frozen import frozen_adapter_prediction
from .interventions import impact_mask_reset
from .jobs import EvaluationJob
from .metrics import input_hash, score, validate_record


def tensor_hash(*values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def pack_prior(a, z, logits) -> dict:
    import base64
    import zlib
    return {name: {"shape": list(value.shape), "float32_zlib_base64": base64.b64encode(zlib.compress(
        value.detach().cpu().contiguous().numpy().tobytes())).decode()}
        for name, value in (("a", a), ("z", z), ("logits", logits))}


def impact(old, new) -> list[bool]:
    if hasattr(old, "maze"):
        from state_repair.oracles.maze import impact_metadata
        return impact_metadata(old.maze, new.maze)["action_set_changed"]
    from state_repair.oracles.circuit import evaluate
    a, b = evaluate(old.circuit), evaluate(new.circuit)
    return [a[u] != b[u] for u in new.node_order]


class MechanismJob(EvaluationJob):
    def __init__(self, job: dict, root: Path):
        super().__init__(job, root)
        self.entries = read_json(Path(job["dataset"])).get("interventions")
        if self.entries is None:
            self.entries = [{"old": asdict(r), "new": asdict(frames([r], 1, job["stream_seed"])[1][0]),
                             "branch": "ordinary"} for r in self.roots]
        self.schedule = [(i, 0) for i in range(len(self.entries))]
        self.adapters = {"spatial_gate": self.adapter}
        self.policies = job.get("intervention_policies", ["spatial_gate", "global_gate", "answer_only", "restart", "carry", "random_reset",
            "local_reset_1", "local_reset_2", "local_reset_3", "noisy_carry", "shuffled_gate", "impact_mask", "answer_uniform", "answer_shuffled_nodes"])
        self.adapter_sources = {}
        for arm in (a for a in ("global_gate", "answer_only", "gru_adapter", "residual_adapter") if a in self.policies):
            adapter = adapter_for(arm, job).to(job["device"])
            selected = job.get("adapter_sources", {}).get(arm)
            if not selected and job.get("selection"):
                from .jobs import resolve_source
                selected = resolve_source({**job, "owner": arm}, root)
            if selected:
                checkpoint = torch.load(root / selected / "checkpoint.pt", weights_only=False, map_location="cpu")
                adapter.load_state_dict({k.removeprefix("adapter."): v for k, v in checkpoint["model"].items() if k.startswith("adapter.")})
            elif arm == "global_gate":
                adapter.load_state_dict(self.adapter.state_dict())
            self.adapter_sources[arm] = selected or "synthetic_fixture_or_pooled_spatial"
            self.adapters[arm] = adapter.eval()

    @torch.no_grad()
    def step(self, index: int, batch: int, k: int) -> dict:
        job = self.job
        entry = self.entries[batch]
        old, new = decode(entry["old"]), decode(entry["new"])
        o = collate([old])[0].to(job["device"])
        n = collate([new], [old])[0].to(job["device"])
        initial = self.solver(o, job["source_K"])
        prior = initial.state.detach()
        prior_hash = tensor_hash(prior.a, prior.z, initial.prediction.logits)
        prior_tensors = pack_prior(prior.a, prior.z, initial.prediction.logits)
        changed = impact(old, new)
        fraction = sum(changed)/len(changed)
        transition = TransitionInput(n, o, prior, ObservedEdit.between(o, n))
        retained = self.adapter(transition, self.solver.fresh_state(n))
        reset_rate = 1-float((retained.retain_a.mean()+retained.retain_z.mean())/2)
        adapters = {**self.adapters, **{arm: adapter_for(arm, {**job, "reset_rate": reset_rate}, spatial=self.adapter).to(job["device"])
            for arm in ("restart", "carry", "random_reset", "local_reset_1", "local_reset_2", "local_reset_3", "noisy_carry", "shuffled_gate") if arm in self.policies}}
        records, calls = [], initial.block_calls
        for budget in job["budgets"]:
            for arm in self.policies:
                privileged = arm == "impact_mask"
                prediction = initial.prediction
                if arm in ("answer_uniform", "answer_shuffled_nodes"):
                    logits = prediction.logits.clone()
                    if arm == "answer_uniform":
                        logits.zero_()
                    else:
                        logits = logits[:, torch.randperm(logits.shape[1], device=logits.device)]
                    prediction = PredictionBatch(logits)
                    adapter = adapters["answer_only"]
                elif not privileged:
                    adapter = adapters[arm]
                if privileged:
                    mask = torch.tensor([changed], device=job["device"], dtype=torch.bool)
                    initialized = impact_mask_reset(prior, self.solver.fresh_state(n), mask)
                    result = self.solver(n, budget, initialized.state)
                    gate_a = gate_z = None
                else:
                    fork = frozen_adapter_prediction(self.solver, adapter, o, n, prior, budget,
                        previous_prediction=prediction if adapter.requires_previous_prediction else None)
                    result = fork.solver
                    gate_a, gate_z = fork.adapter.retain_a, fork.adapter.retain_z
                    if arm == "shuffled_gate":
                        for actual, expected in ((gate_a, retained.retain_a), (gate_z, retained.retain_z)):
                            if not torch.equal(actual.flatten().sort().values, expected.flatten().sort().values):
                                raise ValueError("gate shuffling changed retention distribution")
                calls += result.block_calls
                actions = result.prediction.logits.argmax(-1)[0].cpu().tolist()
                row = {"suite": job["suite"], "seed": job["seed"], "root_id": old.root_id,
                    "branch": entry["branch"], "frame": 1, "split": old.split, "K": budget, "policy": arm,
                    "synthetic": job["synthetic"], "record_kind": "privileged_diagnostic" if privileged else "frozen_state_intervention",
                    "privileged": privileged, "deployable_policy": False, "checkpoint_sha256": self.checkpoint_hash,
                    "source_K": job["source_K"], "source_policy": "restart", "prior_state_sha256": prior_hash,
                    "input_sha256": input_hash(new), "prior_input_sha256": input_hash(old),
                    "actions": actions, "prediction_sha256": digest_json(actions), "block_calls": result.block_calls,
                    "impact_fraction": fraction, "stratum": "low" if fraction <= .1 else "high" if fraction >= .4 else "middle",
                    "state_distance_from_prior": {"a": float((result.state.a-prior.a).norm()), "z": float((result.state.z-prior.z).norm())},
                    "retention_a": None if gate_a is None else gate_a.flatten().cpu().tolist(),
                    "retention_z": None if gate_z is None else gate_z.flatten().cpu().tolist(),
                    "adapter_sources": self.adapter_sources, "dataset_sha256": self.dataset_hash, "config_sha256": digest_json(job),
                    **score(new, actions)}
                validate_record(row, empirical=not job["synthetic"])
                records.append(row)
        return {"loss": 0., "forward_calls": calls, "records": records, "saved_prior": prior_tensors,
                "prior_state_sha256": prior_hash, "root_id": old.root_id, "branch": entry["branch"]}
