"""Resumable independent evaluation units dispatched only by the matrix driver."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from state_repair.execution.datasets import collate, decode, frames
from state_repair.execution.durable import digest_json, read_json
from state_repair.execution.jobs import model_for, adapter_for, load_weights
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.provenance import file_hash
from .metrics import input_hash, score, validate_record
from .timing import STAGES, measured_frame


def saved_records(directory: Path):
    from state_repair.execution.records import step_rows
    for row in step_rows(directory):
        yield from row.get("records", [])


def resolve_source(job: dict, root: Path) -> str:
    if job.get("selection"):
        row = read_json(root / job["selection"] / "summary.json")
        import json
        selected = json.loads((root / job["selection"] / "steps.jsonl").read_text())["selection"]
        return selected[job.get("owner", job["arm"])][str(job["seed"])]
    return job["source"]


class WorkUnit:
    """Common checkpoint shape for deterministic CPU work and inference jobs."""
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.schedule = [(0, 0)]

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("unexpected work-unit state")


class EvaluationJob(WorkUnit):
    def __init__(self, job: dict, root: Path):
        super().__init__()
        self.job, self.root = job, root
        self.source = resolve_source(job, root)
        self.solver = model_for(job)
        source_job = read_json(root / self.source / "job.json")
        self.adapter = adapter_for(source_job.get("arm", "restart"), job).to(job["device"])
        load_weights(self.solver, self.adapter, root, self.source,
                     backbone_only=source_job["recipe"] == "static")
        if job.get("lesion"):
            from .lesions import AnswerLesion
            self.adapter = AnswerLesion(self.adapter, job["lesion"]).to(job["device"])
        if job["arm"] != source_job.get("arm", "restart"):
            if job["arm"] == "random_reset" and job.get("selection"):
                import json
                selected = json.loads((root / job["selection"] / "steps.jsonl").read_text())
                job = {**job, "reset_rate": selected["random_reset_rates"][str(job["seed"]) ]}
            self.adapter = adapter_for(job["arm"], job,
                spatial=self.adapter if job["arm"] == "shuffled_gate" else None).to(job["device"])
        self.model = torch.nn.ModuleDict({"backbone": self.solver, "adapter": self.adapter})
        self.model.eval().requires_grad_(False)
        self.checkpoint_hash = file_hash(root / self.source / "checkpoint.pt")
        self.dataset_hash = file_hash(Path(job["dataset"]))
        self.roots = [decode(e) for e in read_json(Path(job["dataset"]))[job["split"]]][:job.get("root_limit")]
        if any(e.split != job["split"] or e.synthetic != job["synthetic"] for e in self.roots):
            raise ValueError("evaluation root flags mismatch")
        self.units = [(offset, repetition) for repetition in range(job.get("repetitions", 1))
                      for offset in range(0, len(self.roots), job["batch_size"])]
        self.schedule = [(unit, k) for k in job["budgets"] for unit in range(len(self.units))]

    @torch.no_grad()
    def step(self, index: int, batch: int, k: int) -> dict:
        job = self.job
        offset, repetition = self.units[batch]
        examples = self.roots[offset:offset+job["batch_size"]]
        stream = frames(examples, job["edits"], job["stream_seed"])
        records, calls = [], 0
        policy = FixedBudgetPolicy(self.solver, self.adapter, k)
        first_obs = collate(stream[0])[0]
        for _ in range(job.get("warmup", 1)):
            policy(first_obs.to(job["device"]))
        policy.reset()
        for f, values in enumerate(stream):
            obs = collate(values, None if f == 0 else stream[f-1])[0]
            result, actions, times = measured_frame(policy, obs, job["device"])
            calls += result.block_calls
            for e, action in zip(values, actions):
                row = {"suite": job["suite"], "seed": job["seed"], "policy": job["arm"],
                    "K": k, "root_id": e.root_id, "frame": f, "split": e.split,
                    "synthetic": job["synthetic"], "record_kind": "fixed_budget_stream", "privileged": False,
                    "checkpoint_sha256": self.checkpoint_hash, "checkpoint_job": self.source,
                    "config_sha256": digest_json(job), "dataset_sha256": self.dataset_hash,
                    "input_sha256": input_hash(e), "actions": action, "prediction_sha256": digest_json(action),
                    "state_budget": result.state.budget, "source_budget": None if f == 0 else k,
                    "inner_cycles": job["inner_cycles"], "block_calls": result.block_calls,
                    "milliseconds": {name: value/len(values) for name, value in times.items()},
                    "operations": {name: value/len(values) for name, value in result.operations.items()},
                    "measurement": "batch_one_latency" if len(values) == 1 else "batched_throughput",
                    "batch_size": len(values), "dtype": "float32", "device": job["device"],
                    "repetition": repetition, "warmup_frames": job.get("warmup", 1),
                    "backend": "pytorch_dense_masked_neighbor", **score(e, action)}
                if job.get("comparison_track"):
                    row["policy"] = job["comparison_track"]+"/"+row["policy"]
                row["deployable_policy"] = not bool(job.get("lesion"))
                if result.adapter.retain_a is not None:
                    row["mean_retention"] = float((result.adapter.retain_a.mean()+result.adapter.retain_z.mean())/2)
                validate_record(row, empirical=not job["synthetic"])
                records.append(row)
        policy.reset()
        warmup_calls = job.get("warmup", 1)*k*(job["inner_cycles"]+1)
        return {"records": records, "loss": 0., "forward_calls": calls+warmup_calls, "warmup_forward_calls": warmup_calls}


class ReferenceJob(WorkUnit):
    def __init__(self, job: dict, root: Path):
        super().__init__()
        self.job = job
        self.roots = [decode(e) for e in read_json(Path(job["dataset"]))[job["split"]]]
        self.schedule = [(i, 0) for i in range(len(self.roots))]

    def step(self, index: int, batch: int, k: int) -> dict:
        from state_repair.oracles.incremental_references import DStarLite
        from state_repair.oracles.circuit import EventDrivenEvaluator
        job, records = self.job, []
        stream = frames([self.roots[batch]], job["edits"], job["stream_seed"])
        for f, values in enumerate(stream):
            e = values[0]
            start = perf_counter()
            if job["family"] == "maze":
                if f == 0:
                    solver = DStarLite(e.maze)
                    action = solver.actions()
                else:
                    action = solver.update(e.maze)
            else:
                if f == 0:
                    solver = EventDrivenEvaluator(e.circuit)
                    stable = solver.values
                else:
                    stable = solver.update(e.circuit)
                action = [stable[u] for u in e.node_order]
            milliseconds = (perf_counter()-start)*1000
            times = {name: 0. for name in STAGES}
            times.update(core=milliseconds, total=milliseconds)
            records.append({"suite": job["suite"], "seed": 0, "policy": "dstar_lite_full_policy" if job["family"] == "maze" else "event_driven",
                "K": 0, "root_id": e.root_id, "frame": f, "split": e.split, "synthetic": job["synthetic"],
                "record_kind": "reference_solver", "privileged": False, "input_sha256": input_hash(e),
                "actions": action, "prediction_sha256": digest_json(action), "milliseconds": times,
                "operations": {"total_macs": 0}, "batch_size": 1, "measurement": "batch_one_latency",
                "timing_scope": "combined observed-edit scan, queue repair, decoding and result copy in core",
                "device": "cpu_on_rtx5070_host", **score(e, action)})
        return {"records": records, "loss": 0., "forward_calls": 0}


class SelectionJob(WorkUnit):
    def __init__(self, job: dict, root: Path):
        super().__init__()
        self.job, self.root = job, root

    def step(self, index: int, batch: int, k: int) -> dict:
        from .metrics import episodes
        job, values = self.job, {}
        for candidate in job["candidates"]:
            from .checker import check_job
            candidate_job = read_json(self.root / candidate["evaluation"] / "job.json")
            check_job(self.root / candidate["evaluation"], candidate_job)
            rows = list(saved_records(self.root / candidate["evaluation"]))
            if any(r["split"] != "val" for r in rows):
                raise ValueError("selection must use validation only")
            if job.get("static_gate"):
                means = [sum(r["exact_correct"] for r in rows if r["K"] == budget)/sum(r["K"] == budget for r in rows)
                         for budget in job["budgets"]]
                if means[-1] < .85 or any(a>b for a,b in zip(means, means[1:])):
                    raise ValueError("static backbone failed frozen competence/monotonicity gate: " + candidate["source"])
                values[candidate["source"]] = means
            else:
                ep = episodes(rows, job["edits"], empirical=not job["synthetic"])
                values[candidate["source"]] = sum(e["post_accuracy"] for e in ep)/len(ep)
        if job.get("static_gate"):
            return {"loss": 0., "forward_calls": 0, "static_gates": values}
        selection = {}
        grids = {}
        for arm in job["arms"]:
            candidates = [c for c in job["candidates"] if c["arm"] == arm]
            grid_means = {g: sum(values[c["source"]] for c in candidates if c["grid"] == g)/sum(c["grid"] == g for c in candidates)
                          for g in (0, 1)}
            grid = max(grid_means, key=lambda g: (grid_means[g], -g))
            grids[arm] = {"grid": grid, "candidate_validation_means": grid_means}
            selection[arm] = {str(c["seed"]): c["source"] for c in candidates if c["grid"] == grid}
        reset_rates = {}
        for seed, source in selection.get("spatial_gate", {}).items():
            candidate = next(c for c in job["candidates"] if c["source"] == source)
            retained = [r["mean_retention"] for r in saved_records(self.root / candidate["evaluation"]) if r["frame"] > 0]
            reset_rates[seed] = 1-sum(retained)/len(retained)
        return {"loss": 0., "forward_calls": 0, "selection": selection, "grids": grids, "random_reset_rates": reset_rates,
            "rule": "mean post-edit exact accuracy over all training seeds and declared tuning K; ties lower grid"}


def build_evaluation_job(job: dict, root: Path):
    if job["kind"] == "evaluation":
        if job.get("mode") == "dynamics":
            from .dynamics import DynamicsJob
            return DynamicsJob(job, root)
        if job.get("mode") == "mechanism":
            from .mechanism import MechanismJob
            return MechanismJob(job, root)
        return EvaluationJob(job, root)
    if job["kind"] == "reference":
        return ReferenceJob(job, root)
    if job["kind"] == "selection":
        return SelectionJob(job, root)
    if job["kind"] == "report":
        from .report import ReportJob
        return ReportJob(job, root)
    raise ValueError("unsupported production job kind")
