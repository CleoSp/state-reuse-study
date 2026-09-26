"""Experiment job graph and measured capacity projection."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import statistics

from state_repair.train.pilot import PRINCIPAL, AUXILIARY, schedule
from state_repair.provenance import file_hash
from .datasets import identities
from .durable import read_json, digest_json

SEEDS = [29, 43, 71, 101, 137]
BUDGETS = [1, 2, 4, 8, 16]
ROOT = "runs/confirmatory_v1"


def suites() -> dict:
    result = {}
    for family, sizes in (("maze", (12, 16, 20)), ("circuit", (32, 48, 64, 96))):
        for size in sizes:
            name = family + str(size)
            trained = size in ((12, 16) if family == "maze" else (32, 64))
            counts = {"train": (1024 if family == "maze" else 2048) if trained else 0, "val": 256, "test": 256}
            ids = {split: identities(name, split, count) for split, count in counts.items()}
            if family == "maze":
                ids["challenge"] = identities(name + "-challenge", "test", 128)
            result[name] = {"family": family, "size": size, "trained": trained, "depth_shift": family == "circuit" and size in (48, 96),
                "inputs": 8 if size in (32, 48) else 16, "data_seed": 62017, "identities": ids,
                "identity_hashes": {split: digest_json(names) for split, names in ids.items()},
                "stream_seed": 64019, "challenge_seed": 64023, "challenge_pairs_per_type": 32}
    return result


def matched_steps(seeds: list[int], train_batches: int) -> tuple[int, dict]:
    targets = {s: sum(k*5 for _, k in schedule(s, train_batches, {"steps": 1024, "budgets": [1,2,4,8]})) for s in seeds}
    best = None
    for steps in range(2200, 2901):
        actual = {s: sum(k*2 for _, k in schedule(s, train_batches*4, {"steps": steps, "budgets": [1,2,4,8]})) for s in seeds}
        differences = {s: actual[s]/targets[s]-1 for s in seeds}
        candidate = (max(abs(v) for v in differences.values()), steps, differences)
        if best is None or candidate[0:2] < best[0:2]:
            best = candidate
    if best[0] > .05:
        raise ValueError("common matched-call control exceeds five percent mismatch")
    return best[1], best[2]


def build_matrix(*, fallback: bool = False) -> dict:
    definitions = suites()
    jobs = []
    controls_steps, control_errors = matched_steps(SEEDS, 16)
    def add(job):
        job.setdefault("seconds", 14400)
        job.setdefault("synthetic", False)
        job.setdefault("threads", 2)
        job.setdefault("dependencies", [])
        job.setdefault("compress_records", job["kind"] in ("evaluation", "reference"))
        job.setdefault("unit_reserve_seconds", 180 if job["kind"] == "evaluation" else 60)
        job.setdefault("device", "cuda")
        job.setdefault("estimated_peak_bytes", 0 if job["device"] == "cpu" else int(9.8*2**30))
        jobs.append(job)
        return job["id"]
    for name, spec in definitions.items():
        if not spec["trained"]:
            continue
        family, size = spec["family"], spec["size"]
        adapter_seeds = SEEDS[:3] if fallback and name == "maze16" else SEEDS
        arms = list(PRINCIPAL) if family == "maze" else ["restart", "carry", "spatial_gate", "answer_only"]
        base = {"suite": name, "family": family, "size": size, "width": 64 if family == "maze" else 128,
            "heads": 2 if family == "maze" else 4, "inner_cycles": 2, "context_width": 16,
            "dataset": f"{ROOT}/data/{name}-development.json", "edit_seed": 54017, "stream_seed": 64019}
        static_gates = []
        for seed in SEEDS:
            source = add({**base, "id": f"static-{name}-s{seed}", "kind": "research_training", "recipe": "static",
                "seed": seed, "steps": 6000, "batch_size": 64 if family == "maze" else 128,
                "microbatch_size": 16 if name == "maze16" else 64 if family == "maze" else 128,
                "budgets": BUDGETS, "learning_rate": .001, "final_learning_rate": .0003,
                "weight_decay": 0., "gradient_clip": 1.})
            validation = add({**base, "id": f"static-validation-{name}-s{seed}", "kind": "evaluation", "arm": "restart", "source": source,
                "seed": seed, "split": "val", "edits": 0, "batch_size": 64, "budgets": BUDGETS, "dependencies": [source]})
            static_gates.append({"source": source, "evaluation": validation})
        gate = add({"id": f"static-gate-{name}", "kind": "selection", "device": "cpu", "static_gate": True,
            "budgets": BUDGETS, "candidates": static_gates, "dependencies": [c["evaluation"] for c in static_gates]})
        candidates = []
        for arm in arms:
            for grid in (0, 1):
                for seed in adapter_seeds:
                    source = f"static-{name}-s{seed}"
                    trained = add({**base, "id": f"adapter-{name}-{arm}-g{grid}-s{seed}", "kind": "research_training", "recipe": "stream",
                        "source": source, "arm": arm, "seed": seed, "steps": 1024, "batch_size": 64, "budgets": [1,2,4,8] if family == "maze" else [1,2],
                        "learning_rate": [.0001,.0003][grid], "adapter_learning_rate": [.0003,.001][grid],
                        "weight_decay": 0., "gradient_clip": 1., "dependencies": [gate, source]})
                    tuning = add({**base, "id": f"tuning-{name}-{arm}-g{grid}-s{seed}", "kind": "evaluation", "arm": arm, "source": trained,
                        "seed": seed, "split": "val", "edits": 4, "batch_size": 64, "budgets": [1,2,4,8] if family == "maze" else [1,2],
                        "dependencies": [trained]})
                    candidates.append({"arm": arm, "grid": grid, "seed": seed, "source": trained, "evaluation": tuning})
        selection = add({"id": f"selection-{name}", "kind": "selection", "device": "cpu", "arms": arms, "edits": 4,
            "candidates": candidates, "dependencies": [c["evaluation"] for c in candidates]})
        controls = {}
        if family == "maze" and not (fallback and name == "maze16"):
            original = read_json(Path("runs/adapter_pilot_v1/selection.json"))
            for arm in ("restart", "carry", "spatial_gate", "answer_only"):
                grid = original["joint-"+arm]["grid"]
                for seed in SEEDS:
                    controls[arm, seed] = add({**base, "id": f"matched-{name}-{arm}-s{seed}", "kind": "research_training", "recipe": "one_edit",
                        "source": f"static-{name}-s{seed}", "arm": arm, "seed": seed, "steps": controls_steps, "batch_size": 64,
                        "microbatch_size": 32 if size == 16 else 64, "budgets": [1,2,4,8], "learning_rate": [.0001,.0003][grid],
                        "adapter_learning_rate": [.0003,.001][grid], "weight_decay": 0., "gradient_clip": 1.,
                        "dependencies": [gate, f"static-{name}-s{seed}"]})
        evaluation_suites = [name, "maze20"] if family == "maze" else [name, "circuit48" if size == 32 else "circuit96"]
        for suite in evaluation_suites:
            evaluated = []
            evaluation_base = {**base, "suite": f"{name}-on-{suite}", "dataset": f"{ROOT}/data/{suite}-test.json",
                               "split": "test", "batch_size": 64, "budgets": BUDGETS, "estimated_peak_bytes": 4*2**30}
            horizons = (32, 128) if suite == name and not (fallback and name == "maze16") else (32,)
            for horizon in horizons:
                ordinary = []
                timing = {"latency": [], "throughput": []}
                for arm in arms + (list(AUXILIARY) if family == "maze" and horizon == 32 else []):
                    for seed in adapter_seeds:
                        suffix = f"{name}-on-{suite}-{arm}-s{seed}-h{horizon}"
                        job = {**evaluation_base, "id": "test-"+suffix, "kind": "evaluation", "arm": arm, "seed": seed, "edits": horizon,
                               "selection": selection, "owner": arm if arm in arms else "spatial_gate", "dependencies": [selection]}
                        ordinary.append(add(job))
                        if horizon == 32 and arm in arms and seed == SEEDS[0] and not (name == "maze16" and suite == "maze20"):
                            for measurement, batch_size, roots in (("latency", 1, 8), ("throughput", 64, 64)):
                                timing[measurement].append(add({**job, "id": measurement+"-"+suffix, "batch_size": batch_size,
                                     "root_limit": roots, "repetitions": 3, "mode": measurement}))
                for (arm, seed), source in controls.items():
                    ordinary.append(add({**evaluation_base, "id": f"matched-test-{name}-on-{suite}-{arm}-s{seed}-h{horizon}", "kind": "evaluation",
                        "arm": arm, "comparison_track": "one_edit", "seed": seed, "edits": horizon, "source": source, "dependencies": [source]}))
                if horizon == 32:
                    for seed in SEEDS[:3]:
                        for lesion in ("uniform", "shuffled_nodes"):
                            ordinary.append(add({**evaluation_base, "id": f"lesion-{name}-on-{suite}-{lesion}-s{seed}", "kind": "evaluation",
                                "arm": "answer_only", "owner": "answer_only", "lesion": lesion, "comparison_track": lesion,
                                "seed": seed, "edits": 32, "selection": selection, "dependencies": [selection]}))
                reference = add({**evaluation_base, "id": f"reference-{name}-on-{suite}-h{horizon}", "kind": "reference",
                    "device": "cpu", "estimated_peak_bytes": 0, "seed": 0, "edits": horizon, "batch_size": 1, "dependencies": [selection]})
                add({"id": f"report-{name}-on-{suite}-h{horizon}", "kind": "report", "device": "cpu", "evaluations": ordinary+[reference],
                     "bootstrap_repetitions": 10000, "dependencies": ordinary+[reference]})
                for measurement, evaluations in timing.items():
                    if evaluations:
                        add({"id": f"report-{measurement}-{name}-on-{suite}", "kind": "report", "device": "cpu",
                            "evaluations": evaluations+[reference], "bootstrap_repetitions": 10000, "dependencies": evaluations+[reference]})
            mechanisms = []
            for seed in SEEDS[:3]:
                mechanisms.append(add({**evaluation_base, "id": f"dynamics-{name}-on-{suite}-s{seed}", "kind": "evaluation", "mode": "dynamics",
                    "arm": "carry", "owner": "carry", "seed": seed, "source_K": 8 if family == "maze" else 2,
                    "selection": selection, "edits": 4, "batch_size": 1, "dependencies": [selection]}))
                mechanisms.append(add({**evaluation_base, "id": f"mechanism-{name}-on-{suite}-s{seed}", "kind": "evaluation", "mode": "mechanism",
                    "arm": "spatial_gate", "owner": "spatial_gate", "seed": seed, "source_K": 8 if family == "maze" else 2,
                    "intervention_policies": arms + ["shuffled_gate", "random_reset", "local_reset_1", "local_reset_2", "local_reset_3",
                        "impact_mask", "answer_uniform", "answer_shuffled_nodes"],
                    "branch_count": 384 if family == "maze" else 256,
                    "selection": selection, "edits": 1, "batch_size": 1, "dependencies": [selection]}))
            add({"id": f"report-mechanism-{name}-on-{suite}", "kind": "report", "device": "cpu", "mechanism": True,
                "evaluations": mechanisms, "dependencies": mechanisms})
    return {"schema": 1, "version": "confirmatory-v1", "authorization_gpu_hours": 500,
            "seeds": SEEDS, "budgets": BUDGETS, "suites": definitions, "jobs": jobs, "maze16_fallback": fallback,
            "matched_control": {"steps": controls_steps, "relative_example_forward_call_errors": control_errors},
            "startup_acceptance": {"path": "runs/confirmatory_restart_boot_v2/boot-check.json",
                "sha256": file_hash("runs/confirmatory_restart_boot_v2/boot-check.json"), "user_accepted_session0_exception": True}}


def project(matrix: dict) -> dict:
    profiles, sources, setup = {}, {}, {}
    for path in Path(ROOT).glob("capacity-*/profile.json"):
        name = path.parent.name
        p = read_json(path)
        profiles[name] = p
        sources[str(path)] = file_hash(path)
        actuals = [read_json(a) for a in Path(ROOT).glob("attempts/*.actual.json")
                   if read_json(a)["matrix_job"] == name]
        if len(actuals) != 1 or actuals[0]["aborted"]:
            raise ValueError("capacity projection requires one successful measured attempt: " + name)
        timed = sum(sum(r["step_seconds"]) for r in p["rows"]) if "step_seconds" in p["rows"][0] else sum(
            r["milliseconds"]["total"]/1000+r["offline_score_seconds"] for r in p["rows"])
        residual = max(0., actuals[0]["wall_seconds"]-timed)
        setup[name] = {"seconds": p.get("setup_seconds", residual),
            "basis": "explicit measured build" if "setup_seconds" in p else "measured wall minus all timed work (includes I/O)",
            "profile_wall_seconds": actuals[0]["wall_seconds"], "timed_seconds": timed}
    totals = {method: Counter() for method in ("median", "maximum")}
    adapter_profiles = {}
    for family, name in (("maze", "adapter_stream_v2_capacity"), ("circuit", "circuit_stream_v2_capacity")):
        path = Path("runs") / name / "profile.json"
        adapter_profiles[family] = read_json(path)
        sources[path.as_posix()] = file_hash(path)
    preparation = Path("reports/foundation/prompt06-preparation-measurement.json")
    if preparation.exists():
        sources[preparation.as_posix()] = file_hash(preparation)
        for row in read_json(preparation)["rows"]:
            setup[row["profile"]]["additional_full_data_preparation_seconds"] = row["full_development_build_cpu_seconds"]
            setup[row["profile"]]["seconds"] += row["full_development_build_cpu_seconds"]
    rows = []
    storage_path = Path("reports/foundation/prompt06-compression-measurement.json")
    storage = read_json(storage_path) if storage_path.exists() else None
    if storage:
        sources[storage_path.as_posix()] = file_hash(storage_path)
    for job in matrix["jobs"]:
        if job["device"] == "cpu":
            continue
        if job["kind"] == "research_training":
            name = f"capacity-{job['family']}{job['size']}-{job['recipe']}"
        else:
            suite = Path(job["dataset"]).stem.removesuffix("-test").removesuffix("-development")
            name = "capacity-evaluation-"+suite
        p = profiles[name]
        record_bytes = 0
        if job["kind"] == "evaluation":
            evaluated_size = matrix["suites"][suite]["size"]
            nodes = evaluated_size**2 if job["family"] == "maze" else evaluated_size
            if job.get("mode") in ("mechanism", "dynamics"):
                branches = job.get("branch_count", 256)
                records = branches*len(job["budgets"])*(len(job["intervention_policies"]) if job.get("mode") == "mechanism" else 1)
                record_bytes = records*(2500+43*nodes)+branches*nodes*job["width"]*12
            else:
                records = job.get("root_limit", 256)*job.get("repetitions", 1)*(job["edits"]+1)*len(job["budgets"])
                record_bytes = records*(2500+3*nodes)
        io_seconds = record_bytes*(storage["seconds_per_raw_byte"]+storage["json_seconds_per_raw_byte"]) if storage else 0
        projected = {}
        for method, aggregate in (("median", statistics.median), ("maximum", max)):
            if job["kind"] == "research_training":
                costs = {r["K"]: aggregate(r["step_seconds"][1:]) for r in p["rows"]}
                if job["recipe"] != "static":
                    old = adapter_profiles[job["family"]]["rows"]
                    for k in job["budgets"]:
                        arm = statistics.median(t for r in old if r["K"] == k and r["arm"] == job["arm"] for t in r["step_seconds"][1:])
                        spatial = statistics.median(t for r in old if r["K"] == k and r["arm"] == "spatial_gate" for t in r["step_seconds"][1:])
                        costs[k] *= max(1., arm/spatial)
                training_roots = len(matrix["suites"][job["family"]+str(job["size"])]["identities"]["train"])
                training_batches = training_roots//job["batch_size"] * (4 if job["recipe"] == "one_edit" else 1)
                seconds = sum(costs[k] for _, k in schedule(job["seed"], training_batches,
                    {"steps": job["steps"], "budgets": job["budgets"]}))
            else:
                batch = job["batch_size"]
                costs = {k: aggregate(r["milliseconds"]["total"]/1000+r["offline_score_seconds"]
                    for r in p["rows"] if r["K"] == k and r["batch_size"] == batch and r["repetition"] > 0)
                    for k in job["budgets"]}
                if job.get("mode") == "dynamics":
                    seconds = 256*(sum(costs.values())+5*costs[job["source_K"]])
                elif job.get("mode") == "mechanism":
                    seconds = job["branch_count"]*(sum(costs.values())*len(job["intervention_policies"])+costs[job["source_K"]])
                else:
                    import math
                    units = math.ceil(job.get("root_limit", 256)/batch)*job.get("repetitions", 1)
                    seconds = units*((job["edits"]+1)*sum(costs.values())+sum(costs.values())*job.get("warmup", 1))
            seconds += setup[name]["seconds"]
            seconds += io_seconds
            category = job["kind"]+"/"+job.get("mode", job.get("recipe", "stream"))
            totals[method][category] += seconds/3600
            projected[method+"_seconds"] = seconds
        rows.append({"job": job["id"], "profile": name, "setup_seconds": setup[name]["seconds"],
            "estimated_raw_record_bytes": record_bytes, "measured_io_projection_seconds": io_seconds, **projected})
    used = sum(r["seconds"] for r in Budget_events() if r["kind"] == "actual")/3600
    median, maximum = (sum(totals[m].values()) for m in ("median", "maximum"))
    counts = Counter(j["kind"]+"/"+j.get("mode", j.get("recipe", "stream")) for j in matrix["jobs"])
    return {"projection": True, "timing_sources_synthetic": True, "empirical_method_result": False,
        "source_hashes": sources, "setup_measurements": setup, "jobs": rows,
        "class_table": [{"class": c, "jobs": counts[c], "median_hours": totals["median"][c], "maximum_hours": totals["maximum"][c]}
                        for c in sorted(counts)],
        "base_hours": median, "maximum_base_hours": maximum,
        "estimated_raw_record_bytes": sum(r["estimated_raw_record_bytes"] for r in rows),
        "record_storage_sensitivity_bytes": sum(r["estimated_raw_record_bytes"] for r in rows)*.35,
        "interruption_allowance_fraction": .15, "interruption_allowance_hours": median*.15,
        "already_consumed_hours": used, "total_hours_with_interruptions": median*1.15+used,
        "maximum_total_hours_with_interruptions": maximum*1.15+used,
        "per_job_projection_exceeds_cap": [r for r in rows if r["median_seconds"] > 14400]}


def Budget_events() -> list[dict]:
    import json
    path = Path(ROOT) / "gpu_time.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
