"""One serial owner for startup reconciliation, quarantine, verification and jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import shutil
import sys
import time

from state_repair.provenance import file_hash
from .budget import Budget
from .durable import DriverLock, append_jsonl, atomic_json, atomic_write, digest_json, read_json, repair_jsonl
from .training import Shutdown, SyntheticTrainer, load_resume, run_training
from .records import step_rows, step_bytes, raw_step_hash, compress_steps


def seal(output: Path) -> None:
    names = [p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "seal.json"]
    atomic_json(output / "seal.json", {"schema": 1, "files": {p.relative_to(output).as_posix(): file_hash(p) for p in names}})


def verify(output: Path, job: dict) -> dict:
    manifest = read_json(output / "seal.json")
    names = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "seal.json"}
    if names != set(manifest["files"]) or not {"summary.json", "checkpoint.pt", "job.json", "resume.pt"} <= names or len(names & {"steps.jsonl", "steps.jsonl.gz"}) != 1:
        raise ValueError("sealed directory has missing or extra artifacts")
    for name, digest in manifest["files"].items():
        if file_hash(output / name) != digest:
            raise ValueError("sealed artifact hash mismatch: " + name)
    if read_json(output / "job.json") != job:
        raise ValueError("sealed job configuration mismatch")
    payload = load_resume(output / "checkpoint.pt", job)
    rows = list(step_rows(output))
    summary = read_json(output / "summary.json")
    if len(rows) != len(payload["schedule"]) or summary["steps"] != len(rows):
        raise ValueError("incomplete optimizer schedule")
    if payload["log_sha256"] != raw_step_hash(output):
        raise ValueError("final checkpoint step-log hash mismatch")
    for index, (row, (batch, k)) in enumerate(zip(rows, payload["schedule"]), 1):
        if (row["step"], row["batch"], row["K"], row["synthetic"]) != (index, batch, k, job["synthetic"]):
            raise ValueError("step schedule mismatch")
    if sum(r["forward_calls"] for r in rows) != payload["forward_calls"] or summary["forward_calls"] != payload["forward_calls"]:
        raise ValueError("forward counters mismatch")
    events = [json.loads(line) for line in (output / "execution.jsonl").read_text().splitlines()] if (output / "execution.jsonl").exists() else []
    resumes = [e for e in events if e["event"] == "resume"]
    if any(not re.fullmatch(r"[0-9a-f]{64}", e["checkpoint_sha256"]) or not 0 <= e["step"] <= len(rows) for e in resumes):
        raise ValueError("invalid resume provenance")
    copied = {e["checkpoint_sha256"]: e for e in events if e["event"] == "copied_resume"}
    for event in resumes:
        original = copied.get(event["checkpoint_sha256"])
        if original is None or file_hash(Path(original["source"]) / "resume.pt") != event["checkpoint_sha256"]:
            raise ValueError("resume source checkpoint missing or changed")
    return {"verified": True, "synthetic": job["synthetic"], "resume_events": resumes, "steps": len(rows)}


def recover_output(output: Path, job: dict, audit: Path) -> None:
    if output.exists():
        quarantine = output.with_name(output.name + ".incomplete-" + str(time.time_ns()))
        output.rename(quarantine)
        append_jsonl(audit, {"event": "quarantine", "source": str(output), "destination": str(quarantine)})
    output.mkdir()
    for candidate in sorted(output.parent.glob(output.name + ".incomplete-*"), reverse=True):
        checkpoint = candidate / "resume.pt"
        if not checkpoint.exists():
            continue
        try:
            payload = load_resume(checkpoint, job)
            raw = step_bytes(candidate) if any((candidate/n).exists() for n in ("steps.jsonl", "steps.jsonl.gz")) else b""
            import hashlib
            if hashlib.sha256(raw[:payload["log_bytes"]]).hexdigest() != payload["log_sha256"]:
                raise ValueError("resume log prefix changed")
        except (ValueError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            append_jsonl(audit, {"event": "invalid_checkpoint", "path": str(checkpoint), "error": str(exc)})
            continue
        atomic_write(output / "steps.jsonl", lambda handle: handle.write(raw))
        for name in ("resume.pt", "execution.jsonl"):
            source = candidate / name
            if source.exists():
                atomic_write(output / name, lambda handle, source=source: handle.write(source.read_bytes()))
        repair_jsonl(output / "execution.jsonl", audit)
        append_jsonl(output / "execution.jsonl", {"event": "copied_resume", "source": str(candidate),
                                                "checkpoint_sha256": file_hash(checkpoint)})
        break
    atomic_json(output / "job.json", job)


def validate_matrix(config: dict, protocol: Path | None) -> None:
    if config.get("schema") != 1 or config.get("authorization_gpu_hours") not in (150, 500):
        raise ValueError("matrix must declare schema 1 and an authorized 150/500-hour cap")
    jobs = config["jobs"]
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("matrix requires distinct jobs")
    for job in jobs:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job["id"]):
            raise ValueError("job id must be a safe directory name")
        if type(job.get("synthetic")) is not bool:
            raise ValueError("explicit synthetic boolean required")
        if job.get("device") not in ("cpu", "cuda"):
            raise ValueError("only local CPU or CUDA jobs are permitted")
    if not all(j["synthetic"] and j.get("kind") == "synthetic_training" and j["device"] == "cpu" for j in jobs):
        if protocol is None or not protocol.is_file():
            raise ValueError("empirical dispatch requires committed frozen protocol")
        from .binding import validate_binding
        validate_binding(config, protocol)


def run(config_path: Path, root: Path, protocol: Path | None = None) -> dict:
    config = read_json(config_path)
    validate_matrix(config, protocol)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = root / "driver.jsonl"
    with DriverLock(root / "driver.lock"):
        repair_jsonl(root / "recovery.jsonl", root / "recovery-truncations.jsonl")
        repair_jsonl(log, root / "recovery.jsonl")
        import os
        session_id = None
        if os.name == "nt":
            import ctypes
            session = ctypes.c_uint()
            if not ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
                raise OSError("cannot record Windows process session")
            session_id = session.value
        append_jsonl(log, {"event": "startup", "unix": time.time(), "pid": os.getpid(),
                          "windows_session_id": session_id})
        budget = Budget(root, config["authorization_gpu_hours"] * 3600)
        budget.reconcile()
        snapshot = root / "matrix.json"
        if snapshot.exists() and read_json(snapshot) != config:
            raise ValueError("matrix changed; use a new version and output root")
        atomic_json(snapshot, config)
        completed = []
        for job in config["jobs"]:
            if not job["synthetic"]:
                from .datasets import verify_job_dataset
                verify_job_dataset(job, config, root)
            for dependency in job.get("dependencies", []):
                dependency_job = next(j for j in config["jobs"] if j["id"] == dependency)
                verify(root / dependency, dependency_job)
            output = root / job["id"]
            if (output / "seal.json").exists():
                completed.append(verify(output, job))
                continue
            if (root / "STOP").exists():
                raise InterruptedError("STOP flag present; remove after shutdown handling")
            if budget.consumed(job["id"]) + job.get("unit_reserve_seconds", 0) >= job["seconds"]:
                atomic_json(root / "STOP", {"reason": "cumulative job cap has no safe work-unit allowance", "job": job["id"]})
                raise TimeoutError("cumulative job cap exhausted")
            if shutil.disk_usage(root).free < config.get("minimum_free_disk_bytes", 0):
                atomic_json(root / "STOP", {"reason": "frozen minimum free disk reserve reached", "job": job["id"]})
                raise OSError("insufficient free disk for preserved artifacts")
            recover_output(output, job, root / "recovery.jsonl")
            receipt = budget.begin(job, output)
            started = time.monotonic()
            append_jsonl(log, {"event": "job_started", "id": job["id"], "attempt": receipt["job_id"],
                               "unix": time.time(), "estimated_peak_bytes": job.get("estimated_peak_bytes", 0)})
            try:
                with Shutdown() as shutdown:
                    if job["kind"] == "synthetic_training":
                        trainer = SyntheticTrainer(job)
                    else:
                        from .jobs import build
                        trainer = build(job, root)
                    summary = run_training(trainer, job, output,
                                           receipt["segment_cap_seconds"] - (time.monotonic()-started), shutdown,
                                           checkpoint_steps=job.get("checkpoint_steps", 500),
                                           checkpoint_seconds=job.get("checkpoint_seconds", 300))
                if job["device"] == "cuda":
                    import torch
                    torch.cuda.synchronize()
                    summary.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
                atomic_write(output / "checkpoint.pt", lambda handle: handle.write((output / "resume.pt").read_bytes()))
                atomic_json(output / "summary.json", summary)
                if job["kind"] == "report":
                    from state_repair.eval.report import materialize_report
                    materialize_report(output)
                if job.get("compress_records"):
                    compress_steps(output)
                budget.finish(receipt, time.monotonic()-started, aborted=False)
                seal(output)
                completed.append(verify(output, job))
            except BaseException as exc:
                budget.finish(receipt, time.monotonic()-started, aborted=True, error=f"{type(exc).__name__}: {exc}")
                append_jsonl(log, {"event": "job_aborted", "id": job["id"], "unix": time.time(), "error": str(exc)})
                if not job["synthetic"] and isinstance(exc, (ValueError, TimeoutError)):
                    atomic_json(root / "STOP", {"job": job["id"], "reason": str(exc), "failed_attempt_preserved": True})
                raise
            append_jsonl(log, {"event": "job_complete", "id": job["id"], "unix": time.time()})
            del trainer
            if job["device"] == "cuda":
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
        result = {"matrix_complete": True, "jobs": len(completed), "matrix_sha256": digest_json(config),
                  "synthetic": all(j["synthetic"] for j in config["jobs"])}
        if not (root / "complete.json").exists():
            atomic_json(root / "complete.json", result)
        append_jsonl(log, {"event": "matrix_complete", "unix": time.time(), **result})
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, args.output, args.protocol)), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
