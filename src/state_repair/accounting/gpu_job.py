"""Local GPU time reservations and resource records."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from state_repair.accounting.ledger import AccountingLedger
from state_repair.accounting.resources import memory_snapshot

JOURNAL = Path("runs/prompt05_gpu_time.jsonl")
AUTHORIZED_GPU_SECONDS = 10 * 3600


def remaining_seconds(events: list[dict], authorized_seconds: float = AUTHORIZED_GPU_SECONDS) -> float:
    reservations, actual = {}, {}
    for row in events:
        target = reservations if row["kind"] == "reserve" else actual if row["kind"] == "actual" else None
        if target is None or row["job_id"] in target or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("invalid GPU time journal")
        if row["kind"] == "actual" and row["job_id"] not in reservations:
            raise ValueError("unreserved GPU job")
        target[row["job_id"]] = row["seconds"]
    return authorized_seconds - sum(actual.get(job, seconds) for job, seconds in reservations.items())


class GPUJob:
    """GPU reservations with a single journal writer.

    Allocation cap is 6 GiB plus a declared 1.5 GiB runtime allowance. Resource
    records count entire context wall time (including CPU setup), conservatively.
    Call check_limit at each training step and evaluation batch.
    """

    def __init__(self, output: Path, seconds: float, purpose: str, *, synthetic: bool,
                 authorization: str = "prompt05") -> None:
        if not 0 < seconds <= 7200:
            raise ValueError("GPU job reservation must be >0 and <=7200 seconds")
        if authorization not in ("prompt05", "prompt05b"):
            raise ValueError("unknown compute authorization")
        self.authorization = authorization
        self.journal = JOURNAL if authorization == "prompt05" else Path("runs/prompt05b_gpu_time.jsonl")
        allowance = AUTHORIZED_GPU_SECONDS if authorization == "prompt05" else 12 * 3600
        events = [json.loads(line) for line in self.journal.read_text().splitlines()] if self.journal.exists() else []
        if seconds > remaining_seconds(events, allowance):
            raise ValueError(f"GPU reservation exceeds remaining {authorization} hours")
        self.output, self.seconds, self.purpose, self.synthetic = output, seconds, purpose, synthetic
        self.job_id = f"{output.name}:{time.time_ns()}"
        self.ledger = AccountingLedger("runs/accounting.sqlite")

    def append(self, kind: str, seconds: float, **extra: Any) -> None:
        self.journal.parent.mkdir(exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "job_id": self.job_id, "seconds": seconds,
                                     "purpose": self.purpose, "synthetic": self.synthetic,
                                     "authorization": self.authorization, **extra}, allow_nan=False) + "\n")

    def __enter__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no training launched")
        self.output.mkdir(parents=True, exist_ok=False)
        self.ledger.reserve(self.job_id, 0, self.purpose)
        self.append("reserve", self.seconds, device="cuda", dtype="float32", estimated_peak_bytes=int(7.5*2**30))
        self.started = time.perf_counter()
        try:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(6*2**30/total)
            torch.cuda.reset_peak_memory_stats()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def check_limit(self) -> None:
        if time.perf_counter() - self.started > self.seconds:
            raise TimeoutError("reserved GPU-job wall time exhausted")

    def __exit__(self, exc_type, exc, traceback):
        wall = time.perf_counter() - self.started
        record = {"job_id": self.job_id, "wall_s": wall, "gpu_hours": wall/3600,
            "reservation_seconds": self.seconds, "complete": exc is None,
            "authorization": self.authorization, "gpu_time_journal": str(self.journal),
            "error": None if exc is None else f"{exc_type.__name__}: {exc}",
            "synthetic": self.synthetic, "purpose": self.purpose, "device": "cuda",
            "device_name": torch.cuda.get_device_name(), "dtype": "float32",
            "torch": str(torch.__version__), "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(), "memory": memory_snapshot(),
            "external_cost_usd": 0, "electricity_cost_usd": None,
            "ledger": self.ledger.reconcile(self.job_id, 0)}
        self.append("actual", wall, **{k: record[k] for k in ("device", "dtype", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes", "error")})
        (self.output / "resources.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"resources": record}), flush=True)
        return False
