"""Recover both accounting stores from a write-ahead attempt receipt."""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import time
import uuid

from state_repair.accounting.gpu_job import remaining_seconds
from state_repair.accounting.ledger import AccountingLedger
from .durable import append_jsonl, atomic_json, read_json, repair_jsonl

AUTHORIZED_SECONDS = 500 * 3600
MAX_JOB_SECONDS = 4 * 3600


class Budget:
    """Caller must hold the driver lock through recovery, reservation and actual."""

    def __init__(self, root: Path, authorized_seconds: float = AUTHORIZED_SECONDS):
        if not 0 < authorized_seconds <= AUTHORIZED_SECONDS:
            raise ValueError("authorization must fit the 500 local GPU-hour ceiling")
        self.authorized_seconds = authorized_seconds
        self.root = root
        self.receipts = root / "attempts"
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.journal = root / "gpu_time.jsonl"
        self.audit = root / "recovery.jsonl"
        self.ledger = AccountingLedger(root / "accounting.sqlite")

    def events(self) -> list[dict]:
        repair_jsonl(self.journal, self.audit)
        return [__import__("json").loads(line) for line in self.journal.read_text().splitlines()] if self.journal.exists() else []

    def _ledger_kinds(self, attempt: str) -> set[str]:
        with sqlite3.connect(self.ledger.path) as conn:
            return {row[0] for row in conn.execute("SELECT kind FROM events WHERE job_id=?", (attempt,))}

    def _reserve_stores(self, receipt: dict) -> None:
        attempt = receipt["job_id"]
        if "reserve" not in self._ledger_kinds(attempt):
            self.ledger.reserve(attempt, 0, "prompt06_local")
        if not any(e["job_id"] == attempt and e["kind"] == "reserve" for e in self.events()):
            append_jsonl(self.journal, {**receipt, "kind": "reserve", "seconds": receipt["reserved_seconds"]})

    def begin(self, job: dict, output: Path) -> dict:
        seconds = job["seconds"]
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_JOB_SECONDS:
            raise ValueError("job cap must be positive and at most four hours")
        consumed = self.consumed(job["id"])
        remaining = seconds - consumed
        if remaining <= 0:
            raise TimeoutError("cumulative job cap exhausted across interruptions")
        gpu_seconds = remaining if job["device"] == "cuda" else 0
        if gpu_seconds > remaining_seconds(self.events(), self.authorized_seconds):
            raise ValueError("local GPU-hour authorization exhausted")
        receipt = {"job_id": "p06-" + uuid.uuid4().hex, "matrix_job": job["id"],
                   "started_unix": time.time(), "output": str(output.resolve()),
                   "reserved_seconds": gpu_seconds, "segment_cap_seconds": remaining,
                   "synthetic": job["synthetic"], "device": job["device"],
                   "estimated_peak_bytes": job.get("estimated_peak_bytes", 0)}
        atomic_json(self.receipts / (receipt["job_id"] + ".json"), receipt)
        self._reserve_stores(receipt)
        return receipt

    def finish(self, receipt: dict, wall_seconds: float, *, aborted: bool,
               time_basis: str = "monotonic", error: str | None = None) -> None:
        result = {**receipt, "wall_seconds": max(0, wall_seconds), "aborted": aborted,
                  "time_basis": time_basis, "error": error, "finished_unix": time.time()}
        result_path = self.receipts / (receipt["job_id"] + ".actual.json")
        if result_path.exists():
            result = read_json(result_path)
        else:
            atomic_json(result_path, result)
        self._reserve_stores(receipt)
        if "actual" not in self._ledger_kinds(receipt["job_id"]):
            self.ledger.reconcile(receipt["job_id"], 0)
        if not any(e["job_id"] == receipt["job_id"] and e["kind"] == "actual" for e in self.events()):
            append_jsonl(self.journal, {**result, "kind": "actual",
                                      "seconds": result["wall_seconds"] if receipt["device"] == "cuda" else 0})

    def reconcile(self) -> None:
        self.events()
        for path in sorted(self.receipts.glob("*.json")):
            if path.name.endswith(".actual.json"):
                continue
            receipt = read_json(path)
            result_path = path.with_name(path.stem + ".actual.json")
            if result_path.exists():
                result = read_json(result_path)
                self.finish(receipt, result["wall_seconds"], aborted=result["aborted"])
                continue
            output = Path(receipt["output"])
            latest = max([path.stat().st_mtime, receipt["started_unix"],
                          *(p.stat().st_mtime for p in output.rglob("*") if p.is_file())])
            self.finish(receipt, latest - receipt["started_unix"], aborted=True,
                        time_basis="latest_durable_file_timestamp_lower_bound",
                        error="interrupted; time after last durable write is unobservable")
            append_jsonl(self.audit, {"event": "reconciled_aborted", "job_id": receipt["job_id"],
                                     "recoverable_wall_seconds": latest - receipt["started_unix"]})
        known = {read_json(p)["job_id"] for p in self.receipts.glob("*.json")}
        with sqlite3.connect(self.ledger.path) as conn:
            dangling = {r[0] for r in conn.execute("SELECT job_id FROM events r WHERE kind='reserve' AND NOT EXISTS (SELECT 1 FROM events a WHERE a.job_id=r.job_id AND a.kind='actual')")}
        if dangling - known:
            raise ValueError("ledger reservation without write-ahead receipt; manual audit needed")
        journal_pending = {e["job_id"] for e in self.events() if e["kind"] == "reserve"} - {e["job_id"] for e in self.events() if e["kind"] == "actual"}
        if journal_pending:
            raise ValueError("journal reservation without recoverable receipt")

    def consumed(self, job: str) -> float:
        return sum(r["wall_seconds"] for p in self.receipts.glob("*.actual.json")
                   if (r := read_json(p))["matrix_job"] == job)
