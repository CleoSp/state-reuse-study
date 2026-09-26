"""Append-only events with transactional reservations in integer cents."""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


def _cents(amount: float) -> int:
    try:
        value = Decimal(str(amount))
        if not value.is_finite() or value < 0 or value > Decimal(2**63 - 1) / 100 or value != value.quantize(Decimal("0.01")):
            raise ValueError("invalid amount")
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("USD amount must be finite, nonnegative, with at most two decimals and fit SQLite integer cents") from exc
    if isinstance(amount, bool):
        raise ValueError("USD amount must be finite, nonnegative, with at most two decimals")
    return int(value * 100)


def _append_only(conn: sqlite3.Connection, table: str) -> None:
    for operation in ("UPDATE", "DELETE"):
        conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} "
                     f"BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'append-only journal'); END")


class AccountingLedger:
    """One shared database controls the total allocation; categories are labels.

    Actual overruns are recorded, even when they exceed allocation. A negative
    remaining amount then blocks future reservations instead of hiding spending.
    """

    def __init__(self, path: str | Path, total_usd: float = 950):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS allocation (singleton INTEGER PRIMARY KEY CHECK(singleton=1), cents INTEGER NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')), kind TEXT NOT NULL, job_id TEXT NOT NULL, cents INTEGER NOT NULL, category TEXT NOT NULL)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS events_job_kind ON events(job_id,kind)")
            _append_only(conn, "events")
            _append_only(conn, "allocation")
            total = _cents(total_usd)
            conn.execute("INSERT OR IGNORE INTO allocation VALUES (1, ?)", (total,))
            if conn.execute("SELECT cents FROM allocation").fetchone()[0] != total:
                raise ValueError("existing ledger allocation differs; explicit budget update required")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _snapshot(conn: sqlite3.Connection) -> dict[str, float]:
        allocated = conn.execute("SELECT cents FROM allocation").fetchone()[0]
        actual = conn.execute("SELECT COALESCE(SUM(cents),0) FROM events WHERE kind='actual'").fetchone()[0]
        reserved = conn.execute("SELECT COALESCE(SUM(cents),0) FROM events r WHERE kind='reserve' AND NOT EXISTS (SELECT 1 FROM events a WHERE a.job_id=r.job_id AND a.kind='actual')").fetchone()[0]
        return {"allocated_usd": allocated / 100, "reserved_usd": reserved / 100, "actual_usd": actual / 100, "remaining_usd": (allocated - actual - reserved) / 100}

    def reserve(self, job_id: str, planned_usd: float, category: str = "local") -> dict[str, float]:
        amount = _cents(planned_usd)
        if not isinstance(job_id, str) or not job_id.strip() or not isinstance(category, str) or not category.strip():
            raise ValueError("job_id and category are required")
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM events WHERE job_id=?", (job_id,)).fetchone():
                raise ValueError(f"job already recorded: {job_id}")
            allocated = conn.execute("SELECT cents FROM allocation").fetchone()[0]
            committed = conn.execute("SELECT COALESCE(SUM(cents),0) FROM events e WHERE kind='actual' OR (kind='reserve' AND NOT EXISTS (SELECT 1 FROM events a WHERE a.job_id=e.job_id AND a.kind='actual'))").fetchone()[0]
            if amount > allocated - committed:
                raise ValueError("reservation exceeds remaining allocation")
            conn.execute("INSERT INTO events(kind,job_id,cents,category) VALUES ('reserve',?,?,?)", (job_id, amount, category))
            return self._snapshot(conn)

    def reconcile(self, job_id: str, actual_usd: float) -> dict[str, float]:
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id is required")
        amount = _cents(actual_usd)
        with self._transaction() as conn:
            rows = conn.execute("SELECT kind,category FROM events WHERE job_id=?", (job_id,)).fetchall()
            if len(rows) != 1 or rows[0][0] != "reserve":
                raise ValueError("job must have exactly one unreconciled reservation")
            conn.execute("INSERT INTO events(kind,job_id,cents,category) VALUES ('actual',?,?,?)", (job_id, amount, rows[0][1]))
            return self._snapshot(conn)

    def snapshot(self) -> dict[str, float]:
        with self._transaction() as conn:
            return self._snapshot(conn)


class RunManifest:
    """SQLite event journal; transaction-safe append and ordered JSONL export."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS manifest (seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL)")
            _append_only(conn, "manifest")

    def append(self, record: dict[str, Any]) -> None:
        if type(record.get("synthetic")) is not bool:
            raise ValueError("manifest records require an explicit synthetic boolean")
        payload = json.dumps(record, sort_keys=True, allow_nan=False)
        with sqlite3.connect(self.path, timeout=30) as conn:
            conn.execute("INSERT INTO manifest(payload) VALUES (?)", (payload,))

    def records(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload FROM manifest ORDER BY seq")]

    def export_jsonl(self, destination: str | Path) -> None:
        path = Path(destination)
        with path.open("x", encoding="utf-8") as stream:
            for record in self.records():
                stream.write(json.dumps(record, sort_keys=True) + "\n")
