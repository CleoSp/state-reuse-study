"""Preserve the budget-stop evidence, without generating or launching test data."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import zipfile

from state_repair.execution.durable import DriverLock, atomic_json
from state_repair.provenance import file_hash


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    root = workspace / "runs/confirmatory_v1"
    target = workspace / "reports/foundation/prompt06-capacity-budget-stop.zip"
    if target.exists():
        raise ValueError("preserve existing evidence archive; use a new explicit version")
    with DriverLock(root / "driver.lock"):
        files = [p for d in root.glob("capacity-*") for p in d.rglob("*") if p.is_file()]
        files += [*sorted((root / "attempts").glob("*.json")), root / "gpu_time.jsonl"]
        files += [workspace / name for name in (
            "runs/prompt06-capacity-console.txt", "runs/prompt06-evaluation-capacity-console.txt",
            "runs/prompt06-validation-regeneration.txt", "runs/prompt06-validation-regeneration-retry.txt",
            "runs/prompt06-validation-regeneration.json", "runs/prompt06-budget-stop-final-regression.txt",
            "runs/prompt06-regeneration-incomplete-stream-results.json",
            "configs/confirmatory_tier2_capacity_draft.json", "reports/foundation/prompt06-capacity-projection.json")]
        files += [p for p in (workspace / "src").rglob("*.py")]
        files += [p for p in (workspace / "scripts").glob("*.py")]
        files += [p for p in (workspace / "tests").glob("*.py")]
        files = sorted(set(p for p in files if p.exists()))
        snapshot = root / "accounting-budget-stop.sqlite"
        with sqlite3.connect(root / "accounting.sqlite") as source, sqlite3.connect(snapshot) as destination:
            source.backup(destination)
        files.append(snapshot)
        manifest = {p.relative_to(workspace).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in files}
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, path.relative_to(workspace).as_posix())
            archive.writestr("EVIDENCE_MANIFEST.json", json.dumps({"files": manifest,
                "source_snapshot_scope": "current integration source at budget-stop packaging; profiles record their own timing provenance",
                "confirmatory_launched": False, "test_generated": False}, indent=2))
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError("archive CRC check failed")
        atomic_json(target.with_suffix(".manifest.json"), {"archive_sha256": file_hash(target), "bytes": target.stat().st_size,
            "files": len(files), "gpu_job_seconds": 340.345, "synthetic_capacity_jobs": 14,
            "confirmatory_launched": False, "test_generated": False})
        print(json.dumps({"archive": str(target), "bytes": target.stat().st_size, "files": len(files)}))


if __name__ == "__main__":
    main()
