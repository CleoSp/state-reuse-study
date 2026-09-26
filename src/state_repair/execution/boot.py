"""Check observed boot evidence; a synthetic subprocess restart is not a reboot."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .driver import verify
from .durable import read_json


def check_boot(root: Path, logon_audit: dict, *, observe_automatic_logon: bool = False) -> dict:
    if not logon_audit.get("query_succeeded") or logon_audit.get("source") != "System/Microsoft-Windows-Winlogon/7001":
        raise ValueError("real Windows logon-event audit required")
    boot = float(logon_audit["boot_unix"])
    wrappers = [json.loads(line) for line in (root / "startup.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    wrappers = [w for w in wrappers if w["event"] == "startup_wrapper"]
    if not any(abs(datetime.fromisoformat(w["boot_utc"]).timestamp()-boot) < 1 for w in wrappers):
        raise ValueError("startup wrapper has no matching real boot record")
    driver = [json.loads(line) for line in (root / "driver.jsonl").read_text().splitlines()]
    starts = [r["unix"] for r in driver if r["event"] == "startup" and r["unix"] >= boot]
    completions = [r["unix"] for r in driver if r["event"] == "matrix_complete" and r["unix"] >= boot]
    if not starts or not completions:
        raise ValueError("no driver startup and completion after the boot")
    complete = min(completions)
    if float(logon_audit["queried_through_unix"]) < complete:
        raise ValueError("logon audit ends before job completion")
    logon_count = sum(boot <= t <= complete for t in logon_audit["interactive_logon_unix"])
    session_zero = all(r.get("windows_session_id") == 0 for r in driver
                       if r["event"] == "startup" and boot <= r["unix"] <= complete)
    if logon_count and not observe_automatic_logon:
        raise ValueError("a user logged on before the synthetic job completed")
    if observe_automatic_logon and not session_zero:
        raise ValueError("automatic-logon observation requires a recorded noninteractive session 0")
    matrix = read_json(root / "matrix.json")
    checks = []
    for job in matrix["jobs"]:
        if not job["synthetic"]:
            raise ValueError("boot acceptance must use synthetic jobs")
        check = verify(root / job["id"], job)
        events = [json.loads(line) for line in (root / job["id"] / "execution.jsonl").read_text().splitlines()]
        resumed_after = [e for e in check["resume_events"] if boot <= e["unix"] <= complete]
        copied = {e["checkpoint_sha256"]: e["source"] for e in events if e["event"] == "copied_resume"}
        if not any((Path(copied[e["checkpoint_sha256"]]) / "resume.pt").stat().st_mtime < boot for e in resumed_after):
            raise ValueError("no pre-boot checkpoint resumed after boot")
        checks.append(check)
    return {"verified": logon_count == 0, "synthetic": True, "boot_unix": boot,
            "noninteractive_resume_verified": session_zero,
            "strict_no_logon_passed": logon_count == 0,
            "automatic_logon_declared_by_user": observe_automatic_logon,
            "boot_to_first_driver_log_seconds": min(starts)-boot,
            "boot_to_completion_seconds": complete-boot, "jobs": checks,
            "interactive_logons_before_completion": logon_count,
            "logon_audit_source": logon_audit["source"]}
