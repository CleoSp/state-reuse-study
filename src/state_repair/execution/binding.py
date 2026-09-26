"""Content and Git binding for the production matrix; no command dispatch."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from state_repair.provenance import file_hash
from .durable import digest_json

KINDS = {"research_training", "evaluation", "selection", "reference", "report"}


def git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def validate_binding(config: dict, protocol: Path) -> None:
    if config.get("launch_authorized") is not True or config.get("status") != "FROZEN":
        raise ValueError("draft matrices cannot generate test data or launch")
    root = Path(git_bytes(protocol.parent, "rev-parse", "--show-toplevel").decode().strip())
    relative = protocol.resolve().relative_to(root.resolve()).as_posix()
    committed = git_bytes(root, "show", f"HEAD:{relative}")
    if committed.decode().replace("\r\n", "\n") != protocol.read_text(encoding="utf-8"):
        raise ValueError("frozen protocol is not committed unchanged")
    marker = "<!-- matrix-sha256: " + digest_json(config) + " -->"
    if marker not in protocol.read_text(encoding="utf-8"):
        raise ValueError("frozen matrix hash does not match protocol")
    for name, digest in config["source_hashes"].items():
        path = root / name
        if file_hash(path) != digest:
            raise ValueError("frozen implementation changed: " + name)
        if git_bytes(root, "show", "HEAD:" + name).decode().replace("\r\n", "\n") != path.read_text(encoding="utf-8"):
            raise ValueError("production source is not committed: " + name)
    for name, digest in config.get("artifact_hashes", {}).items():
        if file_hash(root / name) != digest:
            raise ValueError("frozen development artifact changed: " + name)
    evidence = root / config["startup_acceptance"]["path"]
    if file_hash(evidence) != config["startup_acceptance"]["sha256"]:
        raise ValueError("startup acceptance evidence changed")
    boot = json.loads(evidence.read_text())
    if not boot["noninteractive_resume_verified"] or config["startup_acceptance"].get("user_accepted_session0_exception") is not True:
        raise ValueError("startup acceptance missing")
    seen = set()
    for job in config["jobs"]:
        if job["kind"] not in KINDS or job["synthetic"]:
            raise ValueError("production matrix requires supported empirical job kinds")
        if not set(job.get("dependencies", [])) <= seen:
            raise ValueError("dependency must precede its consumer")
        if job["device"] == "cuda" and not 0 < job.get("estimated_peak_bytes", 0) <= 10 * 2**30:
            raise ValueError("GPU job requires a declared peak within 10 GiB")
        seen.add(job["id"])
    if config["projection"]["total_hours_with_interruptions"] > config["authorization_gpu_hours"]:
        raise ValueError("projected matrix exceeds authorization")
    if config["projection"].get("per_job_projection_exceeds_cap"):
        raise ValueError("a projected job exceeds its cumulative cap")
