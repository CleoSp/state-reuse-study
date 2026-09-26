"""Package verified pilot evidence in size-bounded, byte-exact shards."""
from __future__ import annotations

import json
from pathlib import Path

from package_run_evidence import package, verify
from state_repair.provenance import file_hash
from run_adapter_pilot import read, write

DIRECTORIES = (
    "static_generalization_v1", "frozen_crossover_v1", "scaled_maze_v1", "static_circuit_v1", "static_circuit_v12",
    "scaled_cuda_profile_w128", "scaled_cuda_profile_w64", "static_circuit_capacity", "static_circuit_capacity_b128",
    "circuit_transfer_profile", "circuit_transfer_profile_v2", "adapter_pilot_capacity_v1", "adapter_pilot_capacity_v2",
    "pilot_evaluation_capacity_v1", "pilot_evaluation_capacity_v2", "adapter_pilot_v1", "pilot_replay_diagnostic",
)


def main() -> None:
    root = Path.cwd()
    checks = read("runs/prompt05-pilot-check.json")
    if not checks["complete"] or not checks["verified"]:
        raise ValueError("complete verified pilot required")
    files = []
    for name in DIRECTORIES:
        directory = root / "runs" / name
        if not directory.is_dir():
            raise ValueError(f"missing explicit evidence source: {name}")
        files.extend(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files.extend(p for p in (root / "runs").glob("prompt05*") if p.is_file())
    files.append(root / "runs/accounting.sqlite")
    files = sorted(set(files))
    shards, current, size = [], [], 0
    for path in files:
        count = path.stat().st_size


        if count > 50*2**20 and path.suffix != ".json":
            raise ValueError(f"large non-JSON artifact needs explicit sharding: {path}")
        if current and size+count > 50*2**20:
            shards.append(current)
            current, size = [], 0
        current.append(path)
        size += count
    if current:
        shards.append(current)
    output = root / "reports/evidence/prompt05"
    manifests = []
    for i, paths in enumerate(shards):
        destination = output / f"milestone05-part{i:03d}.zip"
        if destination.exists():
            verify(destination)
            manifest = read(destination.with_suffix(".manifest.json"))
            expected = [(p.relative_to(root).as_posix(), file_hash(p)) for p in paths]
            if expected != [(m["path"], m["sha256"]) for m in manifest["members"]]:
                raise ValueError("existing shard differs; never silently replace evidence")
        else:
            manifest = package(root, paths, destination)
        if manifest["archive_bytes"] > 50*2**20:
            raise ValueError(f"compressed archive exceeds release size limit: {destination}")
        manifests.append(manifest)
        print(json.dumps({"part": i, "bytes": manifest["archive_bytes"], "members": len(manifest["members"])}), flush=True)
    inventory = {"schema_version": 1, "scope": "complete milestone05 evidence, including failed and synthetic diagnostic jobs",
        "restore": "Extract only the archives listed below into the repository root; members restore runs/ paths. Older named ZIPs in this folder are historical subsets and are not part of this release.",
        "synthetic_policy": "Synthetic capacity/fixture records remain tagged and excluded from empirical tables.",
        "archives": [{k: m[k] for k in ("archive", "archive_sha256", "archive_bytes")} for m in manifests],
        "file_count": sum(len(m["members"]) for m in manifests), "archive_bytes": sum(m["archive_bytes"] for m in manifests),
        "pilot_check_sha256": file_hash("runs/prompt05-pilot-check.json"),
        "report_sha256": file_hash("reports/PILOT_REPORT.md"), "machine_report_sha256": file_hash("reports/pilot_results.json")}
    write(output / "INDEX.json", inventory)
    print(json.dumps({k: v for k, v in inventory.items() if k not in ("archives", "restore")}), flush=True)


if __name__ == "__main__":
    main()
