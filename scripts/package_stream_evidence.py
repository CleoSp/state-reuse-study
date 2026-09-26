"""Package verified stream-training evidence in immutable, size-bounded shards."""
from __future__ import annotations

import argparse
from pathlib import Path

from package_run_evidence import package, verify
from run_adapter_pilot import read, write
from state_repair.provenance import file_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",choices=("maze","circuit","diagnostics","longer","depth"),required=True)
    args = parser.parse_args()
    root = Path.cwd()
    if args.step in ("longer","depth"):
        name = "adapter_pilot_v1_longer" if args.step == "longer" else "circuit_stream_v2_depth48"
        checked = read(root / "runs" / name / "check.json")
        if not checked["verified"] or not checked["complete"]:
            raise ValueError("complete addendum checker required")
        dirs = [root / "runs" / name]
        files = []
    elif args.step in ("maze","circuit"):
        name = "adapter_stream_v2" if args.step == "maze" else "circuit_stream_v2"
        checked = read(root / "runs" / name / "check.json")
        if not checked["verified"] or not checked["complete"]:
            raise ValueError("complete step checker required before packaging")
        dirs = [root / "runs" / name / p for p in ("prepared","training","streams")]
        if args.step == "maze":
            dirs.append(root / "runs" / name / "interventions")
        dirs.append(root / "runs" / f"{name}_capacity")
        dirs.extend(sorted((root / "runs").glob(f"{name}_capacity_failed_*")))
        files = [root / "runs" / name / p for p in ("check.json","selection.json")]
        files.extend((root / "runs" / name).glob("*.json"))
        g1 = root / "runs" / name / "g1_check.json"
        if not read(g1)["verified"]:
            raise ValueError("verified G1 dynamics required for each task family")
        files.append(g1)
        dirs.extend(sorted((root / "runs" / name / "diagnostics").glob("*-dynamics-seed*")))
    else:
        dirs = [root / "runs" / name / "diagnostics" for name in ("adapter_stream_v2","circuit_stream_v2")]
        files = [root / "runs" / name / "diagnostic_check.json" for name in ("adapter_stream_v2","circuit_stream_v2")]
        files.append(root / "runs/circuit_stream_v2/k18_check.json")
        if not all(read(p)["verified"] for p in files):
            raise ValueError("both diagnostic checkers required")
    for directory in dirs:
        if not directory.exists():
            raise ValueError(f"missing evidence directory: {directory}")
        files.extend(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files += [p for p in (root / "runs").glob("prompt05b-*.txt") if p.is_file()]


    snapshot = root / "runs" / f"prompt05b-{args.step}-accounting"
    if not snapshot.exists():
        snapshot.mkdir()
        import sqlite3
        with sqlite3.connect(root / "runs/accounting.sqlite") as source, sqlite3.connect(snapshot / "accounting.sqlite") as target:
            source.backup(target)
        (snapshot / "gpu_time.jsonl").write_bytes((root / "runs/prompt05b_gpu_time.jsonl").read_bytes())
    files.extend(snapshot.iterdir())
    files = sorted(set(files))
    shards,current,size = [],[],0
    for path in files:
        count = path.stat().st_size
        if count > 50*2**20 and path.suffix != ".json":
            raise ValueError(f"large non-JSON evidence needs explicit sharding: {path}")
        if current and size+count > 50*2**20:
            shards.append(current);current=[];size=0
        current.append(path);size+=count
    if current:
        shards.append(current)
    output = root / "reports/evidence/prompt05b"
    manifests = []
    for i,paths in enumerate(shards):
        dest = output / f"{args.step}-part{i:03d}.zip"
        if dest.exists():
            verify(dest)
            manifest = read(dest.with_suffix(".manifest.json"))
            if [(p.relative_to(root).as_posix(),file_hash(p)) for p in paths] != [(r["path"],r["sha256"]) for r in manifest["members"]]:
                raise ValueError("existing evidence shard differs")
        else:
            manifest = package(root,paths,dest)
        if manifest["archive_bytes"] > 50*2**20:
            raise ValueError("compressed release shard exceeds size limit")
        manifests.append({k:manifest[k] for k in ("archive","archive_bytes","archive_sha256")})
    write(output / f"{args.step}-INDEX.json",{"step":args.step,"archives":manifests,"files":len(files),
        "restore":"Extract listed shards into repository root, preserving runs/ paths.",
        "synthetic_policy":"Synthetic profiles and tests stay labeled; empirical tables exclude them."})


if __name__ == "__main__":
    main()
