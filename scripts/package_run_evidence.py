"""Preserve explicitly selected run artifacts byte-for-byte in a portable ZIP.

This checks byte integrity, not scientific correctness. Run the experiment's
checker first. ZIP member names retain repository-relative paths so a clean
checkout can restore the recorded run layout without changing artifact hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from state_repair.provenance import file_hash


SUFFIXES = {".py", ".toml", ".json", ".jsonl", ".pt", ".md", ".txt", ".gz", ".sqlite"}


def package(root: Path, sources: list[Path], output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    allowed = (root / "runs").resolve()
    if not sources or output.suffix != ".zip" or output.is_relative_to(allowed):
        raise ValueError("require explicit run sources and a ZIP output outside runs")
    members: dict[str, Path] = {}
    for source in sources:
        source = source.resolve()
        if not source.is_relative_to(allowed) or source == allowed or not source.exists():
            raise ValueError("sources must exist strictly inside this repository's runs directory")
        paths = source.rglob("*") if source.is_dir() else [source]
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            if any(part.startswith(".") or part.startswith("venv") for part in path.relative_to(allowed).parts):
                raise ValueError(f"hidden/runtime path is not a run artifact: {path}")
            if path.is_dir():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(allowed) or path.suffix not in SUFFIXES:
                raise ValueError(f"unsupported artifact path: {path}")
            members[path.relative_to(root).as_posix()] = path
    if not members:
        raise ValueError("no artifact files selected")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("evidence package already exists; choose a new name")
    inventory = []
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(members.items()):
            payload = path.read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compresslevel=6)
            inventory.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {"schema_version": 1, "validation_scope": "byte-integrity-only",
                "archive": output.name, "archive_sha256": file_hash(output),
                "archive_bytes": output.stat().st_size, "members": inventory}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    verify(output)
    return manifest


def verify(archive_path: Path) -> dict:
    manifest = json.loads(archive_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if manifest["archive"] != archive_path.name or file_hash(archive_path) != manifest["archive_sha256"]:
        raise ValueError("archive identity/hash mismatch")
    if archive_path.stat().st_size != manifest["archive_bytes"]:
        raise ValueError("archive size mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        expected = [row["path"] for row in manifest["members"]]
        if len(set(expected)) != len(expected) or archive.namelist() != expected:
            raise ValueError("archive member list mismatch")
        for row in manifest["members"]:
            path = Path(row["path"])
            if path.is_absolute() or path.parts[0] != "runs" or ".." in path.parts or "\\" in row["path"]:
                raise ValueError("unsafe archive member path")
            payload = archive.read(row["path"])
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"archive member integrity mismatch: {row['path']}")
    return {"verified": True, "validation_scope": "byte-integrity-only", "files": len(expected),
            "archive_sha256": manifest["archive_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.source:
            parser.error("--verify does not accept --source")
        print(json.dumps(verify(args.output)))
    else:
        result = package(Path.cwd(), args.source, args.output)
        print(json.dumps({k: v for k, v in result.items() if k != "members"}))
