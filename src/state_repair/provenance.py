"""Hash local artifacts and the executable source without reading credentials."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def file_hash(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict:
    root=Path(__file__).resolve().parents[2]
    paths=sorted((root/"src").rglob("*.py"))
    paths += [root/"pyproject.toml"]
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest() for p in paths}
    tree_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    try:
        result=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=False)
        commit=result.stdout.strip() if result.returncode==0 else None
        status=subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True,check=False)
        dirty=bool(status.stdout.strip()) if status.returncode==0 else None
    except OSError:
        commit=None
        dirty=None
    return {"code_commit":commit,"working_tree_dirty":dirty,"source_tree_sha256":tree_hash,"source_files":hashes}
