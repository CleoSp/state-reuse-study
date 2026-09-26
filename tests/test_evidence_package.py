"""Byte integrity and path boundaries; these fixtures contain no measurements."""
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_run_evidence import package, verify


def test_evidence_roundtrip_preserves_exact_bytes_and_rejects_corruption(tmp_path):
    source = tmp_path / "runs" / "synthetic"
    source.mkdir(parents=True)
    payload = b'{"synthetic":true}\r\n'
    (source / "summary.json").write_bytes(payload)
    (source / "checkpoint.pt").write_bytes(b"synthetic-binary-fixture\x00\xff")
    (source / "predictions.jsonl.gz").write_bytes(b"synthetic-compressed-fixture")
    (source / "accounting.sqlite").write_bytes(b"synthetic-ledger-fixture")
    output = tmp_path / "evidence.zip"
    manifest = package(tmp_path, [source], output)
    assert verify(output)["files"] == 4
    with zipfile.ZipFile(output) as archive:
        assert archive.read("runs/synthetic/summary.json") == payload
    assert manifest["validation_scope"] == "byte-integrity-only"
    with pytest.raises(FileExistsError):
        package(tmp_path, [source], output)
    duplicate = tmp_path / "duplicate.zip"
    assert package(tmp_path, [source], duplicate)["archive_sha256"] == manifest["archive_sha256"]
    manifest["members"][0]["sha256"] = "0"*64
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="member integrity"):
        verify(output)


def test_packaging_refuses_broad_sources_and_runtime_directories(tmp_path):
    runs = tmp_path / "runs"
    runtime = runs / "venv-cuda"
    runtime.mkdir(parents=True)
    (runtime / "runtime.py").write_text("# synthetic runtime fixture")
    for source in (tmp_path, runs, runtime):
        with pytest.raises(ValueError):
            package(tmp_path, [source], tmp_path / "evidence.zip")
