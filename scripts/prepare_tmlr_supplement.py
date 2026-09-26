"""Create a bounded, anonymous TMLR supplement from existing evidence.

No training, inference, test selection, upload or license grant. Primary
operating-point episode aggregates are exported from verified saved records.
The saved original rows and frozen files are never rewritten.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from synthesize_confirmatory import job_episodes, PRIMARY

DEST = ROOT / "runs/paper_tmlr/supplement"
RUNS = ROOT / "runs/confirmatory_v1"
PAPER = ROOT / "reports/paper"


def copy(source: Path, relative: str | Path) -> None:
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    start = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "reports/confirmatory/summary.json").read_text())
    primary = []
    inputs = {}
    repack = "--repack" in sys.argv
    existing_data = DEST / "review_data/primary_episodes.json.gz"
    if repack:
        previous = json.loads((DEST / "PACKAGE_MANIFEST.json").read_text())
        if sha(existing_data) != previous["files"]["review_data/primary_episodes.json.gz"]:
            raise ValueError("Existing episode export hash changed")
        with gzip.open(existing_data, "rt") as handle:
            exported = json.load(handle)
        primary, inputs = exported["episodes"], exported["source_hashes"]
    for suite in (() if repack else PRIMARY):
        for policy in ("restart", "answer_only"):
            k = 8 if suite.startswith("maze") else (2 if policy == "restart" else 1)
            for seed in (29, 43, 71, 101, 137):
                job = f"test-{suite}-{policy}-s{seed}-h32"
                rows = job_episodes(RUNS / job, ROOT / "runs/synthesis_cache")
                selected = [r for r in rows if r["K"] == k]
                if len(selected) != 256 or any(r["synthetic"] for r in selected):
                    raise ValueError(f"Unexpected episode coverage: {job}")
                primary.extend(selected)
                inputs[job] = json.loads((ROOT / "runs/synthesis_cache" / f"{job}.json").read_text())["records_sha256"]
            print(f"Exported {suite} {policy} K{k}", flush=True)
    data_path = DEST / "review_data/primary_episodes.json.gz"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            out.write(json.dumps({"synthetic": False, "derived_from_saved_records": True,
                                  "source_hashes": inputs, "episodes": primary}, separators=(",", ":")).encode())

    for folder in ("src", "tests"):
        for source in sorted((ROOT / folder).rglob("*.py")):
            copy(source, source.relative_to(ROOT))
    copy(ROOT / "pyproject.toml", "pyproject.toml")
    for name in ("build_paper.py", "build_review_revision.py", "analyze_previous_predictions.py", "measure_lean_streams.py", "continue_lean_streams.py", "validate_revision_timing.py", "export_original_latency.py", "reproduce_revision_diagnostics.py", "extract_budget_diagnostic.py", "synthesize_confirmatory.py", "audit_confirmatory.py", "reproduce_review_tables.py"):
        copy(ROOT / "scripts" / name, "scripts/" + name)
    for name in ("summary.json", "manifest.json", "RESULTS.md"):
        copy(ROOT / "reports/confirmatory" / name, "reports/confirmatory/" + name)
    for source in (ROOT / "reports/confirmatory/figures").glob("*.svg"):
        copy(source, source.relative_to(ROOT))
    for name in ("budget_diagnostic.json", "seed_matched_lesions.json", "maze_crossover_display.json"):
        copy(PAPER / name, "reports/paper/" + name)
    for source in (PAPER / "review_revision").rglob("*"):
        if source.is_file() and source.name not in ("timing_extension_approval.json", "timing_reservation.json"):
            copy(source, source.relative_to(ROOT))
    for name in ("FROZEN_PROTOCOL.md", "MODEL_DEVIATIONS.md"):
        copy(ROOT / "reports" / name, "reports/" + name)
    copy(ROOT / "configs/confirmatory_tier2_frozen.json", "configs/confirmatory_tier2_frozen.json")


    reports = ["report-" + s + "-h32" for s in (*PRIMARY, "circuit64-on-circuit96")]
    diagnostic = json.loads((PAPER / "budget_diagnostic.json").read_text())
    names = reports + [j["job"] for j in diagnostic["jobs"]]
    for name in names:
        for source in (RUNS / name).rglob("*"):
            if source.is_file():
                copy(source, Path("runs/confirmatory_v1") / name / source.relative_to(RUNS / name))

    for source in (PAPER / "tmlr").iterdir():
        if source.name in ("preprint.tex", "official-template.tex") or source.suffix not in (".tex", ".bib", ".bst", ".sty", ".txt", ".md", ".json"):
            continue
        copy(source, "reports/paper/tmlr/" + source.name)


    for relative in ("reports/paper/tmlr/preprint.tex", "reports/paper/tmlr_build_manifest.json"):
        (DEST / relative).unlink(missing_ok=True)
    manuscript = (PAPER / "MANUSCRIPT.md").read_text(encoding="utf-8").splitlines()
    manuscript[4] = "Anonymous authors."
    (DEST / "reports/paper/MANUSCRIPT.md").write_text("\n".join(manuscript) + "\n", encoding="utf-8")
    readme = """# Anonymous review supplement

This package accompanies the TMLR review-format manuscript. It contains code,
the frozen protocol/configuration, full saved numerical summaries and figures,
derived episode aggregates for the three primary comparisons, four sealed report
directories, and four sealed evaluation jobs for the exploratory K16 case study.
It is a bounded review artifact, not the complete 1,314-job archive.

## Reproduce the primary table

In a Python virtual environment with NumPy installed, from this directory:

    python scripts/reproduce_review_tables.py

This recomputes paired-root and crossed accuracy intervals and paired cost
ratios from the included seed/root episode aggregates and checks them against
the saved primary decisions. It uses 10,000 draws and seed 64037. These inputs
are derived from raw predictions; they are not raw node-action records.

## Inspect and regenerate the paper

Install the package in a virtual environment (`python -m pip install -e .`).
Then run `python scripts/build_paper.py`, followed by `tectonic main.tex` from
`reports/paper/tmlr/` (or use a standard TeX distribution and latexmk).
The paper builder verifies the included report seals and saved summary hashes.
`python scripts/extract_budget_diagnostic.py` rescans the included four raw
evaluation jobs and regenerates the frame-zero diagnostic; it runs no model.
The anonymous manuscript will also generate an anonymous preprint wrapper.

The review revision includes 96,000 four-way previous-prediction episode counts,
1,125 raw-transition audit examples, the six original test datasets, and complete
exploratory timing blocks/predictions. Run `python scripts/reproduce_revision_diagnostics.py`
to reproduce diagnostic gains and intervals, original policy-versus-restart
contrasts, ordinary consequence frequencies and the raw audit sample from this
package alone. Full extraction from all saved predictions requires the original
run archive. Timing repetitions and root/seed samples remain separately labeled.
Run `python scripts/validate_revision_timing.py` to verify the completed 900-block
schedule, matched predictions and all 150 timing point estimates.

The full synthesis and independent-audit scripts are included for inspection,
but require additional raw runs not included in this bounded supplement. The
full raw-action audit, all trained checkpoints, training/validation datasets, training history,
and full ledgers cannot be independently reconstructed from this ZIP alone.
The complete original run archive, approximately 18 GB, will be deposited in a
public archive at publication; until then, it is available from the authors.
No external replication is claimed. The frozen scientific settings remain
unchanged; the K16 case study was selected exploratorily after seeing curves.

Third-party TMLR template files retain their upstream Apache-2.0 license and
notices. Repository licensing is handled separately from review distribution.
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")


    forbidden = (b"cleosp", b"cleo spiers", b"ghp_", b"github_pat_", b"-----BEGIN PRIVATE KEY-----")
    files = sorted(p for p in DEST.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json"
                   and "__pycache__" not in p.parts and p.suffix != ".pyc")
    for path in files:
        payload = path.read_bytes().lower()
        if any(token.lower() in payload for token in forbidden):
            raise ValueError(f"Identity/credential review needed: {path.relative_to(DEST)}")
    manifest = {"synthetic": False, "contents": "bounded anonymous review supplement",
                "episodes": len(primary), "source_hashes": inputs,
                "files": {p.relative_to(DEST).as_posix(): sha(p) for p in files},
                "elapsed_seconds": time.perf_counter() - start, "gpu_seconds": 0}
    (DEST / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = PAPER / "tmlr-supplement.zip"
    staged_archive = PAPER / "tmlr-supplement.staging.zip"
    with zipfile.ZipFile(staged_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(DEST.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                z.write(path, path.relative_to(DEST).as_posix())
    with zipfile.ZipFile(staged_archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive CRC failed")
    if staged_archive.stat().st_size >= 100_000_000:
        raise ValueError("Supplement exceeds TMLR 100MB limit")
    staged_archive.replace(archive)
    print(f"Created {archive.name}: {archive.stat().st_size:,} bytes; {len(files)} files; {len(primary)} paired episode rows", flush=True)


if __name__ == "__main__":
    main()
