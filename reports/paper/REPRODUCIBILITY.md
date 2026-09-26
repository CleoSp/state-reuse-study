# Reproducibility checklist for the confirmatory study

For installation and commands supported by the publication source, start with
[REPRODUCING.md](../../REPRODUCING.md). The frozen protocol binds the original
source bytes; running its training driver requires that original snapshot and
its archived prerequisites. The source cleanup does not update those bindings.

Every command runs from the repository root. The CPU virtual environment is `.venv/Scripts/python.exe` (Windows); the CUDA environment used for all GPU jobs was `runs/foundation/venv-cuda/Scripts/python.exe` (PyTorch 2.11.0+cu128, RTX 5070, 12 GiB, driver-level allocator cap 10 GiB, TF32 off, float32). Environment locks: `reports/environment-cpu-windows.lock.txt`; CUDA install logs under `reports/foundation/install-cuda-*.txt`. Raw run artifacts live under `runs/`, which is Git-ignored; the sealed confirmatory tree `runs/confirmatory_v1` (about 18 GB) and its evidence archives under `reports/foundation/prompt06-*.zip` are the primary data.

## What was frozen, and when

| Item | Where | Hash |
|---|---|---|
| Protocol | `reports/FROZEN_PROTOCOL.md`, commit `cdaf843` (2026-09-09 22:46:31 −0700) | SHA-256 `ea3020f73c78d71601034849fa8d5a4a5cea4c17616fb058c92243b81da237a1` (also bound in `runs/confirmatory_v1/data/test-manifest.json`) |
| Matrix and operating points | `configs/confirmatory_tier2_frozen.json` (1,314 jobs, `status: FROZEN`) | canonical-JSON SHA-256 `7c71eb257b2b00ad05ce805ea7fbe1b3a9e910ebbd6ab0d5b15c5a645390eeba` |
| Test identities | per-suite identity lists and hashes in the protocol table and config | e.g. maze12 test `5ffe2132…944fe`, circuit48 test `dae659fd…0950a` |
| Test generation | `scripts/generate_confirmatory_test.py`, one-time claims in `runs/confirmatory_v1/data/*-test-claim.json`; files created 34 to 788 s after the protocol commit | per-file SHA-256 in `test-manifest.json` |
| Selection rule | `src/state_repair/eval/statistics.py::select_operating_points`; development values in `reports/foundation/prompt06-development-selection.json` | `b2e4b9ce…3fb55` |

## Regenerating the results

1. Verify every sealed job and rescore every raw action. The sealed report jobs already did this through `state_repair.eval.checker.check_job` (46,719,642 raw actions across 766 checker rows, all verified). To repeat it read-only, call `state_repair.execution.driver.verify(output, job)` and `check_job(output, job)` per directory; `scripts/synthesize_confirmatory.py` re-verifies every report directory and every raw log it reads, and `scripts/audit_confirmatory.py` rescored a 7,128-row sample with independent code. Re-running the driver itself on the completed matrix is a no-op that reports the matrix complete.
2. Regenerate all tables and figures from sealed records:
   ```powershell
   $env:PYTHONPATH='src'; .venv/Scripts/python.exe scripts/synthesize_confirmatory.py
   ```
   Writes `reports/confirmatory/RESULTS.md`, `summary.json`, `manifest.json` (input record hashes, output hashes) and `figures/*.svg`. The recorded full synthesis took 432.5 s from cache and 984.2 s cold on the original host; it caches per-job episode rows under `runs/synthesis_cache/`. `--no-raw` omits exploratory contrasts and per-frame curves and must not replace the full synthesis for the manuscript.
3. Independent audit regeneration (own bootstrap, own route follower and circuit evaluator):
   ```powershell
   $env:PYTHONPATH='src'; .venv/Scripts/python.exe scripts/audit_confirmatory.py > runs/prompt07-audit-confirmatory.txt
   ```
   About 185 s on CPU; report in `reports/ADVERSARIAL_AUDIT_CONFIRMATORY.md`.
4. CPU test suite (synthetic fixtures only; includes gradient-contract, budget-provenance, timing, pairing, checker and synthesis tests):
   ```powershell
   $env:PYTHONPATH='src'; .venv/Scripts/python.exe -m pytest -q
   ```

## Building the TMLR review and preprint PDFs

From the repository root, with the completed saved results present:

```powershell
.venv/Scripts/python.exe scripts/extract_budget_diagnostic.py
.venv/Scripts/python.exe scripts/build_paper.py
$env:TECTONIC_CACHE_DIR = Join-Path (Get-Location) 'runs/paper_tools/tex-cache'
runs/paper_tools/tectonic/tectonic.exe --only-cached --untrusted --keep-logs --outdir runs/paper_tmlr reports/paper/tmlr/main.tex
runs/paper_tools/tectonic/tectonic.exe --only-cached --untrusted --keep-logs --outdir runs/paper_tmlr reports/paper/tmlr/preprint.tex
```

The local compiler used for the initial PDF is Tectonic 0.17.0, installed only
under `runs/paper_tools/`; a standard TeX distribution can instead run
`latexmk -pdf main.tex` from `reports/paper/tmlr/`. The final outputs are
`output/pdf/state-reuse-tmlr-review.pdf` and `output/pdf/state-reuse-preprint.pdf`.
The source ZIP excludes the named preprint wrapper. Official style files are
pinned unchanged to `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`, with Apache-2.0
attribution. The earlier venue-neutral PDF and source are historical drafts.

The paper builder verifies the synthesis output hashes and four sealed h32
reports, compares their curves with `summary.json`, and verifies the manuscript's
matched-seed table. It writes `seed_matched_lesions.json` and `tmlr_build_manifest.json`.
It also verifies the manuscript's maze contrast table against the saved paired
contrasts and writes `maze_crossover_display.json`: 60 non-self contrasts across
35 policy/budget rows, plus the minimum and maximum of five seed means at each
point. Table 6 uses the existing paired-root intervals; crossed intervals remain
in the supplement. Figure 1's shaded seed ranges are descriptive, not confidence
intervals. No new bootstrap or experimental evaluation is run for these displays.
The matched table uses seeds 29, 43 and 71 for both lesions and intact/restart
comparators; no new bootstrap intervals are introduced. This build is a
presentation check, not a repeat of the full raw-action audit.
Numbered manuscript headings carry stable section anchors, and each Markdown
table has a semantic marker. Cross-references are emitted as LaTeX `\ref`
commands; the build rejects missing targets, duplicate labels and hard-coded
section/table/figure references. All vector figure input files accompany
both the generated sources and the editable `paper-latex` copy.

The K16 extractor verifies four sealed evaluation jobs, counts every frame for
K8/K16, rejects duplicate or missing roots, and checks post-edit means against
the saved per-seed curves. It wrote `budget_diagnostic.json` in 12.31 s on CPU
with no model inference. The selected cases are exploratory.

Build and check the anonymous supplement:

```powershell
.venv/Scripts/python.exe scripts/prepare_tmlr_supplement.py
.venv/Scripts/python.exe runs/paper_tmlr/supplement/scripts/reproduce_review_tables.py
```

It includes 7,680 paired episode aggregates, four full diagnostic evaluation
jobs, four report jobs, source/tests, the frozen protocol/config and numerical
summaries. It is not the full raw-run archive. Primary-table reproduction
matches all three stored decisions and both interval types to 1e-12.
The complete original run archive, approximately 18 GB, will be deposited in a
public archive at publication; until then, it is available from the authors.
`--repack` hash-checks the existing episode export while refreshing sources;
it does not rescan the unchanged primary raw jobs.

## Exploratory review revision

The revision does not replace the frozen experiment. To re-extract the
previous-prediction diagnostic from the original saved predictions:

```powershell
.venv/Scripts/python.exe scripts/analyze_previous_predictions.py
.venv/Scripts/python.exe scripts/export_original_latency.py
```

The first command verifies all files against the seals of 75 original evaluation
jobs, checks prediction/configuration/dataset/checkpoint provenance and complete
own-budget sequences, and rescores 3,168,000 saved predictions.
It exports 96,000 seed/root/budget episode rows covering 3,072,000 edited
transitions, plus 1,125 raw audit examples and six ordinary consequence-frequency
summaries. These are repeated measurements within 256 roots and five seeds per
point, not millions of independent replications. Extraction took 228.8 seconds
on CPU. The second command exports 75 existing sealed batch-one curves; it runs
no inference.

The bounded supplement includes the derived episodes, raw audit examples, six
original test datasets and regeneration code. From either the repository or the
extracted package, run:

```powershell
.venv/Scripts/python.exe scripts/reproduce_revision_diagnostics.py
```

Use the package's own virtual-environment Python if `.venv` is elsewhere. The
checker reconstructs diagnostic point estimates and crossed/root intervals,
original restart and available output-reuse contrasts, ordinary consequence
frequencies, and raw audit scores. Full extraction still needs omitted original
raw jobs. The two reproduction levels are explicitly distinguished in the PDF.

The additional timing sensitivity is recorded in
`review_revision/timing_blocks.jsonl`, `timing_predictions.json`
and `timing_summary.json`. It uses existing seed-29 checkpoints, two fixed roots
per primary suite, all principal arms/budgets, complete 32-edit streams and three
interleaved repetitions. No partial sweep enters a figure or table: the builder
requires `complete=true` before including the new timing displays. The original
attempt stopped and was resumed on the exact saved schedule; both portions
remain in the timing records.

## Re-running the frozen experiment from scratch

Not required to check any number in the paper, and not cheap: 239.8 GPU-hours on one RTX 5070. The unattended driver is `scripts/run_confirmatory.py` with the frozen config; it reconciles ledgers, quarantines unsealed directories, resumes exactly from checkpoints and refuses to run if the committed protocol, matrix or bound datasets changed. Register it as a startup task with `scripts/install_confirmatory_startup.ps1` (exact XML exports under `reports/foundation/confirmatory-production*.xml`). Cross-runtime bitwise reproducibility is not guaranteed: a prior CPU spot replay of CUDA predictions differed on a few route decisions after long streams (see `reports/PILOT_REPORT.md`), so a fresh run will reproduce the protocol, not the exact bytes.

## Data and evaluator lineage

- Maze bases: `state_repair.data.maze.generate_maze` (randomized Kruskal tree plus independent extra passages, probability 0.15), every fourth base vertically cut; edits: `generate_episode` uniform passage toggles; oracle: BFS in `state_repair.oracles.maze`, independently checked against Floyd–Warshall and by exhaustive enumeration of 2x3 grids in the static audit.
- Circuit bases: `state_repair.data.circuit.generate_circuit` (ordinary) and `state_repair.data.deep_circuits.generate_deep_circuit` (depth-shift suites); edits: input flips and binary-operator substitutions; oracle: topological evaluation checked against an independent event-driven evaluator.
- Splits: base roots assigned before descendants; `check_isolation` rejects cross-split canonical duplicates; test identities reserved before generation.
- Record modes and denominators: `state_repair.eval.metrics` (`episodes` keeps every frame including failures; `validate_record` enforces `state_budget == source_budget == K`).

## Statistics

`state_repair.eval.statistics.paired_contrast` and `paired_cost_ratio`: seeds averaged within each root, 256 roots resampled with replacement, 10,000 draws, `numpy.random.default_rng(64037)`; crossed intervals also resample the five seeds with the same root draw. Cost ratio is the ratio of paired mean amortized episode costs. Decision thresholds are in `reports/FROZEN_PROTOCOL.md` and implemented in `scripts/synthesize_confirmatory.py::decide`.

## Compute and cost ledger

`runs/confirmatory_v1/gpu_time.jsonl` and `runs/confirmatory_v1/accounting.sqlite` (read-only queries): 1,335 reserve/actual pairs, 239.790751 GPU-job hours and 13.680733 CPU-job hours, two aborted attempts (372.11 s graceful checkpoint; 0.757 s lower bound after a reset) both quarantined and excluded, $0 external charges, electricity unmeasured. Evidence archives with SHA-256 manifests: `reports/foundation/prompt06-*-archive.json` and `.zip`.
