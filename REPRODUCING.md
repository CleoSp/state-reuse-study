# Reproducing the experiments

Run commands from the repository root in a Python virtual environment. The
package requires Python 3.11 or later, PyTorch, NumPy and PyYAML. CPU execution
supports correctness checks and the small maze example below. The full study
used CUDA; its runtime and source files are pinned in the frozen configuration.

## Available artifacts

| Artifact | Location | Scope |
|---|---|---|
| Experiment protocol | `reports/FROZEN_PROTOCOL.md` | Frozen hypotheses, splits, operating points and decision rules |
| Experiment configuration | `configs/confirmatory_tier2_frozen.json` | 1,314 jobs, source/data hashes and runtime requirements |
| Results | `reports/confirmatory/` | Saved tables, numerical summaries, figures and synthesis manifest |
| Manuscript | `reports/paper/MANUSCRIPT.md` | Methods, findings and limitations |
| Primary-comparison data | `review_data/primary_episodes.json.gz` | Derived episode aggregates included in the public checkout |
| Bounded evidence | [`bounded-evidence.zip`](https://github.com/CleoSp/state-reuse-study/releases/tag/v0.1.0) | Four evaluation jobs, four report jobs and detailed pilot summaries; separate download |
| Complete available experiment data | [`experiment-data-*.zip`](https://github.com/CleoSp/state-reuse-study/releases/tag/v0.1.0) | Full confirmatory and pilot records, supporting runs, checkpoints, predictions, datasets and resources |
| Exact original source | [`original-experiment-source.zip`](https://github.com/CleoSp/state-reuse-study/releases/tag/v0.1.0) | All 101 original source files bound by the frozen configuration, recovered and verified against their hashes |
| Additional diagnostics | `reports/paper/review_revision/` | Derived episodes, raw audit examples, datasets and timing records |
| Full experiment output | `runs/confirmatory_v1/` | Local, Git-ignored raw jobs, checkpoints and ledgers; approximately 18 GB |

A source checkout does not include the full `runs/` tree. The included data and
bounded evidence download support the checks below, but do not contain all
training data, checkpoints or raw predictions. Full result synthesis and full
training replay need the complete data download and its bound prerequisites.
See [DATA.md](DATA.md) for download, integrity checks, restoration and exclusions.

Saved records and their hashes retain their original filenames. Those paths
are artifact identifiers and must remain unchanged when checking the original
experiment.

## Installation and CPU checks

On Windows PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev,reports]"
.venv/Scripts/python.exe -m state_repair.cli doctor --device cpu
.venv/Scripts/python.exe -m pytest -q
```

On Linux or macOS, use `.venv/bin/python` in place of
`.venv/Scripts/python.exe`. An editable install exposes the `state_repair`
package to scripts without a separate `PYTHONPATH` setting. The `dev` extra
installs pytest and `reports` installs ReportLab for report generation and the
full test suite; `pip install -e .` installs the core package alone. The recorded CPU
environment is in `reports/environment-cpu-windows.lock.txt`; it is a record of
one Windows environment, not a portable dependency lock for every platform.

For CUDA, install a PyTorch build compatible with the local GPU before the
editable install. The frozen study specifies Python 3.12.14 and PyTorch
2.11.0+cu128, float32 with TF32 disabled, on an RTX 5070 with 12 GiB of VRAM and
a 10 GiB allocator limit. Its jobs reject a mismatched PyTorch version. CPU and
CUDA predictions are not guaranteed to be bitwise identical.

## Small data-generation and training example

This example generates editable 8-by-8 mazes, trains a static solver on frame
zero, and evaluates its saved checkpoint. It is a software check, not a
reproduction of the paper's main training runs.

```powershell
.venv/Scripts/python.exe -m state_repair.cli generate --config configs/smoke.yaml
.venv/Scripts/python.exe -m state_repair.cli train --config configs/smoke.yaml --max-minutes 20
.venv/Scripts/python.exe -m state_repair.cli evaluate --run runs/smoke --suite smoke
```

`configs/smoke.yaml` requests 64 roots, 200 optimizer steps, two CPU threads and
a 4 GiB process-memory limit. The time limit is cooperative; a limit-triggered
stop is not a completed training run. The output contains the dataset manifest,
configuration, checkpoint, metrics, prediction hashes and resource records. Preserve
existing outputs: to repeat the example, copy the configuration and set a new
`output_dir` before running all three commands with that configuration and path.
Evaluation requires a completed run and the same source files used for training;
it writes `evaluation.jsonl` and `evaluation_manifest.json`, and prints aggregate
validation metrics. It refuses to overwrite an existing evaluation.

The small CLI's `report` subcommand is disabled and always returns an error.
Use the printed evaluation aggregates or saved metric records for this example.
The study's report and figure generation uses `scripts/synthesize_confirmatory.py`
with the original raw archive, as described below.

The small CLI implements static training. Adapter and full-study training use
the experiment scripts and job matrix described below; passing a repair
configuration to the small CLI does not launch adapter training.

## Reproduce the primary table

The public repository includes the required episode aggregates. Run from its
root, using a virtual environment with NumPy installed:

```powershell
python scripts/reproduce_review_tables.py
```

The checker reads the included `review_data/primary_episodes.json.gz`, recomputes
the three primary comparisons and requires agreement with the saved point
estimates and both interval types to an absolute tolerance of `1e-12`. These
inputs are 7,680 derived seed/root episode aggregates. This check does not
rescore all original node predictions or retrain a model.

With the package dependencies installed, the following commands use the
diagnostic files included in the public repository:

```powershell
python scripts/reproduce_revision_diagnostics.py
python scripts/validate_revision_timing.py
```

The first command checks the previous-prediction analysis using 96,000 derived
episode rows, 1,125 raw audit examples and six test datasets. The timing checker
validates the saved schedule, predictions and timing summaries. Neither command
performs new model inference. These diagnostic and timing analyses are
exploratory and do not replace the frozen confirmatory comparisons.

For the additional saved raw jobs and detailed pilot summaries, download
`bounded-evidence.zip` from the [v0.1.0 release](https://github.com/CleoSp/state-reuse-study/releases/tag/v0.1.0).
Check its SHA-256 against the release's `SHA256SUMS.txt`, then extract it at the
root of a fresh checkout:

```powershell
python -m zipfile -e bounded-evidence.zip .
```

The archive contains data and records, not replacement source code. Its
`EVIDENCE_MANIFEST.json` lists hashes of every included file. It contains only
four evaluation jobs and four report jobs from the full matrix, plus the pilot
summaries; it cannot support a complete raw-action audit or retraining by itself.

## Regenerate all results from the original raw archive

Restore the complete original run tree at `runs/confirmatory_v1/`, including
sealed job directories and the accounting records. Then run:

```powershell
.venv/Scripts/python.exe scripts/synthesize_confirmatory.py
.venv/Scripts/python.exe scripts/audit_confirmatory.py
```

Synthesis verifies input seals and raw-record hashes, then writes
`reports/confirmatory/RESULTS.md`, `summary.json`, `manifest.json` and the SVG
figures. It caches episode aggregates under `runs/synthesis_cache/`. The
independent audit uses separate scoring and statistical checks and inspects the
original protocol commit `cdaf843` in Git history. These commands
need raw jobs omitted from the bounded supplement. `--no-raw` produces an
incomplete synthesis without the exploratory contrasts and per-frame curves;
do not substitute it for the manuscript's full synthesis.

To regenerate the previous-prediction and original-latency exports from their
source jobs, run:

```powershell
.venv/Scripts/python.exe scripts/analyze_previous_predictions.py
.venv/Scripts/python.exe scripts/export_original_latency.py
```

Both commands require additional original sealed evaluations. Checking the
included derived diagnostic data and extracting it again from every original
prediction are separate levels of reproduction.

## Repeat static, adapter and evaluation jobs

The main training implementation is dispatched by `scripts/run_confirmatory.py`.
The frozen matrix supplies the static-solver jobs, adapter training jobs,
validation selection, fixed-budget stream evaluations, reference solvers and
reports in dependency order. Training seeds are 29, 43, 71, 101 and 137. Each
evaluation policy carries the state produced at its own fixed budget into the
next edit; a high-budget trajectory must not supply a lower-budget stream.

Exact replay of the sealed matrix requires the original source snapshot. The
protocol was committed as `cdaf843` and binds source-file bytes, including
comments, as well as datasets and prerequisite records. The publication source
cleanup changes those bytes. The frozen driver deliberately rejects the cleaned
checkout; its hash checks must not be disabled or its frozen hashes updated to
make the old experiment accept new sources.

The public repository has a fresh release history. The separate
`original-experiment-source.zip` restores all 101 historical files named by the
frozen source hashes, plus the original configuration and protocol. Extract
that archive into a separate directory and restore the complete data archives
there for an original-source replay. The archive includes historical
documentation verbatim because its hashes are part of the frozen record.
It does not include the original Git database; commands that inspect historical
Git commits still require the author's retained research history.

Use a separate copy of the original snapshot with the archived bound
prerequisites restored at their recorded paths. These include the original
training/validation datasets, selection records, startup acceptance record and
other artifacts named by the frozen configuration. A source-only checkout or
the bounded supplement is insufficient. The launch guard requires the restored
source and protocol to be committed locally. If using the source ZIP rather
than the original Git checkout, initialize a new Git repository there, disable
line-ending conversion (`git config core.autocrlf false`), and commit the restored
source files and protocol. This new local commit identifies the reproduction
checkout; it does not reconstruct the original research Git history. Keep
`runs/` untracked. In that original-source environment, validate
the binding without launching a job:

```powershell
python -c "from pathlib import Path; from state_repair.execution.binding import validate_binding; from state_repair.execution.durable import read_json; validate_binding(read_json(Path('configs/confirmatory_tier2_frozen.json')), Path('reports/FROZEN_PROTOCOL.md'))"
```

After successful validation, the original entry points are:

```powershell
python scripts/generate_confirmatory_test.py --config configs/confirmatory_tier2_frozen.json --protocol reports/FROZEN_PROTOCOL.md --data runs/confirmatory_v1/data
python scripts/run_confirmatory.py --config configs/confirmatory_tier2_frozen.json --protocol reports/FROZEN_PROTOCOL.md --output runs/confirmatory_v1
```

Test generation uses the reserved identities and requires the bound
training/validation data. It verifies existing complete files; an incomplete
claimed generation stops for inspection. The driver verifies and skips sealed
jobs, resumes supported incomplete jobs and retains interrupted output. Running
it on the completed archive verifies completion rather than retraining. A full
rerun needs a separate experiment directory with its original prerequisite data
and without completed training/evaluation job outputs; preserve the original
archive separately.

The recorded study consumed approximately 239.8 GPU-job hours and 13.7 CPU-job
hours. These are historical measurements, not runtime estimates for another
machine. The frozen matrix records per-job peak-memory estimates and resource
limits. Inspect these before launch; training is not part of the quick checks.

Additional script entry points are retained for the smaller experiments:

| Script | Interface | Prerequisites |
|---|---|---|
| `scripts/train_scaled_maze.py` | `--config`, `--output`, `--seed` | Saved verified crossover records |
| `scripts/train_static_circuit.py` | `--config`, `--output`, `--seed` | Static maze checks and matching circuit capacity/transfer records |
| `scripts/run_adapter_pilot.py` | `prepare`, `cache`, `train`, `select`, `streams`, `interventions`; `--config`, `--root`, `--track`, `--seed` | Source checkpoint and its hash, static checks, prepared data and source seals |
| `scripts/run_circuit_stream.py` | `prepare`, `train`, `select`, `streams`; `--seed` | `configs/circuit_stream_v2.json`, source checkpoint and capacity records |

Use each script's `--help` for its exact argument syntax. Their configurations
refer to original checkpoint hashes and saved prerequisite checks; they are not
standalone training commands for an empty checkout. A new experiment using the
cleaned source needs newly versioned configurations and provenance, while the
original frozen records remain unchanged.

## Rebuild the manuscript

With the required saved evaluation and report jobs present, run:

```powershell
python scripts/extract_budget_diagnostic.py
python scripts/build_paper.py
```

These scripts verify the numerical inputs and generate LaTeX under
`reports/paper/tmlr/`. Compile `main.tex` with a TeX installation, for example
`latexmk -pdf main.tex` from that directory. `preprint.tex` is the named wrapper.
The checked-in source can also be compiled without rerunning the data extractors.
The TMLR style files retain their upstream license and provenance.

## Interpretation

The solver is a TRM-inspired adaptation, not an exact reproduction of an
upstream model. The held-out unit is the base problem/episode. Statistical
resampling preserves paired methods and complete streams; training-seed
variation is reported separately from root variation. Synthetic test fixtures
must not enter empirical tables. The frozen primary experiment did not establish
the targeted accuracy-preserving 25% cost reduction or a general advantage for
selective repair. See the manuscript and saved results for the positive,
negative and exploratory findings and their limits.
