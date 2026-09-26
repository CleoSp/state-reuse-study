# State reuse in small recursive solvers

Code and experimental records for **When Does Reusing Computation Help Small Recursive Solvers on Changing Problems?**

[Code](https://github.com/CleoSp/state-reuse-study) · [Reproduction guide](REPRODUCING.md) · [Data downloads](DATA.md) · [Saved results](reports/confirmatory/RESULTS.md) · [MIT license](LICENSE)

The package compares restarting, carrying latent state, reusing previous output probabilities, and learned state adapters on editable mazes and Boolean circuits. The solver is TRM-inspired; it is not an exact TRM or RSM reproduction. Selective repair is a hypothesis evaluated by the experiments, not an assumed improvement.

## Installation and tests

Requires Python 3.11 or later. Create a virtual environment, activate it, then install the package:

```sh
python -m venv .venv
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. On Linux or macOS, use `source .venv/bin/activate`.

```sh
python -m pip install -e ".[dev,reports]"
python -m state_repair.cli doctor --device cpu
python -m pytest -q
```

Core installation needs only `python -m pip install -e .`. The `dev` extra provides pytest; `reports` provides ReportLab for the reporting scripts and full test suite. CPU execution is supported. GPU experiments require a compatible PyTorch installation; the recorded environments are described in [REPRODUCING.md](REPRODUCING.md).

## Reproducing the experiments

The public checkout includes the derived episode data needed to reproduce the primary comparison table, without training a model:

```sh
python scripts/reproduce_review_tables.py
```

This checks saved episode aggregates against the reported estimates and intervals. Full raw-prediction checks require additional evidence; see the guide for the distinction.

[REPRODUCING.md](REPRODUCING.md) covers a small CPU run, experiment configurations, saved-record verification, table generation, and the requirements for repeating training. Use new output directories for new experiments. Published results depend on the original configurations, checkpoints, dataset identities, and complete edit sequences.

The frozen experiment binds the exact original source bytes. Its configuration and protocol remain unchanged; this cleaned source tree is not that historical source snapshot. The reproduction guide explains how to verify saved records and locate the original source for an exact protocol replay.

## Repository contents

| Path | Contents |
|---|---|
| `src/state_repair/` | Typed observations, generators, exact evaluators, recursive solver, adapters, training, evaluation, and resource accounting |
| `configs/` | Experiment recipes and the frozen confirmatory matrix |
| `tests/` | CPU correctness, gradients, information boundaries, stream state, and artifact verification |
| `scripts/` | Training, evaluation, diagnostics, and paper/table generation |
| `reports/confirmatory/` | Saved confirmatory results, numerical summaries, figures, and provenance manifests |
| `reports/FROZEN_PROTOCOL.md` | Original preregistered evaluation protocol |
| `reports/PILOT_REPORT.md` | Pilot results; full pilot archives remain in the original research workspace |
| `reports/paper/` | Manuscript, reproduction records, and diagnostic datasets |
| `review_data/` | Derived episode aggregates for reproducing the primary table |
| `runs/` | Local datasets, checkpoints, predictions, and resource logs; excluded from Git |

The complete available experimental records are distributed as [downloadable archives](DATA.md), rather than stored in Git. They include generated datasets, checkpoints, raw predictions, pilot and confirmatory runs, failed attempts, resource records, and an exact original source snapshot. Individual reproduction commands list their required artifacts in the guide. Synthetic test fixtures are labeled and excluded from empirical results.

Architecture and adaptation details are in [MODEL_DEVIATIONS.md](reports/MODEL_DEVIATIONS.md); observation and state contracts are in [FOUNDATION_INTERFACES.md](reports/FOUNDATION_INTERFACES.md). See the [claim–evidence map](reports/CLAIM_EVIDENCE_MAP.md), [independent audit](reports/ADVERSARIAL_AUDIT_CONFIRMATORY.md), and [related work](sources/READING_LIST.md) for interpretation and provenance.

The project is released under the [MIT license](LICENSE). Third-party materials retain their own licenses. Citation metadata is provided in [CITATION.cff](CITATION.cff).
