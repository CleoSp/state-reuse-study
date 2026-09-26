# Release validation

Checks performed on the v0.1.0 publication copy on Windows with the existing
CPU Python environment:

| Check | Result |
|---|---|
| `python -m pytest -q -rs` | 352 passed, 1 skipped; the skipped test requires a historical checkpoint supplied in the full data archives |
| `python scripts/reproduce_review_tables.py` | All three primary comparisons and their intervals matched saved values to absolute tolerance `1e-12` |
| `python scripts/reproduce_revision_diagnostics.py` | 96,000 episode rows, 75 diagnostic points, 60 original contrasts, 45 output-reuse contrasts, 1,125 raw audit examples and six consequence-frequency summaries checked |
| `python scripts/validate_revision_timing.py` | 900 timing blocks, 150 stream configurations, 4,950 prediction frames and 150 point estimates checked |
| Original-source recovery | All 101 frozen source hashes matched; the restored snapshot passed the original `validate_binding` checks after a local Git commit and restoration of bound prerequisite records |
| Full data archive integrity | All 16,720 files read back from the 16 ZIP files and matched to their original sizes and SHA-256 hashes |
| Uploaded asset integrity | All 20 GitHub release assets matched local byte counts and server-side SHA-256 values |

No training, new experimental model inference or test-set selection was run for
this release. These checks validate packaging, saved-record reproduction and
the existing tests; they are not an independent replication of the study.
