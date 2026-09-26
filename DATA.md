# Experimental data downloads

Download the archives from the [v0.1.0 release](https://github.com/CleoSp/state-reuse-study/releases/tag/v0.1.0).
The source repository contains the small primary-table export and diagnostic
datasets; the complete available experiment records are separate downloads.

| Download | Contents |
|---|---|
| `experiment-data-001.zip`, `experiment-data-002.zip`, ... | All available confirmatory and pilot run records, supporting experiments, generated datasets, checkpoints, predictions, failures, resource logs, and supplementary environment/provenance records |
| `FULL_DATA_MANIFEST.json` | Every archived relative path, byte count, SHA-256, containing archive, and explicit exclusions |
| `original-experiment-source.zip` | Exact bytes of all 101 source files bound by the frozen configuration, plus that configuration and its protocol |
| `bounded-evidence.zip` | Optional smaller download: four evaluation jobs, four report jobs, and detailed pilot summaries |
| `SHA256SUMS.txt` | SHA-256 checksums for every release asset other than this checksum list |

The numbered ZIP files are independent archives, not fragments of a single ZIP.
Download all numbered files for the complete data and extract each into the
same checkout root. They restore the original `runs/` and `reports/` paths.
The original source archive belongs in a separate directory: do not extract it
over the cleaned publication source.

## Download

Using GitHub CLI, from the source repository root:

```sh
gh release download v0.1.0 --repo CleoSp/state-reuse-study --pattern 'experiment-data-*.zip' --pattern FULL_DATA_MANIFEST.json --pattern SHA256SUMS.txt --dir data-download
```

The same files can be downloaded individually from the release page without
GitHub CLI. Keep them together in `data-download/`.

## Verify and restore

Allow enough free disk space for both the ZIP downloads and the extracted data.
Run this Python code from the repository root after downloading all numbered
archives. It checks the archive hashes before extracting anything. Use a fresh
checkout so extraction does not overwrite existing research runs.

```python
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

downloads = Path("data-download")
manifest = json.loads((downloads / "FULL_DATA_MANIFEST.json").read_text())
for entry in manifest["archives"]:
    path = downloads / entry["name"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
for entry in manifest["archives"]:
    with ZipFile(downloads / entry["name"]) as archive:
        archive.extractall(".")
print(f"Restored {len(manifest['files'])} files.")
```

The manifest's per-file hashes allow checking extracted records as well.
Original job seals, frozen configuration hashes and failed-run records remain
unchanged. See [REPRODUCING.md](REPRODUCING.md) for analysis commands and original
training-replay requirements.

## Scope and exclusions

The archives preserve scientific records as saved, including synthetic
correctness/profile fixtures, failed attempts, and historical source or
preregistration snapshots. Synthetic fixtures remain labeled and do not enter
empirical result tables. The cleaned source is maintained separately from these
historical records.

Installed virtual environments, package/compiler caches, duplicate paper-build
directories, transient process locks, bytecode and the original private Git
database are excluded. These are not additional empirical observations. The
exact excluded directory and file paths are listed in the full-data manifest.
The public release provides the original bound source files without publishing
the original Git database; historical Git-commit audits still need that database.
