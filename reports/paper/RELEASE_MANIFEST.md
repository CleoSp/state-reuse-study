# Public release inventory

This repository contains the cleaned experiment source, all 19 experiment
configurations, tests, scientific results, diagnostic datasets, manuscript,
technical documentation and primary-comparison episode aggregates.

The code and associated project documentation are licensed under MIT by
Cleo Spiers. Third-party materials retain their own licenses; dependency
packages and document-template binaries are not redistributed here.

The numbered `experiment-data-*.zip` release assets contain the complete
available run records, including datasets, checkpoints, raw predictions,
failed attempts and resource accounting. `FULL_DATA_MANIFEST.json` records
their membership, checksums and exact exclusions. The separately distributed
`original-experiment-source.zip` restores all 101 files bound by the frozen
source hashes, plus the frozen configuration and protocol.

The optional smaller `bounded-evidence.zip` release asset contains four original sealed
evaluation jobs, four report jobs, and detailed pilot summaries. It contains
no replacement source code. `EVIDENCE_MANIFEST.json` inside the archive records
every included file hash, and `SHA256SUMS.txt` checks the archive as a whole.

The historical research Git database, installed environments, caches and
duplicate document builds are excluded. Original historical source and
preregistration documentation remain verbatim inside evidence archives because
their hashes are part of the saved records. This release does not claim that the cleaned
source has the original source-file hashes. Historical paths and identifiers
inside frozen scientific records are preserved unchanged.

See [REPRODUCING.md](../../REPRODUCING.md) for commands and artifact requirements.
The root `RELEASE_MANIFEST.json` records the public files' SHA-256 hashes.
