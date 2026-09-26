# Data, state, configuration, and accounting interfaces

The definitions in [`types.py`](../src/state_repair/types.py) separate observed
inputs, learned state, labels, and evaluator-only metadata. Maze and circuit
models share these interfaces.

## Observations and transitions

| Type | Fields and validation |
|---|---|
| `ObservationBatch` | `domain`; finite floating `node_features[B,N,F]`; Long `edge_types[B,N,N]`; Bool `valid_nodes[B,N]`; tuple `episode_ids` and nonnegative integer `frame_indices`; Long `node_ids[B,N]`; domain-specific grid metadata. Tensors share a device; each example has valid nodes. Padding has zero features/edges and ID -1. |
| `ObservedEdit` | Finite floating `node_features_delta[B,N,F]` and Bool `edge_changed[B,N,N]` on the same device. `between(old,new)` checks matching domain/layout/features/device/dtype and stable episode/node correspondence, then computes differences solely from observations. |
| `RecurrentState` | Finite floating `a,z[B,N,D]` with matching shape/dtype/device; episode/frame/node/mask bookkeeping; nonnegative integer scalar `budget`. Padded latent values may be finite nonzero and are ignored by the solver. |
| `TransitionInput` | `new`, optional `old`, `old_state`, and `edit`. Frame zero forbids prior inputs. Edited frames require all prior inputs and consecutive frames. Old state must match the old observation; the edit must exactly match observed differences. Mixed fresh/edited examples need separate batches. |

Maze observations have four features: normalized row/column coordinates and
start/goal indicators. `grid_shapes`, Long `starts[B]`, and Long `goals[B]` are
required. Start/goal fields refer to stable identities, not tensor positions.
Relations use `edge_types[batch, source, target]`: `0=absent`, `1=north`,
`2=east`, `3=south`, `4=west`. Each passage has opposite reverse codes. Geometry
is checked against stable row-major identities and grid width, accepting joint
presentation permutations while rejecting row wraps and one-way passages.

Circuit observations have six features: exact one-hot `INPUT,AND,OR,XOR,NOT`,
followed by the observed external input bit. The bit is zero on internal/padded
nodes. Relations are `0=absent`, `1=source-to-target wire`, and `2=reverse
relation`. Only code-1 edges determine arity and acyclicity; code 2 is not an
additional wire. Validation discards its temporary topological traversal.
Circuit grid metadata is `grid_shapes=(), starts=None, goals=None`. IDs are
contiguous stable identities in arbitrary presentation order, not features or
an execution order.

Bookkeeping IDs, frames, seeds, split/root labels, and oracle data never enter
feature encoders. Inference accepts typed observations/states or `TransitionInput`,
not a complete sample dictionary. The trainer unpacks
`TrainingExample(transition,target)`. Runtime checks reject extra fields and
invalid correspondence, but cannot detect deliberately hidden privileged values
inside a valid-shaped feature tensor; feature constructors remain part of the
information boundary.

## Targets and evaluator metadata

`TargetBatch(valid_actions=None, values=None, scored_mask=None)` accepts exactly
one complete domain variant. Maze targets are Bool `valid_actions[B,N,6]`, with
at least one valid action per real node and no padded labels. Circuit values are
Long `[B,N]`, `0/1` on non-input gates and `-1` on inputs/padding. Their Bool
`scored_mask[B,N]` equals valid non-input gates and must be nonempty for every
example. `check_observation(obs)` validates domain, shape, and device at the
trainer/data boundary; inference never calls it. `.to(device)` handles absent
fields.

Maze `OracleMetadata` stores Long `distances[B,N]` (`-1` unreachable, `-2`
padding), optional Bool changed-action/distance/invalid-prediction masks, and
optional audit data. Circuit
[`CircuitOracleMetadata`](../src/state_repair/data/circuit.py) stores descendant
and actual changed-value masks plus audit data. Circuit masks follow presentation
order and are false at padding/frame zero. Descendants exclude the edited node;
changed values can include an edited input. These masks and reference-solver
caches/queues never enter learned observations or states.

## State storage and gradients

`RecurrentState.clone()` and `.to(device)` return independent tensor storage and
preserve gradients. `.detach()` clones all storage and removes the old latent
graph. `.reset(mask,fresh)` selects per-example latents and metadata into new
storage, preserving selected gradient paths. An all-false mask returns an
independent old-state clone even when fresh budget differs; an all-true mask
takes the fresh budget. Partial resets require equal scalar budgets and matching
dtype/device/shape. Separate different-budget streams rather than borrowing
higher-budget state.

Detach old state before adaptation. Keep adapter output and the current
frozen-backbone rollout on the autograd graph. `TransitionInput` checks
correspondence without detaching tensors. Frozen dataclasses prevent field
rebinding, but their PyTorch tensors remain mutable; independent storage is
provided by state helpers, not by arbitrary caller-supplied aliases.

## Configuration and command interfaces

[`load_config(path)`](../src/state_repair/config.py) reads safe YAML and rejects
duplicate/non-string keys, unknown fields, malformed YAML, unsupported schema or
workflow names, invalid numeric values, dimensions, budgets, split fractions,
and head divisibility. It returns frozen `ExperimentConfig` sections with
`to_dict()` and `digest()`. Paths are relative to the process working directory.
Direct dataclass construction is an internal interface; the YAML loader performs
the strict configuration checks.

The compact [`state-repair` CLI](../src/state_repair/cli.py) uses schema version 1
and defaults to workflow `static`. Although `repair` and `confirmatory` are
recognized configuration values, this CLI's generation/training entry points
reject them before side effects. The separate experiment matrix and training
scripts provide the broader research workflows. The CLI smoke training command
accepts `0 < --max-minutes <= 20`; its report command currently rejects execution.
See [REPRODUCING.md](../REPRODUCING.md) for supported reproduction commands.
Missing artifacts fail rather than generating successful-run placeholders.

Source/config compatibility guards bind artifacts to their recorded code and
configuration. Existing records retain those hashes. Reproducing an older
artifact requires its recorded source commit and environment; do not rewrite its
hashes to make a newer checkout pass. There is no cloud launcher.

## Accounting and resource records

[`AccountingLedger`](../src/state_repair/accounting/ledger.py) provides
`reserve(job_id,planned_usd,category='local')`,
`reconcile(job_id,actual_usd)`, and `snapshot()`, with a default total of $950.
All jobs sharing an allocation must use the same canonical database path.
Independent worktrees do not create independent copies of the allocation.

Reservations run in a SQLite `BEGIN IMMEDIATE` transaction using integer cents.
Remaining funds equal allocated funds minus actual spending and outstanding
reservations. Duplicate job IDs/reconciliations fail. Reconciling a cancelled
zero-charge job with zero releases the reservation. Actual overruns are recorded
when they make the remaining balance negative, blocking subsequent reservations.
Categories are labels, not independently enforced suballocations; the ledger
alone does not enforce the study's $650 GPU allocation.

`RunManifest(path)` provides `append(record)`, `records()`, and exclusive-create
`export_jsonl(path)`. Records require an explicit Boolean `synthetic`; nonfinite
JSON values fail. SQLite serializes appends, and export preserves journal order.
UPDATE/DELETE triggers protect manifests, cost events, and allocation from
accidental mutation; they do not protect against a database owner changing the
schema. Callers supply their own experiment/provenance fields.

[`memory_snapshot()`](../src/state_repair/accounting/resources.py) reports host
and process measurements in bytes where supported, with null/error information
otherwise. Device timing and allocator peaks are recorded separately. Unknown
electricity cost is null, not a measured zero.
