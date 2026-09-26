# Frozen confirmatory protocol: tier 2, version 1

<!-- matrix-sha256: 7c71eb257b2b00ad05ce805ea7fbe1b3a9e910ebbd6ab0d5b15c5a645390eeba -->

This is a prospective protocol, not a test result. This file and
`configs/confirmatory_tier2_frozen.json` must be committed unchanged before the first test root is
generated. The exact ordered job definitions, all checkpoint-producing job names,
selection dependencies, identity lists and implementation hashes are in that
config; its canonical-JSON SHA-256 is the marker above. The driver verifies that
marker against the entire config, verifies committed source and the protocol,
and checks data manifests on every start/job. No test data existed when written.

## Scientific question and primary endpoints

The question is whether validation-selected reuse improves the measured
accuracy-cost frontier beyond the strongest eligible non-reuse alternative.
G2 remains closed: 05b did not establish a maze spatial/latent-reuse advantage
over answer-only. G3 supports an accuracy benefit for output reuse at some K;
this does not establish a latency advantage. Preserve collapsed carry, failed
arms, negative lesions and all failed jobs. No positive result is assumed.

The primary outcome is mean post-edit exact-answer accuracy over 32 edits,
with each base root weighted equally. For maze12 this is exact shortest-route
or correct unreachable success. For circuits, ordinary32 and the separately
reserved depth48 suite are both co-primary, using exact full-circuit correctness
over non-input nodes. Maze16, maze20 transfer, circuit64 and depth96, 128-edit
horizon effects, equal-compute controls, mechanisms and all other contrasts are
prespecified secondary/exploratory analyses. Never pool families or count nodes,
edits, K values or timing repetitions as independent experimental units.

The practical target is at most one percentage point lower accuracy and at
least 25% lower amortized measured inference cost. Report estimates and intervals
whether the target passes or fails. A selected-point statement of noninferiority
uses a lower 95% bound above -0.01; a measured saving requires its cost-ratio upper
bound below 1, and the full 25% target requires that upper bound at or below .75.
The cross-suite headline requires all three primary suites to satisfy the claim
(intersection-union decision); pointwise K curves and secondary comparisons are
exploratory and cannot be searched for a replacement headline.

## Validation-only selection, fixed now

The surviving named reuse candidate is **answer-only** in both families. No
second latent-reuse candidate passed G2. Existing validation data select
**maze answer-only K8 versus stream-trained restart K8**, and
**circuit answer-only K1 versus stream-trained restart K2**. The circuit point
is applied unchanged to both ordinary32 and depth48. Transfer/scale analyses use
these same family budgets and report complete K curves; there is no test-based
operating-point or checkpoint selection. Restart at every K remains published.

The executable rule in `eval/statistics.py` selects the highest-accuracy restart
point, ties by lower measured cost then K, then the least-cost answer-only point
within .01 accuracy; if none qualifies, retain the highest-accuracy answer-only
point and explicitly mark noninferiority infeasible. The measured existing maze
one-edit restart maximum is .999837 versus the selected stream restart's 1.0,
so it does not displace the comparator. Classical exact solvers are reference
rows, not selectable learned baselines. No low-quality operating point is chosen
to manufacture a cost saving. The eligible development K are maze1/2/4/8 and
ordinary-circuit1/2; K16 and ordinary-circuit4/8 had no corresponding saved
ordinary-validation stream accuracy and cannot enter this selection.

All values, root/seed variances, measured validation costs and source hashes are
in `reports/foundation/prompt06-development-selection.json`, SHA-256
`b2e4b9ceabb8be2b7666fc8f2d392ecf41dc172521462a5b3ce779d56ad3fb55`.
The development cost ratio is 0.908238
for mazes and 0.765472
for ordinary circuits; neither demonstrates the full 25% saving. The selected
circuit K1 versus K2 loses about 10.8 percentage points on existing depth48
validation, which remains explicit negative evidence. This is not repaired by
choosing a different depth-test point. Development timing is batch one on four
validation roots, three adaptation seeds and two warmed repetitions. Some CPU
preparation/checking overlapped that preliminary timing; final timing follows the
separate fixed measurement design below.

New-scale learning-rate selection is a frozen deterministic procedure, not a
claim that future checkpoint hashes already exist. Each policy/size uses both
grid points and all five seeds, choosing one common grid by mean four-edit
validation exact accuracy over roots, seeds and training K, ties to lower grid.
Both candidate scores and all source-job names are sealed in `selection-SIZE`.
Every future checkpoint is restricted to a named training job in the bound
config. Before a test consumer starts, the driver verifies its dependency seal,
the selected source and its checkpoint hash. Each record carries that hash;
the checker rejects a checkpoint outside the sealed selection. Test generation
does not grant permission to change these rules after inspecting test data.

## Complete scientific matrix and execution graph

Seeds are **29,43,71,101,137**; K is **1,2,4,8,16**. No fallback is applied.
There are 20 independently initialized static jobs: five each at maze12,
maze16, circuit32 and circuit64, 6,000 optimizer updates, AdamW, LR .001 dropping
to .0003 after 75%, zero weight decay, clip1, inner_cycles2. All jobs must pass
the original static gate: complete training, nondecreasing ordinary-validation
accuracy across K, and at least .85 at K16. A failed seed halts the affected
matrix with its evidence intact; it is not dropped or replaced.

Mazes use width64, two heads, batch64; circuit width128/four heads/batch128 is
the existing static_circuit_v12 recipe, not a new width128 maze arm. Every K
executes three shared block calls and six transformer-layer executions. These
are explicitly adapted TRM-inspired solvers, not exact TRM or RSM reproductions.
Maze16 static microbatches are16 and matched-control microbatches32; loss is
weighted to the original batch64 objective, one optimizer step per full batch.
Float32 reduction ordering can differ from full-batch arithmetic; exact resume
uses the same fixed microbatch schedule. Circuit data are transferred per step
instead of permanently preloading every training batch; this operational
adaptation is included in the measured profiles, without changing the objective.

The 220 joint stream-adapter jobs are: seven maze arms (restart, carry,
spatial_gate, global_gate, gru_adapter, residual_adapter, answer_only), four
circuit arms (restart, carry, spatial_gate, answer_only), two trained sizes per
family, two grid points, five seeds. Each starts from its own same-seed static
backbone and takes 1,024 updates on four-edit/five-frame streams, batch64,
detached previous-version states and differentiable current rollouts. Maze
training K1/2/4/8; circuit K1/2. Backbone LR [.0001,.0003], adapter LR
[.0003,.001], clip1, no weight decay. No online oracle enters any policy.

The 40 matched one-edit controls cover restart/carry/spatial/answer-only at both
trained maze sizes and all five seeds, using the original pilot-selected grid
and four independent edit views. Their common update count is
**2576**, chosen from the deterministic schedules
to keep per-seed example-forward-call mismatch within 5% of stream training;
exact deviations are in the config. Equal-data and equal-compute analyses are
separate. All matched and stream arms remain visible, regardless of outcome.

At every trained size, all selected principal policies, all five seeds and all K
have 32- and 128-edit streams. Maze auxiliary controls (local radii1/2/3,
validation-calibrated random reset, noise .01 and distribution-preserving
shuffled gate) have 32-edit streams, all five seeds/K, on the spatial backbone.
Evaluation-only suites maze20, depth48 and depth96 have 32 edits. Maze12 and
maze16 each transfer to the same reserved maze20 roots, with results separated
by source. Circuit32 transfers to48 and64 to96. Matched maze controls have32
edits on shifts and32/128 on trained sizes. Answer-content uniform and
shuffled-node stream lesions use the first three seeds and32 edits; they are
explicit diagnostic policies, not extra deployment candidates.

The complete driver graph has **1314 jobs**. Every job's source,
seed, grid, size, batch, K list, horizon, dependencies, handler, memory estimate
and cumulative cap is enumerated in the bound config. The class table follows.

| Job class | Jobs | Median hours | Maximum hours |
|---|---:|---:|---:|
| evaluation/dynamics | 24 | 1.662240 | 2.702167 |
| evaluation/latency | 37 | 4.261452 | 7.404513 |
| evaluation/mechanism | 24 | 15.135347 | 25.792255 |
| evaluation/stream | 858 | 42.158024 | 59.845232 |
| evaluation/throughput | 37 | 1.290326 | 1.769045 |
| reference/stream | 12 | 0.000000 | 0.000000 |
| report/stream | 34 | 0.000000 | 0.000000 |
| research_training/one_edit | 40 | 16.883465 | 18.449208 |
| research_training/static | 20 | 17.962677 | 18.903772 |
| research_training/stream | 220 | 82.352249 | 86.502688 |
| selection/stream | 8 | 0.000000 | 0.000000 |

## Test identities, generation and split isolation

| Suite | Train | Validation | Test | Challenge identities | Test identity SHA-256 | Challenge identity SHA-256 |
|---|---:|---:|---:|---:|---|---|
| circuit32 | 2048 | 256 | 256 | 0 | 2e7cc666d890088c6c5f8ad9997c58e80369549f7f2947953530a7929a5896b1 | not applicable |
| circuit48 | 0 | 256 | 256 | 0 | dae659fda1aa266e9bef00d24625fbed5329b2e4bdc1018ec05d5854dce0950a | not applicable |
| circuit64 | 2048 | 256 | 256 | 0 | c85d49c09cc30ad9a62a07eb1115af786466cdf40307c68cade50930de17a580 | not applicable |
| circuit96 | 0 | 256 | 256 | 0 | e4ac902af80bd9a87b97bf5437e5f541e29f905989494eb4506e709fb606b7eb | not applicable |
| maze12 | 1024 | 256 | 256 | 128 | 5ffe21328c2d67e70651c7e676235f19d07ea702691ea6c0b73a6aa026944bfe | 86e60d857fd3ae9f6d9701e2677918ff299b54e9d1b49dfe4e01bd86fd51c7ab |
| maze16 | 1024 | 256 | 256 | 128 | 6ff1411b858f35de6003a5adc8f0e92a4b26c60968082b84d5237ca171f356db | c5c337271c2b0e62c729adbfbe47d484c3e6d0f0115d573f6c266c269670c843 |
| maze20 | 0 | 256 | 256 | 128 | bd0156ad45501264b09c25abb1bd3f4b6a6254676b53fbd6c9126d9f99279bb5 | abc9be46414a6d2591203c99320b222750a19cad2ae2371d91b2ef44114f106e |

There are **1,792 ordinary test roots plus384 reserved challenge identities**,
2,176 distinct test identities total. Identity strings and hashes are reserved
without generating root payloads. Hashes are SHA-256 of sorted-key canonical JSON
using the implementation's `digest_json`; full lists and all train/validation
identity hashes are in the config. Data seed62017 is hashed with each identity
through `child_seed`. Stream seed64019 produces the deterministic uniform
observable edit sequence, with the32-edit prefix shared with128 when applicable.
Mazes are randomized spanning-tree mazes; every fourth base has a vertical
middle cut to retain unreachable cases. Circuit ordinary/deep generators and
fixed root-specific presentation permutations are as implemented. Base roots
are split before descendants; state resets at every root/episode boundary.

Maze challenges reserve128 roots per size:64 constructed disconnected
addition bases and64 ordinary maze bases. The fixed sampler enumerates candidate
edges, requests32 matched low/high pairs of each edit type, uses seed64023 and
impact thresholds <=.1 and >=.4. These are evaluator-conditioned diagnostics,
not prevalence estimates. At most one pair is accepted per root. All rejections,
unmet quotas and actual branch counts are saved; no root is replaced and no
threshold changes. Each maze mechanism suite has256 ordinary branches plus
the realized matched branches (at most128); each circuit suite has256 ordinary
branches, not384. Input/label/evaluator metadata remain separately typed.

`scripts/generate_confirmatory_test.py` first verifies the committed protocol
and matrix. It creates a durable suite claim before generation, writes results
atomically, and binds file SHA-256, counts, identities and challenge audits into
`data/test-manifest.json`. Subsequent invocations only reuse verified complete
files. A claimed but incomplete suite stops for recovery inspection; it cannot
silently regenerate test roots. Duplicate canonical maze problems or circuit
topologies across independent roots/splits stop generation without resampling.
The driver refuses a changed development file or incomplete/unbound test manifest.

## Evaluation, timing, statistics and reproducible reports

The four record modes are fixed_budget_stream, frozen_state_intervention,
reference_solver and privileged_diagnostic. Each fixed-K policy carries its own
K-step state and own prediction at every edit, including the initial solve.
No hidden Kmax state, oracle stopping, future edit or evaluator stratum enters
deployed inference. The checker rescores every raw action and verifies full
root/frame/K/repetition coverage, checkpoint and data hashes, state/source budget
and actual block-call count. Frozen interventions cannot enter deployment tables.

Metrics include exact route/unreachable/full-circuit success, all-node optimal
policy correctness, valid-action/node accuracy, whole-stream success (including
the initial solve), and per-frame/cumulative errors. Incorrect, invalid-policy,
no-path and trivial start=goal cases retain their original denominators. Unreachable
success is additionally reported with its explicit unreachable-case denominator.
An implementation failure is recorded as a failed job, never imputed as a valid
prediction or silently excluded from a completed matrix.

Latency uses one selected seed29 checkpoint per principal policy and size,
8 test roots,32 edits,3 warmed repetitions, all five K. Size20 uses the
maze12-trained checkpoint to avoid duplicate policy/size timing. Batch64
throughput uses the same checkpoints, a64-root prefix and3 repetitions. Auxiliary
cost is assigned the spatial cost under the measurement rule; no independent
auxiliary latency claim is made. Main accuracy jobs also preserve batch64 timing
for full curves, separately labeled from the dedicated batch-one measurements.
All policy timings use the local RTX5070, float32, TF32 off, synchronized stage
boundaries. Encoder, observed edit/validation, fresh state, adapter, recursive
core, decoder, state copies and input/output transfer are recorded. Kernel
warm-up uses discarded state and is excluded from reported latency; the original
solve is timed separately and included in amortized cost:
**(initial solve + sum of all edit costs)/(T+1)**. Offline scoring is excluded
from policy cost but included in occupied GPU-job wall time. Dense MACs and
retained-state bytes are accounting estimates, never a substitute for wall time.

D* Lite and the event-driven circuit evaluator appear as separately labeled
CPU-on-the-5070-host reference rows in every ordinary cost table. D* Lite uses
zero heuristic and exhausts the queue to output an all-node policy, an adaptation
of the classical incremental planner; it is not a claimed exact upstream code
reproduction. Both reference implementations include initial solve, edit scans,
repair, decoding and copies. Their combined algorithm time is recorded in core;
neural-only stages are zero/not applicable. Report this timing attribution and
hardware distinction explicitly. Credit Koenig and Likhachev, D* Lite (AAAI2002),
as cited in PROPOSAL.md; no third-party implementation was imported here.

Statistical code aggregates complete episodes within each base root and seed.
Paired-root bootstrap averages over all five seeds before resampling256 roots,
preserving every method's full sequence. It uses10,000 draws and seed64037.
Report this conditional root interval and, separately, a crossed interval that
resamples seeds and uses the same root draw for every sampled seed. Show each
seed, seed SD, root SD and counts. Cost ratios use the ratio of paired mean
episode costs, not the mean of frame ratios. Timing repetitions are averaged
within root/seed before bootstrap. Dedicated latency uncertainty is conditional
on its single checkpoint and8 roots; it is not promoted to five-seed timing
evidence. Complete pointwise curves stay visible if every selected target fails.

Development-variance power uses SE=sqrt(root_SD^2/256 + seed_SD^2/5), a normal
one-sided .025 noninferiority calculation with margin.01. Both zero-loss and
observed-development-delta assumptions are recorded:

| Primary suite | Root SD | Adapter-seed SD | Power at zero loss | Power at observed delta | Observed validation delta |
|---|---:|---:|---:|---:|---:|
| circuit | 0.03463138 | 0.00554894 | 0.859236 | 0.025554 | -0.00996908 |
| circuit_depth48 | 0.12420981 | 0.00788496 | 0.215599 | 0.000000 | -0.10795085 |
| maze | 0.01414848 | 0.00145634 | 1.000000 | 1.000000 | -0.00179036 |

The old three adapter seeds shared a pretrained backbone. Their variance cannot
identify new backbone variation; high nominal power for saturated mazes is not
a guarantee. Ordinary-circuit selection lies almost exactly at the margin, and
depth's chosen cross-K comparison has effectively zero predicted power under
its observed negative delta. Five independent new backbones and256 roots are
retained for honest replication; sample counts are not increased or changed
after seeing tests. This limitation is a result of planning, not a success claim.

Mechanisms use the first three seeds. They fork identical saved prior tensors
and identical spatial-backbone weights into principal initializers plus
shuffled gates, random/local controls, and answer-content lesions; actual
adapter checkpoints are recorded. Global/GRU/residual projections, where present,
are transferred from their named jointly trained checkpoints to this common
backbone, so the interpretation is conditional on that transfer. Impact-mask
reset is privileged=true, outside ordinary tables, and is not an upper bound on
ideal hidden repair. Gate-value distributions are checked after shuffling.
Separate carry-dynamics jobs build their own4-edit carry history at sourceK,
then fork its saved state for extra K1/2/4/8/16 refinement on the same observation.
Changed actions disprove exact action invariance, not necessarily prove useful
correction. Prior tensors are compressed in the durable records and verified
against their hashes. Root-weighted per-stratum accuracy and state distances
are diagnostics; output-impact masks are not latent invalidation truth.

`eval/report.py` regenerates checked JSON and Markdown tables, paired intervals,
stage costs, errors and per-seed estimates from saved actions; its ReportLab SVG
accuracy-cost curves use the existing local bundled dependency path recorded
in each report job. Report jobs are driver dependencies and run unattended.
Mechanism reports retain separate record modes and root-weighted strata.
The exact05b regeneration passed82,944 episodes/87 directories before freeze,
matching saved table values and report content; original negative results and
original stream-results byte hash are preserved.

## Projection, caps, recovery and startup exception

Median projected future work: **181.705780 GPU-hours**.
Maximum-timing future-work sensitivity: **221.368881 hours**.
One 15% interruption allowance: **27.255867 hours** on the median projection.
Already consumed: **0.400364 hours**.
The launch total is **209.362012 hours**; the maximum sensitivity including its own single 15% allowance and consumed time is **254.974577 hours**.

All per-job medians are below four hours. `reports/PILOT_RUN_PLAN.md` records
all assumptions, class counts, maximum sensitivity and measured setup. No
scientific fallback or purchased compute is used. The user prefers investigating
purchased compute before future cuts, but no purchase is authorized here.

GPU class peak estimates:9.8GiB training;4GiB evaluation/timing/mechanisms.
The allocator cap is10GiB on12GiB. Per-job time includes all active attempts,
setup and interrupted work, with a work-unit safety reserve before the four-hour
limit. The shared500-hour ledger includes all prompt06 capacity and development
timing. Reservations precede work; aborted actuals use recoverable durable
timestamps, explicitly a lower bound after a hard cut. No external charges or
services. A10GiB free-disk reserve stops further jobs before storage exhaustion;
raw predictions, checkpoints and quarantined failed runs are not deleted.
Completed evaluation logs use verified lossless gzip storage; this removes only
the redundant uncompressed copy after checking the decoded byte hash. Resume
and checker logic support both representations, including interruption between
compression and sealing. JSON/compression occupancy is measured and included in
the projection. The record-storage sensitivity is
50.06GiB plus checkpoints;
the generation/launch disk audit must retain the10GiB free reserve.

One driver holds the lock and owns sequencing. It reconciles both reservation
stores, quarantines unsealed output, restores exact model/optimizer/RNG/schedule
state, truncates any partial JSONL, verifies sealed jobs and runs the next
dependency. Checkpoints occur at500 steps or5 minutes; JSONL is fsynced each
work unit. Graceful shutdown saves at an atomic optimizer/work-unit boundary;
hard-cut recovery remains active. The startup wrapper retries nonzero exits;
deterministic validation/cap failures leave a STOP record for inspection.

The user explicitly accepts **S4U Windows session0 startup on this machine with
automatic sign-in** as the unattended-startup acceptance condition, replacing
the literal no-Windows-logon requirement. Original evidence keeps
**strict_no_logon_passed=false**. The verified run resumed at step20, began49.3s
after boot and completed62.0s after boot; the independent uninterrupted run
matched weights, optimizer, RNG, counters and every step-log byte exactly.
Evidence: `runs/confirmatory_restart_boot_v2/` and
`reports/foundation/confirmatory-reboot-v2.zip`. No sign-in settings change and
no additional reboot test is performed. Existing power settings/evidence remain.
Before launch, Paper1-Confirmatory is repointed from its synthetic job to this
config/protocol and the CUDA virtual-environment interpreter. Its exact command
and XML export are saved under reports/foundation; launch uses that task, not
interactive per-job sequencing.

Tier3 is explicitly excluded: no24x24+ trained mazes,128-gate+ trained circuits,
new maze width128, sparse-neighbor execution or third task family. Credit prior
state reuse in HRM-Agent, StreamDEQ, TRM, GRUs, learned warm starts, NCA,
incremental GNNs and classical planning as documented in PROPOSAL.md. Soft
retention still executes dense work; no sparse execution or learned halting
claim is made. Any post-freeze scientific change is a new exploratory version.
