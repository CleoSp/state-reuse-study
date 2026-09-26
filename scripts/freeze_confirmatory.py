"""Write the complete prospective protocol; never generate test data here."""
from collections import Counter
from pathlib import Path
import json

from state_repair.execution.matrix import build_matrix, project
from state_repair.execution.durable import atomic_json, atomic_write, digest_json, read_json
from state_repair.provenance import file_hash

root=Path('runs/confirmatory_v1')
if list((root/'data').glob('*-test.json')):
    raise ValueError('cannot freeze or revise after any test root exists')
analysis=read_json(Path('reports/foundation/prompt06-development-selection.json'))
matrix=build_matrix()
matrix.update(launch_authorized=True,status='FROZEN',minimum_free_disk_bytes=10*2**30,
    development_hashes={p.as_posix():file_hash(p) for p in (root/'data').glob('*-development.json')},
    operating_points={family:value['operating_point'] for family,value in analysis['families'].items()})
for job in matrix['jobs']:
    if job['device']=='cuda':
        job['required_torch_version']='2.11.0+cu128'
    if job['kind']=='report' and not job.get('mechanism'):
        family='maze' if 'maze' in job['id'] else 'circuit'
        job['operating_point']=matrix['operating_points'][family]
matrix['projection']=project(matrix)
projection=matrix['projection']
if projection['total_hours_with_interruptions']>500 or projection['per_job_projection_exceeds_cap']:
    raise ValueError('projection exceeds the configured resource limits')
source_paths=sorted({*Path('src/state_repair').rglob('*.py'), *Path('tests').glob('*.py'),
    *(Path('scripts')/n for n in ('run_confirmatory.py','generate_confirmatory_test.py','freeze_confirmatory.py','confirmatory_startup.ps1')),
    Path('pyproject.toml'),Path('README.md'),Path('REPRODUCING.md')})
matrix['source_hashes']={p.as_posix():file_hash(p) for p in source_paths}
artifact_paths=[Path('reports/foundation/prompt06-development-selection.json'),Path('reports/foundation/prompt06-preparation-measurement.json'),
    Path('reports/foundation/prompt06-compression-measurement.json'),
    Path('runs/prompt06-validation-regeneration.json'),Path('runs/adapter_pilot_v1/selection.json'),Path('reports/stream_results.json')]
matrix['artifact_hashes']={p.as_posix():file_hash(p) for p in artifact_paths}
matrix['runtime']={'GPU':'NVIDIA GeForce RTX 5070','GPU_memory_GiB':12,'allocator_cap_GiB':10,
    'Python':'3.12.14','PyTorch':'2.11.0+cu128','CUDA_runtime':'12.8','dtype':'float32','TF32':False,
    'CPU_threads':2,'CPU_RAM_bytes':34295808000,'external_charges_USD':0}
config=Path('configs/confirmatory_tier2_frozen.json')
atomic_json(config,matrix)


matrix=read_json(config)
atomic_json(Path('reports/foundation/prompt06-frozen-projection.json'),projection)
classes=['| Job class | Jobs | Median hours | Maximum hours |','|---|---:|---:|---:|']
classes += [f"| {r['class']} | {r['jobs']} | {r['median_hours']:.6f} | {r['maximum_hours']:.6f} |" for r in projection['class_table']]
numbers=f"""Median projected future work: **{projection['base_hours']:.6f} GPU-hours**.
Maximum-timing future-work sensitivity: **{projection['maximum_base_hours']:.6f} hours**.
One 15% interruption allowance: **{projection['interruption_allowance_hours']:.6f} hours** on the median projection.
Already consumed: **{projection['already_consumed_hours']:.6f} hours**.
The launch total is **{projection['total_hours_with_interruptions']:.6f} hours**; the maximum sensitivity including its own single 15% allowance and consumed time is **{projection['maximum_total_hours_with_interruptions']:.6f} hours**.
"""
plan=f"""# Confirmatory measurement projection

The matrix includes five independent backbones at every trained size, five
adapter seeds, all principal arms and both maze matched-compute controls.
The resource projection assumes local computation.

{numbers}
{chr(10).join(classes)}

The exact {len(matrix['jobs'])}-job graph is in `{config.as_posix()}`. Sources,
per-job estimates and measured setup values are in
`reports/foundation/prompt06-frozen-projection.json`.

Steady-state estimates discard the first training repetition and inference
repetition zero, then take the median per K (maximum per K for sensitivity).
Inference pools the measured restart/spatial samples at each batch and size and
includes separately measured offline scoring in GPU-job occupancy, though scoring
is excluded from policy-cost tables. Warm-up work is charged to occupancy and
excluded from policy latency. Every training family/size/recipe has a direct
capacity profile. For adapter arms, the larger of one and their measured pilot
steady-state ratio to spatial at the same K accounts for adapter-specific work;
using that relative overhead at larger sizes and for one-edit is an explicit
extrapolation, not a new measured result.

Setup uses explicit measured build time where present; for older profiles it
uses measured job wall time minus every timed step/frame, a conservative setup
and I/O residual. Full-development CPU cache construction is a separately named
component, measured for all ten training classes and charged in addition because
the small capacity fixtures did not prepare all training roots. This can double
count a small CPU constructor component; there is no blanket execution multiplier
and no flat 60-second job charge. These measurements do not guarantee runtime:
disk serialization, WDDM workload and CPU scoring variability remain limitations.
JSON serialization and final lossless compression are separately measured on
saved validation records and extrapolated by a conservative per-record byte
estimate; that named I/O component is included in both totals. Predicted raw
record volume is {projection['estimated_raw_record_bytes']/2**30:.2f} GiB;
a conservative35% storage ratio gives {projection['record_storage_sensitivity_bytes']/2**30:.2f} GiB,
before checkpoints. The measured example compressed to14.35% of its raw bytes.
Completed evaluation logs are gzip-compressed and their decoded bytes are checked
against the checkpoint hash before the redundant plain copy is removed. Active
logs remain plain JSONL; recovery expands a compressed unsealed attempt if needed.

Latency has 37 policy/size conditions, five K values bundled in each job, eight
roots, 32 edits and three warmed repetitions. The 20x20 timing checkpoint comes
from the first (12x12-trained) seed-29 selection so size20 is not timed twice.
Batch64 throughput uses those same checkpoints and a 64-root prefix, three
repetitions. Main 32-edit accuracy still evaluates both trained-maze sources on
size20. Only principal arms on trained sizes have 128-edit streams; matched
controls also retain that comparison at trained maze sizes. Auxiliary controls
and shift suites have 32 edits. Mechanism and carry dynamics use three seeds.
Circuit intervention branch count is 256; maze design is 256 ordinary branches
plus up to 128 matched challenge branches (32 pairs per edit type). The exact
realized counts and unmet quotas are emitted by the one-time generator and used
by the handler; incomplete challenges are retained without replacement. The
pre-generation projection reserves those explicit per-suite maxima rather than
assigning 384 branches to circuits. A realized-count projection is saved after
generation without changing the frozen scientific matrix.

Declared peaks before real GPU classes: static/stream/one-edit training **9.8 GiB**;
all evaluation/latency/throughput/mechanism/dynamics **4 GiB**; validation timing
already ran with a **2 GiB estimate**. These are estimates, not separate allocator
caps. All CUDA jobs enforce the shared **10 GiB** allocator cap. Capacity's maximum
observed allocation was 7,591,468,032 bytes; the single-job cap remains four
cumulative active hours. One 15% allowance is justified by the verified exact
resume path (five-minute/500-step checkpoints), not by assuming interruptions are
free. Both ledgers reserve before launch and reconcile actual/aborted time.

"""
planpath=Path('reports/RUN_PLAN.md')
atomic_write(planpath,lambda h:h.write(plan.encode()))
identity_table=['| Suite | Train | Validation | Test | Challenge identities | Test identity SHA-256 | Challenge identity SHA-256 |',
    '|---|---:|---:|---:|---:|---|---|']
for name,spec in matrix['suites'].items():
    ids=spec['identities']; hashes=spec['identity_hashes']
    identity_table.append(f"| {name} | {len(ids['train'])} | {len(ids['val'])} | {len(ids['test'])} | {len(ids.get('challenge',[]))} | {hashes['test']} | {hashes.get('challenge','not applicable')} |")
power=['| Primary suite | Root SD | Adapter-seed SD | Power at zero loss | Power at observed delta | Observed validation delta |',
       '|---|---:|---:|---:|---:|---:|']
for name,value in analysis['sample_size'].items():
    c=value['development_contrast']
    power.append(f"| {name} | {c['root_sd']:.8f} | {c['training_seed_sd']:.8f} | {value['at_zero_accuracy_loss']['approximate_power']:.6f} | {value['at_observed_development_delta']['approximate_power']:.6f} | {c['delta']:.8f} |")
protocol=f"""# Frozen confirmatory protocol: tier 2, version 1

<!-- matrix-sha256: {digest_json(matrix)} -->

This is a prospective protocol, not a test result. This file and
`{config.as_posix()}` must be committed unchanged before the first test root is
generated. The exact ordered job definitions, all checkpoint-producing job names,
selection dependencies, identity lists and implementation hashes are in that
config; its canonical-JSON SHA-256 is the marker above. The driver verifies that
marker against the entire config, verifies committed source and the protocol,
and checks data manifests on every start/job. No test data existed when written.

## Scientific question and primary endpoints

The question is whether validation-selected reuse improves the measured
accuracy-cost frontier beyond the strongest eligible non-reuse alternative.
G2 remains closed: stream training did not establish a maze spatial/latent-reuse advantage
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
`{file_hash('reports/foundation/prompt06-development-selection.json')}`.
The development cost ratio is {analysis['families']['maze']['operating_point']['validation_cost_ratio']:.6f}
for mazes and {analysis['families']['circuit']['operating_point']['validation_cost_ratio']:.6f}
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
**{matrix['matched_control']['steps']}**, chosen from the deterministic schedules
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

The complete driver graph has **{len(matrix['jobs'])} jobs**. Every job's source,
seed, grid, size, batch, K list, horizon, dependencies, handler, memory estimate
and cumulative cap is enumerated in the bound config. The class table follows.

{chr(10).join(classes)}

## Test identities, generation and split isolation

{chr(10).join(identity_table)}

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
as cited in sources/READING_LIST.md; no third-party implementation was imported here.

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

{chr(10).join(power)}

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
The stream-training regeneration passed 82,944 episodes/87 directories before freeze,
matching saved table values and report content; original negative results and
original stream-results byte hash are preserved.

## Projection, caps, recovery and startup exception

{numbers}
All per-job medians are below four hours. `reports/RUN_PLAN.md` records
all assumptions, class counts, maximum sensitivity and measured setup.
The resource projection uses local computation.

GPU class peak estimates:9.8GiB training;4GiB evaluation/timing/mechanisms.
The allocator cap is10GiB on12GiB. Per-job time includes all active attempts,
setup and interrupted work, with a work-unit safety reserve before the four-hour
limit. The shared500-hour ledger includes all confirmatory capacity and development
timing. Reservations precede work; aborted actuals use recoverable durable
timestamps, explicitly a lower bound after a hard cut. No external charges or
services. A10GiB free-disk reserve stops further jobs before storage exhaustion;
raw predictions, checkpoints and quarantined failed runs are not deleted.
Completed evaluation logs use verified lossless gzip storage; this removes only
the redundant uncompressed copy after checking the decoded byte hash. Resume
and checker logic support both representations, including interruption between
compression and sealing. JSON/compression occupancy is measured and included in
the projection. The record-storage sensitivity is
{projection['record_storage_sensitivity_bytes']/2**30:.2f}GiB plus checkpoints;
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
incremental GNNs and classical planning as documented in sources/READING_LIST.md. Soft
retention still executes dense work; no sparse execution or learned halting
claim is made. Any post-freeze scientific change is a new exploratory version.
"""
atomic_write(Path('reports/FROZEN_PROTOCOL.md'),lambda h:h.write(protocol.encode()))
print(json.dumps({'matrix_sha256':digest_json(matrix),'jobs':len(matrix['jobs']),
    'median_total':projection['total_hours_with_interruptions'],'maximum_total':projection['maximum_total_hours_with_interruptions']}),flush=True)
