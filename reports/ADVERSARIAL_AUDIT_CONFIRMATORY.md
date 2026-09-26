# Independent adversarial audit — completed confirmatory matrix (prompt 07)

Audit date: 2026-09-21. Scope: the sealed 1,314-job tier-2 matrix under `runs/confirmatory_v1`, frozen by `reports/FROZEN_PROTOCOL.md` (matrix SHA-256 `7c71eb25…0eeba`, protocol commit `cdaf843`). The auditor worked read-only on all records, on CPU, with no training, no GPU job and no method change. The earlier static-milestone audit in `reports/ADVERSARIAL_AUDIT.md` is untouched and still applies to what it covered.

Audit code: `scripts/audit_confirmatory.py` (own bootstrap, own maze route follower, own circuit evaluator; it does not call `state_repair.eval.report` or `state_repair.eval.statistics`). Console output: `runs/prompt07-audit-confirmatory.txt`. Exact command from the repository root:

```powershell
$env:PYTHONPATH='src'; .venv/Scripts/python.exe scripts/audit_confirmatory.py > runs/prompt07-audit-confirmatory.txt
```

Measured wall time: **185.0 s** on CPU (3,168,000 principal test records streamed for the three primary suites, plus tuning, dynamics, mechanism, latency and reference records). Exit code 0. The script's own findings list is empty.

## Disposition

**No blocking finding.** No major finding that changes a frozen decision. The three primary operating points, all 75 principal curve cells of the three primary suites, the validation-only grid selections, the test-generation ordering and the ledger reproduce exactly from raw sealed records with independent code. Two major items concern how the results must be described, not whether they are correct: (M1) the protocol names the amortized cost formula but does not single out one timing mode for the 25% target, and the only five-seed paired cost measurement is batch-64 throughput timing; (M2) the mechanism tables' `answer_only`, `global_gate`, `gru_adapter` and `residual_adapter` rows run transferred projections on the spatial-gate backbone and cannot be read as deployed-policy results. The frozen headline decision is unaffected under either timing mode.

## Recomputed primary decisions (frozen protocol rules)

Own bootstrap: five seeds averaged within each root, 256 roots resampled, 10,000 draws, `numpy.random.default_rng(64037)`, quantiles .025/.975; cost ratio is the ratio of paired mean amortized episode costs. All values match `report-*/results.json` `operating_points` to 1e-12 (identical draws) and the intervals are stable under a different bootstrap seed (`alt-seed CI` in the log).

| Primary suite | Frozen point | Accuracy delta (pp) | Paired-root 95% CI | Crossed CI | Cost ratio (batch-64 paired, 5 seeds) | Ratio CI | Noninferior (lo > −1 pp) | Saving (hi < 1) | 25% target (hi ≤ .75) |
|---|---|---:|---|---|---:|---|---|---|---|
| maze12 | answer-only K8 vs restart K8 | +0.068 | [−0.066, +0.227] | [−0.103, +0.386] | 1.0037 | [1.0029, 1.0045] | yes | **no** | **no** |
| circuit32 | answer-only K1 vs restart K2 | −0.645 | [−0.945, −0.386] | [−1.008, −0.356] | 0.9931 | [0.9917, 0.9945] | yes (root CI); crossed lower bound −1.008 crosses the margin | yes (0.7%) | **no** |
| circuit48 (depth shift) | answer-only K1 vs restart K2 | −9.741 | [−10.967, −8.542] | [−14.949, −5.154] | 0.9428 | [0.9422, 0.9434] | **no** | yes (5.7%) | **no** |

**Intersection-union headline: fails.** No primary suite reaches the 25% amortized-cost target, and the depth-shift suite fails noninferiority by about ten points, as the protocol's own development-delta power table predicted. Per-seed deltas are all negative on both circuit suites (circuit32: −0.37 to −0.82 pp; circuit48: −4.7 to −18.6 pp).

Batch-one latency operating points (dedicated latency jobs, one seed-29 checkpoint per policy, 8 test roots, 3 warmed repetitions; the protocol says this uncertainty "is not promoted to five-seed timing evidence"):

| Suite | Cost ratio | 95% CI (8 roots) |
|---|---:|---|
| maze12 | 1.0059 | [1.0052, 1.0067] |
| circuit32 | 0.7723 | [0.7685, 0.7774] |
| circuit48 | 0.7753 | [0.7691, 0.7797] |

Under batch-one timing the circuit K1-versus-K2 saving is about 23%, still short of the 25% target's upper-bound rule (0.7774 > 0.75), and the maze point saves nothing under either mode because both arms run K8.

Amortized cost formula found in `src/state_repair/eval/metrics.py:82` (`episodes`): `amortized_ms = sum(total ms over frames 0..32)/(edits+1)`, i.e. the initial solve is included, exactly the protocol's `(initial solve + sum of all edit costs)/(T+1)`. `src/state_repair/eval/timing.py:41` (`amortized_cost`) raises if frame 0 is missing.

## Blocking findings

None.

## Major findings

**M1 — The primary cost measure's timing mode is not singled out in the protocol; the paired five-seed evidence is batch-64 timing.** `reports/FROZEN_PROTOCOL.md` lines 212–219 state: "Latency uses one selected seed29 checkpoint per principal policy and size, 8 test roots, 32 edits, 3 warmed repetitions, all five K … Main accuracy jobs also preserve batch64 timing for full curves, separately labeled from the dedicated batch-one measurements," and line 246 "Dedicated latency uncertainty is conditional on its single checkpoint and 8 roots; it is not promoted to five-seed timing evidence." The `operating_points` in `report-<suite>-h32/results.json` are computed from the accuracy jobs' records, whose every row is labeled `measurement=batched_throughput, batch_size=64` (verified on all 3,168,000 rows). The per-example batch-64 amortized cost is roughly fifty times smaller than batch-one latency and is dominated by fixed per-frame stages (for circuit32 restart K1, `copy` 1.06 ms and `transfer_in` 0.52 ms of 1.86 ms total; the recursive core is 0.17 ms), which is why one fewer cycle saves 0.7% at batch 64 and 23% at batch one. Any manuscript statement of the cost target must name the timing mode and report both; neither mode reaches the 25% target and the decision is the same under both. This is a reporting requirement, not a defect in the records.

**M2 — Mechanism-table rows for transferred adapters are conditional on the spatial-gate backbone.** `src/state_repair/eval/mechanism.py:45-73`: the `MechanismJob` loads the backbone and spatial weights from the sealed spatial-gate selection (`adapter-<size>-spatial_gate-g1-s29` for seed 29), then loads only the `adapter.*` parameters of `answer_only`, `global_gate`, `gru_adapter` and `residual_adapter` from their own selected checkpoints (`adapter_sources` in every record; verified for all eight `mechanism-*-s29` jobs). The frozen prior is that spatial backbone's own restart solve at `source_K=8`. So the mechanism `answer_only` row (for example maze12 K8 low stratum 0.295) is a projection trained with a different backbone applied to a foreign backbone's state; it is not the deployed answer-only policy (deployed maze12 K8: 0.999). The protocol says this ("the interpretation is conditional on that transfer"), and `deployable_policy=False` is set on every intervention record. The paper must not place mechanism rows for these four adapters beside deployed-stream numbers; restart, carry, spatial_gate and the shuffled/random/local/noisy controls use that backbone's own weights and are interpretable within the mechanism table; `answer_uniform` and `answer_shuffled_nodes` also go through the transferred answer-only projection (`mechanism.py:96-103`) and carry the same caveat.

## Minor findings

**m1 — K16 collapse on circuit32 is a single-seed initial-solve failure, not stream accumulation.** For seed 137 the stream-trained `answer_only` and `spatial_gate` checkpoints (both jointly trained backbones; training K ∈ {1,2}) score 0.830 and 0.847 mean post-edit exact accuracy at K16 versus 1.000 at K8; the same seed's restart checkpoint is 1.000 at K16. The initial solve (frame 0) is already wrong on 50 of 256 roots at K16 (206 correct) versus 0 of 256 at K8, and the per-frame curve is flat (frames 1–8: 0.824–0.859; frame 32: 0.840). The other four seeds are ≥ 0.9993. The five-seed mean of 0.966 at K16 is therefore dominated by one backbone that does not extrapolate to unseen K; the crossed interval, not the root interval, is the honest uncertainty for that cell. On maze16, `spatial_gate` seed 71 at K16 (0.564) shows both an initial-solve degradation (224/256 correct at K16 versus 255/256 at K8) and a within-stream decline (frame 1: 0.633, frame 32: 0.547); its K8 curve also declines (0.988 → 0.883). Exploratory; no protocol rule is affected.

**m2 — Carried-state "exact fixed point" is not supported on test roots; the movement is small in ordinary suites and large under depth shift.** From the 24 `dynamics-*` jobs (own aggregation): forking a deployed carry policy's own frame-4 state (built at K8) and refining on the same observation changes at least one action in 9.6% (K1) to 39.2% (K16) of low-stratum maze12 rows, with mean changed-action fraction 0.10% to 0.44%; on circuit48 low stratum, 59% to 72% of rows change with mean fraction 2.8% to 5.3%, and exact correctness rises from 0.582 (K1) to 0.669 (K8). So carried states do keep moving with extra cycles; the earlier pilot's "identical scores at K1 and K8" was never an exact latent fixed point, and the confirmatory dynamics records say so again. Changed actions do not by themselves prove useful correction (protocol wording), but the accuracy gains under depth shift are measured.

**m3 — Restart's own per-frame curve declines at K1 and is the drift reference.** maze12-on-maze12 restart K1: mean of frames 1–8 = 0.391, frames 25–32 = 0.357; at K8 0.9994 → 0.9978. Every reuse policy's per-frame curve is available in the same records (`per_frame` in the audit script), so decay statements can be stated relative to restart at the same frame. answer_only K1 declines more (0.804 → 0.731) than carry (0.863 → 0.842) or spatial_gate (0.884 → 0.838) over the same frames.

**m4 — The test and tuning streams share `stream_seed=64019`.** Edit sequences are `child_seed(64019, root_id)`; since test root identities differ from development identities the sequences differ, and nothing learned depends on the seed. No leakage, but a reviewer will ask; state it.

**m5 — Mechanism `answer_shuffled_nodes` shuffles all positions.** `src/state_repair/eval/mechanism.py:101` permutes `logits[:, randperm(N)]` without restricting to valid nodes; mechanism jobs run batch 1 with no padding so this is equivalent to the deployed lesion (`src/state_repair/eval/lesions.py:21-24`, which permutes within valid nodes). Harmless here; would matter if batched.

**m6 — Reference solvers are Python implementations timed on the host CPU.** Rows are labeled `record_kind=reference_solver`, `device=cpu_on_rtx5070_host`, `measurement=batch_one_latency`, with all non-core stages zero and `timing_scope` recorded (verified on 8,448 rows per reference job). Their amortized cost (maze12 D* Lite 0.249 ms; circuit32 event-driven 0.015 ms) is not comparable stage-by-stage with GPU policy timings and must be described as a separately labeled CPU reference, as the protocol says.

**m7 — Circuit32 noninferiority rests on the root interval.** The crossed interval's lower bound (−1.008 pp) is just past the −1 pp margin. The protocol names the paired-root bound for the selected-point statement, so the frozen decision is "noninferior," but the seed-inclusive uncertainty should be shown beside it.

## Checks performed

| # | Check | How (command or file:line) | Result |
|---|---|---|---|
| 1a | Curve regeneration, three primary suites, 7/4/4 principal policies × 5 K | `scripts/audit_confirmatory.py` section 1; own streaming aggregation over `test-*-h32/steps.jsonl.gz` | 75/75 cells match `results.json` accuracy to 1e-9 and amortized ms to 1e-6; every seed/root/policy/K cell has exactly 33 frames |
| 1b | Operating-point intervals and cost ratios | section 1, own bootstrap | all three match to 1e-12; decisions above |
| 1c | Amortized cost includes initial solve | `src/state_repair/eval/metrics.py:82`, `timing.py:41` | pass |
| 2 | Independent rescoring of raw actions | section 2: own BFS + route follower (`my_route`), own topological evaluator (`my_circuit_eval`); regenerated observations via `frames`; input hash recomputed | 7,128 rows (2,376 per suite; K1 and K8; restart/answer_only/carry) — 0 mismatches in `exact_correct`, `route_correct`, `reason`, `unreachable`, node accuracy, `input_sha256` |
| 2b | Denominators retain failures | section 1 reason counts on maze12 post-edit rows | `shortest_route` 1,016,787; `correct_unreachable` 301,695; `cycle` 53,434; `incorrect_unreachable` 29,132; `illegal_move` 19,160; `wrong_goal_or_nonoptimal` 13,392 — all in the denominator |
| 3 | Output-reuse leakage trace | `src/state_repair/models/adapters.py:325-355` (AnswerOnlyAdapter reads only `prediction.logits`, detached softmax; comment at 352 "never access transition.old_state"); `src/state_repair/models/policy.py:104-106` passes `previous.prediction`, which is `PredictionBatch(prediction.logits.detach().clone())` of this same policy's previous result (`policy.py:136-137`); `policy.py:89-94` rejects any previous result whose `outer_cycles`/`state.budget` differ from this policy's K or whose stream token/adapter key differ; targets never enter (`eval/jobs.py:78-121` builds `ObservationBatch` only via `collate(...)[0]`) | pass; on all 3,168,000 records `state_budget == K` and `source_budget == K` for frames > 0, `None` at frame 0 |
| 4 | Fixed-budget provenance | `eval/jobs.py:87` constructs a new `FixedBudgetPolicy(self.solver, self.adapter, k)` per K per batch; warm-up state discarded by `policy.reset()` at `jobs.py:92`; regression test `tests/test_policy.py:62 test_substituted_high_budget_state_is_rejected` | pass; `checkpoint_job` equals the sealed selection on every record; `block_calls == 3K` everywhere; five distinct checkpoint hashes per suite/policy |
| 5 | Joint-track selection reproduction | section 5, own episode means over `tuning-*` records (4-edit validation streams) | maze12 and circuit32 grids reproduce to 1e-12 with the sealed tie rule; all candidates trained 1,024 steps; both restart grid points trained (maze12 restart g0 0.7931 vs g1 0.7898; circuit32 restart g0 0.9935 vs g1 0.9923; the other arms chose g1) |
| 6 | Test-set discipline | section 6 | protocol SHA-256 `ea3020f7…237a1` identical at `cdaf843`, HEAD and in `test-manifest.json`; matrix marker present; every suite's claim file 17–540 s and test file 34–788 s after the protocol commit; first driver job 984 s after; identity hashes recomputed with `digest_json(sorted-order identity list)` equal the config, the protocol table and the generated files; challenge hashes likewise; `git diff cdaf843..HEAD` on `src/`, the frozen config and the protocol is empty; all selection candidates read `*-development.json` |
| 7 | Carried-state dynamics | section 7 | see m2 |
| 8 | Drift reference | section 8 | see m3 |
| 9 | Mechanism sources and privileged labels | section 9 | all `impact_mask` rows are `record_kind=privileged_diagnostic`, `privileged=true`; no other row is; transferred sources listed (M2) |
| 10 | Timing integrity | `src/state_repair/eval/timing.py:24-40` (synchronize before start, after transfer-in, after decode+argmax; `validate_timing` rejects stage sum > total); `models/policy.py:76-81` (`mark` synchronizes at every stage boundary); warm-up excluded (`jobs.py:89-92`) | pass; stage sum never exceeds total on 3,168,000 records; mean unmeasured gap 0.00076 ms, max 0.0043 ms; latency job has 3,960 rows = 8 roots × 33 frames × 3 reps × 5 K with 1,050 failed predictions retained (no success-only latency) |
| 11 | Ledger completeness | section 11; `sqlite mode=ro` | 1,335 reserve/actual pairs in both stores, identical id sets, `integrity_check` ok; 239.790751 GPU h + 13.680733 CPU-job h; 1,318 empirical + 17 synthetic attempts; two aborted attempts (gru_adapter g0 s101: 372.11 s graceful; local_reset_3 s137: 0.757 s lower bound) both quarantined, both directories present, neither referenced by any report; `complete.json` 1,314 jobs; 1,314 sealed matrix directories |
| 11b | Packaged checker coverage | sum over all `report-*/results.json` `checks` | 766 checker rows, 46,719,642 raw actions rescored by the packaged checker, all `verified=true`, none synthetic |
| 12 | K16 anomaly | section 12 | see m1 |
| 13a | Oracle answers / impact masks / split labels in features | `src/state_repair/data/maze.py:131-149` (features: row, column, start flag, goal flag; relations from passages only); `execution/datasets.py:60-66` collate returns observation `[0]` only | pass by reading |
| 13b | Future frames | `eval/jobs.py:94`: frame f collates `values` with `stream[f-1]` only | pass |
| 13c | Generator seeds / root ids as features | not in `observation()`; ids are bookkeeping tuples | pass by reading |
| 13d | Cross-split reuse | config: no test-split job reads a non-test dataset, no training job reads a test dataset (section 13) | pass |
| 13e | Oracle stopping | `policy.py:114-115` runs exactly `outer_cycles`; no halting | pass |
| 13f | Throughput vs latency labels | `measurement` field: accuracy jobs `batched_throughput`/64, latency jobs `batch_one_latency`/1, references `batch_one_latency` CPU | pass (M1 for wording) |
| 13g | Dependent edits as independent replicates | bootstrap unit is the root with all 32 edits and five seeds kept together (own implementation matches) | pass |
| 13h | One model as many seeds | five distinct static checkpoints (`static-<size>-s<seed>`) feed five distinct adapter checkpoints; five distinct `checkpoint_sha256` per suite/policy | pass |
| 13i | Missing failed jobs | ledger and driver log: 1,316 `job_started`, 1,314 `job_complete`, 1 `job_aborted`, 3 startups | pass |
| 13j | Gradient contract of the stream recipe | tests exist and are in the full suite: `tests/test_stream_step.py:24,53`, `tests/test_adapter_step.py:22,69,84`, `tests/test_adapters.py:91`, `tests/test_model.py:135` (broken-detach negative controls) | present; not re-run by this audit (last full suite 333 passed per STATUS.md) |
| 13k | RNG for shuffling controls | `src/state_repair/execution/jobs.py:18-30` seeds Python, NumPy, Torch and CUDA per job | pass |

## Scope and what was not checked

- Curve and interval regeneration covered the three primary suites and their principal policies. Secondary suites (maze16, maze20 transfer, circuit64, circuit96, 128-edit horizons), auxiliary controls, matched one-edit controls and lesion policies were verified only by the packaged checker (11b), not by this audit's own code.
- Raw-action rescoring sampled 7,128 rows; the packaged checker rescored every row.
- No GPU replay, no recomputation of a checkpoint's forward pass, and no re-run of the pytest suite were performed. Cross-runtime bitwise reproducibility is known to be limited (STATUS.md, prompt 05 CPU spot replay).
- Timing was audited for consistency and labeling, not for absolute accuracy against an external profiler; WDDM and Python overhead in `copy`/`transfer_in` stages are part of the recorded cost and are not separated here.
- Training-time fairness beyond selection (identical batches, sampled-K sequences and step counts across joint arms) was checked only through job definitions (`adapter-*` jobs share `stream_seed`, `edit_seed`, `steps=1024`, grid and source per seed) and the 1,024-step completion of every candidate, not by replaying training logs.
- The reserved maze challenge-pair generation audit and per-stratum mechanism tables were not re-derived; their counts (384/370/360 branches) match the test manifest.
- Statistical power and the choice of primary endpoints are protocol matters and were not re-litigated.

No file under `runs/` was modified. New files: `scripts/audit_confirmatory.py`, `runs/prompt07-audit-confirmatory.txt`, this report. Nothing was committed.
