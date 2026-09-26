"""Regenerate the CPU smoke report from validated local records."""
import json
from pathlib import Path
from statistics import mean

from state_repair.eval.smoke import report_run


if __name__ == "__main__":
    run=Path("runs/smoke")
    checked=report_run(run)
    summary=json.loads((run/"summary.json").read_text())
    provenance=json.loads((run/"provenance.json").read_text())
    command=json.loads(Path("reports/smoke_command.json").read_text())
    steps=[json.loads(line) for line in (run/"steps.jsonl").read_text().splitlines()]
    table=[]
    for row in checked["aggregates"]:
        if row["stage"]=="post":
            table.append(f'| {row["split"]} | {row["K"]} | {round(row["route_correct"]*row["episodes"])}/{row["episodes"]} | {100*row["valid_action_accuracy"]:.2f}% | {row["total_inference_ms"]:.3f} |')
    report=f'''# CPU static learning smoke

**Decision: DEBUG_STATIC_SOLVER; do not scale repair or confirmatory experiments.** One seed and one fixed 200-step run completed; successful overfitting and a useful accuracy-compute range are not established. This is development evidence, not a paper result or test-set estimate.

## Actual execution and resources

- Source commit `{provenance['code_commit']}`, dirty flag `{provenance['working_tree_dirty']}`; canonical source SHA256 `{summary['source_tree_sha256']}`.
- Command `.venv/Scripts/python.exe scripts/run_smoke.py`; inner command and exact exit0 evidence are in `reports/smoke_command.json`. One learning job; no retry.
- {summary['optimizer_steps']} AdamW steps, 32 train roots and 16 validation roots, batch 8, seed 17, 8x8 grids, width 64, 2 heads, inner 1, trained K=2; {summary['parameter_count']:,} parameters.
- Entire subprocess {command['wall_seconds']:.3f}s; internal run {summary['wall_ms']/1000:.3f}s; training batches including collation/backward/optimizer {summary['training_ms']/1000:.3f}s.
- Batched training throughput {summary['training_examples_per_second']:.3f} examples/s; peak process working set {summary['memory']['peak_rss_bytes']:,} bytes ({summary['memory']['peak_rss_bytes']/1024**2:.2f}MiB). This is CPU process memory, not GPU memory. Ceiling 4 GiB, two CPU threads.
- Total {summary['total_forward_block_calls']} batched F forward calls and {summary['total_forward_transformer_layer_executions']} transformer-layer executions across training and pre/post measurements. Backward cost is included in time but not represented as extra forward calls. Original solve is the entire static inference; no update-stream saving is measured.
- Incremental external charge $0; electricity unmeasured/null. Shared ledger reservation/reconciliation complete, reserved $0, recorded external actual $0. The $950 accounting ceiling does not authorize future spending and excludes unmeasured electricity from actuals so far.
- CUDA and MPS **not tested**. Python 3.12.14 / CPU torch 2.14.0+cpu; details in provenance and environment lock.

## Learning observations

Mean sampled-batch loss over first 20 steps: {mean(s['loss'] for s in steps[:20]):.6f}; last 20 steps: {mean(s['loss'] for s in steps[-20:]):.6f}. These batches differ and are descriptive learning diagnostics, not a paired validation-loss estimate.

Post-training metrics below are per base-root/frame-zero. All-node exact policy correctness was zero at every listed point. No confidence interval is claimed from this small single-seed smoke.

| Split | Outer cycles K | Exact routes | Mean valid-action accuracy | Batch-one mean ms |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

At trained K=2, validation valid-action accuracy increased from 27.1484375% to 47.75390625%, while exact validation routes remained 0/16. Training route accuracy is 4/32 at K=2; this is not successful overfitting. K=1 solves 1/16 validation routes and K=4 solves 0/16; these tiny counts do not establish a recurrence advantage. No operating point was chosen and no held-out test payload was evaluated.

Timing includes observed tensor construction, encoding, fresh state, all requested core work, decoding and output copy. Offline BFS scoring is excluded. There was no warmup, only one measurement per root/budget before and after training, and substantial Python/validation overhead; these are descriptive CPU smoke latencies, not a warmed hardware speedup claim. Reloaded-checkpoint timings are retained separately in `smoke_report.json`.

## Evidence and limitations

The CLI evaluation reloaded checkpoint `{summary['checkpoint_sha256']}` and produced real validation records. `report` checked config/source/checkpoint lineage and raw artifact hashes before aggregation. Raw steps, pre/post records, reload records, config, provenance, summary and split manifest are copied without modification into `reports/evidence/smoke/` for review. Full generated data, checkpoint and live cost ledger remain local under `runs/` and are intentionally Git-ignored. A clean clone can regenerate data and run its own smoke; it cannot claim to reload this checkpoint unless the original local artifact is supplied. The exact environment lock is platform-specific, not a universal wheel lock.

Regenerate this report locally with `.venv/Scripts/python.exe scripts/build_pilot_report.py`. Regenerate checked aggregates with `.venv/Scripts/python.exe -m state_repair.cli report --run runs/smoke`. Missing/tampered raw files must fail; independent audit records the actual checks. Nothing here evaluates restart versus carry or a repair adapter. Static frame-zero spanning-tree mazes are initially connected; unreachable edits were tested for correctness but not learned in this run.

Prior-art decision remains NARROW_OR_REFRAME. CoFRe and other established reuse/gating methods preclude an architectural novelty claim from this prototype. Generic GRU/residual/global controls, edited-state gradient/lifetime tests, fixed-budget streams, circuits, multi-seed statistics and confirmatory freeze are still absent. The synthetic initializer gradient test validates the backbone derivative path only; it is not a tested repair adapter.

## Reproduction

See `REPRODUCING.md` for setup, data generation, training and evidence-verification commands.
'''
    Path("reports/PILOT_REPORT.md").write_text(report,encoding="utf-8")
