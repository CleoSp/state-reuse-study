"""Regenerate the adapter pilot report from verified evidence."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sqlite3

import numpy as np

from state_repair.accounting.gpu_job import remaining_seconds
from state_repair.provenance import file_hash
from state_repair.train.pilot import AUXILIARY, PRINCIPAL
from check_adapter_pilot import require
from run_adapter_pilot import json_rows, read, verify_seal, write


def percent(value: float) -> str:
    return f"{100*value:.2f}"


def table(headers: list[str], rows: list[list]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows) + "\n"


def main() -> None:
    root = Path("runs/adapter_pilot_v1")
    check = read("runs/prompt05-pilot-check.json")
    audit = read("runs/prompt05-pilot-provenance-audit.json")
    diagnostic = read("runs/pilot_replay_diagnostic/summary.json")
    require(check["verified"] and check["complete"] and check["gate"]["scientific_stop"], "requires complete verified stop outcome")
    require(audit["verified_lineage"] and audit["primary_check_sha256"] == file_hash("runs/prompt05-pilot-check.json"), "lineage audit mismatch")
    require(diagnostic["cuda_replay_action_mismatches"] == 0, "original-runtime replay unresolved")
    for out in [root / "prepared", root / "cache", Path("runs/pilot_replay_diagnostic"),
                *sorted((root / "training").iterdir()), *sorted((root / "streams").iterdir()), *sorted((root / "interventions").iterdir())]:
        verify_seal(out)
    config = read(root / "prepared/config.json")
    selection = read(root / "selection.json")
    require(file_hash(root / "selection.json") == check["selection_sha256"], "selection changed")
    stream_summaries, frame_rows = [], []
    for out in sorted((root / "streams").iterdir()):
        summary = read(out / "summary.json")
        require(summary["synthetic"] is False, "synthetic empirical report")
        stream_summaries.append(summary)
        grouped = defaultdict(list)
        for row in json_rows(out / "predictions.jsonl.gz"):
            require(row["synthetic"] is False, "synthetic empirical prediction")
            grouped[row["K"], row["frame"]].append(row)
        for (k, frame), values in sorted(grouped.items()):
            frame_rows.append({"track": summary["track"], "arm": summary["arm"], "seed": summary["seed"], "K": k,
                "frame": frame, "roots": len(values), **{metric: sum(r[metric] for r in values)/len(values)
                for metric in ("route_correct", "all_node_correct", "valid_action_accuracy")}})

    intervention = defaultdict(lambda: defaultdict(list))
    for path in sorted((root / "interventions").glob("*/*.jsonl.gz")):
        for row in json_rows(path):
            for stratum in ("all", row["stratum"]):
                key = row["track"], row["arm"], row["seed"], row["suite"], stratum, row["K"]
                intervention[key][row["root_id"]].append(row["route_correct"])
    intervention_metrics = []
    merged = defaultdict(lambda: defaultdict(list))
    for (track, arm, seed, suite, stratum, k), roots in sorted(intervention.items()):
        values = {rid: sum(v)/len(v) for rid, v in roots.items()}
        intervention_metrics.append({"track": track, "arm": arm, "seed": seed, "suite": suite, "stratum": stratum,
            "K": k, "roots": len(values), "route_correct": sum(values.values())/len(values)})
        for rid, value in values.items():
            merged[track, arm, suite, stratum, k][rid].append(value)
    contrasts = []
    for key, spatial in sorted(merged.items()):
        track, arm, suite, stratum, k = key
        if arm != "spatial_gate":
            continue
        roots = sorted(spatial)
        draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(roots), size=(10000, len(roots)))
        a = np.array([np.mean(spatial[r]) for r in roots])
        for other_key, other in sorted(merged.items()):
            ot, oa, os, ostr, ok = other_key
            if (ot, os, ostr, ok) != (track, suite, stratum, k) or oa == "spatial_gate":
                continue
            require(set(other) == set(spatial), "unpaired intervention roots")
            b = np.array([np.mean(other[r]) for r in roots])
            delta = a-b
            contrasts.append({"track": track, "suite": suite, "stratum": stratum, "K": k, "baseline": oa,
                "roots": len(roots), "spatial": float(a.mean()), "baseline_accuracy": float(b.mean()),
                "spatial_minus_baseline": float(delta.mean()), "ci": np.quantile(delta[draws].mean(1), [.025, .975]).tolist() if len(roots) >= 2 else None})
    maze = [read(Path("runs/scaled_maze_v1") / f"seed-{s}/summary.json") for s in config["seeds"]]
    circuits = [read(Path("runs/static_circuit_v12") / f"seed-{s}/summary.json") for s in config["seeds"]]
    crossover = read("runs/frozen_crossover_v1/summary.json")
    events = list(json_rows(Path("runs/prompt05_gpu_time.jsonl")))
    actual = [e for e in events if e["kind"] == "actual"]
    require({e["job_id"] for e in events if e["kind"] == "reserve"} == {e["job_id"] for e in actual}, "open GPU reservation")
    with sqlite3.connect("file:runs/accounting.sqlite?mode=ro", uri=True) as conn:
        allocation = conn.execute("SELECT cents FROM allocation").fetchone()[0]/100
        ledger_events = [dict(zip(("seq", "timestamp", "kind", "job_id", "cents", "category"), r))
            for r in conn.execute("SELECT seq,timestamp,kind,job_id,cents,category FROM events ORDER BY seq")]
    write(Path("runs/prompt05-accounting-export.json"), {"allocation_usd": allocation, "events": ledger_events,
        "database_sha256": file_hash("runs/accounting.sqlite"), "electricity_cost_usd": None})
    resources = {"cumulative_gpu_job_hours": sum(e["seconds"] for e in actual)/3600,
        "remaining_authorized_hours": remaining_seconds(events)/3600, "closed_gpu_jobs": len(actual),
        "external_cost_usd": sum(e["cents"] for e in ledger_events if e["kind"] == "actual")/100,
        "electricity_cost_usd": None, "failed_jobs": [e for e in actual if e["error"]],
        "max_recorded_reserved_gpu_bytes": max(e["peak_reserved_gpu_bytes"] for e in actual)}
    training = [read(p) for p in sorted((root / "training").glob("*/summary.json"))]
    sources = [Path("runs/prompt05-pilot-check.json"), Path("runs/prompt05-pilot-provenance-audit.json"),
               Path("runs/pilot_replay_diagnostic/manifest.json"), root / "prepared/manifest.json", root / "selection.json",
               Path("runs/prompt05-scaled-check.json"), Path("runs/prompt05-circuit-v12-check.json"),
               Path("runs/prompt05-frozen-check.json"), Path("runs/prompt05_gpu_time.jsonl")]
    data = {"synthetic": False, "status": "STOP_AFTER_MILESTONE_05", "scope": "exploratory validation, no held-out test",
        "config": config, "selection": selection, "gate": check["gate"], "verification": check, "provenance_audit": audit,
        "numerical_replay_diagnostic": diagnostic, "resources": resources, "training": training,
        "streams": stream_summaries, "per_frame_metrics": frame_rows, "intervention_metrics": intervention_metrics,
        "intervention_paired_contrasts": contrasts, "source_hashes": {p.as_posix(): file_hash(p) for p in sources}}
    write(Path("reports/pilot_results.json"), data)
    lines = ["# Measured adapter pilot", "",
        "**Decision for adapter_pilot_v1: stop.** The declared pilot gate is met: in both frozen and joint tracks, every selective-repair budget is beaten by restart or carry at the same or lower K, with a positive lower 95% paired-root bound. This recipe does not support a selective-repair paper claim. Subsequent exploratory protocols do not revise this outcome.", "",
        "This is exploratory validation evidence for this training recipe, not proof that all selective-repair methods fail. No held-out test roots were generated or evaluated. Full raw predictions, including failures and unfavorable arms, are retained.", "",
        "## Declared stopping comparison", "",
        "Metric: mean post-edit exact-route accuracy across 32 edits. Average the three adapter seeds within each root, then resample 256 paired roots with all their frames (10,000 draws, seed 54037). Intervals are pointwise and conditional on these training seeds; the multi-budget screening is exploratory. The table shows the eligible baseline with the largest observed mean advantage; every comparison is in `pilot_results.json`. Gaps and intervals are percentage points.", ""]
    stop_rows = []
    for track in ("frozen", "joint"):
        for k in config["budgets"]:
            candidates = [r for r in check["gate"]["comparisons"] if r["track"] == track and r["spatial_K"] == k]
            best = max(candidates, key=lambda r: r["baseline_minus_spatial"])
            stop_rows.append([track, k, best["baseline"], best["baseline_K"], percent(best["baseline_minus_spatial"]),
                              f"[{percent(best['ci'][0])}, {percent(best['ci'][1])}]"])
    lines += [table(["Track", "Repair K", "Baseline", "Baseline K", "Baseline − repair", "95% paired interval"], stop_rows)]
    lines += ["## Protocol and completed work", "",
        "All 72 training runs completed 256 optimizer steps (18,432 total), batch 64, uniform K in {1,2,4,8}. The five frozen learned policies and seven joint policies each received two learning rates and seeds 29/43/71. Frozen restart/carry are parameter-free curves. Joint restart/carry received both backbone learning rates. The adapter and current frozen rollout remain differentiable; prior state is detached before adaptation.", "",
        "All adapter seeds use the same passing static maze seed-29 checkpoint. These are adapter-seed replications conditional on one backbone, not three independent backbone replications. The backbone has width 64, two heads, inner count 2: each outer cycle executes three shared-block calls and six transformer layers. It is TRM-inspired, not an exact TRM/RSM reproduction.", "",
        "Data: 1,024 training roots, four independently seeded one-toggle views per root; 256 validation roots and 32 uniform-toggle edits per root. Ancestry is kept together; test payloads stay unopened. The selected learning rate maximizes four-edit validation accuracy across all roots, budgets and three seeds, with a lower-rate tie break. Both tuning candidates remain published.", "",
        "Shared frozen training caches contain model-generated states at the current run's own K, with observed-input/root/checkpoint/policy provenance. Joint priors are recomputed after every weight update. Deployed streams use their own fixed-K state, including the initial solve. The separate initializer intervention deliberately forks a common K=8 prior; it is never used as cheap stream history.", "",
        "Local resets use radii 1/2/3. Random reset is calibrated to each selected spatial model's four-edit mean reset rate; noise scale is 0.01; shuffled gates use the trained spatial module. Answer-only is trained. Joint auxiliary initializer controls use the joint spatial backbone, whereas principal joint policies own separately trained backbones. These comparison types must remain distinct.", "",
        table(["Track / policy", "Selected grid", "Grid 0 score (%)", "Grid 1 score (%)"],
              [[name, r["grid"], percent(r["candidates"][0]["score"]), percent(r["candidates"][1]["score"])] for name, r in sorted(selection.items())])]
    for track in ("frozen", "joint"):
        lines += [f"## {track.capitalize()} 32-edit curves", "",
            "Mean post-edit route accuracy (%), with the training/control-seed minimum–maximum in parentheses. The frozen restart/carry curves are single fixed models. All-node, valid-action, unreachable/trivial denominators, whole-stream outcomes and all 33 per-frame curves are saved in `pilot_results.json`.", ""]
        curves = []
        for arm in (*PRINCIPAL, *AUXILIARY):
            row = [arm]
            for k in config["budgets"]:
                values = [next(m["route_correct"] for m in s["validation"]["metrics"] if m["K"] == k and m["frames"] == [1,32])
                    for s in stream_summaries if s["track"] == track and s["arm"] == arm]
                row.append(f"{percent(float(np.mean(values)))} ({percent(min(values))}–{percent(max(values))})")
            curves.append(row)
        lines += [table(["Policy", "K=1", "K=2", "K=4", "K=8"], curves)]
    lines += ["## Every principal training-seed curve", "", "Post-edit route accuracy (%) over 32 edits; no seed is omitted.", ""]
    lines += [table(["Track", "Policy", "Seed", "K=1", "K=2", "K=4", "K=8"],
        [[s["track"], s["arm"], s["seed"], *[percent(next(m["route_correct"] for m in s["validation"]["metrics"] if m["K"] == k and m["frames"] == [1,32]))
          for k in config["budgets"]]] for s in stream_summaries if s["arm"] in PRINCIPAL])]
    challenge = read(root / "prepared/challenge_audit.json")
    lines += ["## Frozen initializer interventions", "",
        f"The challenge sampler retained {challenge['roots_included']} low/high pairs: 54 additions and 64 removals. Ten requested addition pairs were unavailable; thresholds and roots were not changed. Ordinary toggles and conditioned challenge pairs are separate suites; challenge prevalence is not ordinary prevalence.", "",
        "The following selected contrast compares spatial gates with shuffled versions of the same trained gates on the same prior and frozen weights. Values are spatial-minus-shuffled route accuracy in percentage points, averaged across seeds before root resampling. This is an exploratory placement diagnostic, not evidence of ground-truth latent invalidation. Every baseline, stratum, budget and seed is available in the machine-readable report.", "",
        table(["Track", "Suite", "Stratum", "K", "Roots", "Spatial − shuffled", "95% paired interval"],
            [[r["track"], r["suite"], r["stratum"], r["K"], r["roots"], percent(r["spatial_minus_baseline"]),
              "unavailable" if r["ci"] is None else f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"]
             for r in contrasts if r["baseline"] == "shuffled_gate" and r["stratum"] in ("low", "high")])]
    lines += ["## Earlier gates and failures", "",
        "The crossover criterion passed on the original 8×8 frozen-state challenge at K=4: carry-minus-restart was +12.20 pp on low-impact edits (95% interval +3.66 to +21.95) and −24.39 pp on high-impact edits (−34.15 to −15.85). Ordinary validation had only two high-impact roots, so it did not independently establish that crossover. This supported asking the adapter question, not assuming a positive answer.", "",
        "All three scaled maze seeds passed the ordinary nondecreasing/≥85% gate. Exact-route counts below use 256 roots per suite.", "",
        table(["Seed", "Suite", "K=1", "K=2", "K=4", "K=8", "K=16"],
            [[s["seed"], suite, *[next(m["routes"] for m in s["validation"] if m["suite"] == suite and m["K"] == k)
              for k in (1,2,4,8,16)]] for s in maze for suite in ("ordinary", "rooms", "size16")]),
        "All three completed circuit v1.2 seeds passed ordinary nondecreasing/≥85% exact correctness. Deeper 48-node circuits retained a large gap. Counts below use 256 roots.", "",
        table(["Seed", "Suite", "K=1", "K=2", "K=4", "K=8", "K=16"],
            [[s["seed"], suite, *[next(m["exact_correct"] for m in s["validation"] if m["suite"] == suite and m["K"] == k)
              for k in (1,2,4,8,16)]] for s in circuits for suite in ("ordinary", "depth48")]),
        "Preserved failures: width-128 maze capacity and batch-256 circuit capacity exhausted the configured allocator allowance; the maze used the declared width-64 fallback and circuits used batch 128. Circuit v1.1 seed 29 hit its training-time cap at 2,905/6,000 steps and is excluded from the completed three-seed matrix. Version 1.2 cached immutable training observations and restarted every circuit seed from scratch. A first transfer-parity diagnostic failed a bitwise criterion; the subsequent declared 1e-5 tolerance passed with identical labels. No held-out threshold or generator was retuned.", "",
        "The 256-step adapter pilot does not establish convergence. Answer-only losses fell substantially; GRU and residual adapters were sensitive to learning rate, and one-edit training did not guarantee stable long streams. First/last-32-step losses, all gradients and both candidate checkpoints are retained. The negative decision applies to this bounded recipe.", "",
        "## Verification and numerical limits", "",
        f"The independent checker verified {check['training_runs']} training runs, {check['tuning_predictions']:,} tuning predictions, {check['stream_predictions']:,} stream predictions and {check['intervention_predictions']:,} initializer predictions. It reconstructed training schedules, optimizer steps, fixed-K provenance, all raw action scores and paired coverage. It did not replay every forward pass.", "",
        "A separate CPU audit exactly regenerated challenge selection and checked all 64 training-cache batches and 48 intervention-prior batches against original observed inputs. The deterministic principal-policy spot replay used the first two validation roots, all seeds, endpoint K=1/8 and all 33 frames: 5,016 predictions / 722,304 node actions. It found 1,846 action differences, three route-decision differences and no all-node-correctness differences between CPU torch 2.14 and saved CUDA torch 2.11 results.", "",
        "A repeat of every discrepant case in the original CUDA runtime matched all 25,344 predictions exactly at batch 64. Within torch 2.11, initial CPU/CUDA logit differences were at most 4.864e-5; they grew in some long streams, reaching 12.30 at the final frame. CPU torch 2.11 and CPU torch 2.14 also differed on some actions. Numerical sensitivity across backends/versions is therefore a real limitation; cross-platform bitwise reproducibility is not claimed. The saved CUDA result and stop screen were not replaced with a favorable replay.", "",
        "The full CPU suite passed 277 tests before the pilot; the final verification log is `runs/prompt05-final-tests.txt`. Tests cover typed information boundaries, nonzero adapter gradients, intentional broken detach, padding, stream budget ownership, raw-artifact rejection and synthetic-record exclusion. Evaluator memoization was added only after GPU runs completed and preserves exact scoring of every raw action record.", "",
        "## Resources and reproducibility", "",
        f"Cumulative local GPU-job wall time, including failed jobs, profiles and the replay diagnostic: **{resources['cumulative_gpu_job_hours']:.4f} hours**, leaving **{resources['remaining_authorized_hours']:.4f}** of the user-approved ten hours. All {resources['closed_gpu_jobs']} reservations are reconciled. Recorded external charges: ${resources['external_cost_usd']:.2f}; electricity remains unmeasured. Whole-job time includes CPU setup/scoring inside each job context. This is not a hardware speedup measurement.", "",
        "Float32, RTX 5070, torch 2.11.0+cu128, TF32 disabled; six-GiB allocator cap and declared 7.5-GiB estimate including runtime headroom. Batch sizes are 64 for the maze pilot, 128 for completed circuits. Source snapshots, checkpoints, RNG/optimizer state, configurations, raw predictions and resource records are packaged under `reports/evidence/prompt05/`, with per-file SHA-256 manifests. Synthetic capacity profiles remain explicitly separate from empirical result tables.", "",
        "Restore the evidence ZIPs into a clean checkout to recreate their `runs/` paths. Use the recorded environment for numerical replay. Recheck with `.venv/Scripts/python.exe scripts/check_adapter_pilot.py --output runs/recheck.json`; regenerate this report with `.venv/Scripts/python.exe scripts/build_adapter_pilot_report.py`. The JSON report contains source hashes, all per-frame/seed/stratum metrics, selection candidates, paired contrasts and the full stopping screen.", "",
        "## Stopping boundary", "",
        "The pilot has a negative selective-repair outcome. Subsequent stream-training and output-reuse experiments use separate exploratory protocols. These validation results do not establish a confirmatory effect or an accuracy-cost frontier. Soft gates still execute dense work; no FLOP/latency reduction is claimed. Established state reuse and gating precedents are credited in sources/READING_LIST.md.", ""]
    Path("reports/PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": "reports/PILOT_REPORT.md", "decision": "STOP_AFTER_MILESTONE_05", "resources": resources,
        "stream_seed_records": len(stream_summaries), "per_frame_metric_rows": len(frame_rows), "intervention_contrasts": len(contrasts)}), flush=True)


if __name__ == "__main__":
    main()
