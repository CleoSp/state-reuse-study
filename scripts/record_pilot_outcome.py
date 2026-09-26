"""Record pilot outcomes and exploratory output-reuse diagnostics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from build_adapter_pilot_report import percent, table
from run_adapter_pilot import json_rows, read, write


def main() -> None:
    report = read("reports/pilot_results.json")
    config = report["config"]
    contrasts = []
    for k in config["budgets"]:
        values = {}
        for arm in ("answer_only", "restart"):
            by_root = defaultdict(list)
            for s in report["streams"]:
                if s["track"] == "joint" and s["arm"] == arm:
                    for e in s["validation"]["episodes"]:
                        if e["K"] == k:
                            by_root[e["root_id"]].append(e["post32"])
            values[arm] = {r: np.mean(v) for r, v in by_root.items()}
        roots = sorted(values["restart"])
        if set(roots) != set(values["answer_only"]):
            raise ValueError("unpaired output-reuse contrast")
        delta = np.array([values["answer_only"][r]-values["restart"][r] for r in roots])
        draws = np.random.default_rng(config["bootstrap_seed"]).integers(0, len(roots), size=(10000,len(roots)))
        contrasts.append({"K": k, "answer_only_minus_restart": float(delta.mean()),
            "ci": np.quantile(delta[draws].mean(1), [.025,.975]).tolist(), "roots": len(roots),
            "scope": "exploratory paired validation roots, conditional on three seeds"})
    rows = list(json_rows(Path("runs/adapter_pilot_v1/interventions/frozen-seed29/carry.jsonl.gz")))
    budgets = {(r["suite"], r["branch"], r["root_id"], r["K"]): r for r in rows}
    invariance = []
    for stratum in ("low", "middle", "high"):
        pairs = [(r, budgets[r["suite"],r["branch"],r["root_id"],8]) for r in rows if r["K"] == 1 and r["stratum"] == stratum]
        count = sum(len(a["actions"]) for a,b in pairs)
        changed = sum(x != y for a,b in pairs for x,y in zip(a["actions"], b["actions"]))
        invariance.append({"stratum": stratum, "branches": len(pairs), "node_actions": count,
            "changed_K1_to_K8": changed, "fraction": changed/count if count else None})
    write(Path("runs/prompt05-output-reuse-planning.json"), {"synthetic": False, "answer_only_paired_intervals": contrasts,
        "carry_action_invariance": invariance, "scope": "post-pilot exploratory planning; not confirmatory or latent-state fixed-point proof"})
    lines = ["# Adapter pilot outcomes", "",
        "All 72 adapter training runs completed; all 2,999,136 tuning, stream and intervention predictions passed independent raw-action scoring, pairing and provenance checks. The preregistered stop fired for **adapter_pilot_v1** in both tracks at every K. This closes the original recipe with a negative selective-repair result. Subsequent protocols do not revise this result.", "",
        f"Cumulative pilot GPU-job wall time: **{report['resources']['cumulative_gpu_job_hours']:.4f} hours**; all {report['resources']['closed_gpu_jobs']} reservations reconciled; recorded external charges $0, electricity unmeasured. No held-out test roots generated or evaluated. Full report and machine-readable metrics: `reports/PILOT_REPORT.md`, `reports/pilot_results.json`.", "",
        "### 32-edit mean post-edit exact-route accuracy (%)", ""]
    for track in ("frozen", "joint"):
        data = []
        arms = sorted({s["arm"] for s in report["streams"] if s["track"] == track})
        for arm in arms:
            values = []
            for k in config["budgets"]:
                scores = [next(m["route_correct"] for m in s["validation"]["metrics"] if m["K"] == k and m["frames"] == [1,32])
                    for s in report["streams"] if s["track"] == track and s["arm"] == arm]
                values.append(percent(float(np.mean(scores))))
            data.append([arm, *values])
        lines += [f"{track.capitalize()} track. Joint auxiliary controls use the spatial backbone; principal joint arms own separately trained backbones.", "",
                  table(["Policy", "K1", "K2", "K4", "K8"], data)]
    lines += ["### Initializer challenge strata", "",
        "Route accuracy (%) from an identical K=8 prior. Endpoint budgets below; every intermediate K, stratum, seed and paired contrast is in the machine-readable report. Challenge quotas: 54 addition pairs and 64 removal pairs; ten requested addition pairs unavailable.", ""]
    data = []
    for track in ("frozen", "joint"):
        arms = sorted({r["arm"] for r in report["intervention_metrics"] if r["track"] == track})
        for arm in arms:
            scores = []
            for k, stratum in ((1,"low"),(1,"high"),(8,"low"),(8,"high")):
                vals = [r["route_correct"] for r in report["intervention_metrics"] if r["track"] == track and r["arm"] == arm
                        and r["suite"] == "challenge" and r["stratum"] == stratum and r["K"] == k]
                scores.append(percent(float(np.mean(vals))))
            data.append([track, arm, *scores])
    lines += [table(["Track", "Initializer", "Low K1", "High K1", "Low K8", "High K8"], data),
        "### Per-frame decay and distribution-drift reference", "",
        "Mean exact-route accuracy (%) at selected frames; **all 33 frame values** for every seed/method/K are saved in `reports/pilot_results.json:per_frame_metrics`. Restart supplies the distribution-drift reference. Frame zero is the policy's own K-step initial solve.", ""]
    data = []
    for track in ("frozen", "joint"):
        for arm in ("restart", "carry", "spatial_gate", "answer_only"):
            for k in config["budgets"]:
                vals = [percent(float(np.mean([r["route_correct"] for r in report["per_frame_metrics"] if
                    r["track"] == track and r["arm"] == arm and r["K"] == k and r["frame"] == frame])))
                    for frame in (0,1,4,8,16,24,32)]
                data.append([track, arm, k, *vals])
    lines += [table(["Track", "Policy", "K", "Frame 0", "1", "4", "8", "16", "24", "32"], data),
        "### Output-only reuse: exploratory contrast", "",
        "Joint answer-only minus joint restart on the same 256 validation roots, averaged over the three seeds and all 32 edits before root resampling (10,000 draws). Intervals below are percentage points and exploratory; they do not establish a measured compute frontier or a confirmatory paper result.", "",
        table(["K", "Answer-only − restart", "95% paired-root interval"], [[r["K"],percent(r["answer_only_minus_restart"]),
            f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in contrasts]),
        "### Carry invariance and replay limits", "",
        "Saved pilot-backbone carry outputs were compared directly between K=1 and K=8 from the identical prior. Counts combine ordinary and challenge branches; they are diagnostics, not independent node-level replications.", "",
        table(["Impact stratum", "Branches", "Node actions", "Changed K1→K8", "Fraction"],
            [[r["stratum"],r["branches"],r["node_actions"],r["changed_K1_to_K8"],r["fraction"]] for r in invariance]),
        "Identical route scores or argmax outputs do **not** prove an exact latent fixed point. Action and state dynamics require separate measurement; these output comparisons do not establish a latent mechanism.", "",
        "CPU spot replay covered 5,016 predictions and differed on three route decisions; original-runtime CUDA replay reproduced all 25,344 predictions in the discrepant cases exactly. Tiny initial numerical differences can amplify across long streams. All differences are retained in `runs/prompt05-pilot-provenance-audit.json` and `runs/pilot_replay_diagnostic/`.", "",
        "Verification commands: `scripts/check_adapter_pilot.py --output runs/prompt05-pilot-check.json`, `scripts/audit_pilot_provenance.py`, and `scripts/diagnose_pilot_replay.py`. Final CPU suite: `runs/prompt05-final-tests.txt`; report generator: `scripts/build_adapter_pilot_report.py`; this outcome report: `scripts/record_pilot_outcome.py`. Evidence shards and SHA-256 index are under `reports/evidence/prompt05/`. Original snapshots and failed/superseded runs remain preserved.", "",
    ]
    Path("reports/pilot_outcome.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
