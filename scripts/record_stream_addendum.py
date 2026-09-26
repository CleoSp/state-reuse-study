"""Record a verified stream-training addendum."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from build_adapter_pilot_report import percent, table
from check_adapter_pilot import require
from run_adapter_pilot import json_rows, read
from run_stream_addenda import ROOTS


def record(kind: str) -> None:
    checked = read(ROOTS[kind] / "check.json")
    require(checked["verified"] and checked["complete"], "complete checked addendum required")
    config = read(ROOTS[kind] / "prepared/config.json")
    marker = "# " + ("Depth-shift evaluation" if kind == "depth" else "Matched-compute control")
    lines = [marker, ""]
    if kind == "depth":
        lines += ["All 24 previously trained checkpoints were evaluated without retraining on the existing 256 48-node validation roots, 32 edits, K1/2/4/8. The primary comparisons below retain the ordinary-validation grid per policy; all grids, seeds and per-frame results are preserved. No depth-shift selection or test-root evaluation occurred.", "",
            f"**Declared gate:** a positive lower paired-root 95% bound for answer-only minus restart at any tested K supports the second family under this structural shift. **Outcome: {checked['gate']}.** " +
            ("The second-family support is specific to this exploratory structural-shift regime; ordinary circuits remain negative." if checked["gate"] else "No positive interval was established under shift; the positive output-reuse claim remains maze-specific."), ""]
    else:
        lines += ["Twelve joint one-edit runs used the unchanged pilot recipe and selected pilot grid, with 626 optimizer steps. All were evaluated on the identical 256-root 32-edit validation streams at K1/2/4/8; no new grid selection.", "",
            f"**Declared gate:** matched-compute spatial_gate minus stream spatial_gate intervals contain zero at both K1 and K4. **Outcome: {checked['gate']}.** The declared operational attribution is " +
            ("compute." if checked["gate"] else "stream exposure.") + " This screening attribution is not a causal or equivalence proof; all arm contrasts remain reported. The original G2 latent-reuse-versus-answer-only result is unchanged.", "",
            table(["Policy","Seed","Steps","One-edit F calls","Stream F calls","Mismatch (%)"],
                [[r["arm"],r["seed"],r["steps"],r["example_F_calls"],r["stream_example_F_calls"],percent(r["relative_difference"])] for r in checked["training_matches"]]), ""]
    metrics = [("post32","Mean post-edit exact correctness (%)")]
    if kind == "depth":
        metrics.append(("node32","Mean post-edit non-input-node accuracy (%)"))
    for metric,label in metrics:
        lines += [label, "", table(["Policy"]+[f"K{k}" for k in config["budgets"]],
            [[a]+[percent(mean(r[metric] for r in checked["curves"] if r["primary"] and r["arm"] == a and r["K"] == k)) for k in config["budgets"]] for a in config["arms"]]), ""]
    lines += [table(["Arm minus comparator","K","Difference (pp)","Paired 95% interval (pp)"],
        [[f"{r['arm']} minus {r['baseline']}",r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in checked["contrasts"]]), ""]
    if kind == "depth":
        exploratory = checked["exploratory_contrasts"]
        lines += ["**Exploratory, requested after the sweep began:** spatial_gate minus answer_only was not a declared depth-shift contrast and does not reopen G2.", "",
            table(["Spatial gate minus answer-only","K","Difference (pp)","Paired 95% interval (pp)"],
                [["exploratory",r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in exploratory]), "",
            "Regime-dependent observation: output reuse wins the declared comparisons on in-distribution mazes. " +
            ("Under circuit structural shift, the learned gate's mean accuracy matches or exceeds answer-only at every tested K." if all(r["delta"] >= 0 for r in exploratory) else
             "Under circuit structural shift, the learned gate's comparison with answer-only is budget dependent; the intervals above give the measured outcome.") +
            " This is an exploratory observation, not a prespecified noninferiority test or a reopening of the maze G2 gate.", ""]
    events = list(json_rows(Path("runs/prompt05b_gpu_time.jsonl")))
    actual = [r for r in events if r["kind"] == "actual"]
    lines += [f"Verified {checked['stream_predictions']:,} raw stream predictions and {checked['tuning_predictions']:,} tuning predictions; every action rescored with full paired coverage. Check: `{ROOTS[kind].as_posix()}/check.json`. Cumulative stream-training GPU-job wall time: {sum(r['seconds'] for r in actual)/3600:.6f} hours. External charges $0; electricity unmeasured. Three earlier failed synthetic capacity attempts remain included.", ""]
    Path(f"reports/stream_{kind}_outcome.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("longer","depth"))
    record(parser.parse_args().kind)
