"""Regenerate stream-training results from verified matrices and diagnostics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3

import numpy as np

from run_adapter_pilot import read, write, json_rows
from check_adapter_pilot import require
from build_adapter_pilot_report import table, percent
from state_repair.provenance import file_hash
from state_repair.train.pilot import PRINCIPAL, AUXILIARY


def seed_accuracy_table(records: list[dict], arms: list[str], budgets: list[int], seeds: list[int]) -> str:
    return table(["Policy","Training seed"]+[f"K{k} (%)" for k in budgets],
        [[arm,seed]+[percent(np.mean([r["post32"] for r in records if r["arm"] == arm and r["seed"] == seed and r["K"] == k]))
            for k in budgets] for arm in arms for seed in seeds])


def curves(root: Path, config: dict, arms: list[str], family: str) -> dict:
    results,per_frame,training = [],[],[]
    for arm in arms:
        for seed in config["seeds"]:
            out = root / "streams" / f"joint-{arm}-seed{seed}"
            summary = read(out / "summary.json")["validation"]
            for k in config["budgets"]:
                episodes = [e for e in summary["episodes"] if e["K"] == k]
                results.append({"arm":arm,"seed":seed,"K":k,"roots":len(episodes),
                    "post32":float(np.mean([e["post32"] for e in episodes])),
                    "whole32":float(np.mean([e["whole32"] for e in episodes])),"episodes":episodes})
                if family == "circuit":
                    results[-1]["node32"] = float(np.mean([e["node32"] for e in episodes]))
            grouped = defaultdict(list)
            for row in json_rows(out / "predictions.jsonl.gz"):
                grouped[row["K"],row["frame"]].append(row)
            metric,node = ("route_correct","valid_action_accuracy") if family == "maze" else ("exact_correct","node_accuracy")
            for (k,f),rows in sorted(grouped.items()):
                per_frame.append({"arm":arm,"seed":seed,"K":k,"frame":f,"roots":len(rows),
                    "exact_accuracy":float(np.mean([r[metric] for r in rows])),"node_accuracy":float(np.mean([r[node] for r in rows]))})
    for out in sorted((root / "training").glob("*")):
        summary = read(out / "summary.json")
        steps = list(json_rows(out / "steps.jsonl"))
        training.append({**{k:summary[k] for k in ("arm","seed","grid","optimizer_steps","training_example_F_calls","training_seconds","parameter_count")},
            "first32_post_edit_loss":float(np.mean([np.mean(r["frame_losses"][1:]) for r in steps[:32]])),
            "last32_post_edit_loss":float(np.mean([np.mean(r["frame_losses"][1:]) for r in steps[-32:]])),
            "steps":[{k:r[k] for k in ("step","K","loss","frame_losses","adapter_gradient_norm","backbone_gradient_norm","gradient_norm")} for r in steps]})
    return {"curves":results,"per_frame":per_frame,"training":training}


def lesion_contrasts(diag: dict, family: str, config: dict) -> list[dict]:
    result = []
    for version in sorted({r["source_version"] for r in diag["lesions"]}):
        root = Path("runs/adapter_pilot_v1" if family == "maze" and version == "pilot" else
                    "runs/adapter_stream_v2" if family == "maze" else "runs/circuit_stream_v2")
        for lesion in ("uniform","shuffled_nodes"):
            for comparator in ("answer_only","restart"):
                for k in config["budgets"]:
                    differences = []
                    for seed in config["seeds"]:
                        source = next(r for r in diag["lesions"] if r["source_version"] == version and r["kind"] == lesion and r["seed"] == seed)
                        a = {e["root_id"]:e["post32"] for e in source["episodes"] if e["K"] == k}
                        baseline = read(root / "streams" / f"joint-{comparator}-seed{seed}" / "summary.json")["validation"]["episodes"]
                        b = {e["root_id"]:e["post32"] for e in baseline if e["K"] == k}
                        require(set(a) == set(b),"lesion comparison lost root pairing")
                        differences.append([a[r]-b[r] for r in sorted(a)])
                    means = np.mean(differences,axis=0)
                    draws = np.random.default_rng(config["bootstrap_seed"]).integers(0,len(means),size=(config["bootstrap_repetitions"],len(means)))
                    result.append({"source_version":version,"lesion":lesion,"comparator":comparator,"K":k,
                        "lesion_minus_comparator":float(means.mean()),"ci":np.quantile(means[draws].mean(1),[.025,.975]).tolist(),
                        "per_seed_delta":np.mean(differences,axis=1).tolist(),"roots":len(means)})
    return result


def intervention_metrics(config: dict) -> list[dict]:
    records = []
    for seed in config["seeds"]:
        out = Path("runs/adapter_stream_v2/interventions") / f"joint-seed{seed}"
        for arm in read(out / "summary.json")["arms"]:
            groups = defaultdict(list)
            for row in json_rows(out / f"{arm}.jsonl.gz"):
                groups[row["suite"],row["stratum"],row["K"]].append(row)
            for (suite,stratum,k),rows in sorted(groups.items()):
                records.append({"seed":seed,"arm":arm,"suite":suite,"stratum":stratum,"K":k,
                    "branches":len(rows),"roots":len({r["root_id"] for r in rows}),
                    "route_accuracy":float(np.mean([r["route_correct"] for r in rows])),
                    "node_accuracy":float(np.mean([r["valid_action_accuracy"] for r in rows])),
                    "backbone_owner":"spatial_gate","source_K":config["source_K"]})
    return records


def main() -> None:
    maze_check = read("runs/adapter_stream_v2/check.json")
    circuit_check = read("runs/circuit_stream_v2/check.json")
    md = read("runs/adapter_stream_v2/diagnostic_check.json")
    cd = read("runs/circuit_stream_v2/diagnostic_check.json")
    longer = read("runs/adapter_pilot_v1_longer/check.json")
    depth = read("runs/circuit_stream_v2_depth48/check.json")
    k18 = read("runs/circuit_stream_v2/k18_check.json")
    require(k18["verified"] and k18["extended_k"] and len(k18["dynamics"]) == 7 and
        all(r["budgets"] == [1,8] for r in k18["dynamics"]), "verified circuit K1/K8 supplement required")
    require(all(c["verified"] and c["complete"] for c in (maze_check,circuit_check,md,cd,longer,depth)),"complete verified stream-training evidence required")
    configs = {f:read(f"runs/{'adapter' if f == 'maze' else 'circuit'}_stream_v2/prepared/config.json") for f in ("maze","circuit")}
    data = {"maze":curves(Path("runs/adapter_stream_v2"),configs["maze"],list(PRINCIPAL+AUXILIARY),"maze"),
        "circuit":curves(Path("runs/circuit_stream_v2"),configs["circuit"],configs["circuit"]["arms"],"circuit")}
    data["maze"]["gate"] = maze_check["gate"]
    data["maze"]["interventions"] = intervention_metrics(configs["maze"])
    data["maze"]["training_work_comparison"] = []
    for row in data["maze"]["training"]:
        old = read(Path("runs/adapter_pilot_v1/training") / f"joint-{row['arm']}-seed{row['seed']}-grid{row['grid']}" / "summary.json")
        data["maze"]["training_work_comparison"].append({"arm":row["arm"],"seed":row["seed"],"grid":row["grid"],
            "pilot_example_F_calls":old["training_example_F_calls"],"stream_example_F_calls":row["training_example_F_calls"],
            "difference":row["training_example_F_calls"]-old["training_example_F_calls"],
            "ratio":row["training_example_F_calls"]/old["training_example_F_calls"]})
    data["circuit"]["contrasts"] = circuit_check["contrasts"]
    data["addenda"] = {"longer":longer,"depth":depth}
    data["circuit"]["K18_dynamics"] = k18
    data["maze"]["diagnostics"],data["circuit"]["diagnostics"] = md,cd
    for family,diag in (("maze",md),("circuit",cd)):
        data[family]["lesion_contrasts"] = lesion_contrasts(diag,family,configs[family])
    data["drift_references"] = {"maze_pilot":read("reports/pilot_results.json")["per_frame_metrics"],
        "circuit_pilot":read("runs/circuit_stream_v2/diagnostics/pilot-restart-reference/summary.json")["validation"]["per_frame"]}
    events = list(json_rows(Path("runs/prompt05b_gpu_time.jsonl")))
    reserve = {r["job_id"] for r in events if r["kind"] == "reserve"}
    actual = [r for r in events if r["kind"] == "actual"]
    require(reserve == {r["job_id"] for r in actual},"unreconciled stream-training job")
    data["resources"] = {"gpu_job_hours":sum(r["seconds"] for r in actual)/3600,"jobs":len(actual),
        "authorization_hours":12,"external_cost_usd":0,"electricity_cost_usd":None,
        "peak_allocated_gpu_bytes":max(r["peak_allocated_gpu_bytes"] for r in actual),
        "peak_reserved_gpu_bytes":max(r["peak_reserved_gpu_bytes"] for r in actual),"failed_jobs":[r for r in actual if r["error"]]}
    data["provenance"] = {"generator_sha256":file_hash(__file__),"validation_only":True,"synthetic":False,
        "checks":{p:file_hash(p) for p in ("runs/adapter_stream_v2/check.json","runs/circuit_stream_v2/check.json",
            "runs/adapter_stream_v2/diagnostic_check.json","runs/circuit_stream_v2/diagnostic_check.json",
            "runs/adapter_pilot_v1_longer/check.json","runs/circuit_stream_v2_depth48/check.json",
            "runs/circuit_stream_v2/k18_check.json")}}
    write(Path("reports/stream_results.json"),data)
    lines = ["## Stream training and output reuse", "",
        "All results below are exploratory validation on paired roots, conditional on the three training seeds. No held-out test roots were used. The original pilot result above remains unchanged.", "",
        "Training uses four sequential edits after a fresh initial solve; five frame losses are weighted equally, state is detached at each version, and one optimizer update follows the stream. The 256 steps match v1 optimizer steps, not data or compute: five rollouts replace two. Seed schedules and exact block-call totals are in `stream_results.json`.", ""]
    gate = maze_check["gate"]
    lines += [f"**G2, latent reuse:** {'reopened for ' + ', '.join(gate['candidates']) if gate['G2_latent_reuse_reopened'] else 'no latent-reuse arm has a positive lower 95% bound versus answer-only at any tested K; close this claim for this backbone family and tested recipe.'}","",
        f"**G3, maze output reuse at K1/2/4:** {'all three lower bounds are positive.' if gate['G3_output_reuse_all_K124'] else 'the three-budget criterion is not met; report the individual contrasts below.'}",""]
    for family in ("maze","circuit"):
        config = configs[family]
        arms = list(PRINCIPAL+AUXILIARY) if family == "maze" else config["arms"]
        lines += [f"### {family.capitalize()} 32-edit accuracy", "",
            "Mean post-edit exact-route/unreachable correctness (%). Auxiliary maze controls use each trained spatial backbone." if family == "maze" else
            "Mean post-edit full-circuit correctness (%); input copies are excluded from scoring. Node accuracy and every frame/seed are included in the machine-readable results.", ""]
        rows = [[a]+[percent(np.mean([r["post32"] for r in data[family]["curves"] if r["arm"] == a and r["K"] == k])) for k in config["budgets"]] for a in arms]
        lines += [table(["Policy"]+[f"K{k}" for k in config["budgets"]],rows),""]
        lines += ["Training-seed variability is shown separately below. The subsequent paired-root intervals describe example variability conditional on these seeds; they are not intervals over training randomness.", "",
            seed_accuracy_table(data[family]["curves"],arms,config["budgets"],config["seeds"]), ""]
        if family == "circuit":
            lines += ["Mean post-edit non-input-node accuracy (%):", "",
                table(["Policy"]+[f"K{k}" for k in config["budgets"]],
                    [[a]+[percent(np.mean([r["node32"] for r in data[family]["curves"] if r["arm"] == a and r["K"] == k]))
                        for k in config["budgets"]] for a in arms]), ""]
        lines += [f"![{family.capitalize()} exact accuracy across edits at K{k}](figures/prompt05b/{family}-decay-K{k}.svg)\n"
                  for k in config["budgets"]]
        comparisons = gate["contrasts"] if family == "maze" else circuit_check["contrasts"]
        lines += [table(["Arm − comparator","K","Difference (pp)","Paired 95% interval (pp)"],
            [[f"{r['arm']} − {r['baseline']}",r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in comparisons]),""]
        if family == "maze":
            lines += ["### Maze frozen-prior initializer interventions","",
                "Every initializer below uses the selected spatial backbone's identical K8 prior. Rates are descriptive branch averages within a stratum, averaged over the three seeds. Ordinary and conditioned challenge suites remain separate; every intermediate K, stratum and seed is in the machine-readable results.",""]
            im = data["maze"]["interventions"]
            rows = []
            for suite in ("ordinary","challenge"):
                for arm in ("restart","carry","spatial_gate",*AUXILIARY):
                    values = []
                    for k,stratum in ((1,"low"),(1,"high"),(8,"low"),(8,"high")):
                        chosen = [r["route_accuracy"] for r in im if r["suite"] == suite and r["arm"] == arm and r["K"] == k and r["stratum"] == stratum]
                        values.append(percent(np.mean(chosen)) if chosen else "unavailable")
                    rows.append([suite,arm,*values])
            lines += [table(["Suite","Initializer","Low K1 (%)","High K1 (%)","Low K8 (%)","High K8 (%)"],rows),""]
        lines += [f"### {family.capitalize()} carried-state dynamics (G1)","",
            "Each comparison forks an identical prior. Changed argmax actions disprove exact action invariance; they do not by themselves show useful correction or an exact latent fixed point.","",
            table(["Backbone","Seed","Stratum","Branches","All-node change fraction","Scored-node change fraction","Low-K accuracy","High-K accuracy"],
                [[f"{d['source_version']}/{d['owner']}",d["seed"],r["stratum"],r["branches"],r["fraction"],r["scored_fraction"],r["accuracy_low"],r["accuracy_high"]]
                 for d in data[family]["diagnostics"]["dynamics"] for r in d["strata"]]),""]
        if family == "circuit":
            lines += ["Circuit K1-versus-K8 supplement, for the same source-K2 priors and ordinary edits. This preserves the original K1/K2 result above and tests the larger budget explicitly; repeated retention values are not pooled again.", "",
                table(["Backbone","Seed","Stratum","All-node change fraction","Scored-node change fraction","K1 exact accuracy","K8 exact accuracy"],
                    [[f"{d['source_version']}/{d['owner']}",d["seed"],r["stratum"],r["fraction"],r["scored_fraction"],r["accuracy_low"],r["accuracy_high"]]
                     for d in k18["dynamics"] for r in d["strata"]]), ""]
        lines += [f"### {family.capitalize()} answer-content lesions", "",
            "Evaluation-time lesions replace previous probabilities with uniform or shuffled-node probabilities. These are not deployable policies. A fall relative to intact answer-only supports dependence on answer content; comparison to separately trained restart is descriptive.","",
            table(["Version","Lesion","Comparator","K","Lesion − comparator (pp)","95% interval (pp)"],
                [[r["source_version"],r["lesion"],r["comparator"],r["K"],percent(r["lesion_minus_comparator"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in data[family]["lesion_contrasts"]]),""]
        if family == "maze":
            residual = next(r for r in data[family]["lesion_contrasts"] if r["source_version"] == "stream" and r["lesion"] == "uniform" and r["comparator"] == "restart" and r["K"] == 4)
            intact = next(r for r in gate["contrasts"] if r["arm"] == "answer_only" and r["baseline"] == "restart" and r["K"] == 1)
            lines += [f"Audit caveat: the uniform-probability lesion at K4 retains a {percent(residual['lesion_minus_comparator'])}-percentage-point advantage over restart, with paired 95% interval [{percent(residual['ci'][0])}, {percent(residual['ci'][1])}]. This small content-free residual is consistent with a stronger jointly trained backbone/constant initializer; this lesion does not isolate the backbone from its learned projection. It is minor beside the intact answer-only K1 advantage of {percent(intact['delta'])} points, but the full gain cannot be attributed solely to informative previous answers.", ""]
        lines += [f"### {family.capitalize()} retention by stratum", "",
            "Descriptive pooled-node retention, averaged over a/z. Twenty-bin histograms and per-branch means are saved in `stream_results.json`; nodes are not independent experimental replicates. Distant nodes are beyond two hops in current observed undirected adjacency.","",
            table(["Stratum","Node group","Nodes","Mean retention","Standard deviation"],
                [[r["stratum"],r["group"],r["nodes"],r["mean"],r["std"]] for r in data[family]["diagnostics"]["retention_histograms"]]),""]
        means = {(r["stratum"],r["group"]):r["mean"] for r in data[family]["diagnostics"]["retention_histograms"]}
        def difference(a,b):
            return None if means[a] is None or means[b] is None else means[a]-means[b]
        deltas = {"low_minus_high_all_nodes":difference(("low","all"),("high","all")),
            "edited_minus_distant_low":difference(("low","edited"),("low","distant")),
            "edited_minus_distant_high":difference(("high","edited"),("high","distant"))}
        data[family]["retention_mean_differences"] = deltas
        fmt = lambda v: "unavailable (empty group)" if v is None else f"{v:.6f}"
        lines += [f"Observed pooled mean-retention differences: low minus high impact = {fmt(deltas['low_minus_high_all_nodes'])}; edited minus distant = {fmt(deltas['edited_minus_distant_low'])} on low-impact branches and {fmt(deltas['edited_minus_distant_high'])} on high-impact branches. These describe the saved gates and do not establish a causal mechanism.",""]
        lines += [f"![{family.capitalize()} retention histogram for {group} nodes](figures/prompt05b/{family}-retention-{group}.svg)\n"
                  for group in ("all","edited","distant")]
    lines += ["### Circuit depth shift", "",
        "This exploratory addendum was requested after observing ordinary-circuit saturation. All 24 existing checkpoints (four policies, two grids, three seeds) were evaluated without retraining on the existing 256 48-node depth-shift validation roots. The primary table and contrasts use each policy's previously frozen ordinary-validation grid. Both grids, all seeds and all frames remain in the machine-readable records; depth-shift results were not used for selection. Circuit training used K1/2; K4/8 also extrapolate the rollout budget.", ""]
    for metric,label in (("post32","Mean post-edit full-circuit correctness (%)"),("node32","Mean post-edit non-input-node accuracy (%)")):
        lines += [label,"",table(["Policy","K1","K2","K4","K8"],
            [[a]+[percent(np.mean([r[metric] for r in depth["curves"] if r["primary"] and r["arm"] == a and r["K"] == k]))
                for k in (1,2,4,8)] for a in configs["circuit"]["arms"]]),""]
    lines += [table(["Arm minus restart","K","Difference (pp)","Paired 95% interval (pp)"],
        [[r["arm"],r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in depth["contrasts"]]),"",
        "The positive answer-only criterion under structural shift is " + ("met at one or more tested budgets; this supports the second family specifically under this structural shift." if depth["gate"] else "not met; the positive output-reuse claim remains restricted to mazes."),""]
    lines += ["Depth-shift accuracy by training seed, using the frozen ordinary-validation grid:", "",
        seed_accuracy_table([r for r in depth["curves"] if r["primary"]],configs["circuit"]["arms"],[1,2,4,8],configs["circuit"]["seeds"]), ""]
    lines += ["The following spatial_gate minus answer_only contrasts were requested after the sweep began. They are exploratory, were not declared depth-shift contrasts, and do not reopen G2.", "",
        table(["Exploratory contrast","K","Difference (pp)","Paired 95% interval (pp)"],
            [["spatial_gate minus answer_only",r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in depth["exploratory_contrasts"]]), "",
        "Regime-dependent observation: answer-only beats the tested latent-reuse arms on in-distribution mazes; " +
        ("the learned gate's mean accuracy matches or exceeds answer-only at every tested K under circuit structural shift." if all(r["delta"] >= 0 for r in depth["exploratory_contrasts"]) else
         "the learned gate's comparison with answer-only under circuit structural shift depends on K, as shown above.") +
        " This does not establish prespecified noninferiority or change the declared G2 decision.", ""]
    lines += [f"![Depth-shift circuit accuracy and restart drift reference, K{k}](figures/prompt05b/circuit-depth-decay-K{k}.svg)\n" for k in (1,2,4,8)]
    lines += ["### Matched forward-call control", "",
        "The original one-edit recipe was rerun from the common pretrained checkpoint for 626 updates, with the pilot-selected grid fixed per policy. The common step count was chosen from deterministic schedules, before training, to minimize the largest call-count mismatch across seeds. It matches total training solver forward calls within 5%; it does not equate backward-pass cost or wall time. No learning-rate selection used these addendum results.", "",
        table(["Policy","Seed","Steps","One-edit F calls","Stream F calls","Difference (%)"],
            [[r["arm"],r["seed"],r["steps"],r["example_F_calls"],r["stream_example_F_calls"],percent(r["relative_difference"])] for r in longer["training_matches"]]),"",
        table(["Policy","K1 (%)","K2 (%)","K4 (%)","K8 (%)"],
            [[a]+[percent(np.mean([r["post32"] for r in longer["curves"] if r["arm"] == a and r["K"] == k])) for k in (1,2,4,8)] for a in configs["circuit"]["arms"]]),"",
        table(["Longer policy minus stream spatial gate","K","Difference (pp)","Paired 95% interval (pp)"],
            [[r["arm"],r["K"],percent(r["delta"]),f"[{percent(r['ci'][0])}, {percent(r['ci'][1])}]"] for r in longer["contrasts"]]),"",
        "Declared operational attribution: " + ("compute-compatible: the matched-compute spatial gate's differences include zero at both K1 and K4." if longer["gate"] else "stream exposure: the matched-compute spatial gate does not satisfy the zero-containing paired-interval criterion at both K1 and K4."),"",
        "That attribution is the requested screening rule, not a causal or equivalence proof. Failure to detect a difference does not establish equal performance, and a difference can reflect optimization or sample weighting as well as exposure to carried states. The original latent-reuse-versus-answer-only G2 result remains separately reported.",""]
    lines += ["Matched-call control accuracy by training seed:", "",
        seed_accuracy_table(longer["curves"],configs["circuit"]["arms"],[1,2,4,8],configs["maze"]["seeds"]), ""]
    circuit_pass = all(r["ci"][0] > 0 for r in circuit_check["contrasts"] if r["arm"] == "answer_only")
    lines += ["### Scope and resources", "",
        f"Ordinary circuit output reuse {'replicates a positive answer-only minus restart interval at both K1 and K2.' if circuit_pass else 'does not establish positive answer-only minus restart intervals at both tested budgets. Restart is near ceiling, limiting headroom; the separate depth-shift result above defines whether there is support under structural shift.'}", "",
        f"Stream-training GPU-job wall time: {data['resources']['gpu_job_hours']:.4f} hours across {len(actual)} reconciled jobs, under the separate 12-hour allowance. This includes {len(data['resources']['failed_jobs'])} failed jobs, whose errors and synthetic labels remain in the resource records. Peak allocated GPU memory was {data['resources']['peak_allocated_gpu_bytes']/2**30:.3f} GiB; peak reserved memory was {data['resources']['peak_reserved_gpu_bytes']/2**30:.3f} GiB. External charges $0; electricity unmeasured. Job wall time includes CPU work inside GPU contexts. Retention values and K counts do not establish a speedup; cost comparisons are reported separately in the confirmatory study.","",
        "Only one pretrained source checkpoint per family, three adaptation seeds, 256 stream-training updates (626 one-edit updates in the matched-call control), short training streams and exploratory pointwise validation intervals were studied. This does not establish convergence, broad task generality, arbitrary-K stability or a causal latent-repair mechanism. The per-frame restart curves are the distribution-drift reference. All arms, failures, full sequences, source snapshots and raw actions are preserved.",""]
    report = Path("reports/PILOT_REPORT.md")
    original = report.read_text(encoding="utf-8").split("## Prompt 05b: stream training and output reuse")[0]
    original = original.split("## Stream training and output reuse")[0].rstrip()
    report.write_text(original+"\n\n"+"\n".join(lines),encoding="utf-8")
    write(Path("reports/stream_results.json"),data)


if __name__ == "__main__":
    main()
