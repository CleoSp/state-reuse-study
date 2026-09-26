"""Tables and cost curves generated only from sealed, rescored records."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from state_repair.execution.durable import atomic_json, atomic_write, read_json
from .jobs import WorkUnit
from .statistics import paired_contrast, paired_cost_ratio


def summarize_curves(episodes: list[dict], *, empirical: bool = True) -> list[dict]:
    grouped = defaultdict(list)
    for row in episodes:
        if type(row.get("synthetic")) is not bool or (empirical and row["synthetic"]):
            raise ValueError("empirical report rejects synthetic records")
        grouped[row["suite"], row["record_kind"], row["policy"], row["K"]].append(row)
    return [{"suite": suite, "record_kind": mode, "policy": policy, "K": k,
        "post_accuracy": sum(r["post_accuracy"] for r in rows)/len(rows),
        "amortized_ms": sum(r["amortized_ms"] for r in rows)/len(rows),
        "whole_stream_success": sum(r["whole_stream_success"] for r in rows)/len(rows),
        "node_accuracy": sum(r["node_accuracy"] for r in rows)/len(rows),
        "all_node_accuracy": sum(r["all_node_accuracy"] for r in rows)/len(rows),
        "unreachable_cases": sum(r["unreachable_count"] for r in rows),
        "unreachable_correct": sum(r["unreachable_correct"] for r in rows),
        "initial_ms": sum(r["initial_ms"] for r in rows)/len(rows),
        "update_ms": sum(r["update_ms"] for r in rows)/len(rows),
        "amortized_macs": sum(r["amortized_macs"] for r in rows)/len(rows),
        "stage_ms": {s: sum(r["stage_ms"][s] for r in rows)/len(rows) for s in rows[0]["stage_ms"]},
        "per_seed_accuracy": {s: sum(r["post_accuracy"] for r in rows if r["seed"] == s)/sum(r["seed"] == s for r in rows) for s in sorted({r["seed"] for r in rows})},
        "mean_cumulative_errors": [sum(r["cumulative_errors"][i] for r in rows)/len(rows) for i in range(len(rows[0]["cumulative_errors"]))],
        "roots": len({r["root_id"] for r in rows}), "seeds": sorted({r["seed"] for r in rows})}
        for (suite, mode, policy, k), rows in sorted(grouped.items())]


class ReportJob(WorkUnit):
    def __init__(self, job: dict, root: Path):
        super().__init__()
        self.job, self.root = job, root

    def step(self, index: int, batch: int, k: int) -> dict:
        from .checker import check_job
        from state_repair.execution.driver import verify
        if self.job.get("mechanism"):
            return mechanism_report(self.job, self.root)
        episodes, checks = [], []
        for name in self.job["evaluations"]:
            out = self.root / name
            job = read_json(out / "job.json")
            verify(out, job)
            checked = check_job(out, job)
            episodes.extend(checked.pop("episodes"))
            checks.append({"job": name, **checked})
        spatial_costs = {(r["suite"], r["seed"], r["root_id"], r["K"]): r for r in episodes if r["policy"] == "spatial_gate"}
        from state_repair.train.pilot import AUXILIARY
        for row in episodes:
            if row["policy"] in AUXILIARY:
                source = spatial_costs[row["suite"], row["seed"], row["root_id"], row["K"]]
                for metric in ("amortized_ms", "amortized_macs", "initial_ms", "update_ms", "stage_ms"):
                    row[metric] = source[metric]
        curves = summarize_curves(episodes, empirical=not self.job["synthetic"])
        episodes = collapse_repetitions(episodes)
        comparisons = []
        for suite, k in sorted({(r["suite"], r["K"]) for r in episodes if r["policy"] == "restart"}):
            baseline = [r for r in episodes if r["suite"] == suite and r["K"] == k and r["policy"] == "restart"]
            for policy in sorted({r["policy"] for r in episodes if r["suite"] == suite and r["K"] == k and r["record_kind"] == "fixed_budget_stream"} - {"restart"}):
                current = [r for r in episodes if r["suite"] == suite and r["K"] == k and r["policy"] == policy]
                paired_baseline = [r for r in baseline if r["seed"] in {v["seed"] for v in current}]
                comparisons.append({"suite": suite, "K": k, "policy": policy,
                    "accuracy": paired_contrast(current, paired_baseline, empirical=not self.job["synthetic"],
                                      repetitions=self.job.get("bootstrap_repetitions", 10000)),
                    "cost": paired_cost_ratio(current, paired_baseline, empirical=not self.job["synthetic"],
                                      repetitions=self.job.get("bootstrap_repetitions", 10000))})
        operating = []
        for suite in sorted({r["suite"] for r in episodes}):
            point = self.job.get("operating_point")
            if point:
                left = [r for r in episodes if r["suite"] == suite and r["policy"] == point["reuse"]["policy"] and r["K"] == point["reuse"]["K"]]
                right = [r for r in episodes if r["suite"] == suite and r["policy"] == point["comparator"]["policy"] and r["K"] == point["comparator"]["K"]]
                if left and right:
                    operating.append({"suite": suite, "frozen_point": point,
                        "accuracy": paired_contrast(left, right, empirical=not self.job["synthetic"], repetitions=self.job.get("bootstrap_repetitions", 10000)),
                        "cost": paired_cost_ratio(left, right, empirical=not self.job["synthetic"], repetitions=self.job.get("bootstrap_repetitions", 10000))})
        return {"loss": 0., "forward_calls": 0, "curves": curves, "contrasts": comparisons, "operating_points": operating, "checks": checks}


def collapse_repetitions(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["suite"], row["policy"], row["K"], row["seed"], row["root_id"]].append(row)
    return [{**values[0], "post_accuracy": sum(r["post_accuracy"] for r in values)/len(values),
             "amortized_ms": sum(r["amortized_ms"] for r in values)/len(values)} for values in grouped.values()]


def materialize_report(output: Path) -> None:
    """Crash-safe, regenerable JSON, Markdown and ReportLab scientific SVG."""
    import json
    import sys
    row = json.loads((output / "steps.jsonl").read_text())
    value = {k: row[k] for k in ("curves", "contrasts", "operating_points", "checks")}
    if "mechanism" in row:
        value["mechanism"] = row["mechanism"]
    atomic_json(output / "results.json", value)
    lines = ["# Confirmatory records", "", "Generated from saved actions; reference solvers are separately labeled. Intervals are pointwise unless the frozen protocol says otherwise.", "",
        "| Mode | Policy | K | Post-edit exact | Whole stream | Initial ms | Amortized ms | MACs |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in value["curves"]:
        lines.append(f'| {r["record_kind"]} | {r["policy"]} | {r["K"]} | {r["post_accuracy"]:.6f} | {r["whole_stream_success"]:.6f} | {r["initial_ms"]:.4f} | {r["amortized_ms"]:.4f} | {r["amortized_macs"]:.0f} |')
    lines += ["", "Full stage costs, error accumulation, node/valid-action scores, unreachable denominators, seed estimates, paired accuracy intervals and cost-ratio intervals are in results.json.",
        "Auxiliary-control cost is assigned the spatial gate cost under the fixed measurement design; no independent auxiliary latency claim is made."]
    atomic_write(output / "results.md", lambda h: h.write(("\n".join(lines)+"\n").encode()))
    if not value["curves"]:
        return
    try:
        import reportlab
    except ImportError:
        package = read_json(output / "job.json").get("report_packages")
        if not package:
            raise RuntimeError("frozen local ReportLab dependency path is missing")
        sys.path.append(package)
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics import renderSVG
    from reportlab.lib import colors
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 70, 65, 650, 340
    policies = sorted({r["policy"] for r in value["curves"]})
    plot.data = [sorted((r["amortized_ms"], r["post_accuracy"]) for r in value["curves"] if r["policy"] == p) for p in policies]
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = 0, 1
    drawing = Drawing(1050, 455)
    drawing.add(plot)
    drawing.add(String(70, 430, "Accuracy versus measured amortized cost (initial solve included)", fontSize=14))
    drawing.add(String(280, 20, "Milliseconds per frame", fontSize=12))
    palette = [colors.red, colors.blue, colors.green, colors.purple, colors.orange, colors.brown, colors.black]
    for i, policy in enumerate(policies):
        plot.lines[i].strokeColor = palette[i % len(palette)]
        drawing.add(String(745, 400-i*19, policy, fillColor=palette[i % len(palette)], fontSize=10))
    svg = renderSVG.drawToString(drawing)
    atomic_write(output / "accuracy-cost.svg", lambda h: h.write(svg.encode() if isinstance(svg, str) else svg))


def mechanism_report(job: dict, root: Path) -> dict:
    from .checker import check_job
    from .jobs import saved_records
    from state_repair.execution.driver import verify
    grouped, checks = defaultdict(list), []
    for name in job["evaluations"]:
        config = read_json(root / name / "job.json")
        verify(root / name, config)
        checks.append(check_job(root / name, config))
        for row in saved_records(root / name):
            grouped[row["suite"], row["record_kind"], row["policy"], row["K"], row["stratum"]].append(row)
    result = []
    for (suite, mode, policy, k, stratum), rows in sorted(grouped.items()):
        roots = defaultdict(list)
        for row in rows:
            roots[row["root_id"]].append(row)
        result.append({"suite": suite, "mode": mode, "policy": policy, "K": k, "stratum": stratum,
            "independent_roots": len(roots), "branches_times_seeds": len(rows),
            "accuracy_root_weighted": sum(sum(r["exact_correct"] for r in v)/len(v) for v in roots.values())/len(roots),
            "exploratory": True, "privileged": mode == "privileged_diagnostic",
            "mean_state_distance_a": sum(r["state_distance_from_prior"]["a"] for r in rows)/len(rows),
            "mean_state_distance_z": sum(r["state_distance_from_prior"]["z"] for r in rows)/len(rows)})
        if "changed_action_fraction" in rows[0]:
            result[-1]["changed_action_fraction"] = sum(r["changed_action_fraction"] for r in rows)/len(rows)
    return {"loss": 0., "forward_calls": 0, "curves": [], "contrasts": [], "operating_points": [], "checks": checks, "mechanism": result}
