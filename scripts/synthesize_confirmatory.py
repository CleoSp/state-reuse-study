"""Regenerate every confirmatory table and figure from sealed records only.

Inputs are the driver-sealed report, evaluation, lesion, reference and ledger
artifacts under runs/confirmatory_v1. Every report directory is re-verified
against its seal before its row is read; raw evaluation logs are re-verified
before their records are aggregated. Synthetic records are rejected. Nothing
here trains, evaluates or selects; the frozen operating points come from the
bound protocol config. Outputs go to reports/confirmatory.

Usage (CPU, from the repository root):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/synthesize_confirmatory.py
Optional: --no-raw skips the raw-record pass (exploratory contrasts and
per-frame curves are then omitted); --cache DIR caches per-job episode rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_repair.eval.jobs import saved_records  # noqa: E402
from state_repair.eval.metrics import episodes  # noqa: E402
from state_repair.eval.statistics import paired_contrast  # noqa: E402
from state_repair.execution.driver import verify  # noqa: E402
from state_repair.execution.durable import atomic_json, atomic_write, read_json  # noqa: E402
from state_repair.execution.records import raw_step_hash, step_rows, stored_steps  # noqa: E402
from state_repair.provenance import file_hash  # noqa: E402

RUNS = ROOT / "runs" / "confirmatory_v1"
OUT = ROOT / "reports" / "confirmatory"
CONFIG = ROOT / "configs" / "confirmatory_tier2_frozen.json"
PROTOCOL = ROOT / "reports" / "FROZEN_PROTOCOL.md"

PRINCIPAL = ["restart", "carry", "spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only"]
LESIONS = ["uniform/answer_only", "shuffled_nodes/answer_only"]
AUXILIARY = ["local_reset_1", "local_reset_2", "local_reset_3", "random_reset", "noisy_carry", "shuffled_gate"]
ONE_EDIT = ["one_edit/restart", "one_edit/carry", "one_edit/spatial_gate", "one_edit/answer_only"]
SUITES = ["maze12-on-maze12", "maze16-on-maze16", "maze12-on-maze20", "maze16-on-maze20",
          "circuit32-on-circuit32", "circuit32-on-circuit48", "circuit64-on-circuit64", "circuit64-on-circuit96"]
PRIMARY = ["maze12-on-maze12", "circuit32-on-circuit32", "circuit32-on-circuit48"]
TRAINED = ["maze12-on-maze12", "maze16-on-maze16", "circuit32-on-circuit32", "circuit64-on-circuit64"]
RAW_SUITES = ["maze12-on-maze12", "maze16-on-maze16", "circuit32-on-circuit48", "circuit64-on-circuit96"]
KS = [1, 2, 4, 8, 16]

TRANSFERRED = {"global_gate", "gru_adapter", "residual_adapter", "answer_only", "answer_uniform", "answer_shuffled_nodes"}
MARGIN, SAVING_TARGET = 0.01, 0.75
COLORS = {"restart": "#1B365D", "carry": "#C25431", "spatial_gate": "#00806C", "global_gate": "#8264A8",
          "gru_adapter": "#B8860B", "residual_adapter": "#7A7A7A", "answer_only": "#D81B60",
          "uniform/answer_only": "#E58F65", "shuffled_nodes/answer_only": "#4C9BE8", "reference": "#000000",
          "one_edit/restart": "#1B365D", "one_edit/carry": "#C25431", "one_edit/spatial_gate": "#00806C",
          "one_edit/answer_only": "#D81B60", "shuffled_gate": "#4C9BE8", "impact_mask": "#000000"}





def load_report(name: str) -> dict:
    output = RUNS / name
    job = read_json(output / "job.json")
    verify(output, job)
    row = next(iter(step_rows(output)))
    if job["synthetic"] or row["synthetic"]:
        raise ValueError(f"synthetic report rejected: {name}")
    return {"name": name, "job": job, "row": row, "records_sha256": raw_step_hash(output),
            "stored": stored_steps(output).name}


def report_name(suite: str, kind: str) -> str:
    return {"h32": f"report-{suite}-h32", "h128": f"report-{suite}-h128", "latency": f"report-latency-{suite}",
            "throughput": f"report-throughput-{suite}", "mechanism": f"report-mechanism-{suite}"}[kind]


def curve_index(row: dict) -> dict:
    return {(c["policy"], c["K"]): c for c in row["curves"]}


def contrast_index(row: dict) -> dict:
    return {(c["policy"], c["K"]): c for c in row["contrasts"]}





def decide(op: dict) -> dict:
    acc, cost = op["accuracy"], op["cost"]
    noninferior = acc["root_ci"][0] > -MARGIN
    noninferior_crossed = acc["crossed_ci"][0] > -MARGIN
    saving = cost["root_ci"][1] < 1
    target = cost["root_ci"][1] <= SAVING_TARGET
    return {"suite": op["suite"], "reuse": op["frozen_point"]["reuse"], "comparator": op["frozen_point"]["comparator"],
            "accuracy_delta": acc["delta"], "accuracy_root_ci": acc["root_ci"], "accuracy_crossed_ci": acc["crossed_ci"],
            "per_seed_delta": acc["per_seed_delta"], "training_seed_sd": acc["training_seed_sd"], "root_sd": acc["root_sd"],
            "cost_ratio": cost["ratio"], "cost_root_ci": cost["root_ci"], "cost_crossed_ci": cost["crossed_ci"],
            "per_seed_ratio": cost["per_seed_ratio"], "roots": acc["roots"], "seeds": acc["seeds"],
            "noninferior_root": noninferior, "noninferior_crossed": noninferior_crossed,
            "measured_saving": saving, "target_saving_25pct": target,
            "claim_met": noninferior and target}





def evaluation_jobs(suite: str, edits: int = 32) -> list[Path]:
    result = []
    for prefix in ("test-", "lesion-"):
        for path in sorted(RUNS.glob(f"{prefix}{suite}-*")):
            if not path.is_dir() or ".incomplete-" in path.name:
                continue
            job = read_json(path / "job.json")
            if job["suite"] != suite or job["edits"] != edits or job["kind"] != "evaluation":
                continue
            result.append(path)
    return result


def job_episodes(path: Path, cache: Path | None) -> list[dict]:
    job = read_json(path / "job.json")
    verify(path, job)
    digest = raw_step_hash(path)
    cached = cache / f"{path.name}.json" if cache else None
    if cached and cached.exists():
        value = read_json(cached)
        if value["records_sha256"] == digest:
            return value["episodes"]
    rows = episodes(list(saved_records(path)), job["edits"], empirical=not job["synthetic"])
    for row in rows:
        row.pop("stage_ms", None)
    if cached:
        atomic_json(cached, {"records_sha256": digest, "episodes": rows})
    return rows


def raw_pass(suites: list[str], cache: Path | None, log) -> dict:
    result = {}
    for suite in suites:
        started = time.perf_counter()
        pool: dict[str, list[dict]] = defaultdict(list)
        jobs = evaluation_jobs(suite)
        for path in jobs:
            for row in job_episodes(path, cache):
                pool[row["policy"]].append(row)
        contrasts, frames = [], {}
        base = pool.get("answer_only", [])
        for policy, rows in sorted(pool.items()):
            seeds = sorted({r["seed"] for r in rows})
            for k in KS:
                current = [r for r in rows if r["K"] == k]
                if not current:
                    continue
                per_frame = [1 - sum(r["errors_by_frame"][i] for r in current) / len(current) for i in range(32)]
                frames[(policy, k)] = per_frame
                if policy != "answer_only" and base:
                    reference = [r for r in base if r["K"] == k and r["seed"] in seeds]
                    contrasts.append({"suite": suite, "K": k, "policy": policy, "comparator": "answer_only",
                                      "exploratory": True, **paired_contrast(current, reference)})
        result[suite] = {"jobs": [p.name for p in jobs], "contrasts": contrasts,
                         "per_frame": {f"{p}|K{k}": v for (p, k), v in frames.items()}}
        log(f"raw pass {suite}: {len(jobs)} jobs, {len(contrasts)} exploratory contrasts, {time.perf_counter()-started:.1f}s")
    return result





def ledger() -> dict:
    rows = [json.loads(line) for line in (RUNS / "gpu_time.jsonl").read_text().splitlines() if line.strip()]
    actual = [r for r in rows if r["kind"] == "actual"]
    reserve = [r for r in rows if r["kind"] == "reserve"]
    if len(actual) != len(reserve):
        raise ValueError("unbalanced ledger")
    by_device = defaultdict(float)
    for r in actual:
        by_device[r["device"]] += r["wall_seconds"] / 3600
    aborted = [{"job": r["matrix_job"], "wall_seconds": r["wall_seconds"], "error": r["error"]} for r in actual if r["aborted"] or r["error"]]
    complete = read_json(RUNS / "complete.json")
    quarantined = sorted(p.name for p in RUNS.glob("*.incomplete-*"))
    startup = [json.loads(l) for l in (RUNS / "startup.jsonl").read_text().splitlines() if l.strip()]
    return {"attempts": len(actual), "empirical_attempts": sum(1 for r in actual if not r["synthetic"]),
            "gpu_hours": by_device.get("cuda", 0.0), "cpu_job_hours": by_device.get("cpu", 0.0),
            "total_hours": sum(by_device.values()), "aborted_attempts": aborted,
            "jobs": complete["jobs"], "matrix_complete": complete["matrix_complete"], "matrix_sha256": complete["matrix_sha256"],
            "quarantined_directories": quarantined, "startup_events": len(startup),
            "authorization_gpu_hours": read_json(CONFIG)["authorization_gpu_hours"]}





def _fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.ceil(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 1)]


def line_chart(path: Path, series: list[dict], title: str, subtitle: str, xlabel: str, ylabel: str, *,
               xlog: bool = False, ylim=(0, 1), width=760, height=430) -> None:
    left, right, top, bottom = 64, 200, 58, 56
    pw, ph = width - left - right, height - top - bottom
    xs = [x for s in series for x, _ in s["points"]]
    fx = (lambda v: math.log2(v)) if xlog else (lambda v: v)
    xlo, xhi = min(map(fx, xs)), max(map(fx, xs))
    if xhi == xlo:
        xhi = xlo + 1
    ylo, yhi = ylim
    def X(v): return left + (fx(v) - xlo) / (xhi - xlo) * pw
    def Y(v): return top + ph - (v - ylo) / (yhi - ylo) * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="15" font-weight="bold" fill="#1B365D">{title}</text>',
           f'<text x="{left}" y="42" font-size="10" fill="#333">{subtitle}</text>',
           f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
    for t in _ticks(ylo, yhi):
        y = Y(t)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+pw}" y2="{y:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{left-6}" y="{y+3.5:.1f}" font-size="9" text-anchor="end">{_fmt(t)}</text>')
    xticks = sorted(set(xs)) if xlog else _ticks(min(xs), max(xs))
    for t in xticks:
        x = X(t)
        out.append(f'<line x1="{x:.1f}" y1="{top+ph}" x2="{x:.1f}" y2="{top+ph+4}" stroke="#999"/>')
        out.append(f'<text x="{x:.1f}" y="{top+ph+15}" font-size="9" text-anchor="middle">{_fmt(t)}</text>')
    out.append(f'<text x="{left+pw/2:.1f}" y="{height-14}" font-size="10" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="16" y="{top+ph/2:.1f}" font-size="10" text-anchor="middle" transform="rotate(-90 16 {top+ph/2:.1f})">{ylabel}</text>')
    for i, s in enumerate(series):
        color = s.get("color", "#000")
        pts = sorted(s["points"])
        if len(pts) > 1 and not s.get("marker_only"):
            d = " ".join(f'{"M" if j == 0 else "L"}{X(x):.1f},{Y(y):.1f}' for j, (x, y) in enumerate(pts))
            out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{s.get("width", 1.8)}" stroke-dasharray="{s.get("dash", "")}"/>')
        for x, y in pts:
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="{s.get("r", 2.6)}" fill="{color}"/>')
        ly = top + 12 + i * 16
        out.append(f'<line x1="{left+pw+14}" y1="{ly}" x2="{left+pw+34}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{left+pw+40}" y="{ly+3.5}" font-size="9">{s["label"]}</text>')
    out.append("</svg>\n")
    atomic_write(path, lambda h: h.write("\n".join(out).encode()))


def stacked_bars(path: Path, groups: list[dict], stages: list[str], title: str, subtitle: str, ylabel: str,
                 width=760, height=430) -> None:
    left, right, top, bottom = 64, 190, 58, 70
    pw, ph = width - left - right, height - top - bottom
    palette = ["#1B365D", "#C25431", "#00806C", "#8264A8", "#B8860B", "#7A7A7A", "#D81B60", "#4C9BE8", "#E58F65"]
    total = max(sum(g["values"][s] for s in stages) for g in groups)
    yticks = _ticks(0, total)
    yhi = yticks[-1] if yticks[-1] >= total else total
    def Y(v): return top + ph - v / yhi * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="15" font-weight="bold" fill="#1B365D">{title}</text>',
           f'<text x="{left}" y="42" font-size="10" fill="#333">{subtitle}</text>',
           f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
    for t in yticks:
        out.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{left+pw}" y2="{Y(t):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{left-6}" y="{Y(t)+3.5:.1f}" font-size="9" text-anchor="end">{_fmt(t)}</text>')
    slot = pw / len(groups)
    for gi, g in enumerate(groups):
        x0 = left + gi * slot + slot * 0.2
        bw = slot * 0.6
        base = 0.0
        for si, s in enumerate(stages):
            v = g["values"][s]
            out.append(f'<rect x="{x0:.1f}" y="{Y(base+v):.1f}" width="{bw:.1f}" height="{Y(base)-Y(base+v):.1f}" fill="{palette[si % len(palette)]}"/>')
            base += v
        out.append(f'<text x="{x0+bw/2:.1f}" y="{top+ph+14}" font-size="9" text-anchor="middle">{g["label"]}</text>')
    for si, s in enumerate(stages):
        ly = top + 12 + si * 16
        out.append(f'<rect x="{left+pw+14}" y="{ly-6}" width="12" height="12" fill="{palette[si % len(palette)]}"/>')
        out.append(f'<text x="{left+pw+32}" y="{ly+3.5}" font-size="9">{s}</text>')
    out.append(f'<text x="16" y="{top+ph/2:.1f}" font-size="10" text-anchor="middle" transform="rotate(-90 16 {top+ph/2:.1f})">{ylabel}</text>')
    out.append("</svg>\n")
    atomic_write(path, lambda h: h.write("\n".join(out).encode()))


def grouped_bars(path: Path, categories: list[str], series: list[dict], title: str, subtitle: str, ylabel: str,
                 width=760, height=430) -> None:
    left, right, top, bottom = 64, 190, 58, 70
    pw, ph = width - left - right, height - top - bottom
    def Y(v): return top + ph - v * ph
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
           f'<text x="{left}" y="24" font-size="15" font-weight="bold" fill="#1B365D">{title}</text>',
           f'<text x="{left}" y="42" font-size="10" fill="#333">{subtitle}</text>',
           f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>']
    for t in (0, .2, .4, .6, .8, 1):
        out.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{left+pw}" y2="{Y(t):.1f}" stroke="#eee"/>')
        out.append(f'<text x="{left-6}" y="{Y(t)+3.5:.1f}" font-size="9" text-anchor="end">{_fmt(t)}</text>')
    slot = pw / len(categories)
    bw = slot * 0.8 / len(series)
    for ci, c in enumerate(categories):
        for si, s in enumerate(series):
            v = s["values"].get(c)
            if v is None:
                continue
            x = left + ci * slot + slot * 0.1 + si * bw
            out.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{Y(0)-Y(v):.1f}" fill="{s["color"]}"/>')
        out.append(f'<text x="{left+ci*slot+slot/2:.1f}" y="{top+ph+14}" font-size="9" text-anchor="middle">{c}</text>')
    for si, s in enumerate(series):
        ly = top + 12 + si * 16
        out.append(f'<rect x="{left+pw+14}" y="{ly-6}" width="12" height="12" fill="{s["color"]}"/>')
        out.append(f'<text x="{left+pw+32}" y="{ly+3.5}" font-size="9">{s["label"]}</text>')
    out.append(f'<text x="16" y="{top+ph/2:.1f}" font-size="10" text-anchor="middle" transform="rotate(-90 16 {top+ph/2:.1f})">{ylabel}</text>')
    out.append("</svg>\n")
    atomic_write(path, lambda h: h.write("\n".join(out).encode()))





def pct(v: float) -> str:
    return f"{100*v:.2f}"


def pp(v: float) -> str:
    return f"{100*v:+.2f}"


def ci(v: list[float], scale: float = 100, sign: bool = True) -> str:
    a, b = v
    f = (lambda x: f"{scale*x:+.2f}") if sign else (lambda x: f"{scale*x:.3f}")
    return f"[{f(a)}, {f(b)}]"


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("---|" + "---:|" * (len(header) - 1))
    return "\n".join(["| " + " | ".join(header) + " |", "|" + align] + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"





def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs" / "synthesis_cache")
    parser.add_argument("--raw-suites", nargs="*", default=RAW_SUITES)
    args = parser.parse_args()
    started = time.perf_counter()
    log = lambda m: print(m, flush=True)

    config = read_json(CONFIG)
    if config["status"] != "FROZEN" or config["launch_authorized"] is not True:
        raise ValueError("expected the frozen, launched confirmatory config")
    protocol_hash = file_hash(PROTOCOL)
    manifest_hash = read_json(RUNS / "data" / "test-manifest.json")["protocol_sha256"]
    if manifest_hash != protocol_hash:
        raise ValueError("FROZEN_PROTOCOL.md does not match the hash bound at test generation")

    reports = {}
    for suite in SUITES:
        for kind in ("h32", "latency", "throughput", "mechanism"):
            if kind in ("latency", "throughput") and suite == "maze16-on-maze20":
                continue
            reports[(suite, kind)] = load_report(report_name(suite, kind))
        if suite in TRAINED:
            reports[(suite, "h128")] = load_report(report_name(suite, "h128"))
    log(f"verified {len(reports)} sealed report jobs")

    decisions = {s: decide(reports[(s, "h32")]["row"]["operating_points"][0]) for s in SUITES}
    latency_points = {s: decide(reports[(s, "latency")]["row"]["operating_points"][0]) for s in SUITES if (s, "latency") in reports}
    headline = all(decisions[s]["claim_met"] for s in PRIMARY)
    all_noninferior = all(decisions[s]["noninferior_root"] for s in PRIMARY)

    raw = {} if args.no_raw else raw_pass(args.raw_suites, args.cache, log)
    led = ledger()

    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)


    written = []
    for suite in SUITES:
        row = reports[(suite, "h32")]["row"]
        idx = curve_index(row)
        ref = next(c for c in row["curves"] if c["record_kind"] == "reference_solver")
        series = [{"label": p, "color": COLORS[p], "points": [(idx[(p, k)]["amortized_ms"], idx[(p, k)]["post_accuracy"]) for k in KS if (p, k) in idx]} for p in PRINCIPAL if (p, 1) in idx]
        series.append({"label": f"{ref['policy']} (CPU reference)", "color": COLORS["reference"], "points": [(ref["amortized_ms"], ref["post_accuracy"])], "marker_only": True, "r": 4})
        name = f"accuracy-cost-{suite}.svg"
        line_chart(figures / name, series, f"{suite}: post-edit exact accuracy versus amortized cost",
                   "Batch-64 timing from the accuracy jobs; initial solve included; K in 1,2,4,8,16 left to right; five seeds, 256 test roots.",
                   "Amortized milliseconds per frame (batch 64)", "Mean post-edit exact accuracy")
        written.append(name)
        series = [{"label": p, "color": COLORS[p], "points": [(k, idx[(p, k)]["post_accuracy"]) for k in KS if (p, k) in idx]} for p in PRINCIPAL if (p, 1) in idx]
        name = f"accuracy-K-{suite}.svg"
        line_chart(figures / name, series, f"{suite}: post-edit exact accuracy by fixed budget K",
                   "32-edit streams, each K carrying its own state; five seeds averaged; 256 test roots.", "Outer cycles K (log scale)", "Mean post-edit exact accuracy", xlog=True)
        written.append(name)
        lesion_series = [{"label": p, "color": COLORS[p], "points": [(k, idx[(p, k)]["post_accuracy"]) for k in KS if (p, k) in idx]} for p in ("restart", "answer_only", *LESIONS)]
        name = f"lesion-{suite}.svg"
        line_chart(figures / name, lesion_series, f"{suite}: answer-content lesions",
                   "uniform/shuffled use the answer-only checkpoints of seeds 29/43/71 with lesioned previous probabilities; restart and answer-only use five seeds.",
                   "Outer cycles K (log scale)", "Mean post-edit exact accuracy", xlog=True)
        written.append(name)
        if suite in TRAINED:
            idx128 = curve_index(reports[(suite, "h128")]["row"])
            series = []
            for p in ("restart", "carry", "spatial_gate", "answer_only"):
                series.append({"label": f"{p} 32 edits", "color": COLORS[p], "points": [(k, idx[(p, k)]["post_accuracy"]) for k in KS]})
                series.append({"label": f"{p} 128 edits", "color": COLORS[p], "dash": "5,3", "points": [(k, idx128[(p, k)]["post_accuracy"]) for k in KS]})
            name = f"horizon-{suite}.svg"
            line_chart(figures / name, series, f"{suite}: 32-edit versus 128-edit streams",
                       "Mean post-edit exact accuracy over the whole stream; dashed lines are the 128-edit streams; five seeds.", "Outer cycles K (log scale)", "Mean post-edit exact accuracy", xlog=True)
            written.append(name)
        if (suite, "latency") in reports:
            lidx = curve_index(reports[(suite, "latency")]["row"])
            stages = ["edit_and_validation", "encoder", "fresh", "adapter", "core", "decoder", "copy", "transfer_in", "transfer_out"]
            groups = [{"label": f"{p} K{k}", "values": lidx[(p, k)]["stage_ms"]} for k in (1, 2, 8) for p in ("restart", "answer_only", "spatial_gate")]
            name = f"latency-stages-{suite}.svg"
            stacked_bars(figures / name, groups, stages, f"{suite}: batch-one latency by stage",
                         "Seed-29 checkpoints, 8 test roots, 3 warmed repetitions, device-synchronized; amortized over the initial solve and 32 edits.", "Milliseconds per frame (batch 1)")
            written.append(name)
        mech = reports[(suite, "mechanism")]["row"]["mechanism"]
        for k in (1, 8):
            cats = ["low", "middle", "high"]
            pol = ["restart", "carry", "spatial_gate", "global_gate", "shuffled_gate", "impact_mask"]
            series = [{"label": p + (" (privileged)" if p == "impact_mask" else " (transferred)" if p in TRANSFERRED else ""), "color": COLORS[p],
                       "values": {r["stratum"]: r["accuracy_root_weighted"] for r in mech if r["policy"] == p and r["K"] == k}} for p in pol]
            name = f"mechanism-{suite}-K{k}.svg"
            grouped_bars(figures / name, cats, series, f"{suite}: frozen-prior interventions at K={k}",
                         "Identical saved spatial-backbone priors forked into each initializer; root-weighted exact accuracy by evaluator-only impact stratum; three seeds.", "Exact accuracy after one edit")
            written.append(name)
    for suite, value in raw.items():
        for k in (1, 8):
            series = [{"label": p, "color": COLORS[p], "points": [(f + 1, a) for f, a in enumerate(value["per_frame"][f"{p}|K{k}"])], "r": 1.6}
                      for p in ("restart", "carry", "spatial_gate", "answer_only") if f"{p}|K{k}" in value["per_frame"]]
            name = f"per-frame-{suite}-K{k}.svg"
            line_chart(figures / name, series, f"{suite}: exact accuracy at each edit, K={k}",
                       "Frame-level accuracy over 256 roots and five seeds; restart is the drift reference for the edit process itself.", "Edit index (frame)", "Exact accuracy at this frame")
            written.append(name)


    summary = {"synthetic": False, "generated_by": "scripts/synthesize_confirmatory.py",
               "protocol_sha256": protocol_hash, "matrix_sha256": config["artifact_hashes"].get("matrix") if isinstance(config.get("artifact_hashes"), dict) else None,
               "frozen_operating_points": config["operating_points"], "decision_rule": {
                   "noninferiority": "paired-root 95% lower bound of (reuse - comparator) post-edit exact accuracy above -0.01",
                   "measured_saving": "paired-root 95% upper bound of the amortized cost ratio below 1",
                   "target": "paired-root 95% upper bound of the amortized cost ratio at or below 0.75",
                   "headline": "all three primary suites satisfy noninferiority and the 25% target (intersection-union)"},
               "headline_met": headline, "all_primary_noninferior": all_noninferior,
               "decisions_batch64": decisions, "decisions_batch_one_latency_seed29_8roots": latency_points,
               "reports": {f"{s}|{k}": {"name": r["name"], "records_sha256": r["records_sha256"], "stored": r["stored"]} for (s, k), r in reports.items()},
               "curves": {f"{s}|{k}": reports[(s, k)]["row"]["curves"] for (s, k) in reports if k != "mechanism"},
               "contrasts_vs_restart": {f"{s}|{k}": reports[(s, k)]["row"]["contrasts"] for (s, k) in reports if k != "mechanism"},
               "mechanism": {s: reports[(s, "mechanism")]["row"]["mechanism"] for s in SUITES},
               "exploratory_contrasts_vs_answer_only": {s: v["contrasts"] for s, v in raw.items()},
               "per_frame_accuracy": {s: v["per_frame"] for s, v in raw.items()},
               "raw_jobs": {s: v["jobs"] for s, v in raw.items()},
               "ledger": led, "figures": written}
    atomic_json(OUT / "summary.json", summary)


    md = ["# Confirmatory results synthesized from sealed records", "",
          "Generated by `scripts/synthesize_confirmatory.py` from driver-sealed artifacts under `runs/confirmatory_v1`. "
          "Every report directory was re-verified against its seal before use; raw evaluation logs used for the exploratory "
          "contrasts were re-verified and their records validated. No number here comes from anything except those records. "
          f"Protocol SHA-256 `{protocol_hash}`; matrix SHA-256 `{led['matrix_sha256']}`. Accuracy is mean post-edit exact "
          "correctness over 32 edits (exact shortest route or correct unreachable verdict for mazes; every non-input node correct for circuits), "
          "each base root weighted equally, five independent training seeds averaged, 256 test roots per suite.", "",
          "## 1. Frozen primary decisions", "",
          "The protocol froze **answer-only K8 versus stream-trained restart K8** for mazes and **answer-only K1 versus restart K2** for circuits, "
          "applied unchanged to the ordinary and depth-shift circuit suites. The target was at most one percentage point lower accuracy with at least "
          "25% lower amortized measured cost; the headline needed all three primary suites (intersection-union). Costs below are the batch-64 accuracy-job "
          "timings that the sealed report jobs used for the operating-point cost ratio; the dedicated batch-one latency operating points follow separately.", ""]
    rows = []
    for s in SUITES:
        d = decisions[s]
        rows.append([("**" + s + "**" if s in PRIMARY else s) + (" (primary)" if s in PRIMARY else " (secondary)"),
                     f'{d["reuse"]["policy"]} K{d["reuse"]["K"]} vs {d["comparator"]["policy"]} K{d["comparator"]["K"]}',
                     pp(d["accuracy_delta"]), ci(d["accuracy_root_ci"]), ci(d["accuracy_crossed_ci"]),
                     f'{d["cost_ratio"]:.4f}', ci(d["cost_root_ci"], 1, False),
                     "yes" if d["noninferior_root"] else "no", "yes" if d["measured_saving"] else "no", "yes" if d["target_saving_25pct"] else "no"])
    md.append(table(["Suite", "Frozen point", "Δ accuracy (pp)", "Root 95% CI (pp)", "Crossed 95% CI (pp)", "Cost ratio", "Root 95% CI", "Noninferior", "Saving < 1", "Target ≤ 0.75"], rows))
    md += ["", f"**Headline (all three primary suites noninferior with a 25% saving): {'met' if headline else 'not met'}.** "
           f"All three primary suites noninferior: {'yes' if all_noninferior else 'no'}.", ""]
    rows = []
    for s, d in latency_points.items():
        rows.append([s, f'{d["reuse"]["policy"]} K{d["reuse"]["K"]} vs {d["comparator"]["policy"]} K{d["comparator"]["K"]}',
                     f'{d["cost_ratio"]:.4f}', ci(d["cost_root_ci"], 1, False), pp(d["accuracy_delta"]), ci(d["accuracy_root_ci"])])
    md += ["Batch-one latency operating points (seed-29 checkpoint only, 8 test roots, three warmed repetitions; conditional on one checkpoint and not five-seed evidence):", "",
           table(["Suite", "Frozen point", "Latency ratio", "Root 95% CI", "Δ accuracy on these 8 roots (pp)", "Root 95% CI (pp)"], rows)]
    md += ["", "Per-seed operating-point deltas (pp) and cost ratios for the primary suites:", ""]
    rows = []
    for s in PRIMARY:
        d = decisions[s]
        rows.append([s, ", ".join(f"{100*x:+.2f}" for x in d["per_seed_delta"]), f'{100*(d["training_seed_sd"] or 0):.2f}', f'{100*d["root_sd"]:.2f}',
                     ", ".join(f"{x:.3f}" for x in d["per_seed_ratio"])])
    md.append(table(["Suite", "Per-seed Δ (seeds 29,43,71,101,137 order as recorded)", "Seed SD (pp)", "Root SD (pp)", "Per-seed cost ratio"], rows))

    md += ["", "## 2. Fixed-budget accuracy curves (32 edits)", "",
           "Mean post-edit exact accuracy (%) for the principal policies at every K. Each K is a separate stream that carries its own K-cycle state; no policy sees a higher-budget state.", ""]
    for s in SUITES:
        idx = curve_index(reports[(s, "h32")]["row"])
        rows = [[p] + [pct(idx[(p, k)]["post_accuracy"]) for k in KS] for p in PRINCIPAL if (p, 1) in idx]
        md += [f"### {s}", "", table(["Policy"] + [f"K{k}" for k in KS], rows)]
        ref = next(c for c in reports[(s, "h32")]["row"]["curves"] if c["record_kind"] == "reference_solver")
        md += [f"Whole-stream success (%) at K8: " + ", ".join(f"{p} {pct(idx[(p, 8)]['whole_stream_success'])}" for p in PRINCIPAL if (p, 8) in idx) + ". "
               f"Classical reference `{ref['policy']}` (CPU on the same host): exact on every frame, amortized {ref['amortized_ms']:.4f} ms per frame.", ""]

    md += ["## 3. Paired contrasts against restart at the same K", "",
           "Difference in mean post-edit exact accuracy (percentage points) and the paired-root 95% interval (10,000 draws, seeds averaged within each root); the crossed interval resamples seeds too. Cost ratios are ratios of paired mean amortized batch-64 cost.", ""]
    for s in SUITES:
        cidx = contrast_index(reports[(s, "h32")]["row"])
        rows = []
        for p in PRINCIPAL[1:]:
            for k in KS:
                c = cidx.get((p, k))
                if c:
                    rows.append([p, str(k), pp(c["accuracy"]["delta"]), ci(c["accuracy"]["root_ci"]), ci(c["accuracy"]["crossed_ci"]),
                                 f'{100*(c["accuracy"]["training_seed_sd"] or 0):.2f}', f'{c["cost"]["ratio"]:.4f}', ci(c["cost"]["root_ci"], 1, False)])
        md += [f"### {s}", "", table(["Policy − restart", "K", "Δ (pp)", "Root 95% CI", "Crossed 95% CI", "Seed SD (pp)", "Cost ratio", "Root 95% CI"], rows)]

    if raw:
        md += ["## 4. Exploratory contrasts against answer-only (not prespecified)", "",
               "Computed from the same sealed episode records with the same paired-root bootstrap. These comparisons were not frozen and cannot replace the headline; they are reported to characterize when latent reuse beats output reuse.", ""]
        for s, value in raw.items():
            rows = []
            for c in value["contrasts"]:
                if c["policy"] in PRINCIPAL or c["policy"] in LESIONS:
                    rows.append([c["policy"], str(c["K"]), pp(c["delta"]), ci(c["root_ci"]), ci(c["crossed_ci"]), f'{100*(c["training_seed_sd"] or 0):.2f}', str(len(c["seeds"]))])
            md += [f"### {s}", "", table(["Policy − answer_only", "K", "Δ (pp)", "Root 95% CI", "Crossed 95% CI", "Seed SD (pp)", "Seeds"], rows)]

    md += ["## 5. Measured cost", "",
           "Amortized milliseconds per frame including the initial solve, (initial + Σ edits)/(T+1). Batch-64 values come from the accuracy jobs (five seeds); batch-one latency from the dedicated seed-29 jobs on eight roots. Classical references run on the host CPU and are separately labeled.", ""]
    rows = []
    for s in SUITES:
        idx = curve_index(reports[(s, "h32")]["row"])
        ref = next(c for c in reports[(s, "h32")]["row"]["curves"] if c["record_kind"] == "reference_solver")
        lidx = curve_index(reports[(s, "latency")]["row"]) if (s, "latency") in reports else {}
        for p in ("restart", "answer_only", "spatial_gate"):
            rows.append([s, p] + [f'{idx[(p, k)]["amortized_ms"]:.3f}' for k in KS] + ([f'{lidx[(p, k)]["amortized_ms"]:.1f}' for k in KS] if lidx else [""] * 5))
        rows.append([s, f"{ref['policy']} (reference, CPU)", f"{ref['amortized_ms']:.4f}", "", "", "", "", "", "", "", "", ""])
    md.append(table(["Suite", "Policy"] + [f"b64 K{k}" for k in KS] + [f"b1 K{k}" for k in KS], rows))
    md += ["", "Batch-one stage breakdown at K1 and K8 (ms, seed 29, eight roots):", ""]
    rows = []
    for s in ("maze12-on-maze12", "circuit32-on-circuit32"):
        lidx = curve_index(reports[(s, "latency")]["row"])
        for p in ("restart", "answer_only", "spatial_gate"):
            for k in (1, 8):
                st = lidx[(p, k)]["stage_ms"]
                rows.append([s, p, str(k)] + [f'{st[x]:.2f}' for x in ("edit_and_validation", "encoder", "fresh", "adapter", "core", "decoder", "copy", "transfer_in", "transfer_out", "total")])
    md.append(table(["Suite", "Policy", "K", "edit+valid", "encoder", "fresh", "adapter", "core", "decoder", "copy", "in", "out", "total"], rows))

    md += ["", "## 6. Answer-content lesions", "",
           "The answer-only checkpoints of seeds 29, 43 and 71 deployed with lesioned previous probabilities: `uniform` replaces them with a uniform distribution (content-free), `shuffled_nodes` permutes them across nodes. Mean post-edit exact accuracy (%).", ""]
    rows = []
    for s in SUITES:
        idx = curve_index(reports[(s, "h32")]["row"])
        for p in ("restart", "answer_only", *LESIONS):
            rows.append([s, p] + [pct(idx[(p, k)]["post_accuracy"]) for k in KS])
    md.append(table(["Suite", "Policy", "K1", "K2", "K4", "K8", "K16"], rows))

    md += ["", "## 7. Equal-compute one-edit controls (mazes)", "",
           "Adapters trained on single-edit examples for 2,576 updates (forward calls matched within 2% of the stream recipe per seed), evaluated on the same 32-edit test streams. Mean post-edit exact accuracy (%).", ""]
    rows = []
    for s in ("maze12-on-maze12", "maze16-on-maze16", "maze12-on-maze20", "maze16-on-maze20"):
        idx = curve_index(reports[(s, "h32")]["row"])
        for p in ONE_EDIT:
            base = p.split("/")[1]
            rows.append([s, p] + [pct(idx[(p, k)]["post_accuracy"]) for k in KS] + [pct(idx[(base, k)]["post_accuracy"]) for k in KS])
    md.append(table(["Suite", "Policy", "one-edit K1", "K2", "K4", "K8", "K16", "stream K1", "K2", "K4", "K8", "K16"], rows))

    md += ["", "## 8. Auxiliary controls on the spatial backbone (mazes)", ""]
    rows = []
    for s in ("maze12-on-maze12", "maze16-on-maze16"):
        idx = curve_index(reports[(s, "h32")]["row"])
        for p in ("spatial_gate", *AUXILIARY):
            rows.append([s, p] + [pct(idx[(p, k)]["post_accuracy"]) for k in KS])
    md.append(table(["Suite", "Policy", "K1", "K2", "K4", "K8", "K16"], rows))

    md += ["", "## 9. 128-edit streams (trained sizes)", "", "Mean post-edit exact accuracy (%) and whole-stream success (%) over 128 edits.", ""]
    rows = []
    for s in TRAINED:
        idx = curve_index(reports[(s, "h128")]["row"])
        for p in PRINCIPAL:
            if (p, 1) in idx:
                rows.append([s, p] + [pct(idx[(p, k)]["post_accuracy"]) for k in KS] + [pct(idx[(p, k)]["whole_stream_success"]) for k in KS])
    md.append(table(["Suite", "Policy", "acc K1", "K2", "K4", "K8", "K16", "whole K1", "K2", "K4", "K8", "K16"], rows))
    rows = []
    for s in TRAINED:
        d = decisions[s]
        d128 = decide(reports[(s, "h128")]["row"]["operating_points"][0])
        rows.append([s, pp(d["accuracy_delta"]), ci(d["accuracy_root_ci"]), pp(d128["accuracy_delta"]), ci(d128["accuracy_root_ci"]), f'{d["cost_ratio"]:.4f}', f'{d128["cost_ratio"]:.4f}'])
    md += ["", "Frozen operating point on 32-edit versus 128-edit streams:", "", table(["Suite", "Δ 32 (pp)", "Root CI", "Δ 128 (pp)", "Root CI", "Cost ratio 32", "Cost ratio 128"], rows)]

    md += ["", "## 10. Mechanism: frozen-prior interventions and carried-state dynamics", "",
           "Interventions fork identical saved prior tensors (spatial-backbone stream at source K8) into each initializer under identical weights and score one edit; root-weighted exact accuracy by evaluator-only impact stratum, three seeds. "
           "Rows for global_gate, gru_adapter, residual_adapter and answer_only use projections transferred from their own jointly trained checkpoints onto the common spatial backbone, so they are conditional on that transfer and are not the deployed policies. "
           "`impact_mask` is a privileged oracle diagnostic and is not an upper bound. `carry_refinement` continues carry's own four-edit history with extra cycles on the same observation; `changed` is the fraction of node actions that change relative to the carried prediction.", ""]
    for s in ("maze12-on-maze12", "maze16-on-maze16", "circuit32-on-circuit48", "circuit64-on-circuit96"):
        mech = reports[(s, "mechanism")]["row"]["mechanism"]
        rows = []
        for p in ("restart", "carry", "spatial_gate", "global_gate", "shuffled_gate", "local_reset_1", "random_reset", "impact_mask", "carry_refinement"):
            for k in (1, 8, 16):
                cells = {r["stratum"]: r for r in mech if r["policy"] == p and r["K"] == k}
                if not cells:
                    continue
                row = [p + (" (privileged)" if p == "impact_mask" else " (transferred)" if p in TRANSFERRED else ""), str(k)]
                for st in ("low", "middle", "high"):
                    r = cells.get(st)
                    row.append(f'{pct(r["accuracy_root_weighted"])} (n={r["independent_roots"]})' + (f', changed {pct(r["changed_action_fraction"])}%' if r and "changed_action_fraction" in r else "") if r else "")
                rows.append(row)
        md += [f"### {s}", "", table(["Initializer", "K", "Low impact", "Middle", "High impact"], rows)]

    md += ["## 11. Backbones and selection", "", "Every static backbone passed the frozen gate (nondecreasing ordinary-validation exact accuracy across K and at least 0.85 at K16). Learning-rate grids were chosen on validation only; both candidate means are sealed in the selection jobs.", ""]
    rows = []
    for size in ("maze12", "maze16", "circuit32", "circuit64"):
        gate = next(iter(step_rows(RUNS / f"static-gate-{size}")))["static_gates"]
        sel = next(iter(step_rows(RUNS / f"selection-{size}")))
        for name, values in sorted(gate.items()):
            rows.append([size, name.split("-")[-1], ", ".join(f"{v:.4f}" for v in values), ""])
        for policy, g in sorted(sel["grids"].items()):
            rows.append([size, policy, "", f'grid {g["grid"]} (g0 {g["candidate_validation_means"]["0"]:.4f}, g1 {g["candidate_validation_means"]["1"]:.4f})'])
    md.append(table(["Size", "Seed / policy", "Static validation exact at K1,2,4,8,16", "Selected stream grid (validation means)"], rows))

    md += ["", "## 12. Resources and execution", "",
           f"Recorded attempts: {led['attempts']} ({led['empirical_attempts']} empirical). GPU-job hours {led['gpu_hours']:.3f} of {led['authorization_gpu_hours']} authorized; CPU-job hours {led['cpu_job_hours']:.3f}; {led['jobs']} sealed jobs; matrix complete: {led['matrix_complete']}. "
           f"Aborted or interrupted attempts: {len(led['aborted_attempts'])} ({'; '.join(a['job'] + ' ' + str(round(a['wall_seconds'], 1)) + 's' for a in led['aborted_attempts'])}); quarantined directories preserved: {len(led['quarantined_directories'])}; unattended startup events: {led['startup_events']}. "
           "External charges $0; electricity unmeasured. Hardware: one RTX 5070 (12 GiB), float32, TF32 off, PyTorch 2.11.0+cu128.", "",
           "## 13. Figures", "", "All figures are written by this script from the same sealed rows; the manifest records their hashes.", ""]
    md += [f"- `figures/{n}`" for n in written]
    atomic_write(OUT / "RESULTS.md", lambda h: h.write(("\n".join(md) + "\n").encode()))

    manifest = {"synthetic": False, "generator": "scripts/synthesize_confirmatory.py",
                "generator_sha256": file_hash(Path(__file__)), "protocol_sha256": protocol_hash,
                "inputs": {r["name"]: r["records_sha256"] for r in reports.values()},
                "raw_jobs": {s: v["jobs"] for s, v in raw.items()},
                "outputs": {n: file_hash(OUT / n) for n in ("RESULTS.md", "summary.json")},
                "figures": {n: file_hash(figures / n) for n in written},
                "elapsed_seconds": time.perf_counter() - started}
    atomic_json(OUT / "manifest.json", manifest)
    log(f"wrote {OUT/'RESULTS.md'}, summary.json, manifest.json and {len(written)} figures in {manifest['elapsed_seconds']:.1f}s; headline met: {headline}")


if __name__ == "__main__":
    main()
