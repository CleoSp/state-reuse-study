"""Build TMLR review/preprint LaTeX from verified saved summaries.

CPU only. Does not run inference, training, or new statistical comparisons.
Run from the repository root with the existing Python virtual environment.
Compile reports/paper/tmlr/main.tex and preprint.tex with Tectonic or latexmk.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from state_repair.execution.driver import verify  # noqa: E402
from state_repair.execution.records import step_rows, raw_step_hash  # noqa: E402

PAPER = ROOT / "reports/paper"
LATEX = PAPER / "tmlr"
RESULTS = ROOT / "reports/confirmatory"
RUNS = ROOT / "runs/confirmatory_v1"
SEEDS = ("29", "43", "71")
POLICIES = ("restart", "carry", "spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only")
LABELS = {p: p.replace("_", " ") for p in POLICIES}
COLORS = ("1B365D", "C25431", "00806C", "8264A8", "B8860B", "7A7A7A", "D81B60")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tex(text: str) -> str:
    special = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "\\": r"\textbackslash{}",
               "^": r"\textasciicircum{}", "−": "-", "–": "--", "—": "---",
               "≤": r"\(\leq\)", "≥": r"\(\geq\)", "±": r"\(\pm\)",
               "Δ": r"\(\Delta\)", "Σ": r"\(\Sigma\)", "σ": r"\(\sigma\)",
               "←": r"\(\leftarrow\)", "→": r"\(\rightarrow\)", "·": r"\(\cdot\)",
               "…": r"\ldots{}", "∈": r"\(\in\)", "≠": r"\(\neq\)", "≈": r"\(\approx\)",
               "č": r"\v{c}", "ć": r"\'{c}"}
    return "".join(special.get(c, c) for c in text)


def inline(text: str, refs: set[str]) -> str:
    pieces = re.split(r"(\$[^$]+\$|\\ref\{(?:sec|tab|fig):[a-z0-9-]+\}|`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\])", text)
    out = []
    for part in pieces:
        if part.startswith((r"\ref{", "$")):
            out.append(part)
        elif part.startswith("`"):
            raw = part[1:-1]
            if "/" in raw and not any(c in raw for c in "Σ+ ·"):
                out.append(r"\path{" + raw + "}")
            else:
                out.append(r"\texttt{" + tex(raw) + "}")
        elif part.startswith("**"):
            out.append(r"\textbf{" + inline(part[2:-2], refs) + "}")
        elif part.startswith("[") and all(x.strip() in refs for x in part[1:-1].split(",")):
            out.append(r"\citep{" + ",".join(x.strip().replace(" ", "") for x in part[1:-1].split(",")) + "}")
        else:
            out.append(tex(part))
    return "".join(out)


CAPTIONS = [
    "Initialization under each reuse policy. Circuits compare restart, carry, the spatial gate and output reuse. The remaining learned adapters are tested on mazes.",
    "Protocol choices fixed before test generation. Both circuit suites use the same operating point, including under structural shift. The exploratory curves are separate from this comparison.",
    "Primary accuracy comparisons at the frozen operating points. Mazes compare output reuse and restart at K8. Both circuit suites compare output reuse at K1 with restart at K2. The interval over roots determines the frozen noninferiority decision. For circuit32, the crossed interval extends below the -1 pp margin.",
    "Cost of output reuse divided by restart cost at the frozen operating points. Timing at batch size 64 uses five seeds. Latency at batch size one uses seed 29, eight roots and three repetitions. The 95% intervals resample paired roots. Every upper bound exceeds 0.75.",
    "Exact accuracy after edits on 12x12 mazes, averaged over five seeds, 256 roots and 32 edits per root. The table includes every principal policy. K counts outer cycles, with three calls to the shared block in each cycle.",
    "Exploratory paired accuracy differences on 12x12 mazes at equal K, in percentage points. Each cell gives the mean difference and its 95% interval from paired roots, conditional on the five training seeds and 256 roots. The intervals apply to individual comparisons rather than all comparisons simultaneously. A positive difference favors the row policy; Reference denotes comparison with itself. The supplement also gives crossed intervals that resample training seeds.",
    "Interventions starting from the same prior state on 12x12 mazes. Entries give exact accuracy (%) at K1 / K8 for three seeds, using the spatial backbone's own K8 state before the edit. Counts within each stratum refer to roots with eligible branches. They do not count nodes or seeds as independent samples. The impact mask uses privileged information and is excluded from deployed comparisons.",
    "Interventions on previous answer content, with seeds (29, 43, 71) and roots matched across policies within each suite and budget. Restart and intact output reuse use the same seeds as the lesions. All entries give mean exact accuracy (%) after edits; this table contains no means over five seeds.",
    "Selected exploratory differences between jointly trained maze systems, in percentage points. Crossed intervals resample complete roots and training seeds. Each seed range spans the five paired differences between seed means; it is not an uncertainty interval. The appendix gives all contrasts with intervals conditional on the observed seeds.",
    "Accuracy before and after updating, using each policy's own history. Old scores the previous argmax prediction on the current problem. New scores the refined prediction. Repair is the percentage corrected among cases where Old was wrong; Break is the percentage made wrong among cases where Old was correct. Percentages are rounded to one decimal. Gain gives New minus Old with crossed 95% intervals. Each row uses five seeds, 256 roots and 32 edits. The old predictions come from actively updated streams, so these rows do not evaluate a separately deployed cache.",
    "Edit consequences in ordinary streams of 32 edits: 8,192 transitions from 256 roots per suite, counted once across all policies and seeds. Challenge counts refer to separately conditioned branches, shown as low/middle/high. They cannot estimate the frequency of consequences in ordinary streams. Low means at most 10% of scored targets changed; high means at least 40%.",
    "Exploratory timing check at the frozen budget choices. Entries give time per frame at batch size one, timing complete streams including the initial solve. The sample uses two fixed roots, seed 29 and three interleaved repetitions per mode. Original remeasures the implementation with clocks for individual stages. Lean removes those clocks and unnecessary history copies while keeping the existing core validation. The frozen timings and decisions remain unchanged.",
]

TABLE_LABELS = (
    "policies", "frozen-protocol", "primary-accuracy", "primary-cost",
    "maze-accuracy", "maze-contrasts", "common-prior", "answer-lesions",
    "key-contrasts", "previous-prediction", "consequence-frequencies", "timing-sensitivity",
)


def make_table(lines: list[str], number: int, refs: set[str]) -> str:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        raise ValueError("Ragged manuscript table")
    label = TABLE_LABELS[number]
    if label == "key-contrasts":
        spec = r"@{}XcrXX@{}"
    elif label == "previous-prediction":
        spec = r"@{}p{0.26\linewidth}crrrrX@{}"
    elif label == "consequence-frequencies":
        spec = r"@{}XrrrrX@{}"
    elif label == "timing-sensitivity":
        spec = r"@{}Xlcrrr@{}"
    elif number == 0:
        spec = r"@{}>{\raggedright\arraybackslash}p{0.20\linewidth}Y>{\raggedright\arraybackslash}p{0.20\linewidth}@{}"
    elif number == 1:
        spec = r"@{}>{\raggedright\arraybackslash}p{0.20\linewidth}Y@{}"
    elif number == 5:
        spec = r"@{}lc>{\raggedleft\arraybackslash}X>{\raggedleft\arraybackslash}X@{}"
    elif n == 3:
        spec = r"@{}Xrr@{}"
    elif number == 4:
        spec = r"@{}Xrrrrr@{}"
    else:
        spec = "@{}" + "X" * n + "@{}"
    placement = "H" if label in ("maze-accuracy", "frozen-protocol", "primary-accuracy", "primary-cost") else "!htbp"
    out = [r"\begin{table}["+placement+"]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
           r"\renewcommand{\arraystretch}{" + ("1.0" if number == 5 else "1.2") + "}", r"\caption{" + tex(CAPTIONS[number]) + "}",
           r"\label{tab:" + TABLE_LABELS[number] + "}", r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule"]
    for i, row in enumerate(rows):
        if i == 1:
            out.append(r"\midrule")
            continue
        if number == 5 and i > 2 and row[0]:
            out.append(r"\addlinespace[3pt]")
        cells = [inline(c, refs) for c in row]
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out)


def seed_range(row: dict) -> tuple[float, float]:
    """Descriptive observed seed spread; never a confidence interval."""
    values = row["per_seed_accuracy"]
    expected = {"29", "43", "71", "101", "137"}
    if set(values) != expected:
        raise ValueError("Seed-range figure requires all five principal training seeds")
    if any(not 0 <= x <= 1 for x in values.values()):
        raise ValueError("Accuracy outside [0, 1]")
    if abs(sum(values.values()) / 5 - row["post_accuracy"]) > 1e-12:
        raise ValueError("Plotted mean differs from saved seed means")
    return min(values.values()), max(values.values())


def maze_contrast_table(summary: dict) -> tuple[str, list[dict]]:
    restart = {(r["policy"], r["K"]): r["accuracy"]
               for r in summary["contrasts_vs_restart"]["maze12-on-maze12|h32"]}
    answer = {(r["policy"], r["K"]): r
              for r in summary["exploratory_contrasts_vs_answer_only"]["maze12-on-maze12"]}
    lines = ["| Policy | K | Δ vs restart (pp), 95% CI | Δ vs answer-only (pp), 95% CI |",
             "|---|---:|---:|---:|"]
    rows = []
    def value(contrast: dict | None) -> str:
        if contrast is None:
            return "reference"
        if contrast["synthetic"] or contrast["seeds"] != [29, 43, 71, 101, 137] or contrast["roots"] != 256:
            raise ValueError("Unexpected contrast provenance")
        return (f"{100*contrast['delta']:+.2f} "
                f"[{100*contrast['root_ci'][0]:+.2f}, {100*contrast['root_ci'][1]:+.2f}]")
    for policy in POLICIES:
        for k in (1, 2, 4, 8, 16):
            a = None if policy == "restart" else restart[policy, k]
            b = None if policy == "answer_only" else answer[policy, k]
            lines.append(f"| {policy if k == 1 else ''} | {k} | {value(a)} | {value(b)} |")
            rows.append({"policy": policy, "K": k, "versus_restart": a, "versus_answer_only": b})
    return "\n".join(lines), rows


def curve_figure(summary: dict, suite: str, caption: str, filename: str, *, seed_bands: bool = False) -> None:
    curves = summary["curves"][suite + "|h32"]
    out = [r"\begin{figure}[!htbp]", r"\centering", r"\begin{tikzpicture}",
           r"\begin{axis}[width=0.92\linewidth,height=6.5cm,xmode=log,log basis x=2,",
           r"xmin=0.9,xmax=18,xtick={1,2,4,8,16},xticklabels={1,2,4,8,16},ymin=0,ymax=103,",
           r"xlabel={Refinement budget $K$},ylabel={Post-edit exact accuracy (\%)},",
           r"grid=major,grid style={gray!15},tick label style={font=\small},",
           r"legend style={at={(0.5,-0.25)},anchor=north,draw=none,font=\small},legend columns=3]"]
    if seed_bands:

        for policy in POLICIES:
            rows = sorted((r for r in curves if r["policy"] == policy), key=lambda r: r["K"])
            if not rows:
                continue
            bounds = [(r["K"], *seed_range(r)) for r in rows]
            coords = " ".join(f"({k},{high*100:.8f})" for k, low, high in bounds)
            coords += " " + " ".join(f"({k},{low*100:.8f})" for k, low, high in reversed(bounds))
            color = f"policy{POLICIES.index(policy)}"
            out.append(rf"\addplot[draw=none,fill={color},fill opacity=0.09,forget plot] coordinates {{{coords}}} \closedcycle;")
    for policy in POLICIES:
        rows = sorted((r for r in curves if r["policy"] == policy), key=lambda r: r["K"])
        if not rows:
            continue
        coords = " ".join(f"({r['K']},{100*r['post_accuracy']:.8f})" for r in rows)
        color = f"policy{POLICIES.index(policy)}"
        styles = ("solid", "dashed", "solid", "dashed", "densely dotted", "dashdotted", "solid")
        marks = ("*", "square*", "triangle*", "diamond*", "pentagon*", "x", "o")
        i = POLICIES.index(policy)
        out += [rf"\addplot+[color={color},{styles[i]},mark={marks[i]},line width=0.9pt,mark size=2pt] coordinates {{{coords}}};",
                r"\addlegendentry{" + tex(LABELS[policy]) + "}"]
    out += [r"\end{axis}", r"\end{tikzpicture}", r"\caption{" + inline(caption, set()) + "}",
            r"\label{fig:" + filename + "}", r"\end{figure}"]
    (LATEX / (filename + ".tex")).write_text("\n".join(out) + "\n", encoding="utf-8")


def budget_figure() -> None:
    diagnostic = json.loads((PAPER / "budget_diagnostic.json").read_text())
    if diagnostic["synthetic"] or not diagnostic["exploratory"]:
        raise ValueError("Incorrect budget diagnostic provenance")
    out = [r"\begin{figure}[!htbp]", r"\centering", r"\begin{tikzpicture}",
           r"\begin{axis}[width=0.92\linewidth,height=6.3cm,xmin=0,xmax=32,ymin=70,ymax=102,",
           r"xtick={0,8,16,24,32},xlabel={Frame (0 = initial solve)},ylabel={Exact accuracy (\%)},",
           r"grid=major,grid style={gray!15},tick label style={font=\small},",
           r"legend style={at={(0.5,-0.25)},anchor=north,draw=none,font=\small},legend columns=2]"]
    for job in diagnostic["jobs"]:
        if job["suite"] != "circuit32-on-circuit32":
            continue
        for curve in job["curves"]:
            k = curve["K"]
            if k == 8 and job["policy"] != "answer_only":
                continue
            policy = job["policy"]
            coords = " ".join(f"({r['frame']},{r['accuracy']*100:.8f})" for r in curve["frames"])
            line_style = "densely dashed" if k == 8 else "solid"
            i = POLICIES.index(policy)
            out += [rf"\addplot+[color=policy{i},{line_style},mark=none,line width=0.9pt] coordinates {{{coords}}};",
                    r"\addlegendentry{" + tex(LABELS[policy]) + f", K={k}" + "}"]
    out += [r"\end{axis}", r"\end{tikzpicture}",
            r"\caption{Accuracy can fall beyond the budgets used in joint stream training before any edit occurs. For circuit32 seed 137, output reuse solves 256/256 initial roots at K8 and 206/256 at K16. At K16, the spatial gate solves 221/256 and restart solves 256/256. Later frames fluctuate around this lower accuracy. The omitted restart and spatial gate curves at K8 remain at 100\%. Each line gives the exact fraction over the same 256 roots for one checkpoint; these are not estimates over five seeds. We selected this case after inspecting the budget curves, so it is exploratory. The vertical axis spans 70--102\%.}",
            r"\label{fig:budget-extrapolation}", r"\end{figure}"]
    (LATEX / "budget-extrapolation.tex").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if digest(RESULTS / name) != expected:
            raise ValueError(f"Synthesis output hash mismatch: {name}")
    summary = json.loads((RESULTS / "summary.json").read_text())
    if summary["synthetic"]:
        raise ValueError("Synthetic summary cannot enter the paper")
    checked = []
    suites = ("maze12-on-maze12", "circuit32-on-circuit32", "circuit32-on-circuit48", "circuit64-on-circuit96")
    for suite in suites:
        name = "report-" + suite + "-h32"
        output = RUNS / name
        job = json.loads((output / "job.json").read_text())
        verify(output, job)
        record = next(iter(step_rows(output)))
        if job["synthetic"] or record["synthetic"]:
            raise ValueError(f"Synthetic report: {name}")
        if raw_step_hash(output) != manifest["inputs"][name]:
            raise ValueError(f"Report provenance mismatch: {name}")
        if record["curves"] != summary["curves"][suite + "|h32"]:
            raise ValueError(f"Summary curves differ from sealed report: {suite}")
        checked.append(name)
        print(f"Verified {name}", flush=True)
    LATEX.mkdir(parents=True, exist_ok=True)
    matched = []
    for suite in suites:
        curves = summary["curves"][suite + "|h32"]
        for k in (1, 2, 4, 8, 16):
            values = {}
            for policy in ("restart", "answer_only", "uniform/answer_only", "shuffled_nodes/answer_only"):
                row = next(r for r in curves if r["policy"] == policy and r["K"] == k)
                values[policy] = sum(row["per_seed_accuracy"][seed] for seed in SEEDS) / len(SEEDS)
            matched.append({"suite": suite, "K": k, "seeds": list(map(int, SEEDS)), "roots": 256,
                            "post_accuracy": values,
                            "uniform_minus_restart": values["uniform/answer_only"] - values["restart"],
                            "uniform_minus_intact": values["uniform/answer_only"] - values["answer_only"]})
    dump(PAPER / "seed_matched_lesions.json", {"synthetic": False, "source_sha256": digest(RESULTS / "summary.json"),
         "statistic": "mean of saved per-seed post-edit accuracies; no new intervals", "rows": matched})

    manuscript = (PAPER / "MANUSCRIPT.md").read_text(encoding="utf-8")
    expected_table, contrasts = maze_contrast_table(summary)
    actual_table = manuscript.split("<!-- BEGIN GENERATED MAZE CONTRASTS -->", 1)[1].split("<!-- END GENERATED MAZE CONTRASTS -->", 1)[0].strip()
    if actual_table != expected_table:
        raise ValueError("Manuscript paired-contrast table differs from saved evidence")
    bands = [{"policy": r["policy"], "K": r["K"], "mean": r["post_accuracy"],
              "minimum_seed_accuracy": seed_range(r)[0], "maximum_seed_accuracy": seed_range(r)[1]}
             for r in summary["curves"]["maze12-on-maze12|h32"] if r["policy"] in POLICIES]
    dump(PAPER / "maze_crossover_display.json", {"synthetic": False,
         "source_sha256": digest(RESULTS / "summary.json"), "intervals": "saved paired-root 95% CIs; no new bootstrap",
         "bands": "min/max of five saved per-seed means, not confidence intervals", "seed_ranges": bands,
         "contrasts": contrasts})
    if any(x in manuscript for x in ("reports/", "pending the owner", "subject to author review")):
        raise ValueError("Internal-document residue in manuscript")
    abstract = manuscript.split("## Abstract\n", 1)[1].split("## 1.", 1)[0].strip()
    if "\n\n" in abstract:
        raise ValueError("TMLR requires a single-paragraph abstract")

    for row in matched:
        short = row["suite"].split("-on-")[1]
        prefix = f"| {short}, K{row['K']} |"
        matching = [line for line in manuscript.splitlines() if line.startswith(prefix)]
        if matching:
            expected = prefix + " " + " | ".join(f"{v*100:.2f}" for v in row["post_accuracy"].values()) + " |"
            if matching != [expected]:
                raise ValueError(f"Manuscript lesion table differs from saved data: {prefix}")

    reference_text = manuscript.split("## References\n", 1)[1]
    references = {}
    for line in reference_text.splitlines():
        m = re.match(r"- \[([^]]+)\] (.+)", line)
        if m:
            references[m[1]] = m[2]

    bib_keys = set(re.findall(r"@\w+\{([^,]+),", (LATEX / "references.bib").read_text()))
    if {k.replace(" ", "") for k in references} != bib_keys:
        raise ValueError("Manuscript/BibTeX keys disagree")

    bib = []
    for label, description in references.items():
        arxiv = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", description)
        url = "https://arxiv.org/abs/" + arxiv[1] if arxiv else {
            "NCA": "https://distill.pub/2020/growing-ca/",
            "D* Lite": "https://cdn.aaai.org/AAAI/2002/AAAI02-072.pdf",
        }.get(label)
        bib += [r"\bibitem[" + tex(label) + "]{" + label.replace(" ", "") + "}", tex(description)]
        if url:
            bib.append(r"\url{" + url + "}")
        bib.append("")
    curve_figure(summary, "maze12-on-maze12", r"How the maze policy ordering changes with budget (exploratory). Lines average accuracy over five training seeds and 256 roots. Faint bands run from the lowest to the highest of the five seed means at each K. They describe observed seed variation, without estimating confidence intervals or uncertainty over test roots. The figure includes all seven principal policies, each carrying the state produced at its own budget. Table \ref{tab:maze-contrasts} gives intervals from paired roots; the supplement gives crossed intervals.", "maze-crossover", seed_bands=True)
    curve_figure(summary, "circuit32-on-circuit48", r"Accuracy when circuit size, depth and wiring change together (exploratory). The four principal systems were jointly trained on circuits with 32 nodes and evaluated on circuits with 48 nodes, using five seeds and 256 roots. Bands span the lowest and highest seed means; they are descriptive ranges. Comparisons at equal K ask a different question from the frozen comparison of output reuse at K1 with restart at K2. Section \ref{sec:distribution-shift} reports crossed intervals.", "circuit-depth-shift", seed_bands=True)
    budget_figure()
    from build_review_revision import generate
    revision_tables = generate(ROOT)
    for label, expected in revision_tables.items():
        start_marker, end_marker = f"<!-- BEGIN {label} -->", f"<!-- END {label} -->"
        if manuscript.split(start_marker, 1)[1].split(end_marker, 1)[0].strip() != expected:
            raise ValueError(f"Revision table differs from saved evidence: {label}")

    lines = manuscript.split("## References\n", 1)[0].splitlines()
    out = []
    i = 0
    seen_tables: set[str] = set()
    pending_table: str | None = None
    refs = set(references)
    while i < len(lines):
        line = lines[i]
        if line == "<!-- materials-access -->":
            out.append(r"\materialsavailability")
        if line == "<!-- appendix -->":
            out += [r"\FloatBarrier", r"\bibliographystyle{tmlr}", r"\bibliography{references}", r"\clearpage", r"\appendix"]
        figure_marker = re.fullmatch(r"<!-- figure:([a-z0-9-]+) -->", line)
        if figure_marker:
            if not (LATEX / (figure_marker[1]+'.tex')).is_file():
                raise ValueError("Missing figure input")
            out.append(r"\input{"+figure_marker[1]+"}")
        table_marker = re.fullmatch(r"<!-- table:([a-z0-9-]+) -->", line)
        if table_marker:
            if pending_table is not None:
                raise ValueError("Table marker without a following table")
            pending_table = table_marker[1]
        if line.startswith("<!--"):
            i += 1
            continue
        if i < 6:
            i += 1
            continue
        if line == "## Abstract":
            out.append(r"\begin{abstract}")
        elif line == "## Reproducibility and artifacts":
            out.append(r"\section*{Reproducibility and artifacts}")
        elif line.startswith(("## ", "### ")):
            heading_match = re.fullmatch(r"(#{2,3}) (?:(?:\d+(?:\.\d+)?|[A-Z])\.?\s+)(.+) \{#(sec:[a-z0-9-]+)\}", line)
            if heading_match is None:
                raise ValueError(f"Numbered heading needs a stable section label: {line}")
            depth, heading, section_label = heading_match.groups()
            if section_label == "sec:introduction":
                out.append(r"\end{abstract}")
            command = "section" if depth == "##" else "subsection"
            out.append("\\" + command + "{" + tex(heading) + "}")
            out.append(r"\label{" + section_label + "}")
        elif line.startswith("|"):
            if pending_table not in TABLE_LABELS or pending_table in seen_tables:
                raise ValueError(f"Missing, unknown or duplicate table label: {pending_table}")
            table_no = TABLE_LABELS.index(pending_table)
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(make_table(block, table_no, refs))
            seen_tables.add(pending_table)
            pending_table = None
            continue
        elif re.match(r"\d+\. ", line):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                out.append(r"\item " + inline(re.sub(r"^\d+\. ", "", lines[i]), refs))
                i += 1
            out.append(r"\end{enumerate}")
            continue
        else:
            out.append(inline(line, refs))
        i += 1
    expected_tables = set(TABLE_LABELS)
    if 'timing-sensitivity' not in revision_tables:
        expected_tables.remove('timing-sensitivity')
    if seen_tables != expected_tables or pending_table is not None:
        raise ValueError(f"Table labels do not match expected tables: {seen_tables}")
    (LATEX / "body.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    figure_files = [name+'.tex' for name in re.findall(r'\\input\{([^}]+)\}', '\n'.join(out))]

    for name in ['body.tex', *figure_files]:
        display_path = LATEX / name
        display = display_path.read_text(encoding='utf-8')
        display = display.replace('answer\\_only', 'output reuse').replace('answer-only', 'output reuse').replace('answer only', 'output reuse')
        for identifier, name in (
            ('spatial\\_gate', 'spatial gate'), ('global\\_gate', 'global gate'),
            ('gru\\_adapter', 'GRU adapter'), ('residual\\_adapter', 'residual adapter'),
        ):
            display = display.replace(identifier, name)
        display = display.replace('Answer-only', 'Output reuse')
        display = re.sub(r'\b(Sections?|Tables?|Figures?) +(?=\\ref\{)', r'\1~', display)
        display = display.replace('Output reuse (output reuse in the saved records)', 'Output reuse')
        display_path.write_text(display, encoding='utf-8')
    content = "\n".join((LATEX / name).read_text(encoding="utf-8") for name in ['body.tex', *figure_files])
    labels = re.findall(r"\\label\{([^}]+)\}", content)
    cross_refs = re.findall(r"\\ref\{([^}]+)\}", content)
    if len(labels) != len(set(labels)) or set(cross_refs) - set(labels):
        raise ValueError("Duplicate labels or unresolved cross-reference targets")
    if re.search(r"\b(?:Sections?|Tables?|Figures?)\s+\d", content):
        raise ValueError("Hard-coded cross-reference number remains in generated LaTeX")
    preamble = r"""% Generated by scripts/build_paper.py; official tmlr.sty remains unmodified.
\usepackage{amsmath,amssymb,booktabs,tabularx,array}
\usepackage{microtype}
\usepackage{graphicx,xcolor}
\usepackage{float}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{xurl}
\usepackage[breaklinks=true,colorlinks=true,linkcolor=blue!45!black,citecolor=blue!45!black,urlcolor=blue!45!black]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[section]{placeins}
\setlength{\emergencystretch}{2em}
\urlstyle{tt}
\renewcommand{\UrlFont}{\ttfamily\small}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\widowpenalty=10000
\clubpenalty=10000
\title{When Does Reusing Computation Help Small Recursive Solvers on Changing Problems?}
\hypersetup{pdftitle={When Does Reusing Computation Help Small Recursive Solvers on Changing Problems?}}
"""
    preamble += "\n".join(rf"\definecolor{{policy{i}}}{{HTML}}{{{color}}}" for i, color in enumerate(COLORS))
    (LATEX / "preamble.tex").write_text(preamble + "\n", encoding="utf-8")
    document = r"""
\begin{document}
\maketitle
\input{body}
\end{document}
"""
    main_tex = r"""% Anonymous TMLR review-format draft. Not yet submitted.
\documentclass[10pt]{article}
\usepackage{tmlr}
\input{preamble}
\author{Anonymous authors}
\hypersetup{pdfauthor={}}
\newcommand{\materialsavailability}{The code and bounded evidence package are provided in the accompanying supplement.}
""" + document
    (LATEX / "main.tex").write_text(main_tex, encoding="utf-8")
    author_name = manuscript.splitlines()[4].strip().rstrip(".")
    preprint = r"""% Named public-preprint format. Not yet submitted or published.
\documentclass[10pt]{article}
\usepackage[preprint]{tmlr}
\input{preamble}
""" + r"\author{\name " + tex(author_name) + "}\n" + r"\hypersetup{pdfauthor={" + tex(author_name) + "}}\n" + r"\newcommand{\materialsavailability}{Public materials URL: \underline{\hspace{0.65\linewidth}}}" + "\n" + document
    (LATEX / "preprint.tex").write_text(preprint, encoding="utf-8")
    outputs = [PAPER / "seed_matched_lesions.json", PAPER / "budget_diagnostic.json", PAPER / "maze_crossover_display.json",
               *sorted(p for p in LATEX.iterdir() if p.suffix in (".tex", ".bib", ".bst", ".sty"))]
    dump(PAPER / "tmlr_build_manifest.json", {"synthetic": False, "generator": "scripts/build_paper.py",
         "generator_sha256": digest(Path(__file__)), "manuscript_sha256": digest(PAPER / "MANUSCRIPT.md"),
         "summary_sha256": digest(RESULTS / "summary.json"), "synthesis_manifest_sha256": digest(RESULTS / "manifest.json"),
         "sealed_reports_verified": checked, "outputs": {str(p.relative_to(ROOT)): digest(p) for p in outputs},
         "elapsed_seconds": time.perf_counter() - started, "gpu_seconds": 0,
         "cross_references": {"labels": len(labels), "references": len(cross_refs), "unresolved": 0, "hard_coded_numbers": 0},
         "tables": len(seen_tables), "figures": len(figure_files),
         "review_revision_inputs": {p.name: digest(p) for p in (PAPER / "review_revision").iterdir() if p.is_file()},
         "validation": "summary hashes, four report seals, exact curve equality, matched-seed and paired-contrast tables, five-seed ranges, bibliography keys, one-paragraph abstract, stable cross-reference labels"})
    print(f"Built TMLR LaTeX, {len(seen_tables)} tables and {len(figure_files)} figures in {time.perf_counter()-started:.2f}s; no GPU work.")


if __name__ == "__main__":
    main()
