"""Generate exploratory review tables/plots from hash-bound saved evidence."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

POLICIES = ('restart', 'carry', 'spatial_gate', 'global_gate', 'gru_adapter', 'residual_adapter', 'answer_only')
MARKS = ('*', 'square*', 'triangle*', 'diamond*', 'pentagon*', 'x', 'o')
STYLES = ('solid', 'dashed', 'solid', 'dashed', 'densely dotted', 'dashdotted', 'solid')
SUITES = ('maze12-on-maze12', 'circuit32-on-circuit32', 'circuit32-on-circuit48')
NAMES = dict(zip(SUITES, ('12x12 mazes', '32-node circuits', '48-node structural shift')))


def table(header: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def signed(value: float) -> str:
    return f'{100*value:+.2f}'


def ci(values: list[float]) -> str:
    return '[' + ', '.join(signed(v) for v in values) + ']'


def generate(root: Path) -> dict[str, str]:
    folder = root / 'reports/paper/review_revision'
    latex = root / 'reports/paper/tmlr'
    summary = json.loads((root / 'reports/confirmatory/summary.json').read_text())
    diagnostic = json.loads((folder / 'previous_prediction_summary.json').read_text())
    timing_path = folder / 'timing_summary.json'
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else None
    if timing is not None and not timing['complete']:
        timing = None
    if timing is not None and (timing['blocks'] != timing['expected_blocks'] or not timing['identical_actions']):
        raise ValueError('Inconsistent timing experiment')
    for name, expected in diagnostic['outputs'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Diagnostic episode/audit data hash mismatch')
    if timing is not None and hashlib.sha256((folder / 'timing_blocks.jsonl').read_bytes()).hexdigest() != timing['records_sha256']:
        raise ValueError('Timing records changed')
    if timing is not None and hashlib.sha256((folder / 'timing_predictions.json').read_bytes()).hexdigest() != timing['predictions_sha256']:
        raise ValueError('Timing predictions changed')
    tables = {}
    rows = []
    contrasts = {(r['policy'], r['K']): r for r in summary['exploratory_contrasts_vs_answer_only']['maze12-on-maze12']}
    restart = {(r['policy'], r['K']): r['accuracy'] for r in summary['contrasts_vs_restart']['maze12-on-maze12|h32']}
    selections = [('answer_only', 1, 'restart'), ('answer_only', 8, 'restart'), ('gru_adapter', 1, 'answer_only'),
                  ('carry', 8, 'answer_only'), ('spatial_gate', 8, 'answer_only'), ('global_gate', 8, 'answer_only'), ('gru_adapter', 8, 'answer_only'),
                  ('residual_adapter', 8, 'answer_only'), ('spatial_gate', 16, 'answer_only')]
    for policy, k, comparator in selections:
        value = restart[policy, k] if comparator == 'restart' else contrasts[policy, k]
        rows.append([policy + ' − ' + comparator, str(k), signed(value['delta']), ci(value['crossed_ci']),
                     ci([min(value['per_seed_delta']), max(value['per_seed_delta'])])])
    tables['key-contrasts'] = table(['Maze comparison', 'K', 'Δ (pp)', 'Crossed 95% CI', 'Seed Δ range'], rows)
    rows = []
    for suite in SUITES:
        for policy in ('answer_only', 'spatial_gate'):
            for k in (1, 8):
                r = next(r for r in diagnostic['results'] if (r['suite'], r['policy'], r['K']) == (suite, policy, k))
                rows.append([suite.split('-on-')[-1] + ', ' + policy, str(k), f"{r['previous_accuracy']*100:.1f}",
                    f"{r['post_accuracy']*100:.1f}", f"{r['repair_rate']*100:.1f}", f"{r['break_rate']*100:.1f}",
                    signed(r['gain']['delta']) + ' ' + ci(r['gain']['crossed_ci'])])
    tables['previous-prediction'] = table(['Suite / policy', 'K', 'Old %', 'New %', 'Repair %', 'Break %', 'Gain pp [95% CI]'], rows)
    rows = []
    for r in diagnostic['frequencies']:
        rows.append([r['suite'].split('-on-')[-1], *[f"{100*r['ordinary'].get(s,0)/r['transitions']:.2f}" for s in ('low','middle','high')],
                     str(r['challenge_branches']), '/'.join(str(r['challenge'].get(s,0)) for s in ('low','middle','high'))])
    tables['consequence-frequencies'] = table(['Suite', 'Low %', 'Middle %', 'High %', 'Challenge n', 'Challenge L/M/H'], rows)
    points = {} if timing is None else {(r['suite'], r['policy'], r['K'], r['mode']): r for r in timing['points']}
    rows = []
    for suite in SUITES:
        if timing is None:
            break
        for policy, k in [('restart', 8 if suite.startswith('maze') else 2), ('answer_only', 8 if suite.startswith('maze') else 1)]:
            a, b = [points[suite, policy, k, mode] for mode in ('instrumented','lean')]
            rows.append([suite.split('-on-')[-1], policy, str(k), f"{a['mean_ms']:.2f}", f"{b['mean_ms']:.2f}",
                         f"{b['mean_ms']/a['mean_ms']:.3f}"])
    if timing is not None:
        tables['timing-sensitivity'] = table(['Suite', 'Policy', 'K', 'Original ms', 'Lean ms', 'Lean / orig.'], rows)


    out = [r'\begin{figure}[!htbp]', r'\centering', r'\begin{tikzpicture}',
           r'\begin{axis}[width=0.90\linewidth,height=5.5cm,xmin=3.5,xmax=16.5,xtick={4,8,16},',
           r'xlabel={Refinement budget $K$},ylabel={Accuracy minus output reuse (pp)},',
           r'grid=major,grid style={gray!15},legend style={at={(0.5,-0.28)},anchor=north,draw=none,font=\small},legend columns=3]']
    out += [r'\addplot[black!40,dashed,forget plot] coordinates {(4,0) (16,0)};']
    for policy in POLICIES[:-1]:
        if policy == 'restart':

            pass
        vals = [contrasts[policy, k] for k in (4,8,16)]
        coords = ' '.join(f"({k},{r['delta']*100:.7f}) += (0,{(r['crossed_ci'][1]-r['delta'])*100:.7f}) -= (0,{(r['delta']-r['crossed_ci'][0])*100:.7f})" for k,r in zip((4,8,16),vals))
        index = POLICIES.index(policy)
        out += [rf'\addplot+[color=policy{index},{STYLES[index]},mark={MARKS[index]},error bars/.cd,y dir=both,y explicit] coordinates {{{coords}}};',
                r'\addlegendentry{' + policy.replace('_', ' ') + '}']
    out += [r'\end{axis}',r'\end{tikzpicture}',
            r'\caption{Maze accuracy relative to output reuse at larger budgets. Points give paired differences between jointly trained systems. Crossed 95\% intervals resample five seeds and 256 roots and apply to each comparison individually. Figure \ref{fig:maze-crossover} gives the means and descriptive seed ranges across all budgets.}',
            r'\label{fig:maze-differences}',r'\end{figure}']
    (latex / 'maze-differences.tex').write_text('\n'.join(out)+'\n')

    original_latency = json.loads((folder/'original_latency.json').read_text())
    for filename, mode in [('accuracy-cost','throughput'), ('accuracy-latency','original'), ('lean-latency','lean')]:
        if mode == 'lean' and timing is None:
            continue
        out = [r'\begin{figure}[!htbp]', r'\centering']
        for suite in SUITES:
            out += [r'\begin{tikzpicture}',r'\begin{axis}[width=0.93\linewidth,height=4.1cm,',
                r'title={' + NAMES[suite] + '},ymin=0,ymax=103,scaled x ticks=false,',
                r'xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},',
                r'xlabel={' + ('Amortized ms per example (batch 64)' if mode=='throughput' else 'Amortized latency at batch size one (ms)') + r'},ylabel={Exact accuracy (\%)},',
                r'grid=major,grid style={gray!15},legend style={at={(0.5,-0.35)},anchor=north,draw=none,font=\small},legend columns=3]']
            for policy in POLICIES:
                if mode == 'throughput':
                    vals = [r for r in summary['curves'][suite+'|h32'] if r['policy']==policy]
                    coords = ' '.join(f"({r['amortized_ms']:.8f},{100*r['post_accuracy']:.8f})" for r in sorted(vals,key=lambda r:r['K']))
                elif mode == 'original':
                    vals = [r for r in original_latency['curves'][suite] if r['policy']==policy]
                    coords = ' '.join(f"({r['amortized_ms']:.8f},{100*r['post_accuracy']:.8f})" for r in sorted(vals,key=lambda r:r['K']))
                else:
                    vals = [r for r in timing['points'] if r['suite']==suite and r['policy']==policy and r['mode']=='lean']
                    coords = ' '.join(f"({r['mean_ms']:.8f},{100*r['post_accuracy']:.8f})" for r in sorted(vals,key=lambda r:r['K']))
                if vals:
                    index = POLICIES.index(policy)
                    out += [rf'\addplot+[color=policy{index},{STYLES[index]},mark={MARKS[index]},line width=0.8pt] coordinates {{{coords}}};']
                    if suite == SUITES[0]:
                        out.append(r'\addlegendentry{' + policy.replace('_',' ') + '}')
            out += [r'\end{axis}',r'\end{tikzpicture}',r'\par\vspace{0.3cm}']
        caption = (r'Accuracy and measured amortized cost for every principal policy at all five budgets. Lines join points in increasing $K$. Both axes use the original runs of 32 edits, with five seeds and 256 roots. Costs cover frame zero and every measured pipeline stage at batch size 64. These measurements describe throughput; latency at batch size one is measured separately. Table \ref{tab:timing-sensitivity} checks sensitivity to instrumentation.'
                   if mode=='throughput' else r'Exploratory accuracy and latency measurements from the lean implementation. Both axes use the same two fixed roots, 32 edits, checkpoint from seed 29 and three timing repetitions. Lines follow $K=1,2,4,8,16$. Repeating the timing does not provide independent accuracy replications. This small sample cannot establish a population frontier or support inference across five seeds. Every action matched the original instrumented implementation.')
        if mode == 'original':
            caption = r'Accuracy and original latency at batch size one for all principal policies at $K=1,2,4,8,16$. Lines join increasing budgets. Both axes use the same eight roots, checkpoints from seed 29, complete streams of 32 edits plus frame zero, and three timing repetitions. The values come from the original records with timing at each stage; they were not remeasured for this figure and do not estimate variation over five seeds. Timing repetitions leave the accuracy sample size unchanged. Table \ref{tab:timing-sensitivity} compares the original and lean implementations.'
        if timing is None:
            caption = caption.replace(r' Table \ref{tab:timing-sensitivity} checks sensitivity to instrumentation.', '')
            caption = caption.replace(r' Table \ref{tab:timing-sensitivity} compares the original and lean implementations.', '')
        out += [r'\caption{'+caption+'}',r'\label{fig:'+filename+'}',r'\end{figure}']
        (latex / (filename+'.tex')).write_text('\n'.join(out)+'\n')
    (folder / 'display_tables.json').write_text(json.dumps(tables, indent=2)+'\n')
    return tables
