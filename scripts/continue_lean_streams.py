"""Resume the seeded timing schedule, verifying each skipped block."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/paper/review_revision'


def main() -> None:
    approval = json.loads((OUT / 'timing_extension_approval.json').read_text())
    if approval.get('approved') is not True or approval.get('max_total_seconds') != 3000:
        raise ValueError('Explicit 50-minute authorization is required')
    prior = json.loads((OUT / 'timing_summary.json').read_text())
    if prior['complete'] or prior['blocks'] >= 900:
        raise ValueError('There is no incomplete schedule to continue')
    backup = ROOT / 'runs/paper_tmlr/timing-initial-cap'
    backup.mkdir(exist_ok=False)
    for name in ('timing_summary.json', 'timing_predictions.json', 'timing_blocks.jsonl', 'timing_reservation.json'):
        shutil.copy2(OUT/name, backup/name)
    source_path = ROOT / 'scripts/measure_lean_streams.py'
    source = source_path.read_text()
    source = source.replace("    if (OUT / 'timing_blocks.jsonl').exists():\n        raise ValueError('Refuse to overwrite a timing run')\n", '')
    source = source.replace("    start = time.perf_counter()", "    previous_summary = json.loads((OUT / 'timing_summary.json').read_text())\n    previous_elapsed = previous_summary['elapsed_seconds']\n    start = time.perf_counter()", 1)
    source = source.replace("'max_wall_seconds': 1200, 'reserved_gpu_hours': 1200 / 3600", "'max_wall_seconds': 3000, 'reserved_gpu_hours': 3000 / 3600")
    source = source.replace("    blocks, expected, predictions = [], {}, []", """    blocks = [json.loads(line) for line in (OUT/'timing_blocks.jsonl').read_text().splitlines()]
    predictions = json.loads((OUT/'timing_predictions.json').read_text())['records']
    expected = {(r['suite'],r['policy'],r['K'],r['root_id']):r['actions'] for r in predictions}
    previous_count = len(blocks)
    task_number = -1""")
    source = source.replace("        for suite, arm, k, root_index, mode in tasks:\n", """        for suite, arm, k, root_index, mode in tasks:
            task_number += 1
            if task_number < previous_count:
                old = blocks[task_number]
                identity = streams[suite][root_index][0]
                if (old['suite'],old['policy'],old['K'],old['root_id'],old['mode'],old['repetition']) != (suite,arm,k,identity,mode,repetition):
                    raise ValueError('Resume schedule changed')
                continue
""")
    source = source.replace("time.perf_counter() - start > 1170", "previous_elapsed + time.perf_counter() - start > 2970")
    source = source.replace("elapsed = time.perf_counter() - start", "elapsed = previous_elapsed + time.perf_counter() - start")
    source = source.replace("'plan_sha256': sha(plan), 'script_sha256': sha(Path(__file__)),", "'plan_sha256': sha(plan), 'script_sha256': sha(Path(__file__)), 'continuation_script_sha256': sha(ROOT/'scripts/continue_lean_streams.py'), 'resumed_after_blocks': previous_count,")
    namespace = {'__name__': '__main__', '__file__': str(source_path)}
    exec(compile(source, str(source_path), 'exec'), namespace)


if __name__ == '__main__':
    main()
