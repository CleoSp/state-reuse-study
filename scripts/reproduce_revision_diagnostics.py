"""Recompute review diagnostics from included episodes, plus raw sample checks.

Uses only the bounded review package. Full raw extraction is a separate command.
"""
from __future__ import annotations
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from state_repair.execution.datasets import decode, frames
from state_repair.eval.metrics import score
from analyze_previous_predictions import PreparedScore, interval


def main() -> None:
    folder = ROOT / 'reports/paper/review_revision'
    diagnostic = json.loads((folder / 'previous_prediction_summary.json').read_text())
    saved = json.loads((ROOT / 'reports/confirmatory/summary.json').read_text())
    with gzip.open(folder / 'previous_prediction_episodes.json.gz', 'rt') as handle:
        episodes = json.load(handle)
    with gzip.open(folder / 'previous_prediction_audit.json.gz', 'rt') as handle:
        samples = json.load(handle)
    grouped = defaultdict(dict)
    for row in episodes:
        if row['synthetic'] or len(row['counts']) != 4 or sum(row['counts']) != 32 or any(v < 0 for v in row['counts']):
            raise ValueError('Invalid episode counts')
        key = row['suite'], row['policy'], row['K']
        pair = row['seed'], row['root_id']
        if pair in grouped[key]:
            raise ValueError('Duplicate seed/root')
        grouped[key][pair] = row['counts']
    accuracies = {}
    for key, cells in grouped.items():
        seeds = sorted({s for s, _ in cells})
        roots = sorted({r for _, r in cells})
        if len(cells) != 1280 or seeds != [29,43,71,101,137] or len(roots) != 256:
            raise ValueError('Incomplete root/seed matrix')
        arr = np.array([[cells[s,r] for r in roots] for s in seeds])
        total = arr.sum((0,1))
        target = next(v for v in diagnostic['results'] if (v['suite'],v['policy'],v['K']) == key)
        assert total.tolist() == target['counts']
        cc,repaired,broken,wrong = total
        checks_rates={'previous_accuracy':(cc+broken)/total.sum(),'post_accuracy':(cc+repaired)/total.sum(),
                      'repair_rate':repaired/(repaired+wrong) if repaired+wrong else None,
                      'break_rate':broken/(cc+broken) if cc+broken else None}
        for name,value in checks_rates.items():
            if value is None:
                assert target[name] is None
            else:
                assert abs(value-target[name])<1e-12
        gain = interval((arr[:,:,1]-arr[:,:,2])/32)
        for name in ('delta','root_ci','crossed_ci','per_seed_delta'):
            np.testing.assert_allclose(gain[name],target['gain'][name],rtol=0,atol=1e-12)
        accuracies[key] = (arr[:,:,0]+arr[:,:,1])/32
        curve = next(v for v in saved['curves'][key[0]+'|h32'] if (v['policy'],v['K']) == key[1:])
        np.testing.assert_allclose(accuracies[key].mean(),curve['post_accuracy'],rtol=0,atol=1e-12)
    checks=0
    answer_checks=0
    for suite, policy, k in accuracies:
        if policy != 'restart':
            value = interval(accuracies[suite,policy,k]-accuracies[suite,'restart',k])
            expected = next(v['accuracy'] for v in saved['contrasts_vs_restart'][suite+'|h32'] if (v['policy'],v['K'])==(policy,k))
            for name in ('delta','root_ci','crossed_ci'):
                np.testing.assert_allclose(value[name],expected[name],rtol=0,atol=1e-12)
            checks+=1
        if policy != 'answer_only' and suite in saved['exploratory_contrasts_vs_answer_only']:
            value = interval(accuracies[suite,policy,k]-accuracies[suite,'answer_only',k])
            expected = next(v for v in saved['exploratory_contrasts_vs_answer_only'][suite] if (v['policy'],v['K'])==(policy,k))
            for name in ('delta','root_ci','crossed_ci'):
                np.testing.assert_allclose(value[name],expected[name],rtol=0,atol=1e-12)
            answer_checks+=1
    for sample in samples:
        example=decode(sample['example'])
        assert not sample['synthetic']
        assert score(example,sample['previous_actions'])['exact_correct']==sample['previous_correct']
        assert score(example,sample['current_actions'])['exact_correct']==sample['current_correct']
    for row in diagnostic['frequencies']:
        filename=row['suite'].split('-on-')[-1]+'-test.json'
        source=folder / 'datasets' / filename
        expected=diagnostic['sources'][str(Path('runs/confirmatory_v1/data')/filename)]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        data=json.loads(source.read_text())
        roots=[decode(r) for r in data['test']]

        stream=frames(roots,32,64019)
        counts=defaultdict(int)
        previous=None
        for examples in stream:
            current=[PreparedScore(e) for e in examples]
            if previous is not None:
                for a,b in zip(previous,current): counts[b.consequence(a)]+=1
            previous=current
        assert dict(counts)==row['ordinary']
    result={'episode_rows':len(episodes),'diagnostic_points':len(grouped),
            'original_contrasts_reproduced':checks,'raw_samples_rescored':len(samples),
            'output_reuse_contrasts_reproduced':answer_checks,
            'ordinary_frequencies_reproduced':len(diagnostic['frequencies']), 'tolerance':1e-12}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
