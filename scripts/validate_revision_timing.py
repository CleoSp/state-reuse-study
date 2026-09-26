"""Validate timing completeness, paired sample identity and saved prediction scores."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from state_repair.execution.datasets import decode, frames
from state_repair.eval.metrics import score


def main() -> None:
    folder=ROOT/'reports/paper/review_revision'
    summary=json.loads((folder/'timing_summary.json').read_text())
    blocks=[json.loads(line) for line in (folder/'timing_blocks.jsonl').read_text().splitlines()]
    predictions=json.loads((folder/'timing_predictions.json').read_text())['records']
    assert summary['complete'] and summary['identical_actions'] and len(blocks)==900 and len(predictions)==150
    assert hashlib.sha256((folder/'timing_blocks.jsonl').read_bytes()).hexdigest()==summary['records_sha256']
    assert hashlib.sha256((folder/'timing_predictions.json').read_bytes()).hexdigest()==summary['predictions_sha256']
    suites=('maze12-on-maze12','circuit32-on-circuit32','circuit32-on-circuit48')
    policies=('restart','carry','spatial_gate','global_gate','gru_adapter','residual_adapter','answer_only')
    identities,examples={},{}
    for suite in suites:
        filename=suite.split('-on-')[-1]+'-test.json'
        payload=json.loads((folder/'datasets'/filename).read_text())
        roots=[decode(r) for r in payload['test'][:2]]
        identities[suite]=[r.root_id for r in roots]
        for root in roots:
            examples[suite,root.root_id]=[f[0] for f in frames([root],32,64019)]
    tasks=[(suite,arm,k,root,mode) for suite in suites
           for arm in (policies if suite.startswith('maze') else ('restart','carry','spatial_gate','answer_only'))
           for k in (1,2,4,8,16) for root in identities[suite] for mode in ('instrumented','lean')]
    rng=random.Random(9222026)
    index=0
    for repeat in range(3):
        rng.shuffle(tasks)
        for key in tasks:
            row=blocks[index]
            assert tuple(row[k] for k in ('suite','policy','K','root_id','mode'))==key
            assert row['repetition']==repeat and row['order_index']==index
            assert row['seed']==29 and row['frames']==33 and row['synthetic'] is False
            assert row['amortized_ms']>0 and abs(row['amortized_ms']*33-row['total_ms'])<1e-8
            index+=1
    accuracy={}
    for row in predictions:
        key=row['suite'],row['policy'],row['K'],row['root_id']
        assert key not in accuracy and len(row['actions'])==33
        truth=[score(e,a)['exact_correct'] for e,a in zip(examples[row['suite'],row['root_id']],row['actions'])]
        accuracy[key]=(sum(truth[1:])/32,all(truth))
    groups=defaultdict(list)
    for row in blocks:
        expected=accuracy[row['suite'],row['policy'],row['K'],row['root_id']]
        assert (row['post_accuracy'],row['whole_stream_success'])==expected
        groups[row['suite'],row['policy'],row['K'],row['mode']].append(row)
    for point in summary['points']:
        rows=groups[point['suite'],point['policy'],point['K'],point['mode']]
        assert len(rows)==point['blocks']==6
        assert abs(sum(r['amortized_ms'] for r in rows)/6-point['mean_ms'])<1e-12
        assert abs(sum(r['post_accuracy'] for r in rows)/6-point['post_accuracy'])<1e-12
    print(json.dumps({'timing_blocks':900,'paired_stream_configurations':150,
        'prediction_frames_rescored':150*33,'point_estimates_checked':len(summary['points']),
        'seeded_interleaved_schedule':'pass','original_frozen_records':'unchanged',
        'gpu_job_seconds':summary['elapsed_seconds'],'peak_tensor_bytes':summary['peak_allocated_bytes']},indent=2))


if __name__=='__main__':
    main()
