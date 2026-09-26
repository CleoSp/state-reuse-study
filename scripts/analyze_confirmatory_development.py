"""Select operating points and estimate training-seed variance from validation records."""
from collections import defaultdict
from pathlib import Path

from state_repair.execution.durable import read_json, atomic_json
from state_repair.provenance import file_hash
from state_repair.eval.statistics import select_operating_points, paired_contrast, sample_size

root=Path('runs/confirmatory_v1')
sources={}
out={"synthetic":False,"split":"val","test_generated":False,"families":{},"sample_size":{}}
all_episodes={}
for family, directory in (("maze","adapter_stream_v2"),("circuit","circuit_stream_v2")):
    costs=root/('development-cost-'+family)/'profile.json'
    profile=read_json(costs)
    sources[costs.as_posix()]=file_hash(costs)
    grouped=defaultdict(list)
    for r in profile['rows']:
        if r['split']!='val' or r['synthetic'] is not False or r['state_budget']!=r['K']:
            raise ValueError('invalid development timing record')
        grouped[r['policy'],r['K']].append(r['milliseconds']['total'])
    curves=[]
    for arm in ('restart','answer_only'):
        ep=[]
        for seed in (29,43,71):
            path=Path('runs')/directory/'streams'/f'joint-{arm}-seed{seed}'/'summary.json'
            summary=read_json(path)
            sources[path.as_posix()]=file_hash(path)
            ep.extend({**r,'seed':seed,'policy':arm,'synthetic':False,'post_accuracy':r['post32']}
                      for r in summary['validation']['episodes'])
        all_episodes[family,arm]=ep
        for k in sorted({r['K'] for r in ep}):
            values=[r['post32'] for r in ep if r['K']==k]
            curves.append({'policy':arm,'K':k,'post_accuracy':sum(values)/len(values),
                'amortized_ms':sum(grouped[arm,k])/len(grouped[arm,k]),'synthetic':False,'split':'val'})
    point=select_operating_points(curves)
    out['families'][family]={'operating_point':point,'curves':curves,
        'reuse_policy_basis':'05b G2 closed; answer-only is the surviving G3 candidate. No latent-reuse claim is reopened.',
        'cost_scope':'first four reserved validation roots, three existing adaptation seeds, two warmed repetitions, batch one, initial solve included',
        'cost_limitations':'development timing under the local WDDM host workload; some CPU correctness/data-preparation work overlapped; not final test latency',
        'ineligible_unmeasured_K':[16] if family=='maze' else [4,8,16]}
    a=[r for r in all_episodes[family,'answer_only'] if r['K']==point['reuse']['K']]
    b=[r for r in all_episodes[family,'restart'] if r['K']==point['comparator']['K']]
    contrast=paired_contrast(a,b)
    out['sample_size'][family]={'development_contrast':contrast,
        'at_zero_accuracy_loss':sample_size(contrast['root_sd'],contrast['training_seed_sd']),
        'at_observed_development_delta':sample_size(contrast['root_sd'],contrast['training_seed_sd'],true_delta=contrast['delta'])}


selection=read_json(Path('runs/circuit_stream_v2/selection.json'))
sources['runs/circuit_stream_v2/selection.json']=file_hash('runs/circuit_stream_v2/selection.json')
point=out['families']['circuit']['operating_point']
paired={}
for arm in ('restart','answer_only'):
    ep=[]
    for seed in (29,43,71):
        path=Path('runs/circuit_stream_v2_depth48/streams')/f"joint-{arm}-seed{seed}-grid{selection[arm]['grid']}"/'summary.json'
        summary=read_json(path); sources[path.as_posix()]=file_hash(path)
        ep.extend({**r,'seed':seed,'post_accuracy':r['post32'],'synthetic':False} for r in summary['validation']['episodes'])
    paired[arm]=[r for r in ep if r['K']==point['reuse' if arm=='answer_only' else 'comparator']['K']]
contrast=paired_contrast(paired['answer_only'],paired['restart'])
out['sample_size']['circuit_depth48']={'development_contrast':contrast,
    'at_zero_accuracy_loss':sample_size(contrast['root_sd'],contrast['training_seed_sd']),
    'at_observed_development_delta':sample_size(contrast['root_sd'],contrast['training_seed_sd'],true_delta=contrast['delta'])}
out['planning_limitation']='The three development adapter seeds shared one static backbone; these variances cannot identify variation over independent new backbones. Five new static seeds are retained; normal power is an approximation, not a guarantee.'
out['source_hashes']=sources
atomic_json(Path('reports/foundation/prompt06-development-selection.json'),out)
print({f:v['operating_point'] for f,v in out['families'].items()},flush=True)
print(out['sample_size'],flush=True)
