"""Export existing sealed latency curves for review; no new measurements."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    curves, sources = {}, {}
    for suite in ('maze12-on-maze12','circuit32-on-circuit32','circuit32-on-circuit48'):
        directory = ROOT/'runs/confirmatory_v1'/('report-latency-'+suite)
        seal = json.loads((directory/'seal.json').read_text())
        report = directory/'results.json'
        digest = hashlib.sha256(report.read_bytes()).hexdigest()
        if digest != seal['files']['results.json']:
            raise ValueError('Sealed latency report changed')
        rows = json.loads(report.read_text())['curves']
        curves[suite] = [r for r in rows if r['K'] > 0]
        assert len(curves[suite]) == (35 if suite.startswith('maze') else 20)
        assert all(r['roots']==8 and r['seeds']==[29] for r in curves[suite])
        sources[str(report.relative_to(ROOT))] = digest
    out = ROOT/'reports/paper/review_revision/original_latency.json'
    out.write_text(json.dumps({'synthetic':False,'new_measurements':False,'sources':sources,'curves':curves},indent=2)+'\n')
    print('Exported 75 original batch-one latency/accuracy points, eight roots and seed 29 each.')


if __name__=='__main__':
    main()
