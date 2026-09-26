"""CPU measurement of the additional full-data training-cache setup component."""
from pathlib import Path
from time import perf_counter
import gc
from state_repair.execution.jobs import build
from state_repair.execution.durable import read_json, atomic_json
from state_repair.provenance import file_hash

root=Path('runs/confirmatory_v1')
rows=[]
for path in sorted(root.glob('capacity-*/profile.json')):
    config=read_json(path)['config']
    if config.get('kind') != 'research_training':
        continue
    data=root/'data'/f"{config['family']}{config['size']}-development.json"
    job={**config,'device':'cpu','dataset':str(data),'synthetic':False}
    start=perf_counter()
    trainer=build(job,root)
    duration=perf_counter()-start
    row={'profile':path.parent.name,'full_development_build_cpu_seconds':duration,
         'dataset_sha256':file_hash(data),'batches':len(trainer.batches),'GPU_work':False,
         'scope':'CPU full-root stream generation, typed collation and CPU model/optimizer construction; no optimizer step'}
    rows.append(row)
    print(row,flush=True)
    del trainer
    gc.collect()
atomic_json(Path('reports/foundation/prompt06-preparation-measurement.json'),{'rows':rows,'script_sha256':file_hash(__file__)})
