"""Bounded exploratory latency sensitivity; frozen records are read-only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from state_repair.execution.datasets import collate, frames
from state_repair.execution.jobs import environment
from state_repair.eval.jobs import EvaluationJob
from state_repair.eval.lean import LeanPolicy
from state_repair.eval.metrics import score
from state_repair.eval.timing import measured_frame
from state_repair.models.policy import FixedBudgetPolicy

RUNS = ROOT / 'runs/confirmatory_v1'
OUT = ROOT / 'reports/paper/review_revision'
POLICIES = ('restart', 'carry', 'spatial_gate', 'global_gate', 'gru_adapter', 'residual_adapter', 'answer_only')
SUITES = ('maze12-on-maze12', 'circuit32-on-circuit32', 'circuit32-on-circuit48')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, data: dict) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n')


@torch.no_grad()
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'timing_blocks.jsonl').exists():
        raise ValueError('Refuse to overwrite a timing run')
    plan = ROOT / 'reports/paper/EXPLORATORY_REVISION_PLAN.md'
    write('timing_reservation.json', {'plan_sha256': sha(plan), 'authorized_by': 'user requested scoped review revisions',
        'max_wall_seconds': 1200, 'reserved_gpu_hours': 1200 / 3600, 'estimated_peak_bytes': 4 * 2**30,
        'external_charge_usd': 0, 'electricity': 'unmeasured', 'status': 'reserved'})
    start = time.perf_counter()
    models, streams, metadata = {}, {}, {}
    environment({'device': 'cuda', 'threads': 2, 'seed': 29, 'required_torch_version': '2.11.0+cu128'})
    torch.cuda.set_per_process_memory_fraction(4 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    for suite in SUITES:
        for arm in POLICIES if suite.startswith('maze') else ('restart', 'carry', 'spatial_gate', 'answer_only'):
            path = RUNS / f'test-{suite}-{arm}-s29-h32/job.json'
            frozen = json.loads(path.read_text())
            worker = EvaluationJob({**frozen, 'batch_size': 1, 'root_limit': 2}, RUNS)
            models[suite, arm] = (worker.solver, worker.adapter)
            metadata[suite, arm] = {'checkpoint_sha256': worker.checkpoint_hash,
                'checkpoint_job': worker.source, 'original_job_sha256': sha(path),
                'dataset_sha256': worker.dataset_hash}
            if suite not in streams:
                streams[suite] = []
                for root in worker.roots:
                    examples = frames([root], 32, frozen['stream_seed'])
                    observations = [collate(e, None if f == 0 else examples[f-1])[0] for f, e in enumerate(examples)]
                    streams[suite].append((root.root_id, observations, examples))
    tasks = [(suite, arm, k, root_index, mode)
             for suite, arm in models for k in (1, 2, 4, 8, 16) for root_index in (0, 1)
             for mode in ('instrumented', 'lean')]
    rng = random.Random(9222026)
    blocks, expected, predictions = [], {}, []
    complete = True
    for repetition in range(3):
        rng.shuffle(tasks)
        for suite, arm, k, root_index, mode in tasks:
            if time.perf_counter() - start > 1170:
                complete = False
                break
            solver, adapter = models[suite, arm]
            root_id, observations, examples = streams[suite][root_index]
            policy = (FixedBudgetPolicy(solver, adapter, k) if mode == 'instrumented'
                      else LeanPolicy(solver, adapter, k))
            policy(observations[0].to('cuda'))
            policy.reset()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            actions = []
            for obs in observations:
                if mode == 'instrumented':
                    _, output, _ = measured_frame(policy, obs, 'cuda')
                else:
                    result = policy(obs.to('cuda'))
                    output = result.logits.argmax(-1).cpu().tolist()
                actions.append(output[0])
            torch.cuda.synchronize()
            milliseconds = (time.perf_counter() - begin) * 1000
            policy.reset()
            key = suite, arm, k, root_id
            if key in expected and actions != expected[key]:
                raise ValueError(f'Action mismatch across paths/repetitions: {key}, {mode}')
            if key not in expected:
                expected[key] = actions
                predictions.append({'suite': suite, 'policy': arm, 'K': k, 'root_id': root_id, 'actions': actions})
            correctness = [bool(score(e[0], a)['exact_correct']) for e, a in zip(examples, actions)]
            row = {'suite': suite, 'policy': arm, 'K': k, 'root_id': root_id, 'mode': mode,
                'repetition': repetition, 'order_index': len(blocks), 'seed': 29, 'frames': 33,
                'total_ms': milliseconds, 'amortized_ms': milliseconds / 33,
                'post_accuracy': sum(correctness[1:]) / 32, 'whole_stream_success': all(correctness),
                'synthetic': False, **metadata[suite, arm]}
            blocks.append(row)
            with (OUT / 'timing_blocks.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if len(blocks) % 25 == 0:
                print(f'{len(blocks)}/900 stream blocks; {(time.perf_counter()-start):.1f}s', flush=True)
        if not complete:
            break
    grouped = {}
    for row in blocks:
        key = row['suite'], row['policy'], row['K'], row['mode']
        grouped.setdefault(key, []).append(row)
    points = []
    for (suite, arm, k, mode), rows in grouped.items():
        points.append({'suite': suite, 'policy': arm, 'K': k, 'mode': mode, 'blocks': len(rows),
            'mean_ms': sum(r['amortized_ms'] for r in rows) / len(rows),
            'min_ms': min(r['amortized_ms'] for r in rows), 'max_ms': max(r['amortized_ms'] for r in rows),
            'post_accuracy': sum(r['post_accuracy'] for r in rows) / len(rows)})
    elapsed = time.perf_counter() - start
    write('timing_predictions.json', {'synthetic': False, 'records': predictions})
    write('timing_summary.json', {'synthetic': False, 'exploratory': True, 'complete': complete,
        'plan_sha256': sha(plan), 'script_sha256': sha(Path(__file__)),
        'lean_implementation_sha256': sha(ROOT / 'src/state_repair/eval/lean.py'),
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(0), 'threads': torch.get_num_threads(),
        'tf32': False, 'seed': 29, 'roots_per_suite': 2, 'repetitions': 3,
        'order_seed': 9222026, 'blocks': len(blocks), 'expected_blocks': 900,
        'points': points, 'identical_actions': True, 'elapsed_seconds': elapsed,
        'gpu_job_hours': elapsed / 3600, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'external_charge_usd': 0,
        'electricity': 'unmeasured', 'records_sha256': sha(OUT / 'timing_blocks.jsonl'),
        'predictions_sha256': sha(OUT / 'timing_predictions.json')})
    print(f'Finished: complete={complete}, {len(blocks)} blocks, {elapsed:.1f}s, peak {torch.cuda.max_memory_allocated()} bytes', flush=True)
    if not complete:
        raise RuntimeError('Timing cap reached; retained all partial records, no complete result claimed')


if __name__ == '__main__':
    main()
