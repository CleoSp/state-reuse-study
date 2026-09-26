"""Exploratory saved-prediction diagnostics; no inference or test selection."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from state_repair.data.maze import MazeExample
from state_repair.data.circuit import Operator
from state_repair.execution.datasets import decode, frames
from state_repair.execution.durable import digest_json
from state_repair.execution.records import step_rows
from state_repair.eval.metrics import score, input_hash, validate_record
from state_repair.eval.jobs import resolve_source
from state_repair.oracles.maze import solve_maze
from state_repair.oracles.circuit import evaluate

RUNS = ROOT / 'runs/confirmatory_v1'
OUT = ROOT / 'reports/paper/review_revision'
PRIMARY = ('maze12-on-maze12', 'circuit32-on-circuit32', 'circuit32-on-circuit48')
SUITES = (*PRIMARY, 'maze16-on-maze16', 'circuit64-on-circuit64', 'circuit64-on-circuit96')
SEEDS = (29, 43, 71, 101, 137)
KS = (1, 2, 4, 8, 16)
POLICIES = ('restart', 'carry', 'spatial_gate', 'global_gate', 'gru_adapter', 'residual_adapter', 'answer_only')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class PreparedScore:
    """Cache evaluator targets; follow old argmax maps without any repair."""
    def __init__(self, example):
        self.example = example
        self.is_maze = isinstance(example, MazeExample)
        if self.is_maze:
            self.dist, self.targets = solve_maze(example.maze)
            self.passages = set(example.maze.edges)
        else:
            values = evaluate(example.circuit)
            self.targets = [values[u] for u in example.node_order]
            self.scored = [i for i, u in enumerate(example.node_order)
                           if example.circuit.operators[u] != Operator.INPUT]

    def exact(self, actions: list[int]) -> bool:
        if not self.is_maze:
            return all(actions[i] == self.targets[i] for i in self.scored)
        maze = self.example.maze
        u, steps, visited = maze.start, 0, set()
        while u not in visited and steps <= maze.n:
            visited.add(u)
            a = actions[u]
            if a == 5:
                return u == maze.start and self.dist[u] == -1
            if a == 4:
                return u == maze.goal and steps == self.dist[maze.start]
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[a]
            r, c = u // maze.width + dr, u % maze.width + dc
            v = r * maze.width + c
            if not (0 <= r < maze.height and 0 <= c < maze.width):
                return False
            if (min(u, v), max(u, v)) not in self.passages:
                return False
            u, steps = v, steps + 1
        return False

    def consequence(self, previous: PreparedScore) -> str:
        ids = range(len(self.targets)) if self.is_maze else self.scored
        fraction = sum(self.targets[i] != previous.targets[i] for i in ids) / len(ids)
        return 'low' if fraction <= .1 else 'high' if fraction >= .4 else 'middle'


def interval(delta: np.ndarray) -> dict:
    rng = np.random.default_rng(64037)
    roots = rng.integers(0, delta.shape[1], (10000, delta.shape[1]))
    seeds = rng.integers(0, delta.shape[0], (10000, delta.shape[0]))
    conditional = delta.mean(0)[roots].mean(1)
    crossed = np.concatenate([delta[seeds[i:i+100, :, None], roots[i:i+100, None, :]].mean((1, 2))
                              for i in range(0, 10000, 100)])
    return {'delta': float(delta.mean()), 'root_ci': np.quantile(conditional, [.025, .975]).tolist(),
            'crossed_ci': np.quantile(crossed, [.025, .975]).tolist(),
            'per_seed_delta': delta.mean(1).tolist(), 'bootstrap_seed': 64037, 'draws': 10000}


def main() -> None:
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / 'reports/confirmatory/summary.json').read_text())
    sources, results, episodes, samples, frequencies = {}, [], [], [], []
    for suite in SUITES:
        anchor = RUNS / f'test-{suite}-restart-s29-h32/job.json'
        anchor_job = json.loads(anchor.read_text())
        dataset = ROOT / anchor_job['dataset']
        sources[str(dataset.relative_to(ROOT))] = sha(dataset)
        payload = json.loads(dataset.read_text())
        roots = [decode(r) for r in payload['test']]
        stream = frames(roots, 32, anchor_job['stream_seed'])
        prepared = {(e.root_id, f): PreparedScore(e) for f, examples in enumerate(stream) for e in examples}
        hashes = {key: input_hash(value.example) for key, value in prepared.items()}
        ordinary = Counter(prepared[e.root_id, f].consequence(prepared[e.root_id, f-1])
                           for f, examples in enumerate(stream) if f for e in examples)
        challenge = Counter()
        for entry in payload.get('interventions', []):
            if entry['branch'] != 'ordinary':
                challenge[PreparedScore(decode(entry['new'])).consequence(PreparedScore(decode(entry['old'])))] += 1
        frequencies.append({'suite': suite, 'ordinary': dict(ordinary), 'transitions': 256 * 32,
                            'roots': 256, 'challenge': dict(challenge),
                            'challenge_branches': sum(challenge.values())})
        print(f'{suite}: ordinary consequence counts {dict(ordinary)}', flush=True)
        if suite not in PRIMARY:
            continue
        policies = POLICIES if suite.startswith('maze') else ('restart', 'carry', 'spatial_gate', 'answer_only')
        suite_counts = {}
        root_ids = sorted(e.root_id for e in roots)
        for policy in policies:
            for seed in SEEDS:
                directory = RUNS / f'test-{suite}-{policy}-s{seed}-h32'
                seal = json.loads((directory / 'seal.json').read_text())
                for name, expected in seal['files'].items():
                    if sha(directory / name) != expected:
                        raise ValueError(f'Seal mismatch: {directory.name}/{name}')
                sources[str((directory / 'seal.json').relative_to(ROOT))] = sha(directory / 'seal.json')
                sources[str((directory / 'steps.jsonl.gz').relative_to(ROOT))] = seal['files']['steps.jsonl.gz']
                job = json.loads((directory / 'job.json').read_text())
                source = resolve_source(job, RUNS)
                checkpoint_hash = sha(RUNS / source / 'checkpoint.pt')
                config_hash, data_hash = digest_json(job), sha(dataset)
                previous, counts, seen = {}, defaultdict(lambda: [0, 0, 0, 0]), set()
                for unit in step_rows(directory):
                    for row in unit['records']:
                        validate_record(row)
                        k, f, root = row['K'], row['frame'], row['root_id']
                        key = (k, root)
                        if (k, root, f) in seen or k not in KS or root not in root_ids:
                            raise ValueError('Duplicate or unexpected frame')
                        seen.add((k, root, f))
                        if (row['config_sha256'] != config_hash or row['dataset_sha256'] != data_hash
                            or row['checkpoint_sha256'] != checkpoint_hash or row['checkpoint_job'] != source
                            or row['input_sha256'] != hashes[root, f]):
                            raise ValueError('Saved row provenance mismatch')
                        target = prepared[root, f]
                        after = target.exact(row['actions'])
                        if after != row['exact_correct']:
                            raise ValueError('Independent cached scorer disagrees with saved score')
                        if f:
                            old_frame, old_actions = previous[key]
                            if old_frame != f - 1:
                                raise ValueError('Nonconsecutive own-budget history')
                            before = target.exact(old_actions)

                            category = 0 if before and after else 1 if after else 2 if before else 3
                            counts[key][category] += 1
                            if root == root_ids[0] and f in (1, 16, 32):
                                assert before == score(target.example, old_actions)['exact_correct']
                                assert after == score(target.example, row['actions'])['exact_correct']
                                samples.append({'suite': suite, 'policy': policy, 'seed': seed, 'K': k,
                                    'frame': f, 'example': asdict(target.example), 'previous_actions': old_actions,
                                    'current_actions': row['actions'], 'previous_correct': before,
                                    'current_correct': after, 'synthetic': False})
                        previous[key] = (f, row['actions'])
                if len(seen) != 5 * 256 * 33 or len(counts) != 5 * 256:
                    raise ValueError('Incomplete diagnostic matrix')
                for k in KS:
                    rows = [counts[k, root] for root in root_ids]
                    if any(sum(v) != 32 for v in rows):
                        raise ValueError('Incomplete episode')
                    suite_counts[policy, seed, k] = rows
                    mean = sum(v[0] + v[1] for v in rows) / (256 * 32)
                    saved = next(v for v in summary['curves'][suite + '|h32'] if v['policy'] == policy and v['K'] == k)
                    assert abs(mean - saved['per_seed_accuracy'][str(seed)]) < 1e-12
                    episodes.extend({'suite': suite, 'policy': policy, 'seed': seed, 'K': k, 'root_id': root,
                                     'counts': counts[k, root], 'synthetic': False} for root in root_ids)
                print(f'  checked {policy} seed {seed}: {len(seen)} predictions', flush=True)
            for k in KS:
                arr = np.asarray([suite_counts[policy, seed, k] for seed in SEEDS])
                total = arr.sum((0, 1)); n = int(total.sum())
                cc, repaired, broken, ww = [int(v) for v in total]
                results.append({'suite': suite, 'policy': policy, 'K': k, 'seeds': list(SEEDS), 'roots': 256,
                    'transitions': n, 'counts': total.tolist(), 'previous_accuracy': (cc + broken) / n,
                    'post_accuracy': (cc + repaired) / n, 'repair_rate': repaired / (repaired + ww) if repaired + ww else None,
                    'break_rate': broken / (cc + broken) if cc + broken else None,
                    'gain': interval((arr[:, :, 1] - arr[:, :, 2]) / 32)})
    for name, values in [('previous_prediction_episodes.json.gz', episodes), ('previous_prediction_audit.json.gz', samples)]:
        with gzip.open(OUT / name, 'wt', encoding='utf-8') as handle:
            json.dump(values, handle, separators=(',', ':'))
    result = {'synthetic': False, 'exploratory': True, 'scope': 'own-history previous argmax diagnostic, not cache-only deployment',
        'sources': sources, 'results': results, 'frequencies': frequencies, 'episode_rows': len(episodes),
        'audit_samples': len(samples), 'elapsed_seconds': time.perf_counter() - start,
        'gpu_seconds': 0, 'outputs': {p.name: sha(p) for p in OUT.glob('previous_prediction_*.json.gz')}}
    (OUT / 'previous_prediction_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Complete: {len(results)} points, {len(episodes)} episodes, {len(samples)} audit samples; {result["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
