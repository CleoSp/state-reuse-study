"""Synthetic CPU checks for route scoring and carry-state diagnostics."""
import pytest
import torch

from state_repair.data.maze import MazeExample, generate_maze
from state_repair.data.circuit import CircuitExample, generate_circuit
from state_repair.execution.datasets import collate, frames
from state_repair.models.adapters import make_adapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.eval.lean import LeanPolicy
from state_repair.types import Domain


@pytest.mark.parametrize('family', ['maze', 'circuit'])
@pytest.mark.parametrize('arm', ['restart', 'carry', 'answer_only', 'spatial_gate', 'global_gate', 'gru_adapter', 'residual_adapter'])
def test_lean_matches_original_complete_stream(family, arm):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    domain = Domain(family)
    solver = RecursiveSolver(width=16, heads=2, inner_cycles=2,
                             attention_mode='masked_neighbor', domain=domain).eval()
    kwargs = {} if arm in ('restart', 'carry') else {'width': 16, 'domain': domain}
    adapter = make_adapter(arm, **kwargs).eval()
    root = (MazeExample(generate_maze(3, 3, 19), 'synthetic-test', 0, 'test', True)
            if family == 'maze' else CircuitExample(generate_circuit(8, 19, 2),
                'synthetic-test', 0, 'test', tuple(range(8)), True))
    stream = frames([root], 3, 31)
    for k in (1, 2):
        original, lean = FixedBudgetPolicy(solver, adapter, k), LeanPolicy(solver, adapter, k)
        with torch.no_grad():
            for f, examples in enumerate(stream):
                obs = collate(examples, None if f == 0 else stream[f-1])[0]
                expected = original(obs).prediction.logits
                assert torch.equal(expected, lean(obs).logits)
        if arm == 'restart':
            assert lean.old is lean.state is lean.prediction is None
        elif arm == 'answer_only':
            assert lean.old is lean.state is None and lean.prediction is not None
        else:
            assert lean.prediction is None and lean.state is not None
        lean.reset()
        with torch.no_grad(), pytest.raises(ValueError, match='start at zero'):
            lean(collate(stream[1], stream[0])[0])


@pytest.mark.parametrize('family', ['maze', 'circuit'])
def test_cached_diagnostic_scorer_matches_reference(family):
    import random
    from scripts.analyze_previous_predictions import PreparedScore
    from state_repair.eval.metrics import score
    root = (MazeExample(generate_maze(3, 3, 31), 'synthetic-scorer', 0, 'test', True)
            if family == 'maze' else CircuitExample(generate_circuit(8, 31, 2),
                'synthetic-scorer', 0, 'test', tuple(reversed(range(8))), True))
    rng = random.Random(17)
    for examples in frames([root], 3, 23):
        prepared = PreparedScore(examples[0])
        for _ in range(50):
            actions = [rng.randrange(6 if family == 'maze' else 2) for _ in prepared.targets]
            assert prepared.exact(actions) == score(examples[0], actions)['exact_correct']
        if family == 'maze':
            actions = [next(i for i, allowed in enumerate(v) if allowed) for v in prepared.targets]
        else:
            actions = prepared.targets.copy()
        assert prepared.exact(actions)


def test_old_maze_route_must_remain_shortest_not_merely_valid():
    from scripts.analyze_previous_predictions import PreparedScore
    from state_repair.data.maze import Maze
    old = Maze(2, 2, ((0,2),(1,3),(2,3)), 0, 1)
    current = old.toggle(0,1)
    old_actions = [2,4,1,0]
    before = MazeExample(old, 'synthetic-route', 0, 'test', True)
    after = MazeExample(current, 'synthetic-route', 1, 'test', True)
    assert PreparedScore(before).exact(old_actions)
    assert not PreparedScore(after).exact(old_actions)
    assert PreparedScore(after).exact([1,4,5,5])
    assert not PreparedScore(after).exact([5,4,5,5])
