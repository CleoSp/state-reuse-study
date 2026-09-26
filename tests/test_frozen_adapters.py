"""Synthetic paired forks, observation boundary and current-gradient checks."""
import pytest
import torch

from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.eval.frozen import frozen_adapter_prediction, frozen_prediction
from state_repair.models.adapters import make_adapter
from state_repair.models.recursive import RecursiveSolver


def inputs():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
    old, _, _ = collate([MazeExample(maze, "synthetic", 0, "val", True)])
    new, _, _ = collate([MazeExample(maze.toggle(2, 3), "synthetic", 1, "val", True)])
    return old, new


@pytest.mark.parametrize("arm", ["restart", "carry"])
def test_loaded_adapter_matches_original_crossover_and_preserves_prior(arm):
    torch.manual_seed(7)
    model = RecursiveSolver(width=8, heads=2)
    old, new = inputs()
    prior = model(old, 4).state
    before = prior.a.detach().clone(), prior.z.detach().clone()
    result = frozen_adapter_prediction(model, make_adapter(arm), old, new, prior, 1)
    expected = frozen_prediction(model, old, new, prior, arm, 1)
    torch.testing.assert_close(result.solver.prediction.logits, expected.prediction.logits, rtol=0, atol=0)
    assert result.source_budget == 4 and result.solver.state.budget == 1
    assert result.record_kind == "frozen_state_intervention"
    assert torch.equal(prior.a, before[0]) and torch.equal(prior.z, before[1])


@pytest.mark.parametrize("arm", ["spatial_gate", "global_gate", "gru_adapter", "residual_adapter", "answer_only"])
def test_frozen_fork_keeps_current_adapter_gradient_without_prior_graph(arm, monkeypatch):
    torch.manual_seed(7)
    old, new = inputs()
    model = RecursiveSolver(width=8, heads=2).requires_grad_(False)
    initial = model(old, 4)
    initial.state.a.requires_grad_(True)
    initial.state.z.requires_grad_(True)
    options = {"width": 8} if arm == "answer_only" else {"width": 8, "context_width": 4}
    adapter = make_adapter(arm, **options)
    kwargs = {"previous_prediction": initial.prediction} if arm == "answer_only" else {}
    import state_repair.oracles.maze as oracle
    def forbidden(*args, **kwargs):
        raise AssertionError("oracle must not enter inference")
    monkeypatch.setattr(oracle, "solve_maze", forbidden)
    result = frozen_adapter_prediction(model, adapter, old, new, initial.state, 1, **kwargs)
    result.solver.prediction.logits.square().mean().backward()
    assert initial.state.a.grad is None and initial.state.z.grad is None
    assert sum(float(p.grad.abs().sum()) for p in adapter.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.parameters())
    if arm == "answer_only":
        with pytest.raises(TypeError, match="previous model prediction"):
            frozen_adapter_prediction(model, adapter, old, new, initial.state, 1)
