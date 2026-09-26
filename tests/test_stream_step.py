"""Synthetic checks against a separately accumulated full-episode reference."""
from copy import deepcopy

import pytest
import torch

from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.models.adapters import make_adapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.adapter_step import supervised_loss
from state_repair.train.stream_step import stream_backward


def episode():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
    frames = (maze, maze.toggle(2, 3), maze.toggle(0, 1), maze)
    batches = [collate([MazeExample(m, "synthetic", t, "train", True)]) for t, m in enumerate(frames)]
    return [b[0] for b in batches], [b[1] for b in batches]


@pytest.mark.parametrize("track", ["frozen", "joint"])
@pytest.mark.parametrize("arm", ["spatial_gate", "gru_adapter", "carry", "restart"])
def test_incremental_backward_matches_full_episode_gradient(track, arm):
    torch.manual_seed(91)
    model = RecursiveSolver(width=8, heads=2).requires_grad_(track == "joint")
    adapter = make_adapter(arm, **({"width": 8, "context_width": 4} if arm in ("spatial_gate", "gru_adapter") else {}))
    reference_model, reference_adapter = deepcopy(model), deepcopy(adapter)
    observations, targets = episode()
    records = stream_backward(model, adapter, observations, targets, 2, track=track)
    policy = FixedBudgetPolicy(reference_model, reference_adapter, 2)
    losses = []
    for obs, target in zip(observations, targets):
        result = policy(obs)
        assert result.state.budget == 2
        assert not policy._previous.state.a.requires_grad
        losses.append(supervised_loss(obs, result.prediction, target))
    objective = torch.stack(losses).mean()
    if objective.requires_grad:
        objective.backward()
    assert [r.frame for r in records] == list(range(4))
    assert all(r.block_calls == 4 for r in records)
    assert [r.loss for r in records] == [float(v.detach()) for v in losses]
    for got, want in zip([*model.parameters(), *adapter.parameters()],
                         [*reference_model.parameters(), *reference_adapter.parameters()]):
        assert (got.grad is None) == (want.grad is None)
        if got.grad is not None:
            torch.testing.assert_close(got.grad, want.grad, rtol=2e-5, atol=2e-6)
    if arm in ("spatial_gate", "gru_adapter"):
        assert sum(float(p.grad.abs().sum()) for p in adapter.parameters() if p.grad is not None) > 0


def test_stream_rejects_missing_frames_and_broken_current_gradient():
    from dataclasses import replace
    model = RecursiveSolver(width=8, heads=2).requires_grad_(False)
    adapter = make_adapter("spatial_gate", width=8, context_width=4)
    observations, targets = episode()
    with pytest.raises(ValueError, match="consecutive"):
        stream_backward(model, adapter, observations[1:], targets[1:], 2, track="frozen")
    def detach_output(module, args, output):
        return replace(output, state=output.state.detach())
    hook = adapter.register_forward_hook(detach_output)
    with pytest.raises(RuntimeError, match="gradient path"):
        stream_backward(model, adapter, observations, targets, 2, track="frozen")
    hook.remove()
    with torch.no_grad(), pytest.raises(ValueError, match="autograd"):
        stream_backward(model, adapter, observations, targets, 2, track="frozen")
