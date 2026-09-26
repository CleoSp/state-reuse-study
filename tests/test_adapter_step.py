"""Synthetic optimizer and cache controls for the actual pilot training path."""
from dataclasses import replace

import pytest
import torch

from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.models.adapters import make_adapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.adapter_step import FrozenPrior, one_edit_loss
from state_repair.types import PredictionBatch


def fixture():
    maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
    old, old_target, _ = collate([MazeExample(maze, "synthetic", 0, "train", True)])
    new, new_target, _ = collate([MazeExample(maze.toggle(2, 3), "synthetic", 1, "train", True)])
    return old, new, old_target, new_target


@pytest.mark.parametrize("track", ["frozen", "joint"])
def test_actual_one_edit_path_updates_adapter_and_detaches_prior(track):
    torch.manual_seed(89)
    model = RecursiveSolver(width=8, heads=2)
    model.requires_grad_(track == "joint")
    adapter = make_adapter("spatial_gate", width=8, context_width=4)
    before = adapter.head.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in [*model.parameters(), *adapter.parameters()] if p.requires_grad], lr=.01)
    result = one_edit_loss(model, adapter, *fixture(), 2, track=track)
    assert not result.prior_state.a.requires_grad and not result.prior_state.z.requires_grad
    assert result.initial_block_calls_executed == result.post_edit_block_calls == 4
    assert result.state.budget == result.prior_state.budget == 2
    assert result.initial_loss.requires_grad == (track == "joint")
    result.loss.backward()
    assert torch.isfinite(adapter.head.weight.grad).all() and adapter.head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.parameters()) == (track == "joint")
    optimizer.step()
    assert not torch.equal(before, adapter.head.weight)


def test_frozen_cached_path_matches_recomputation_and_rejects_wrong_provenance():
    torch.manual_seed(89)
    model = RecursiveSolver(width=8, heads=2).requires_grad_(False)
    adapter = make_adapter("spatial_gate", width=8, context_width=4)
    old, new, old_target, new_target = fixture()
    with torch.no_grad():
        initial = model(old, 2)
    cache = FrozenPrior(old, initial.state.detach(), PredictionBatch(initial.prediction.logits.detach().clone()),
                        "synthetic-checkpoint", "restart", 2)
    plain = one_edit_loss(model, adapter, old, new, old_target, new_target, 2, track="frozen")
    cached = one_edit_loss(model, adapter, old, new, old_target, new_target, 2, track="frozen",
                           cache=cache, checkpoint_sha256="synthetic-checkpoint")
    torch.testing.assert_close(plain.prediction.logits, cached.prediction.logits, atol=0, rtol=0)
    assert cached.initial_block_calls_executed == 0
    cached.loss.backward()
    assert adapter.head.weight.grad.abs().sum() > 0
    for bad in (replace(cache, checkpoint_sha256="wrong"), replace(cache, budget=8), replace(cache, source_policy="carry")):
        with pytest.raises(ValueError, match="provenance"):
            one_edit_loss(model, adapter, old, new, old_target, new_target, 2, track="frozen",
                          cache=bad, checkpoint_sha256="synthetic-checkpoint")
    changed = replace(old, node_features=old.node_features+1)
    with pytest.raises(ValueError, match="observed input"):
        cache.check(changed, "synthetic-checkpoint", 2)
    model.requires_grad_(True)
    with pytest.raises(ValueError, match="freshly recomputed"):
        one_edit_loss(model, adapter, old, new, old_target, new_target, 2, track="joint", cache=cache)


def test_training_path_broken_detach_negative_control():
    torch.manual_seed(89)
    model = RecursiveSolver(width=8, heads=2).requires_grad_(False)
    adapter = make_adapter("spatial_gate", width=8, context_width=4)
    def break_path(module, args, output):
        return replace(output, state=output.state.detach())
    hook = adapter.register_forward_hook(break_path)
    broken = one_edit_loss(model, adapter, *fixture(), 2, track="frozen")
    hook.remove()
    assert not broken.loss.requires_grad
    assert all(p.grad is None for p in adapter.parameters())
    with torch.no_grad(), pytest.raises(ValueError, match="requires autograd"):
        one_edit_loss(model, adapter, *fixture(), 2, track="frozen")


def test_circuit_adapter_training_path_has_nonzero_gradient():
    from state_repair.data.circuit import CircuitExample, Operator, collate as circuit_collate, generate_circuit
    from state_repair.types import Domain
    torch.manual_seed(89)
    circuit = generate_circuit(8, 73)
    node = next(u for u, op in enumerate(circuit.operators) if op == Operator.INPUT)
    edited = circuit.flip_input(node)
    old, old_target, _ = circuit_collate([CircuitExample(circuit, "synthetic", 0, "train", tuple(range(8)), True)])
    new, new_target, _ = circuit_collate([CircuitExample(edited, "synthetic", 1, "train", tuple(range(8)), True)], previous=[circuit])
    model = RecursiveSolver(width=8, heads=2, domain=Domain.CIRCUIT).requires_grad_(False)
    adapter = make_adapter("spatial_gate", width=8, context_width=4, domain=Domain.CIRCUIT)
    result = one_edit_loss(model, adapter, old, new, old_target, new_target, 2, track="frozen")
    result.loss.backward()
    assert torch.isfinite(adapter.head.weight.grad).all() and adapter.head.weight.grad.abs().sum() > 0
