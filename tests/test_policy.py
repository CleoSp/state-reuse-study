"""Synthetic=true stream ownership and edit-boundary correctness fixtures."""
from dataclasses import replace

import pytest
import torch

from test_model import observation, deterministic_cpu
from state_repair.models import RecursiveSolver
from state_repair.models.adapters import make_adapter
from state_repair.models.policy import FixedBudgetPolicy


@pytest.mark.parametrize("key", ["restart", "carry"])
@pytest.mark.parametrize("k", [0, 2])
def test_policy_calls_and_exact_budget(key, k):
    model = RecursiveSolver(width=16, attention_mode="masked_neighbor")
    policy = FixedBudgetPolicy(model, make_adapter(key), k)
    actual = dict(encoder_calls=0, adapter_calls=0, decoder_calls=0, block_calls=0,
                  transformer_layer_executions=0)
    def hook(key):
        def count(*_):
            actual[key] += 1
        return count
    modules = [(model.encoder, "encoder_calls"), (policy.adapter, "adapter_calls"),
               (model.head, "decoder_calls"), (model.block, "block_calls")]
    modules += [(layer, "transformer_layer_executions") for layer in model.block.layers]
    handles = [module.register_forward_hook(hook(key)) for module, key in modules]
    first = policy(observation())
    result = policy(observation(frame=1))
    assert result.state.budget == result.outer_cycles == k
    for key, value in actual.items():
        assert value == 2 * result.operations[key]
    assert all(t >= 0 for t in result.milliseconds.values())
    assert policy._previous.state.a.grad_fn is None
    assert policy._previous.prediction.logits.grad_fn is None
    assert result.state.a.data_ptr() != first.state.a.data_ptr()
    assert result.state.a.data_ptr() != policy._previous.state.a.data_ptr()
    for handle in handles:
        handle.remove()


def test_restart_carry_boundary_semantics_and_detach():
    model = RecursiveSolver(width=16, attention_mode="masked_neighbor")
    policy = FixedBudgetPolicy(model, make_adapter("carry"), 2)
    first = policy(observation())
    first.state.a.retain_grad()
    changed = observation(frame=1)
    result = policy(changed, first)
    assert torch.equal(result.adapter.state.a, first.state.a)
    assert result.adapter.state.a.grad_fn is None
    result.prediction.logits.square().mean().backward()
    assert first.state.a.grad is None
    with pytest.raises(ValueError, match="frame zero"):
        policy(observation(episode="synthetic=true-new"), first)
    reset = policy(observation(episode="synthetic=true-new"))
    assert torch.equal(reset.prediction.logits, model(observation(), 2).prediction.logits)
    restart = FixedBudgetPolicy(model, make_adapter("restart"), 2)
    restart(observation())
    assert torch.equal(restart(changed).prediction.logits, model(changed, 2).prediction.logits)


def test_substituted_high_budget_state_is_rejected():
    model = RecursiveSolver(width=16, attention_mode="masked_neighbor")
    low = FixedBudgetPolicy(model, make_adapter("carry"), 2)
    high = FixedBudgetPolicy(model, make_adapter("carry"), 8)
    original, expensive = low(observation()), high(observation())
    with pytest.raises(ValueError, match="budget provenance"):
        low(observation(frame=1), replace(original, state=expensive.state))
    other = FixedBudgetPolicy(model, make_adapter("carry"), 2)
    with pytest.raises(ValueError, match="stream provenance"):
        low(observation(frame=1), other(observation()))
    assert low(observation(frame=1)).state.budget == 2


def test_boundary_errors_and_snapshot_isolation():
    policy = FixedBudgetPolicy(RecursiveSolver(width=16), make_adapter("carry"), 1)
    with pytest.raises(TypeError, match="ObservationBatch"):
        policy({"observation": observation()})
    with pytest.raises(ValueError, match="previous"):
        policy(observation(frame=1))
    first = policy(observation())
    with torch.no_grad():
        first.state.a.add_(100)
        first.observation.node_features.add_(100)
    assert policy._previous.state.a.abs().max() < 100
    assert policy._previous.observation.node_features.max() == 1
    with pytest.raises(ValueError, match="consecutive"):
        policy(observation(frame=2))
