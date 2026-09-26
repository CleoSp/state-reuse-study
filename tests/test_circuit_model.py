"""Synthetic=true circuit head, bidirectional wiring and permutation checks."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from test_model import deterministic_cpu, observation as maze_observation
from state_repair.data.circuit import Circuit, CircuitExample, Operator, collate, observation
from state_repair.models import RecursiveSolver
from state_repair.models.adapters import make_adapter
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.train.losses import circuit_value_loss
from state_repair.types import Domain


def circuit():
    return Circuit((Operator.INPUT, Operator.INPUT, Operator.AND, Operator.NOT),
                   ((), (), (0, 1), (2,)), (0, 1, 0, 0))


def circuit_pair(padding=0):
    old = observation(circuit(), "synthetic=true")
    new = observation(circuit().flip_input(0), "synthetic=true", 1)
    if padding:
        def pad(obs):
            return replace(obs, node_features=torch.nn.functional.pad(obs.node_features, (0,0,0,padding)),
                           edge_types=torch.nn.functional.pad(obs.edge_types, (0,padding,0,padding)),
                           valid_nodes=torch.nn.functional.pad(obs.valid_nodes, (0,padding)),
                           node_ids=torch.nn.functional.pad(obs.node_ids, (0,padding), value=-1))
        old, new = pad(old), pad(new)
    return old, new


@pytest.mark.parametrize("domain,count", [(Domain.MAZE,100954), (Domain.CIRCUIT,100822)])
@pytest.mark.parametrize("k", [0, 1, 3])
def test_counts_and_head_by_domain(domain, count, k):
    model = RecursiveSolver(domain=domain)
    assert model.attention_mode == "masked_neighbor"
    assert sum(p.numel() for p in model.parameters()) == count
    obs = maze_observation() if domain == Domain.MAZE else circuit_pair()[0]
    calls = []
    hooks = [model.block.register_forward_hook(lambda *_: calls.append("block"))]
    hooks += [layer.register_forward_hook(lambda *_: calls.append("layer")) for layer in model.block.layers]
    result = model(obs, k)
    assert result.prediction.logits.shape[-1] == (6 if domain == Domain.MAZE else 2)
    assert result.block_calls == calls.count("block") == 2*k
    assert result.transformer_layer_executions == calls.count("layer") == 4*k
    for hook in hooks:
        hook.remove()


def test_wire_relations_allow_both_directions_and_exclude_non_neighbors():
    model = RecursiveSolver(width=16, domain=Domain.CIRCUIT)
    layer = model.block.layers[0]
    obs, _ = circuit_pair()
    x = torch.randn(1, 4, 16, requires_grad=True)
    output = layer(x, obs.edge_types, obs.valid_nodes)
    for query, neighbor in [(0, 2), (2, 0)]:
        grad = torch.autograd.grad(output[0, query, 0], x, retain_graph=True)[0]
        assert grad[0, neighbor].abs().max() > 1e-7
    grad = torch.autograd.grad(output[0, 0, 0], x)[0]
    assert torch.count_nonzero(grad[0, 1]) == 0


def test_circuit_targets_permutation_padding_and_retained_state():
    c = circuit()
    order = torch.tensor([3, 0, 2, 1])
    example = CircuitExample(c, "synthetic=true", 0, "train", tuple(range(4)), True)
    obs, target, _ = collate([example])
    perm, perm_target, _ = collate([replace(example, node_order=tuple(order.tolist()))])
    assert torch.equal(perm_target.values, target.values[:, order])
    assert torch.equal(perm_target.scored_mask, target.scored_mask[:, order])
    model = RecursiveSolver(width=16, domain=Domain.CIRCUIT)
    original = model(obs, 2)
    retained = replace(original.state, a=original.state.a[:,order], z=original.state.z[:,order],
                       node_ids=original.state.node_ids[:,order], valid_nodes=original.state.valid_nodes[:,order])
    actual = model(perm, 1, retained)
    expected = model(obs, 1, original.state)
    torch.testing.assert_close(actual.prediction.logits, expected.prediction.logits[:,order])
    torch.testing.assert_close(actual.state.z, expected.state.z[:,order])
    torch.testing.assert_close(circuit_value_loss(actual.prediction.logits, perm_target),
                               circuit_value_loss(expected.prediction.logits, target))
    padded, _ = circuit_pair(3)
    state = model.fresh_state(padded)
    state = replace(state, a=state.a.masked_fill(~padded.valid_nodes[...,None], 1e4))
    torch.testing.assert_close(model(padded, 2, state).prediction.logits[:,:4], original.prediction.logits)
    logits = original.prediction.logits.detach().requires_grad_()
    circuit_value_loss(logits, target).backward()
    assert not logits.grad[~target.scored_mask].any()
    logits2 = logits.detach().clone()
    logits2[~target.scored_mask] = 1e4
    assert torch.equal(circuit_value_loss(logits, target), circuit_value_loss(logits2, target))


def test_maze_checkpoint_strict_load_and_snapshot_parity():
    root = Path("runs/static_generalization_v1/masked_neighbor")
    if not root.exists():
        pytest.skip("local historical checkpoint not distributed with repository")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = RecursiveSolver()
    weights = payload["model"]
    model.load_state_dict(weights, strict=True)
    source = root / "source/src/state_repair/models/recursive.py"
    spec = importlib.util.spec_from_file_location("historical_recursive", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.RecursiveSolver(attention_mode="masked_neighbor")
    original.load_state_dict(weights, strict=True)
    assert torch.equal(model(maze_observation(), 2).prediction.logits,
                       original(maze_observation(), 2).prediction.logits)
