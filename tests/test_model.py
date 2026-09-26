"""CPU correctness tests. Every fixture here has synthetic=true provenance.

These constructed tensors test implementation, not learned task competence.
"""
from dataclasses import replace

import pytest
import torch

from state_repair.models import RecursiveSolver
from state_repair.types import Domain, ObservationBatch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(173)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(padding: int = 0, frame: int = 0, episode: str = "synthetic=true") -> ObservationBatch:
    n = 4 + padding
    features = torch.zeros(1, n, 4)
    features[0, :4] = torch.tensor([[0., 0., 1., 0.], [0., 1., 0., 0.],
                                   [1., 0., 0., 0.], [1., 1., 0., 1.]])
    edges = torch.zeros(1, n, n, dtype=torch.long)
    for source, target, direction, reverse in [(0, 1, 2, 4), (0, 2, 3, 1), (2, 3, 2, 4)]:
        edges[0, source, target], edges[0, target, source] = direction, reverse
    valid = torch.arange(n).unsqueeze(0) < 4
    ids = torch.arange(n).unsqueeze(0).masked_fill(~valid, -1)
    return ObservationBatch(Domain.MAZE, features, edges, valid, (episode,), (frame,),
                            ids, ((2, 2),), torch.tensor([0]), torch.tensor([3]))


@pytest.mark.parametrize("inner,cycles", [(1, 0), (1, 1), (1, 3), (2, 2)])
def test_actual_calls_and_shapes(inner, cycles):
    model = RecursiveSolver(width=16, heads=2, inner_cycles=inner)
    counts = {"block": 0, "layers": 0, "encoder": 0, "head": 0}
    def increment(name):
        def hook(*_args):
            counts[name] += 1
        return hook
    handles = [model.block.register_forward_hook(increment("block")),
               model.encoder.register_forward_hook(increment("encoder")),
               model.head.register_forward_hook(increment("head"))]
    handles += [layer.register_forward_hook(increment("layers")) for layer in model.block.layers]
    result = model(observation(), outer_cycles=cycles)
    assert result.prediction.logits.shape == (1, 4, 6)
    assert result.state.a.shape == result.state.z.shape == (1, 4, 16)
    assert result.outer_cycles == result.state.budget == cycles
    assert result.block_calls == counts["block"] == (inner + 1) * cycles
    assert result.transformer_layer_executions == counts["layers"] == 2 * result.block_calls
    assert counts["encoder"] == counts["head"] == 1
    for handle in handles:
        handle.remove()


def test_graph_relations_influence_predictions():
    model = RecursiveSolver(width=16, heads=2)
    obs = observation()
    edges = obs.edge_types.clone()
    edges[0, 0, 1] = edges[0, 1, 0] = 0
    edited = replace(obs, edge_types=edges)
    original = model(obs, 2).prediction.logits
    changed = model(edited, 2).prediction.logits
    assert (original - changed).abs().max() > 1e-6
    with pytest.raises(ValueError, match="edge direction"):
        replace(obs, edge_types=torch.where(obs.edge_types > 0, 5 - obs.edge_types, 0))


def test_node_permutation_equivariance():
    model = RecursiveSolver(width=16, heads=2)
    obs = observation(padding=1)
    order = torch.tensor([3, 0, 4, 2, 1])
    permuted = replace(obs, node_features=obs.node_features[:, order],
                       edge_types=obs.edge_types[:, order][:, :, order],
                       valid_nodes=obs.valid_nodes[:, order], node_ids=obs.node_ids[:, order])
    original = model(obs, 3)
    reordered = model(permuted, 3)
    torch.testing.assert_close(reordered.prediction.logits, original.prediction.logits[:, order], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(reordered.state.z, original.state.z[:, order], atol=2e-6, rtol=2e-5)


def test_padding_isolation_with_corrupted_padded_state():
    model = RecursiveSolver(width=16, heads=2)
    plain, padded = observation(), observation(padding=3)
    state = model.fresh_state(padded)
    mask = ~padded.valid_nodes[:, :, None]
    state = replace(state, a=state.a.masked_fill(mask, 10000), z=state.z.masked_fill(mask, -10000))
    original = model(plain, 2)
    result = model(padded, 2, state)
    torch.testing.assert_close(result.prediction.logits[:, :4], original.prediction.logits, atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(result.prediction.logits[:, 4:]) == 0
    assert torch.count_nonzero(result.state.a[:, 4:]) == 0
    assert torch.count_nonzero(result.state.z[:, 4:]) == 0


def test_fresh_and_returned_state_do_not_alias_and_k0_decodes():
    model = RecursiveSolver(width=16, heads=2)
    obs = observation()
    fresh, another = model.fresh_state(obs), model.fresh_state(obs)
    result = model(obs, 0, fresh)
    assert fresh.a.data_ptr() != another.a.data_ptr() != result.state.a.data_ptr()
    assert fresh.a.data_ptr() != fresh.z.data_ptr()
    assert fresh.a.data_ptr() != model.initial_a.data_ptr()
    assert result.state.node_ids.data_ptr() != fresh.node_ids.data_ptr()
    torch.testing.assert_close(result.prediction.logits, model.decode(obs, fresh).logits)
    before = fresh.a.clone()
    with torch.no_grad():
        result.state.a.add_(1)
    torch.testing.assert_close(fresh.a, before)


def test_state_correspondence_and_fresh_episode_reset():
    model = RecursiveSolver(width=16, heads=2)
    obs = observation()
    state = model(obs, 1).state
    order = torch.tensor([1, 0, 2, 3])
    permuted = replace(obs, node_ids=obs.node_ids[:, order],
                       node_features=obs.node_features[:, order],
                       edge_types=obs.edge_types[:, order][:, :, order])
    for invalid in [replace(obs, episode_ids=("different",)), replace(obs, frame_indices=(1,)),
                    permuted]:
        with pytest.raises(ValueError, match="correspondence|frame"):
            model(invalid, 1, state)
    new_obs = replace(obs, episode_ids=("different",))
    fresh_result = model(new_obs, 1)
    torch.testing.assert_close(fresh_result.prediction.logits, model(obs, 1).prediction.logits)
    assert model(obs, 2, state).state.budget == 3


def test_frozen_backbone_allows_initializer_gradient_and_detects_bad_detach():
    model = RecursiveSolver(width=16, heads=2).requires_grad_(False)
    old_obs = observation()
    edges = old_obs.edge_types.clone()
    edges[0, 0, 1] = edges[0, 1, 0] = 0
    new_obs = replace(old_obs, edge_types=edges, frame_indices=(1,))
    old_state = model(old_obs, 2).state.detach()
    fresh = model.fresh_state(new_obs)
    gate = torch.nn.Parameter(torch.tensor(0.2))
    optimizer = torch.optim.SGD([gate], lr=0.1)
    def initialize():
        retain = gate.sigmoid()
        return replace(fresh, a=retain * old_state.a + (1 - retain) * fresh.a,
                       z=retain * old_state.z + (1 - retain) * fresh.z)
    initialized = initialize()
    result = model(new_obs, 2, initialized)
    result.prediction.logits.square().mean().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad) and gate.grad.abs() > 1e-8
    before = gate.detach().clone()
    optimizer.step()
    assert not torch.equal(gate.detach(), before)
    assert all(parameter.grad is None for parameter in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    broken = model(new_obs, 2, initialize().detach()).prediction.logits.square().mean()
    assert not broken.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        broken.backward()
    assert gate.grad is None


def test_budget_32_finite_and_parameters_shared():
    model = RecursiveSolver(width=16, heads=2)
    count = sum(p.numel() for p in model.parameters())
    result = model(observation(), 32)
    assert torch.isfinite(result.prediction.logits).all()
    assert count == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("cycles", [-1, 1.2, True])
def test_invalid_budget(cycles):
    with pytest.raises(ValueError, match="outer_cycles"):
        RecursiveSolver()(observation(), cycles)


def test_inference_rejects_whole_sample_and_circuit():
    from state_repair.data.circuit import generate_circuit, observation as circuit_observation
    model = RecursiveSolver()
    with pytest.raises(TypeError, match="ObservationBatch"):
        model({"observation": observation(), "target": torch.ones(1)}, 1)
    with pytest.raises(ValueError, match="circuit"):
        model(circuit_observation(generate_circuit(4,17)), 1)
