"""Synthetic=true fixtures only: adapter mechanics, accounting and gradients."""
from dataclasses import replace

import pytest
import torch
from torch import nn

from test_model import observation, deterministic_cpu
from test_circuit_model import circuit_pair
from state_repair.data.circuit import Circuit, Operator, observation as circuit_observation
from state_repair.eval.interventions import impact_mask_reset
from state_repair.models import RecursiveSolver, FixedBudgetPolicy, make_adapter, ADAPTER_REGISTRY
from state_repair.models.adapters import ObservedContext
from state_repair.types import Domain, ObservedEdit, PredictionBatch, TransitionInput


LEARNED = ("spatial_gate", "global_gate", "gru_adapter", "residual_adapter")


def pair(domain, padding=0):
    if domain == Domain.CIRCUIT:
        return circuit_pair(padding)
    old = observation(padding)
    edges = old.edge_types.clone()
    edges[:,0,1] = edges[:,1,0] = 0
    return old, replace(old, edge_types=edges, frame_indices=(1,))


def adapter_for(key, domain, width=16, context_width=8):
    if key in LEARNED:
        return make_adapter(key, width=width, context_width=context_width, domain=domain)
    if key == "shuffled_gate":
        return make_adapter(key, spatial_gate=adapter_for("spatial_gate", domain, width, context_width))
    if key == "answer_only":
        return make_adapter(key, width=width, domain=domain)
    return make_adapter(key)


def transition_for(model, domain, padding=0):
    old, new = pair(domain, padding)
    result = model(old, 2)
    return TransitionInput(new, old, result.state, ObservedEdit.between(old, new)), result.prediction


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("key", list(ADAPTER_REGISTRY))
def test_all_adapter_padding_nonaliasing_and_frame_zero(domain, key):
    model = RecursiveSolver(width=16, domain=domain)
    adapter = adapter_for(key, domain)
    plain, prediction = transition_for(model, domain)
    padded, padded_prediction = transition_for(model, domain, 3)
    corrupt = replace(padded.old_state, a=padded.old_state.a.masked_fill(~padded.new.valid_nodes[...,None], 1e4),
                      z=padded.old_state.z.masked_fill(~padded.new.valid_nodes[...,None], -1e4))
    padded = replace(padded, old_state=corrupt)
    torch.manual_seed(413)
    fresh = model.fresh_state(plain.new)
    result = adapter(plain, fresh, previous_prediction=prediction)
    torch.manual_seed(413)
    padded_result = adapter(padded, model.fresh_state(padded.new), previous_prediction=padded_prediction)
    torch.testing.assert_close(result.state.a, padded_result.state.a[:,:4], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(result.state.z, padded_result.state.z[:,:4], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(model(plain.new, 1, result.state).prediction.logits,
                               model(padded.new, 1, padded_result.state).prediction.logits[:,:4], atol=2e-6, rtol=2e-5)
    assert not padded_result.state.a[:,4:].any() and not padded_result.state.z[:,4:].any()
    assert result.state.a.data_ptr() not in (fresh.a.data_ptr(), plain.old_state.a.data_ptr())
    assert result.state.z.data_ptr() not in (fresh.z.data_ptr(), plain.old_state.z.data_ptr())
    initial_fresh = model.fresh_state(plain.old)
    initialized = adapter(TransitionInput(plain.old), initial_fresh)
    assert torch.equal(initialized.state.a, initial_fresh.a)
    assert not initialized.operations


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("key", ["spatial_gate", "global_gate"])
@pytest.mark.parametrize("retain,baseline", [(1., "carry"), (0., "restart")])
def test_gate_extremes_bitwise(domain, key, retain, baseline):
    model = RecursiveSolver(width=16, domain=domain)
    transition, _ = transition_for(model, domain)
    fresh = model.fresh_state(transition.new)
    adapter = make_adapter(key, width=16, context_width=8, domain=domain, retain_override=retain)
    result = adapter(transition, fresh)
    expected = make_adapter(baseline)(transition, fresh)
    assert torch.equal(result.state.a, expected.state.a)
    assert torch.equal(result.state.z, expected.state.z)
    assert torch.equal(model(transition.new, 1, result.state).prediction.logits,
                       model(transition.new, 1, expected.state).prediction.logits)


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("key", LEARNED)
def test_frozen_backbone_gradient_optimizer_and_broken_detach(domain, key):
    model = RecursiveSolver(width=16, domain=domain).requires_grad_(False)
    transition, _ = transition_for(model, domain)
    old = replace(transition.old_state, a=transition.old_state.a.detach().requires_grad_(),
                  z=transition.old_state.z.detach().requires_grad_())
    transition = replace(transition, old_state=old)
    adapter = adapter_for(key, domain)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    fresh = model.fresh_state(transition.new)

    def loss(broken=False):
        state = adapter.initialize(transition, fresh).state
        if broken:
            state = state.detach()
        logits = model(transition.new, 1, state).prediction.logits
        return logits.square().mean()

    loss().backward()
    gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 1e-6
    assert adapter.context.project.weight.grad.abs().sum() > 1e-7
    before = [p.detach().clone() for p in adapter.parameters()]
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.parameters()))
    assert old.a.grad is old.z.grad is None
    assert all(p.grad is None for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    broken_loss = loss(True)
    assert not broken_loss.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        broken_loss.backward()
    assert all(p.grad is None for p in adapter.parameters())


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("key", list(ADAPTER_REGISTRY))
def test_adapter_operation_counts_against_hooks_and_k0(domain, key):
    model = RecursiveSolver(width=16, domain=domain)
    adapter = adapter_for(key, domain)
    transition, prediction = transition_for(model, domain)
    measured = {"linear": 0, "gru": 0, "macs": 0, "layers": 0, "context": 0}
    def linear(module, args, output):
        measured["linear"] += 1
        measured["macs"] += output.numel()*module.in_features
    def gru(module, args, output):
        measured["gru"] += 1
        measured["macs"] += output.shape[0]*(module.weight_ih.numel()+module.weight_hh.numel())
    def layer(module, args, output):
        b,n,d = output.shape
        measured["layers"] += 1
        measured["macs"] += 2*b*n*n*d
    def context(*_):
        measured["context"] += 1
    from state_repair.models.recursive import GraphTransformerLayer
    hooks = []
    for module in adapter.modules():
        fn = linear if isinstance(module, nn.Linear) else gru if isinstance(module, nn.GRUCell) else layer if isinstance(module, GraphTransformerLayer) else context if isinstance(module, ObservedContext) else None
        if fn:
            hooks.append(module.register_forward_hook(fn))
    result = adapter(transition, model.fresh_state(transition.new), previous_prediction=prediction)
    assert measured["linear"] == result.operations.get("adapter_linear_calls", 0)
    assert measured["gru"] == result.operations.get("adapter_gru_calls", 0)
    assert measured["macs"] == result.operations.get("adapter_macs", 0)
    assert measured["layers"] == result.operations.get("context_layer_executions", 0)
    assert measured["context"] == result.operations.get("context_calls", 0)
    for hook in hooks:
        hook.remove()
    policy = FixedBudgetPolicy(model, adapter, 0)
    first = policy(transition.old)
    second = policy(transition.new)
    assert second.block_calls == second.transformer_layer_executions == second.state.budget == 0
    assert second.operations["encoder_calls"] == second.operations["adapter_calls"] == second.operations["decoder_calls"] == 1
    assert second.operations.get("context_calls", 0) == (1 if key in LEARNED or key == "shuffled_gate" else 0)
    assert first.operations.get("context_calls", 0) == 0


def test_context_receptive_field_measured_two_hops():
    c = Circuit((Operator.INPUT,)+(Operator.NOT,)*5, ((),)+tuple((i,) for i in range(5)), (0,)*6)
    old = circuit_observation(c, "synthetic=true")
    changed = circuit_observation(c.flip_input(0), "synthetic=true", 1)
    unchanged = replace(old, frame_indices=(1,))
    model = RecursiveSolver(width=16, domain=Domain.CIRCUIT)
    state = model(old, 1).state
    context = ObservedContext(16, 8, Domain.CIRCUIT)
    def apply(new):
        return context(TransitionInput(new, old, state, ObservedEdit.between(old,new)))
    difference = (apply(changed)-apply(unchanged)).abs().amax(-1)[0]
    assert (difference[:3] > 1e-7).all()
    assert torch.equal(difference[3:], torch.zeros(3))
    assert context.receptive_field_hops == 2


@pytest.mark.parametrize("domain", list(Domain))
@pytest.mark.parametrize("key", LEARNED)
def test_learned_adapters_permutation_with_retained_state(domain, key):
    model = RecursiveSolver(width=16, domain=domain)
    transition, _ = transition_for(model, domain, 1)
    order = torch.tensor([3,0,4,2,1])
    def permute(obs):
        return replace(obs, node_features=obs.node_features[:,order], edge_types=obs.edge_types[:,order][:,:,order],
                       valid_nodes=obs.valid_nodes[:,order], node_ids=obs.node_ids[:,order])
    old = transition.old_state
    state = replace(old, a=old.a[:,order], z=old.z[:,order], node_ids=old.node_ids[:,order], valid_nodes=old.valid_nodes[:,order])
    pold, pnew = permute(transition.old), permute(transition.new)
    perm = TransitionInput(pnew, pold, state, ObservedEdit.between(pold, pnew))
    adapter = adapter_for(key, domain)
    result = adapter(transition, model.fresh_state(transition.new))
    reordered = adapter(perm, model.fresh_state(pnew))
    torch.testing.assert_close(reordered.state.a, result.state.a[:,order], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(reordered.state.z, result.state.z[:,order], atol=2e-6, rtol=2e-5)


def test_control_semantics_and_privileged_registry_separation():
    c = Circuit((Operator.INPUT,)+(Operator.NOT,)*5, ((),)+tuple((i,) for i in range(5)), (0,)*6)
    old, new = circuit_observation(c, "synthetic=true"), circuit_observation(c.flip_input(0), "synthetic=true", 1)
    model = RecursiveSolver(width=16, domain=Domain.CIRCUIT)
    prior = model(old, 2)
    transition = TransitionInput(new, old, prior.state, ObservedEdit.between(old,new))
    fresh = model.fresh_state(new)
    for radius in (1,2,3):
        result = make_adapter("local_reset", radius=radius)(transition, fresh)
        assert torch.equal(result.retain_a[0,:,0], (torch.arange(6)>radius).float())
    for rate in (0.,0.5,1.):
        result = make_adapter("random_reset", reset_rate=rate)(transition, fresh)
        assert (result.retain_a == 0).sum() == round(6*rate)
    zero_noise = make_adapter("noisy_carry", noise_scale=0)(transition, fresh)
    assert torch.equal(zero_noise.state.a, prior.state.a)
    spatial = adapter_for("spatial_gate", Domain.CIRCUIT)
    original = spatial(transition, fresh)
    shuffled = make_adapter("shuffled_gate", spatial_gate=spatial)(transition, fresh)
    assert torch.equal(original.retain_a.sort(1).values, shuffled.retain_a.sort(1).values)
    assert torch.equal(original.retain_z.sort(1).values, shuffled.retain_z.sort(1).values)
    assert not torch.equal(original.retain_a, shuffled.retain_a)
    for key in set(ADAPTER_REGISTRY) - set(LEARNED) - {"restart", "carry"}:
        assert ADAPTER_REGISTRY[key].is_control
    with pytest.raises(ValueError, match="unknown deployable"):
        make_adapter("impact_mask_reset")
    intervention = impact_mask_reset(prior.state, fresh, torch.tensor([[1,0,0,0,0,0]], dtype=torch.bool))
    assert intervention.privileged is True and intervention.record()["privileged"] is True
    assert torch.equal(intervention.state.a[:,0], fresh.a[:,0])
    assert torch.equal(intervention.state.a[:,1:], prior.state.a[:,1:])


def test_answer_only_cannot_use_latents_and_uses_probabilities():
    model = RecursiveSolver(width=16)
    transition, prediction = transition_for(model, Domain.MAZE)
    adapter = make_adapter("answer_only", width=16)
    fresh = model.fresh_state(transition.new)
    with pytest.raises(ValueError, match="previous model prediction"):
        adapter(transition, fresh)
    first = adapter(transition, fresh, previous_prediction=prediction)
    corrupt = replace(transition.old_state, a=transition.old_state.a+100, z=transition.old_state.z-100)
    second = adapter(replace(transition, old_state=corrupt), fresh, previous_prediction=prediction)
    assert torch.equal(first.state.a, second.state.a) and torch.equal(first.state.z, second.state.z)
    shifted = adapter(transition, fresh, previous_prediction=PredictionBatch(prediction.logits+10))
    torch.testing.assert_close(first.state.a, shifted.state.a)
    altered = prediction.logits.clone()
    altered[...,0] += 10
    assert not torch.equal(first.state.a, adapter(transition, fresh, previous_prediction=PredictionBatch(altered)).state.a)


def test_no_oracle_calls_in_policy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oracle entered deployable policy")
    monkeypatch.setattr("state_repair.oracles.circuit.evaluate", forbidden)
    monkeypatch.setattr("state_repair.oracles.maze.solve_maze", forbidden)
    for key in ADAPTER_REGISTRY:
        old, new = pair(Domain.CIRCUIT)
        policy = FixedBudgetPolicy(RecursiveSolver(width=16, domain=Domain.CIRCUIT), adapter_for(key, Domain.CIRCUIT), 1)
        policy(old)
        policy(new)
