from dataclasses import fields, replace
import itertools
import random

import pytest
import torch

from state_repair.data.circuit import (Circuit, CircuitExample, CircuitOracleMetadata, Operator as Op,
    collate, generate_circuit, generate_episode, observation, topology_fingerprint, reject_duplicate_roots)
from state_repair.oracles.circuit import (EventDrivenEvaluator, descendant_mask, evaluate,
    impact_metadata, recursive_evaluate, score_circuit)
from state_repair.types import Domain, ObservationBatch, ObservedEdit, RecurrentState, TargetBatch, TransitionInput
from state_repair.train.losses import maze_valid_set_loss
from state_repair.models.recursive import RecursiveSolver


def blocked():
    return Circuit((Op.INPUT,Op.INPUT,Op.AND,Op.NOT), ((),(),(0,1),(2,)), (0,0,0,0))


def test_blocked_descendant_and_undo_reference_cache():
    old = blocked(); new = old.flip_input(0)
    assert evaluate(old) == recursive_evaluate(old) == [0,0,0,1]
    assert impact_metadata(old,new) == {"descendant_mask":[False,False,True,True],"changed_value_mask":[True,False,False,False]}
    ref = EventDrivenEvaluator(old)
    assert ref.update(new) == evaluate(new) == recursive_evaluate(new)
    assert ref.last_gate_evaluations == 1 and ref.last_input_updates == 1
    assert ref.update(new.flip_input(0)) == evaluate(old)
    leaked = ref.values; leaked[0] = 99
    assert ref.values == evaluate(old)
    changed = old.substitute(2,Op.OR)
    assert ref.update(changed) == evaluate(changed)
    assert ref.update(changed.substitute(2,Op.AND)) == evaluate(old)
    assert old == new.flip_input(0)


def test_truth_tables_independent_evaluators():
    for bits in itertools.product((0,1),repeat=3):
        for op1,op2 in itertools.product((Op.AND,Op.OR,Op.XOR),repeat=2):
            circuit = Circuit((Op.INPUT,Op.INPUT,Op.INPUT,op1,Op.NOT,op2),
                              ((),(),(),(0,1),(3,),(2,4)),(*bits,0,0,0))
            assert evaluate(circuit) == recursive_evaluate(circuit)
            ref = EventDrivenEvaluator(circuit)
            for node in (0,1,2,1,0,2):
                circuit = circuit.flip_input(node)
                assert ref.update(circuit) == evaluate(circuit) == recursive_evaluate(circuit)


def test_many_seed_streams_relabeling_and_fingerprints():
    for seed in range(64):
        circuit = generate_circuit(6+seed%15,seed)
        assert circuit == generate_circuit(circuit.n,seed)
        order = list(range(circuit.n)); random.Random(seed+300).shuffle(order)
        relabeled = circuit.relabel(order)
        assert topology_fingerprint(circuit) == topology_fingerprint(relabeled)
        assert evaluate(relabeled) == [evaluate(circuit)[u] for u in order]
        episode = generate_episode(circuit,f"test-{seed}","train",8,seed+100,synthetic=True)
        ref = EventDrivenEvaluator(circuit)
        perm_ref = EventDrivenEvaluator(relabeled)
        for example in episode[1:]:
            answer = evaluate(example.circuit)
            assert answer == recursive_evaluate(example.circuit) == ref.update(example.circuit)
            assert perm_ref.update(example.circuit.relabel(order)) == [answer[u] for u in order]
            assert topology_fingerprint(example.circuit) == topology_fingerprint(circuit)
            assert example.node_order == episode[0].node_order
            assert example.synthetic is True


def test_observational_schema_permutation_padding_and_targets(monkeypatch):
    base = generate_circuit(9,10)
    order = tuple(reversed(range(base.n)))
    examples = [CircuitExample(base,"a",0,"train",order,True),
                CircuitExample(blocked(),"b",0,"train",(2,0,3,1),True)]
    obs,target,metadata = collate(examples)
    target.check_observation(obs)
    assert obs.domain == Domain.CIRCUIT and obs.grid_shapes == () and obs.starts is None
    assert target.valid_actions is None
    assert (target.values[~target.scored_mask] == -1).all()
    assert not metadata.descendant_mask.any() and not metadata.changed_value_mask.any()
    assert not obs.valid_nodes[1,4:].any() and (obs.node_ids[1,4:] == -1).all()
    assert not obs.edge_types[1,4:,:].any() and not obs.edge_types[1,:,4:].any()
    assert not obs.node_features[1,4:].any()
    plain = observation(base)
    assert torch.equal(obs.node_features[0],plain.node_features[0,list(order)])
    assert torch.equal(obs.edge_types[0],plain.edge_types[0,list(order)][:,list(order)])
    for u,parents in enumerate(base.parents):
        for parent in parents:
            assert plain.edge_types[0,parent,u] == 1 and plain.edge_types[0,u,parent] == 2
    forbidden = {"target","values","distances","descendant_mask","changed_value_mask","split","root_id","seed","node_order","topological_order"}
    assert not forbidden.intersection(f.name for f in fields(ObservationBatch))
    def forbidden_oracle(*args,**kwargs): raise AssertionError("oracle entered features")
    monkeypatch.setattr("state_repair.oracles.circuit.evaluate",forbidden_oracle)
    assert torch.equal(observation(base).node_features,plain.node_features)
    with pytest.raises(TypeError): observation(base,values=[0]*base.n)
    with pytest.raises(ValueError,match="solver domain maze"):
        RecursiveSolver(width=8,heads=2)(plain,1)
    with pytest.raises(ValueError,match="maze valid-action"):
        maze_valid_set_loss(torch.zeros(2,9,6),target,obs.valid_nodes)


def test_edited_batch_and_state_joint_permutation():
    old = blocked(); new = old.flip_input(0); order = (2,0,3,1)
    old_obs = observation(old,"a",0,order)
    example = CircuitExample(new,"a",1,"train",order,True)
    obs,target,metadata = collate([example],previous=[old])
    assert metadata.changed_value_mask.tolist() == [[False,True,False,False]]
    state = RecurrentState(torch.randn(1,4,3,requires_grad=True),torch.randn(1,4,3,requires_grad=True),
                           old_obs.episode_ids,old_obs.frame_indices,old_obs.node_ids,old_obs.valid_nodes)
    transition = TransitionInput(obs,old_obs,state.detach(),ObservedEdit.between(old_obs,obs))
    assert transition.edit.node_features_delta[...,5].count_nonzero() == 1
    assert not transition.edit.edge_changed.any()
    permutation = [3,1,0,2]
    def perm_obs(o):
        return replace(o,node_features=o.node_features[:,permutation],edge_types=o.edge_types[:,permutation][:,:,permutation],
                       node_ids=o.node_ids[:,permutation],valid_nodes=o.valid_nodes[:,permutation])
    pstate = replace(state,a=state.a[:,permutation],z=state.z[:,permutation],node_ids=state.node_ids[:,permutation],valid_nodes=state.valid_nodes[:,permutation])
    TransitionInput(perm_obs(obs),perm_obs(old_obs),pstate.detach(),ObservedEdit.between(perm_obs(old_obs),perm_obs(obs)))
    ptarget = TargetBatch(values=target.values[:,permutation],scored_mask=target.scored_mask[:,permutation])
    ptarget.check_observation(perm_obs(obs))
    with pytest.raises(ValueError,match="previous"): collate([example])
    with pytest.raises(ValueError,match="correspondence"):
        TransitionInput(perm_obs(obs),old_obs,state,ObservedEdit.between(old_obs,obs))


@pytest.mark.parametrize("mutation",["arity","duplicate","self","cycle","internal_bit","empty"])
def test_invalid_circuit_descriptions(mutation):
    c = blocked()
    with pytest.raises(ValueError):
        if mutation == "arity": replace(c,parents=((),(),(0,),(2,)))
        elif mutation == "duplicate": replace(c,parents=((),(),(0,0),(2,)))
        elif mutation == "self": replace(c,parents=((),(),(0,2),(2,)))
        elif mutation == "cycle": replace(c,parents=((),(),(0,3),(2,)))
        elif mutation == "internal_bit": replace(c,input_bits=(0,0,1,0))
        else: Circuit((Op.INPUT,),((),),(0,))


@pytest.mark.parametrize("mutation",["onehot","bit","reverse","arity","self","cycle","metadata","pad"])
def test_invalid_circuit_observation(mutation):
    obs,_,_ = collate([CircuitExample(blocked(),"a",0,"train",(0,1,2,3),True),
                      CircuitExample(generate_circuit(6,19),"b",0,"train",tuple(range(6)),True)])
    x = obs.node_features.clone(); e = obs.edge_types.clone()
    with pytest.raises(ValueError):
        if mutation == "onehot": x[0,0,1] = 1
        elif mutation == "bit": x[0,2,5] = 1
        elif mutation == "reverse": e[0,2,0] = 0
        elif mutation == "arity": e[0,0,2] = e[0,2,0] = 0
        elif mutation == "self": e[0,2,2] = 1
        elif mutation == "cycle":
            x[0,:4] = 0; x[0,0,0] = 1; x[0,1:4,4] = 1; e[0] = 0
            for a,b in ((1,2),(2,3),(3,1)): e[0,a,b]=1; e[0,b,a]=2
        elif mutation == "metadata": replace(obs,starts=torch.tensor([0,0]))
        else: x[0,4,0] = 1
        replace(obs,node_features=x,edge_types=e)


def test_targets_scoring_and_root_ancestry():
    c = blocked(); values = evaluate(c)
    assert score_circuit(c,[1,1,*values[2:]])["circuit_correct"]
    assert not score_circuit(c,[0,0,1,1])["circuit_correct"]
    with pytest.raises(ValueError): score_circuit(c,[False,0,0,1])
    with pytest.raises(ValueError,match="nonempty"):
        TargetBatch(values=torch.full((1,4),-1),scored_mask=torch.zeros(1,4,dtype=torch.bool))
    a = CircuitExample(c,"a",0,"train",(0,1,2,3),True)
    with pytest.raises(ValueError,match="crosses splits"):
        reject_duplicate_roots([a,replace(a,split="test",frame_index=1,circuit=c.flip_input(0))])
    with pytest.raises(ValueError,match="collision"):
        reject_duplicate_roots([a,replace(a,root_id="renamed",circuit=c.relabel((3,1,2,0)))])
    with pytest.raises(ValueError,match="collision"):
        reject_duplicate_roots([a,replace(a,root_id="variant",circuit=c.substitute(2,Op.XOR))])


def test_target_variants_domain_and_scored_mask_rejections():
    from state_repair.data.maze import Maze, observation as maze_observation
    obs,target,_ = collate([CircuitExample(blocked(),"a",0,"train",(0,1,2,3),True)])
    maze = maze_observation(Maze(1,4,(),0,3))
    with pytest.raises(ValueError,match="circuit observations"): target.check_observation(maze)
    maze_target = TargetBatch(torch.ones(1,4,6,dtype=torch.bool))
    with pytest.raises(ValueError,match="maze observations"): maze_target.check_observation(obs)
    with pytest.raises(ValueError,match="one domain"):
        TargetBatch(maze_target.valid_actions, target.values, target.scored_mask)
    with pytest.raises(TypeError,match="values and scored_mask"):
        TargetBatch(values=target.values)
    values=target.values.clone(); mask=target.scored_mask.clone()
    mask[0,0]=True; values[0,0]=0
    with pytest.raises(ValueError,match="non-input"):
        TargetBatch(values=values,scored_mask=mask).check_observation(obs)
    with pytest.raises(ValueError,match="-1"):
        TargetBatch(values=torch.zeros(1,4,dtype=torch.long),scored_mask=target.scored_mask)
