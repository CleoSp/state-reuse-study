"""Synthetic boundary fixtures; no empirical results."""
from dataclasses import replace

import pytest
import torch

from state_repair.types import (Domain, ObservationBatch, ObservedEdit,
                                RecurrentState, TransitionInput)


def observation(frame=0, episode="synthetic"):
    return ObservationBatch(Domain.MAZE,torch.zeros(1,2,4),torch.zeros(1,2,2,dtype=torch.long),
        torch.ones(1,2,dtype=torch.bool),(episode,),(frame,),torch.arange(2).view(1,2),((1,2),),
        torch.tensor([0]),torch.tensor([1]))


def state(obs):
    return RecurrentState(torch.randn(1,2,3,requires_grad=True),torch.randn(1,2,3,requires_grad=True),
        obs.episode_ids,obs.frame_indices,obs.node_ids.clone(),obs.valid_nodes.clone())


def test_observation_rejects_oracle_and_bad_padding():
    obs=observation()
    with pytest.raises(TypeError):
        replace(obs,distances=torch.zeros(1,2))
    with pytest.raises(ValueError,match="padding"):
        replace(obs,valid_nodes=torch.tensor([[True,False]]))


def test_detach_and_clone_do_not_alias():
    original=state(observation())
    detached=original.detach()
    assert detached.a.grad_fn is None and not detached.a.requires_grad
    assert detached.a.data_ptr()!=original.a.data_ptr()
    assert detached.node_ids.data_ptr()!=original.node_ids.data_ptr()
    cloned=original.clone()
    assert cloned.a.requires_grad and cloned.a.data_ptr()!=original.a.data_ptr()
    moved=original.to("cpu")
    assert moved.a.data_ptr()!=original.a.data_ptr()


def test_episode_reset_and_correspondence():
    old=state(observation())
    fresh=state(observation(episode="new"))
    with pytest.raises(ValueError,match="correspondence"):
        old.check_observation(observation(episode="new"))
    reset=old.reset(torch.tensor([True]),fresh)
    reset.check_observation(observation(episode="new"))
    assert torch.equal(reset.a,fresh.a)
    assert reset.a.data_ptr()!=fresh.a.data_ptr()


def test_transition_rejects_oracle_edit_stale_state_and_frame_zero_carry():
    old=observation()
    new=observation(1)
    edit=ObservedEdit.between(old,new)
    TransitionInput(new,old,state(old),edit)
    with pytest.raises(ValueError,match="derived"):
        TransitionInput(new,old,state(old),ObservedEdit(torch.ones_like(edit.node_features_delta),edit.edge_changed))
    with pytest.raises(ValueError,match="fresh"):
        TransitionInput(old,old,state(old),edit)
    with pytest.raises(ValueError,match="frame"):
        TransitionInput(new,old,state(new),edit)
    with pytest.raises(TypeError,match="ObservationBatch"):
        TransitionInput({"new":new,"target":"forbidden"})
