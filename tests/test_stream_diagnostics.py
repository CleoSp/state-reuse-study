"""Lesions act on answer content only and bootstrap preserves seed/root pairing."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_stream_reuse import lesion_hook
from check_adapter_pilot import stream_gates
from state_repair.types import PredictionBatch


@pytest.mark.parametrize("kind", ["uniform", "shuffled_nodes"])
def test_lesion_policy_starts_fresh_and_continues_an_edit(kind):
    from copy import deepcopy
    from state_repair.data.maze import MazeExample, collate, generate_maze
    from state_repair.models.adapters import make_adapter
    from state_repair.models.policy import FixedBudgetPolicy
    from state_repair.models.recursive import RecursiveSolver
    from state_repair.train.pilot import streams
    from state_repair.types import Domain

    torch.manual_seed(17)
    root = MazeExample(generate_maze(3,3,17), "synthetic-lesion-root", 0, "train", True)
    observations = [collate(frame)[0] for frame in streams([root],1,23)]
    solver = RecursiveSolver(width=8,heads=2,inner_cycles=1).eval()
    adapter = make_adapter("answer_only",width=8,domain=Domain.MAZE).eval()
    intact = FixedBudgetPolicy(solver,deepcopy(adapter),1)
    lesioned = FixedBudgetPolicy(solver,adapter,1)
    handle = adapter.register_forward_pre_hook(lesion_hook(kind),with_kwargs=True)
    try:
        with torch.no_grad():
            expected = intact(observations[0])
            initial = lesioned(observations[0])
            assert torch.equal(initial.prediction.logits,expected.prediction.logits)
            edited = lesioned(observations[1])
        assert edited.state.budget == 1
        assert edited.observation.frame_indices == (1,)
    finally:
        handle.remove()


def test_uniform_and_shuffle_lesions_preserve_source_and_padding():
    logits = torch.arange(24,dtype=torch.float).reshape(2,4,3)
    original = logits.clone()
    valid = torch.tensor([[1,1,1,0],[1,1,1,1]],dtype=torch.bool)
    args = (SimpleNamespace(new=SimpleNamespace(valid_nodes=valid)),)
    kwargs = {"previous_prediction":PredictionBatch(logits)}
    _,uniform = lesion_hook("uniform")(None,args,kwargs)
    assert torch.equal(uniform["previous_prediction"].logits,torch.zeros_like(logits))
    _,shuffled = lesion_hook("shuffled_nodes")(None,args,kwargs)
    out = shuffled["previous_prediction"].logits
    for b in range(2):
        assert torch.equal(out[b,valid[b]].sort(dim=0).values,logits[b,valid[b]].sort(dim=0).values)
    assert torch.equal(out[0,3],logits[0,3]) and torch.equal(logits,original)


def test_gates_keep_all_seeds_and_reject_unpaired_roots():
    cfg = {"seeds":[29,43,71],"budgets":[1,2,4,8],"bootstrap_seed":11,"bootstrap_repetitions":100}
    curves = {( "joint",arm,seed,k):{str(i):value for i in range(8)}
        for arm,value in [("restart",.2),("answer_only",.6),("carry",.4),("spatial_gate",.4),
                          ("global_gate",.4),("gru_adapter",.4),("residual_adapter",.4)]
        for seed in cfg["seeds"] for k in cfg["budgets"]}
    result = stream_gates(curves,cfg)
    assert result["G3_output_reuse_all_K124"] and not result["G2_latent_reuse_reopened"]
    curves["joint","carry",29,1].pop("0")
    with pytest.raises(ValueError,match="unpaired"):
        stream_gates(curves,cfg)
