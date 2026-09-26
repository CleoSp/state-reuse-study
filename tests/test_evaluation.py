from dataclasses import replace

import pytest
import torch

from state_repair.data.maze import Maze,MazeExample
from state_repair.eval.smoke import aggregate,evaluate_examples
from state_repair.models.recursive import RecursiveSolver


def test_cpu_static_latency_and_block_counts_are_measured():
    torch.set_num_threads(1)
    model=RecursiveSolver(width=8,heads=2)
    example=MazeExample(Maze(1,2,((0,1),),0,1),"synthetic-root",0,"val",True)
    observed=[]
    hook=model.block.register_forward_hook(lambda *_:observed.append(1))
    records=evaluate_examples(model,[example],[0,1,2])
    hook.remove()
    assert len(observed)==6
    assert [r["block_calls"] for r in records]==[0,2,4]
    for row in records:
        assert row["synthetic"] is True
        assert row["batch_size"]==1 and row["total_inference_ms"]>0
        assert row["total_inference_ms"]==pytest.approx(sum(row[name] for name in
            ("input_ms","encoder_ms","initialization_ms","core_ms","decode_ms","copy_ms")))
    with pytest.raises(ValueError,match="synthetic"):
        aggregate(records)
    assert len(aggregate(records,empirical=False))==3
    with pytest.raises(ValueError,match="one frame"):
        aggregate(records+records,empirical=False)


def test_smoke_does_not_evaluate_test_or_edited_data():
    model=RecursiveSolver(width=8,heads=2)
    example=MazeExample(Maze(1,1,(),0,0),"synthetic",0,"test",True)
    with pytest.raises(ValueError,match="train/val"):
        evaluate_examples(model,[example],[1])
    with pytest.raises(ValueError,match="frame zero"):
        evaluate_examples(model,[replace(example,split="val",frame_index=1)],[1])
