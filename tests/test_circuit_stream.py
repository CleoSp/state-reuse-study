"""Synthetic circuit stream ancestry, scoring, and real trainer integration."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_circuit_stream as runner
from state_repair.data.circuit import CircuitExample, generate_circuit
from state_repair.models.recursive import RecursiveSolver
from state_repair.types import Domain


def fixture():
    config = {**runner.read("configs/circuit_stream_v2.json"), "synthetic": True,
        "width": 8, "heads": 2, "inner_cycles": 1, "context_width": 4,
        "steps": 2, "batch_size": 2}
    roots = [CircuitExample(generate_circuit(8, s), f"synthetic-{s}", 0, "train", tuple(range(8)), True) for s in (11, 12)]
    frames = runner.stream_frames(roots, 4, config["edit_seed"])
    return config, frames


def test_circuit_stream_keeps_node_correspondence_and_one_edit():
    config, frames = fixture()
    for i in range(2):
        for f in range(1, 5):
            old, new = frames[f-1][i], frames[f][i]
            assert old.root_id == new.root_id and old.node_order == new.node_order
            assert new.frame_index == f and new.synthetic
            changes = sum(a != b for a,b in zip(old.circuit.operators, new.circuit.operators))
            changes += sum(a != b for a,b in zip(old.circuit.input_bits, new.circuit.input_bits))
            assert changes == 1
    data = runner.batches([[asdict(e) for e in f] for f in frames], config, 4)
    assert len(data) == 1 and len(data[0]) == 5
    for _, obs, target in data[0]:
        target.check_observation(obs)


@pytest.mark.parametrize("arm", ["restart", "carry", "spatial_gate", "answer_only"])
def test_circuit_trainer_checkpoint_and_equal_frame_objective(tmp_path, monkeypatch, arm):
    config, frames = fixture()
    data = runner.batches([[asdict(e) for e in f] for f in frames], config, 4)
    model = RecursiveSolver(width=8, heads=2, inner_cycles=1, domain=Domain.CIRCUIT)
    class CPUJob:
        def __init__(self, config, out, seconds, purpose, *, synthetic):
            assert synthetic
            self.out = out
        def __enter__(self):
            self.out.mkdir()
            return self
        def check_limit(self):
            pass
        def __exit__(self, *args):
            runner.write(self.out / "resources.json", {"synthetic": True, "device": "cpu"})
    monkeypatch.setattr(runner, "gpu_job", CPUJob)
    (tmp_path / "prepared").mkdir()
    for name in ("dataset.json", "manifest.json"):
        runner.write(tmp_path / "prepared" / name, {"synthetic": True})
    runner.train_one(tmp_path, config, data, data, model.state_dict(), arm, 29, 0, device="cpu")
    out = tmp_path / "training" / f"joint-{arm}-seed29-grid0"
    runner.verify_seal(out)
    rows = list(runner.json_rows(out / "steps.jsonl"))
    assert all(r["loss"] == sum(r["frame_losses"])/5 for r in rows)
    assert all(r["frame_block_calls"] == [r["K"]*2]*5 for r in rows)
    if arm not in ("restart", "carry"):
        assert all(r["adapter_gradient_norm"] > 0 for r in rows)
    predictions = list(runner.json_rows(out / "tuning.jsonl.gz"))
    assert len(predictions) == 20 and all(r["synthetic"] and r["state_budget"] == r["K"] for r in predictions)
    assert runner.read(out / "summary.json")["validation"] == runner.summarize(predictions)
