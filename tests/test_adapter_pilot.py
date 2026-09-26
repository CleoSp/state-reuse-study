"""Synthetic end-to-end runner checks, budget independence and pairing failures."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_adapter_pilot as runner
from check_adapter_pilot import check_stream_file, stop_screen
from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.adapter_step import FrozenPrior
from state_repair.train.pilot import schedule, streams, training_views
from state_repair.types import PredictionBatch


def config():
    return {**runner.read("configs/adapter_pilot_v1.json"), "synthetic": True,
        "width": 8, "heads": 2, "inner_cycles": 1, "context_width": 4,
        "steps": 2, "batch_size": 2, "budgets": [1, 2], "seeds": [29, 43, 71],
        "source_checkpoint_sha256": "synthetic-checkpoint"}


def roots():
    m = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
    second = Maze(2, 3, ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5)), 0, 5)
    return [MazeExample(value, f"synthetic-{i}", 0, "train", True) for i, value in enumerate((m, second))]


def test_schedule_and_uniform_edits_keep_ancestry_and_do_not_depend_on_arm():
    c = config()
    a = training_views(roots(), c)
    assert a == training_views(roots(), c)
    for view in a:
        for old, new in zip(roots(), view):
            assert new.root_id == old.root_id and new.split == "train" and new.frame_index == 1
            assert len(set(old.maze.edges)^set(new.maze.edges)) == 1
    c["steps"] = 12
    s = schedule(29, 4, c)
    assert s == schedule(29, 4, {**c, "unread_policy": "other"})
    assert s != schedule(43, 4, c)
    assert all(sorted(i for i, k in s[j:j+4]) == list(range(4)) for j in range(0, 12, 4))


@pytest.mark.parametrize("track,arm,protocol", [("frozen", "gru_adapter", "one_edit"), ("joint", "restart", "one_edit"), ("joint", "spatial_gate", "stream"), ("joint", "answer_only", "stream")])
def test_real_runner_cpu_optimizer_checkpoint_and_stream_roundtrip(tmp_path, monkeypatch, track, arm, protocol):
    c = config()
    torch.manual_seed(91)
    model = RecursiveSolver(width=8, heads=2, inner_cycles=1).requires_grad_(False)
    weights = {n: v.clone() for n, v in model.state_dict().items()}
    old = [collate(roots())[:2]]
    new = [collate(v)[:2] for v in training_views(roots(), c)]
    if protocol == "stream":
        c.update(training_protocol="stream", training_edits=4)
        training = streams(roots(), 4, c["edit_seed"])
        new = runner.stream_training_batches({"train": [asdict(e) for e in training[0]],
            "views": [[asdict(e) for e in f] for f in training[1:]]}, c)
    frames = streams([replace(e, split="val") for e in roots()], 4, 63)
    batches = [[(frame, collate(frame)[0])] for frame in frames]
    caches = {}
    with torch.no_grad():
        for k in c["budgets"]:
            result = model(old[0][0], k)
            caches[0, k] = FrozenPrior(old[0][0], result.state.detach(), result.prediction, "synthetic-checkpoint", "restart", k)
    (tmp_path / "prepared").mkdir()
    (tmp_path / "cache").mkdir()
    for p in ("prepared/dataset.json", "prepared/manifest.json", "cache/cache_index.json"):
        runner.write(tmp_path / p, {"synthetic": True})
    class CPUFixtureJob:
        def __init__(self, output, seconds, purpose, *, synthetic):
            assert synthetic is True
            self.output = output
        def __enter__(self):
            self.output.mkdir()
            return self
        def check_limit(self):
            pass
        def __exit__(self, *args):
            runner.write(self.output / "resources.json", {"synthetic": True, "device": "cpu"})
    monkeypatch.setattr(runner, "GPUJob", CPUFixtureJob)
    runner.train_one(tmp_path, c, {}, old, new, batches, caches, weights, track, arm, 29, 0, device="cpu")
    out = tmp_path / "training" / f"{track}-{arm}-seed29-grid0"
    runner.verify_seal(out)
    cp = torch.load(out / "checkpoint.pt", weights_only=False)
    assert cp["steps"] == 2 and cp["cuda_rng"] == []
    assert all(torch.equal(cp["model"][n], v) for n, v in weights.items()) == (track == "frozen")
    rows = list(runner.json_rows(out / "tuning.jsonl.gz"))
    assert len(rows) == 2*5*2 and all(r["synthetic"] for r in rows)
    assert all(r["state_budget"] == r["K"] for r in rows)
    summary = runner.read(out / "summary.json")
    assert summary["validation"] == runner.evaluation_summary(rows)
    if protocol == "stream":
        steps = list(runner.json_rows(out / "steps.jsonl"))
        assert all(r["frames_per_step"] == 5 for r in steps)
        assert all(r["loss"] == sum(r["frame_losses"])/5 for r in steps)
        assert all(r["post_edit_block_calls"] == 4*r["K"]*2 for r in steps)
        assert summary["training_example_F_calls"] == sum(5*r["K"]*2*2 for r in steps)
    examples = {(e.frame_index, e.root_id): e for frame in frames for e in frame}
    with pytest.raises(ValueError, match="empirical"):
        check_stream_file(out / "tuning.jsonl.gz", c, examples, 4, {})
    duplicate = [*rows, rows[0]]
    with pytest.raises(ValueError, match="duplicate"):
        runner.evaluation_summary(duplicate)


def test_stop_screen_requires_both_tracks_every_budget_and_paired_roots():
    c = {**config(), "bootstrap_repetitions": 100}
    curves = {}
    for track in ("frozen", "joint"):
        for arm in ("spatial_gate", "restart", "carry"):
            for seed in c["seeds"]:
                for k in c["budgets"]:
                    curves[track, arm, seed, k] = {f"root-{r}": 0. if arm == "spatial_gate" else 1. for r in range(8)}
    assert stop_screen(curves, c)["scientific_stop"]
    for seed in c["seeds"]:
        curves["joint", "spatial_gate", seed, 2] = {f"root-{r}": 1. for r in range(8)}
    assert not stop_screen(curves, c)["scientific_stop"]
