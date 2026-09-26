"""Synthetic correctness checks, not an empirical learning run."""
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import pytest
import torch

from state_repair.config import ExperimentConfig, ModelConfig
from state_repair.data.maze import Maze, MazeExample, collate
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.checkpoint import save_checkpoint
from state_repair.train import loop
from state_repair.train.losses import maze_valid_set_loss
from state_repair.types import TargetBatch


def test_valid_set_loss_ties_padding_and_gradient():
    logits = torch.zeros((1, 3, 6), requires_grad=True)
    labels = torch.zeros((1, 3, 6), dtype=torch.bool)
    labels[0, 0, :2] = True
    labels[0, 1, 4] = True
    mask = torch.tensor([[True, True, False]])
    loss = maze_valid_set_loss(logits, TargetBatch(labels), mask)
    assert loss.item() == pytest.approx((math.log(3) + math.log(6)) / 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[0, 2], torch.zeros(6))
    assert logits.grad[0, 0, 0] < 0 and logits.grad[0, 0, 2] > 0


def test_missing_labels_and_nonfinite_scored_logits_rejected():
    logits = torch.zeros((1, 1, 6))
    labels = torch.zeros_like(logits, dtype=torch.bool)
    mask = torch.ones((1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="valid action"):
        maze_valid_set_loss(logits, TargetBatch(labels), mask)
    labels[..., 0] = True
    logits[..., 1] = torch.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        maze_valid_set_loss(logits, TargetBatch(labels), mask)


def test_loss_reaches_encoder_and_recurrent_parameters():
    example = MazeExample(Maze(1, 2, ((0, 1),), 0, 1), "synthetic-fixture", 0, "train", synthetic=True)
    obs, targets, _ = collate([example])
    model = RecursiveSolver(width=8, heads=2)
    result = model(obs, 1)
    maze_valid_set_loss(result.prediction.logits, targets, obs.valid_nodes).backward()
    for parameter in (model.encoder.weight, model.block.layers[0].qkv.weight, model.initial_a):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0


def test_static_loader_opens_train_val_only_and_filters_descendants(tmp_path, monkeypatch):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = {"generation_config": {"schema_version": 1, "seed": config.seed, "dataset": config.to_dict()["dataset"]},
                "splits": {name: {"sha256": name} for name in ("train", "val", "test")}}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    opened = []

    def fake_load(path, split):
        assert split != "test"
        opened.append(split)
        return [MazeExample(Maze(1, 1, (), 0, 0), split, frame, split) for frame in (0, 1)]

    monkeypatch.setattr(loop, "load_split", fake_load)
    data, hashes, _ = loop.load_static_data(config)
    assert opened == ["train", "val"]
    assert set(hashes) == {"train", "val"}
    assert all(len(examples) == 1 and examples[0].frame_index == 0 for examples in data.values())


def test_resource_stop_conditions(monkeypatch):
    with pytest.raises(loop.ResourceLimitReached, match="wall-clock"):
        loop.check_limits(time.perf_counter() - 1, 4096)
    monkeypatch.setattr(loop, "memory_snapshot", lambda: {"rss_bytes": 2048 * 1024**2})
    with pytest.raises(loop.ResourceLimitReached, match="memory limit"):
        loop.check_limits(time.perf_counter() + 10, 1024)
    monkeypatch.setattr(loop, "memory_snapshot", lambda: {"rss_bytes": None, "peak_rss_bytes": None})
    with pytest.raises(loop.ResourceLimitReached, match="unavailable"):
        loop.check_limits(time.perf_counter() + 10, 4096)


def test_checkpoint_safe_load_and_no_overwrite(tmp_path):
    config = replace(ExperimentConfig(), model=ModelConfig(width=8, heads=2))
    model = RecursiveSolver(width=8, heads=2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(path, model, optimizer, config, 0, {"train": "x", "val": "y"}, {"synthetic": True})
    checkpoint = torch.load(path, weights_only=True)
    assert len(digest) == 64
    assert checkpoint["optimizer_steps"] == 0
    assert checkpoint["exact_resume_supported"] is False
    assert "optimizer_state_dict" in checkpoint and "torch_rng_state" in checkpoint
    restored = RecursiveSolver(width=8, heads=2)
    restored.load_state_dict(checkpoint["model_state_dict"])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, optimizer, config, 0, {}, {})


def test_training_rejects_excess_budget_accelerator_missing_and_existing(tmp_path):
    config = replace(ExperimentConfig(), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="max_minutes"):
        loop.train_static(config, 21)
    with pytest.raises(ValueError, match="CPU only"):
        loop.train_static(replace(config, device="cuda"))
    with pytest.raises(FileNotFoundError, match="generate"):
        loop.train_static(config)
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="already exist"):
        loop.train_static(config)
