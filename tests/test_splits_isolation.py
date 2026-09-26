from dataclasses import replace
import json

import pytest

from state_repair.config import DatasetConfig, ExperimentConfig
from state_repair.data.maze import Maze, MazeExample, generate_maze
from state_repair.data.serialization import generate_dataset, load_split
from state_repair.data.splits import assign_splits, canonical_hash, reject_cross_split_duplicates


def test_root_and_canonical_duplicate_rejection():
    maze = generate_maze(3, 3, 42)
    a = MazeExample(maze, "a", 0, "train")
    with pytest.raises(ValueError, match="root crosses splits"):
        reject_cross_split_duplicates([a, replace(a, split="val", maze=maze.toggle(*maze.edges[0]))])
    with pytest.raises(ValueError, match="canonical duplicate"):
        reject_cross_split_duplicates([a, replace(a, root_id="b", split="test", maze=replace(maze, start=(maze.start + 1) % maze.n))])
    reject_cross_split_duplicates([a, replace(a, frame_index=2)])
    rotated = Maze(3, 3, tuple(sorted(tuple(sorted((8-u, 8-v))) for u,v in maze.edges)), 8-maze.start, 8-maze.goal)
    assert canonical_hash(rotated) == canonical_hash(maze)


def test_split_determinism_and_all_descendants(tmp_path):
    cfg = ExperimentConfig(output_dir=str(tmp_path / "run"), dataset=DatasetConfig(height=5, width=5, roots=12, edits_per_episode=2))
    assert assign_splits(12, 17, .5, .25) == assign_splits(12, 17, .5, .25)
    directory = generate_dataset(cfg, synthetic=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    all_examples = []
    for split, count in (("train", 6), ("val", 3), ("test", 3)):
        examples = load_split(directory, split)
        assert len(examples) == count * 3
        assert all(manifest["roots"][e.root_id]["split"] == split for e in examples)
        all_examples.extend(examples)
    reject_cross_split_duplicates(all_examples)
    assert generate_dataset(cfg, synthetic=True) == directory
    with pytest.raises(ValueError, match="configuration differs"):
        generate_dataset(replace(cfg, seed=18), synthetic=True)


def test_train_load_does_not_open_test_and_detects_corruption(tmp_path):
    cfg = ExperimentConfig(output_dir=str(tmp_path / "run"), dataset=DatasetConfig(height=5, width=5, roots=8, edits_per_episode=0))
    directory = generate_dataset(cfg, synthetic=True)
    (directory / "test.jsonl").unlink()
    assert load_split(directory, "train")
    with pytest.raises(FileNotFoundError):
        load_split(directory, "test")
    with (directory / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_split(directory, "train")


def test_different_directories_identical_split_bytes(tmp_path):
    cfg = ExperimentConfig(output_dir=str(tmp_path / "first"), dataset=DatasetConfig(height=5, width=5, roots=8, edits_per_episode=1))
    first = generate_dataset(cfg, synthetic=True)
    second = generate_dataset(replace(cfg, output_dir=str(tmp_path / "second")), synthetic=True)
    for name in ("manifest.json", "train.jsonl", "val.jsonl", "test.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
