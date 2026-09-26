"""Versioned, independently hashed split files. Never silently overwrite."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from state_repair.config import ExperimentConfig, require_static_workflow
from state_repair.data.maze import Maze, MazeExample, generate_episode, generate_maze
from state_repair.data.splits import assign_splits, canonical_hash, reject_cross_split_duplicates
from state_repair.oracles.maze import impact_metadata, solve_maze


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _dataset_config(config: ExperimentConfig) -> dict:
    return {"schema_version": 1, "seed": config.seed, "dataset": asdict(config.dataset)}


def generate_dataset(config: ExperimentConfig, *, synthetic: bool = False) -> Path:
    require_static_workflow(config)
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be an explicit Boolean")
    destination = Path(config.output_dir) / "dataset"
    manifest_path = destination / "manifest.json"
    identity = _dataset_config(config)
    if synthetic:
        identity["synthetic"] = True
    if destination.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"incomplete dataset exists; inspect before removing: {destination}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("generation_config") != identity:
            raise ValueError("existing dataset configuration differs; use a new output_dir")
        for split in ("train", "val", "test"):
            load_split(destination, split)
        return destination
    dc = config.dataset
    assignments = assign_splits(dc.roots, config.seed, dc.train_fraction, dc.val_fraction)
    examples = []
    root_manifest = {}
    for root_id, split in sorted(assignments.items()):
        root_seed = int.from_bytes(hashlib.sha256(f"{config.seed}:{root_id}".encode()).digest()[:8], "big")
        maze = generate_maze(dc.height, dc.width, root_seed, dc.extra_edge_probability)
        episode = generate_episode(maze, root_id, split, dc.edits_per_episode, root_seed ^ 0x5EED)
        if synthetic:
            episode = [replace(example, synthetic=True) for example in episode]
        examples.extend(episode)
        root_manifest[root_id] = {"split": split, "root_hash": canonical_hash(maze),
                                  "frames": len(episode)}
    reject_cross_split_duplicates(examples)
    payloads = {}
    entries = {}
    for split in ("train", "val", "test"):
        records = []
        previous: dict[str, Maze] = {}
        for example in examples:
            if example.split != split:
                continue
            distances, actions = solve_maze(example.maze)
            metadata = {"distances": distances}
            if example.root_id in previous:
                metadata.update(impact_metadata(previous[example.root_id], example.maze))
            previous[example.root_id] = example.maze
            records.append({"schema_version": 1, "root_id": example.root_id,
                "frame_index": example.frame_index, "split": split, "synthetic": example.synthetic,
                "status": "developmental", "observation": asdict(example.maze),
                "target": {"valid_actions": actions}, "oracle_metadata": metadata,
                "canonical_hash": canonical_hash(example.maze)})
        payloads[split] = ("".join(_json(record) + "\n" for record in records)).encode()
        entries[split] = {"file": f"{split}.jsonl", "sha256": _digest(payloads[split]), "examples": len(records),
                          "roots": sum(s == split for s in assignments.values())}
    manifest = {"schema_version": 1, "generation_config": identity,
        "distribution": "randomized_kruskal_plus_independent_passages; uniform_potential_edge_toggle",
        "challenge_balancing": "not implemented", "cross_split_duplicate_policy": "reject_entire_generation_without_resampling",
        "splits": entries, "roots": root_manifest}
    destination.mkdir(parents=True, exist_ok=False)
    for split, payload in payloads.items():
        (destination / f"{split}.jsonl").write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_split(dataset_dir: Path | str, split: str) -> list[MazeExample]:
    """Read only the selected split payload; never open another split file."""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    directory = Path(dataset_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset schema")
    info = manifest["splits"][split]
    if info["file"] != f"{split}.jsonl":
        raise ValueError("split manifest filename mismatch")
    payload = (directory / info["file"]).read_bytes()
    if _digest(payload) != info["sha256"]:
        raise ValueError(f"{split} file hash mismatch")
    examples = []
    frames: dict[str, list[int]] = {}
    previous: dict[str, Maze] = {}
    for line in payload.decode().splitlines():
        record = json.loads(line)
        root_id = record["root_id"]
        if record["schema_version"] != 1 or record["split"] != split or manifest["roots"][root_id]["split"] != split:
            raise ValueError("record root/split/schema does not match manifest")
        raw = record["observation"]
        maze = Maze(raw["height"], raw["width"], tuple(tuple(e) for e in raw["edges"]), raw["start"], raw["goal"])
        if record["canonical_hash"] != canonical_hash(maze):
            raise ValueError("canonical observation hash mismatch")
        dist, actions = solve_maze(maze)
        if record["target"]["valid_actions"] != actions or record["oracle_metadata"]["distances"] != dist:
            raise ValueError("stored target/oracle mismatch")
        frame = record["frame_index"]
        if type(frame) is not int or frame < 0 or type(record["synthetic"]) is not bool:
            raise ValueError("invalid frame/synthetic record fields")
        expected_metadata = {"distances": dist}
        if root_id in previous:
            old = previous[root_id]
            if (old.height, old.width, old.start, old.goal) != (maze.height, maze.width, maze.start, maze.goal):
                raise ValueError("ordinary maze stream changed fixed layout/start/goal")
            if len(set(old.edges) ^ set(maze.edges)) != 1:
                raise ValueError("ordinary maze frames require exactly one passage toggle")
            expected_metadata.update(impact_metadata(old, maze))
        if record["oracle_metadata"] != expected_metadata:
            raise ValueError("stored impact metadata mismatch")
        previous[root_id] = maze
        if frame == 0 and canonical_hash(maze) != manifest["roots"][root_id]["root_hash"]:
            raise ValueError("root hash mismatch")
        frames.setdefault(root_id, []).append(frame)
        examples.append(MazeExample(maze, root_id, frame, split, record["synthetic"]))
    if len(examples) != info["examples"] or len(frames) != info["roots"]:
        raise ValueError("split record count mismatch")
    for root, indices in frames.items():
        if indices != list(range(manifest["roots"][root]["frames"])):
            raise ValueError("missing, reordered, or duplicate episode frames")
    if set(frames) != {root for root, entry in manifest["roots"].items() if entry["split"] == split}:
        raise ValueError("manifest root membership mismatch")
    reject_cross_split_duplicates(examples)
    return examples
