"""Deterministic typed data and identity-only test reservations."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import random

from state_repair.data import maze, circuit
from state_repair.data.deep_circuits import generate_deep_circuit
from state_repair.train.pilot import child_seed, decode_example
from .durable import atomic_json, digest_json, read_json


def identities(suite: str, split: str, count: int) -> list[str]:
    return [f"confirmatory-v1-{suite}-{split}-{i:06d}" for i in range(count)]


def decode(row: dict):
    if "maze" in row:
        return decode_example(row)
    c = row["circuit"]
    return circuit.CircuitExample(circuit.Circuit(tuple(circuit.Operator(o) for o in c["operators"]),
        tuple(tuple(p) for p in c["parents"]), tuple(c["input_bits"])), row["root_id"],
        row["frame_index"], row["split"], tuple(row["node_order"]), row["synthetic"])


def generate_roots(spec: dict, split: str, *, synthetic: bool = False) -> list:
    result = []
    for index, root in enumerate(spec["identities"][split]):
        seed = child_seed(spec["data_seed"], root)
        if spec["family"] == "maze":
            problem = maze.generate_maze(spec["size"], spec["size"], seed)
            if index % 4 == 0:
                cut = problem.width // 2
                problem = replace(problem, edges=tuple((u, v) for u, v in problem.edges
                    if (u % problem.width < cut) == (v % problem.width < cut)))
            example = maze.MazeExample(problem, root, 0, split, synthetic)
        else:
            generator = generate_deep_circuit if spec.get("depth_shift") else circuit.generate_circuit
            problem = generator(spec["size"], seed, inputs=spec["inputs"])
            order = list(range(problem.n))
            random.Random(seed ^ 17).shuffle(order)
            example = circuit.CircuitExample(problem, root, 0, split, tuple(order), synthetic)
        result.append(example)
    return result


def frames(roots: list, edits: int, seed: int) -> list[list]:
    episodes = []
    for root in roots:
        if isinstance(root, maze.MazeExample):
            episode = [replace(e, synthetic=root.synthetic) for e in maze.generate_episode(
                root.maze, root.root_id, root.split, edits, child_seed(seed, root.root_id))]
        else:
            episode = [replace(e, node_order=root.node_order) for e in circuit.generate_episode(
                root.circuit, root.root_id, root.split, edits, child_seed(seed, root.root_id), synthetic=root.synthetic)]
        episodes.append(episode)
    return [[episode[f] for episode in episodes] for f in range(edits + 1)]


def collate(examples: list, previous: list | None = None):
    if isinstance(examples[0], maze.MazeExample):
        return maze.collate(examples)
    return circuit.collate(examples, previous=None if previous is None else [e.circuit for e in previous])


def prepare_development(spec: dict, path: Path, *, synthetic: bool = False) -> dict:
    value = {split: [asdict(e) for e in generate_roots(spec, split, synthetic=synthetic)] for split in ("train", "val")}
    check_isolation([decode(e) for values in value.values() for e in values])
    if path.exists():
        if digest_json(read_json(path)) != digest_json(value):
            raise ValueError("development data changed")
    else:
        atomic_json(path, value)
    return value


def check_isolation(examples: list) -> None:
    from state_repair.data.splits import reject_cross_split_duplicates
    reject_cross_split_duplicates([e for e in examples if isinstance(e, maze.MazeExample)])
    circuit.reject_duplicate_roots([e for e in examples if isinstance(e, circuit.CircuitExample)])


def prepare_test_once(matrix: dict, protocol: Path, root: Path) -> dict:
    """Generate only after committed binding; reuse intact completed files.

    A claimed suite with no completed artifact is a hard stop, not permission
    to regenerate test roots. Atomic claims survive power loss independently
    of the final manifest. No quota failure causes replacement root sampling.
    """
    from .binding import validate_binding
    from .durable import DriverLock
    from state_repair.provenance import file_hash
    from state_repair.data.challenges import sample_challenges, addition_challenge_root
    validate_binding(matrix, protocol)
    synthetic = matrix.get("synthetic", False)
    root.mkdir(parents=True, exist_ok=True)
    with DriverLock(root / "generation.lock"):
        manifest_path = root / "test-manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {
            "matrix_sha256": digest_json(matrix), "protocol_sha256": file_hash(protocol), "suites": {}}
        if manifest["matrix_sha256"] != digest_json(matrix):
            raise ValueError("test generation matrix changed")
        all_examples = []
        for name, spec in matrix["suites"].items():
            development = read_json(root / f"{name}-development.json")
            all_examples.extend(decode(e) for split in ("train", "val") for e in development[split])
            path = root / f"{name}-test.json"
            claim = root / f"{name}-test-claim.json"
            if name in manifest["suites"]:
                if file_hash(path) != manifest["suites"][name]["sha256"]:
                    raise ValueError("generated test data changed")
                value = read_json(path)
            else:
                if claim.exists():
                    raise ValueError("test generation already claimed; inspect preserved incomplete generation: " + name)
                atomic_json(claim, {"suite": name, "matrix_sha256": digest_json(matrix), "identities": spec["identity_hashes"]})
                roots = generate_roots(spec, "test", synthetic=synthetic)
                entries = [{"old": asdict(e), "new": asdict(frames([e], 1, spec["stream_seed"])[1][0]), "branch": "ordinary"} for e in roots]
                challenge, audit = [], None
                if spec["family"] == "maze":
                    for i, identity in enumerate(spec["identities"]["challenge"]):
                        seed = child_seed(spec["data_seed"], identity)
                        e = addition_challenge_root(spec["size"], spec["size"], seed, identity, "test", synthetic=synthetic) if i < 64 else maze.MazeExample(
                            maze.generate_maze(spec["size"], spec["size"], seed), identity, 0, "test", synthetic)
                        challenge.append(e)
                    sampled = sample_challenges(challenge, pairs_per_type=spec["challenge_pairs_per_type"], seed=spec["challenge_seed"],
                        addition_only_roots=frozenset(e.root_id for e in challenge[:64]))
                    audit = sampled.audit
                    for pair in sampled.pairs:
                        for label, problem in (("low", pair.low), ("high", pair.high)):
                            entries.append({"old": asdict(pair.root), "new": asdict(replace(pair.root, maze=problem, frame_index=1)),
                                            "branch": pair.edit_type+"-"+label})
                value = {"test": [asdict(e) for e in roots], "challenge": [asdict(e) for e in challenge],
                         "interventions": entries, "challenge_audit": audit, "synthetic": synthetic}
                check_isolation(all_examples+roots+challenge)
                atomic_json(path, value)
                manifest["suites"][name] = {"path": path.as_posix(), "sha256": file_hash(path),
                    "roots": len(roots), "reserved_challenge_roots": len(challenge), "actual_intervention_branches": len(entries),
                    "identity_hashes": spec["identity_hashes"], "challenge_audit": audit}
                atomic_json(manifest_path, manifest)
            all_examples.extend(decode(e) for split in ("test", "challenge") for e in value[split])
        check_isolation(all_examples)
        manifest["complete"] = True
        atomic_json(manifest_path, manifest)
        return manifest


def verify_job_dataset(job: dict, matrix: dict, root: Path) -> None:
    from state_repair.provenance import file_hash
    if "dataset" not in job:
        return
    path = Path(job["dataset"])
    if job.get("split") == "test":
        manifest = read_json(root / "data/test-manifest.json")
        if not manifest.get("complete") or manifest["matrix_sha256"] != digest_json(matrix):
            raise ValueError("complete bound test manifest required")
        expected = {Path(v["path"]).resolve(): v["sha256"] for v in manifest["suites"].values()}
        digest = expected[path.resolve()]
    else:
        digest = matrix["development_hashes"][path.as_posix()]
    if file_hash(path) != digest:
        raise ValueError("bound dataset content changed")
