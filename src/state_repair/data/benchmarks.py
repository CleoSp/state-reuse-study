"""Separate ordinary and challenge datasets.

Run: python -m state_repair.data.benchmarks --output runs/benchmarks --synthetic
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from state_repair.data import circuit as circuits
from state_repair.data import maze as mazes
from state_repair.data.challenges import ChallengeSample, sample_challenges, addition_challenge_root
from state_repair.data.splits import assign_splits, canonical_hash, reject_cross_split_duplicates
from state_repair.oracles import circuit as circuit_oracle
from state_repair.oracles import maze as maze_oracle
from state_repair.types import Domain

SCHEMA_VERSION = 2
CHALLENGE_PROTOCOL = {"low_max": .1, "high_min": .4, "pairs_per_type": 1,
                      "metric": "action_set_changed_fraction", "edit_count": 1,
                      "matching_keys": ["base_root", "operation_type", "edit_count"]}
CONSTRUCTED_CHALLENGE_PROTOCOL = {**CHALLENGE_PROTOCOL, "addition_design": "two_components_v1",
    "constructed_roots_per_split": 1, "ordinary_distribution_unchanged": True}


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 17
    roots: int = 32
    edits: int = 4
    synthetic: bool = False
    maze_height: int = 8
    maze_width: int = 8
    extra_edge_probability: float = .15
    circuit_nodes: int = 32
    circuit_inputs: int = 8
    train_fraction: float = .5
    val_fraction: float = .25
    constructed_additions: bool = True

    def __post_init__(self) -> None:
        for name in ("seed", "roots", "edits", "maze_height", "maze_width", "circuit_nodes", "circuit_inputs"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.roots < 4 or self.edits < 0 or min(self.maze_height, self.maze_width) < 1:
            raise ValueError("require at least four roots, nonnegative edits and positive maze size")
        if type(self.constructed_additions) is not bool:
            raise ValueError("constructed_additions must be Boolean")
        if self.constructed_additions and (self.maze_height < 4 or self.maze_width < 2):
            raise ValueError("constructed addition strata require height >= 4 and width >= 2")
        if not 2 <= self.circuit_inputs < self.circuit_nodes or type(self.synthetic) is not bool:
            raise ValueError("invalid circuit dimensions or synthetic flag")
        if not 0 <= self.extra_edge_probability <= 1 or not 0 < self.train_fraction < 1 or not 0 < self.val_fraction < 1-self.train_fraction:
            raise ValueError("invalid generation/split probabilities")


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_circuit(raw: dict[str, Any]) -> circuits.Circuit:
    if set(raw) != {"operators", "parents", "input_bits"}:
        raise ValueError("circuit observation schema excludes extra fields")
    if any(type(op) is not int for op in raw["operators"]):
        raise ValueError("serialized operators must be integer enum codes")
    return circuits.Circuit(tuple(circuits.Operator(op) for op in raw["operators"]),
        tuple(tuple(p) for p in raw["parents"]), tuple(raw["input_bits"]))


def _raw_maze(raw: dict[str, Any]) -> mazes.Maze:
    if set(raw) != {"height", "width", "edges", "start", "goal"}:
        raise ValueError("maze observation schema excludes extra fields")
    return mazes.Maze(raw["height"], raw["width"], tuple(tuple(e) for e in raw["edges"]), raw["start"], raw["goal"])


def _record(example: mazes.MazeExample | circuits.CircuitExample,
            previous: mazes.Maze | circuits.Circuit | None) -> dict[str, Any]:
    record = {"schema_version": SCHEMA_VERSION, "root_id": example.root_id,
              "frame_index": example.frame_index, "split": example.split,
              "synthetic": example.synthetic, "status": "developmental", "suite": "ordinary"}
    if isinstance(example, mazes.MazeExample):
        distances, actions = maze_oracle.solve_maze(example.maze)
        metadata = {"distances": distances}
        if previous is not None:
            if not isinstance(previous, mazes.Maze):
                raise TypeError("maze record requires maze previous observation")
            metadata.update(maze_oracle.impact_metadata(previous, example.maze))
        record.update(domain=Domain.MAZE.value, observation=asdict(example.maze),
                      target={"valid_actions": actions}, oracle_metadata=metadata,
                      family_hash=canonical_hash(example.maze))
    elif isinstance(example, circuits.CircuitExample):
        values = circuit_oracle.evaluate(example.circuit)
        scored = [op != circuits.Operator.INPUT for op in example.circuit.operators]
        metadata = {"descendant_mask": [False]*example.circuit.n, "changed_value_mask": [False]*example.circuit.n}
        if previous is not None:
            if not isinstance(previous, circuits.Circuit):
                raise TypeError("circuit record requires circuit previous observation")
            metadata = circuit_oracle.impact_metadata(previous, example.circuit)
        record.update(domain=Domain.CIRCUIT.value, observation=asdict(example.circuit),
                      node_order=list(example.node_order), target={"values": [v if m else -1 for v,m in zip(values,scored)], "scored_mask": scored},
                      oracle_metadata=metadata, family_hash=circuits.topology_fingerprint(example.circuit))
    else:
        raise TypeError("unknown benchmark example type")
    record["observation_hash"] = _hash(_json(record["observation"]))
    return record


def _parse_record(record: dict[str, Any], domain: Domain, previous: mazes.Maze | circuits.Circuit | None) -> mazes.MazeExample | circuits.CircuitExample:
    if record.get("domain") != domain.value:
        raise ValueError("explicit benchmark domain mismatch")
    if domain == Domain.MAZE:
        maze = _raw_maze(record["observation"])
        example = mazes.MazeExample(maze, record["root_id"], record["frame_index"], record["split"], record["synthetic"])
        if previous is not None:
            if not isinstance(previous, mazes.Maze) or (previous.height,previous.width,previous.start,previous.goal) != (maze.height,maze.width,maze.start,maze.goal) or len(set(previous.edges)^set(maze.edges)) != 1:
                raise ValueError("ordinary maze requires fixed geometry/start/goal and one toggle")
    else:
        example = circuits.CircuitExample(_raw_circuit(record["observation"]), record["root_id"], record["frame_index"], record["split"], tuple(record["node_order"]), record["synthetic"])
    if (example.frame_index == 0) != (previous is None):
        raise ValueError("record frame and previous-observation mismatch")
    if record != json.loads(_json(_record(example, previous))):
        raise ValueError("stored benchmark target/metadata/schema mismatch")
    return example


def _payload(directory: Path, manifest: dict[str, Any], name: str) -> list[dict[str, Any]]:
    info = manifest["files"][name]
    payload = (directory / name).read_bytes()
    if _hash(payload) != info["sha256"]:
        raise ValueError(f"benchmark payload hash mismatch: {name}")
    records = [json.loads(line) for line in payload.splitlines()]
    if len(records) != info["records"]:
        raise ValueError("benchmark record count mismatch")
    return records


def _challenge_complete(audits: Any) -> bool:
    """Derive completion from quotas/counts, checking redundant status flags.

    Counts themselves are recomputed from roots by load_challenges before a
    cached dataset is returned. Fresh generation receives the sampler's audits.
    """
    if not isinstance(audits, dict) or set(audits) != {"train", "val", "test"} or any(
            not isinstance(a, dict) or type(a.get("complete")) is not bool for a in audits.values()):
        raise ValueError("manifest requires all challenge completion audits")
    statuses = []
    for audit in audits.values():
        quota = audit.get("requested_pairs_per_type")
        counts = audit.get("counts")
        if type(quota) is not int or quota != CHALLENGE_PROTOCOL["pairs_per_type"] or not isinstance(counts,dict) or set(counts) != {"addition","removal"}:
            raise ValueError("inconsistent challenge quota/count manifest")
        included = [counts[kind].get("included_pairs") for kind in ("addition","removal")]
        if any(type(n) is not int or not 0 <= n <= quota for n in included):
            raise ValueError("invalid challenge inclusion counts")
        derived = all(n == quota for n in included)
        if audit["complete"] != derived:
            raise ValueError("challenge completion audit disagrees with inclusion counts")
        statuses.append(derived)
    return all(statuses)


def _manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = CONSTRUCTED_CHALLENGE_PROTOCOL if manifest.get("generation_config", {}).get("constructed_additions", False) else CHALLENGE_PROTOCOL
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("challenge_protocol") != protocol:
        raise ValueError("unsupported benchmark schema/protocol")
    complete = _challenge_complete(manifest.get("challenges"))
    if type(manifest.get("challenge_complete")) is not bool or manifest["challenge_complete"] != complete:
        raise ValueError("manifest challenge_complete disagrees with per-split audits")
    if type(manifest.get("synthetic")) is not bool or manifest["synthetic"] is not manifest["generation_config"]["synthetic"]:
        raise ValueError("manifest synthetic status disagrees with generation configuration")
    return manifest


def _sample_for_split(roots: list[mazes.MazeExample], config: dict, split: str) -> ChallengeSample:
    """Additional challenge identities have their split fixed before generation."""
    seed = config["seed"]
    if not config.get("constructed_additions", False):
        return sample_challenges(roots, seed=seed)
    root_id = f"maze-designed-addition-{split}"
    root_seed = int.from_bytes(hashlib.sha256(f"{seed}:{root_id}".encode()).digest()[:8], "big")
    extra = addition_challenge_root(config["maze_height"], config["maze_width"], root_seed,
        root_id, split, synthetic=config["synthetic"])
    sample = sample_challenges([*roots, extra], seed=seed, addition_only_roots=frozenset([root_id]))
    sample.audit.update(constructed_root_ids=[root_id], constructed_base_hash=canonical_hash(extra.maze),
                        ordinary_roots_proposed=len(roots), addition_design="two_components_v1")
    return sample


def load_split(directory: Path | str, domain: Domain, split: str) -> list[mazes.MazeExample] | list[circuits.CircuitExample]:
    """Open only selected domain/split file and manifest; strip all labels/audits."""
    if not isinstance(domain, Domain) or split not in ("train", "val", "test"):
        raise ValueError("load_split requires explicit Domain enum and known split")
    directory = Path(directory)
    manifest = _manifest(directory)
    records = _payload(directory, manifest, f"{domain.value}/{split}.jsonl")
    previous: dict[str, mazes.Maze | circuits.Circuit] = {}
    examples = []
    frames: dict[str,list[int]] = {}
    orders: dict[str, tuple[int,...]] = {}
    for record in records:
        root = record["root_id"]
        root_info = manifest["roots"][domain.value][root]
        if record["split"] != split or root_info["split"] != split or record["synthetic"] is not manifest["generation_config"]["synthetic"]:
            raise ValueError("benchmark root/split/synthetic ancestry mismatch")
        example = _parse_record(record, domain, previous.get(root))
        if isinstance(example, circuits.CircuitExample):
            if root in orders and orders[root] != example.node_order:
                raise ValueError("circuit presentation correspondence changed within stream")
            orders[root] = example.node_order
            previous[root] = example.circuit
        else:
            previous[root] = example.maze
        if example.frame_index == 0 and record["family_hash"] != root_info["family_hash"]:
            raise ValueError("benchmark root family hash mismatch")
        frames.setdefault(root, []).append(example.frame_index)
        examples.append(example)
    expected_roots = {root for root, entry in manifest["roots"][domain.value].items() if entry["split"] == split}
    if set(frames) != expected_roots or any(v != list(range(manifest["generation_config"]["edits"]+1)) for v in frames.values()):
        raise ValueError("benchmark missing/reordered frames or roots")
    if domain == Domain.MAZE:
        reject_cross_split_duplicates(examples)
    else:
        circuits.reject_duplicate_roots(examples)
    return examples


def _challenge_records(sample: ChallengeSample) -> list[dict[str, Any]]:
    records = []
    for pair in sample.pairs:
        root = pair.root
        branches = []
        for index,(stratum,maze) in enumerate((("low",pair.low),("high",pair.high))):
            branches.append({"branch_id": f"{root.root_id}/branch-{index}",
                "observation": asdict(maze), "oracle_metadata": {
                    **maze_oracle.impact_metadata(root.maze,maze), "stratum": stratum}})
        records.append({"schema_version":SCHEMA_VERSION, "suite":"semantic_impact_challenge",
            "root_id":root.root_id, "split":root.split, "synthetic":root.synthetic,
            "base":asdict(root.maze), "base_hash":canonical_hash(root.maze),
            "edit_type":pair.edit_type, "branches":branches})
    return records


def load_challenges(directory: Path | str, split: str) -> list[dict[str, Any]]:
    """Evaluator-only paired branches; not accepted by model APIs."""
    if split not in ("train", "val", "test"):
        raise ValueError("unknown split")
    directory = Path(directory)
    manifest = _manifest(directory)
    records = _payload(directory, manifest, f"maze/challenge-{split}.jsonl")
    roots = [e for e in load_split(directory, Domain.MAZE, split) if e.frame_index == 0]
    seed = manifest["generation_config"]["seed"]
    expected = _sample_for_split(roots, manifest["generation_config"], split)
    if records != json.loads(_json(_challenge_records(expected))) or manifest["challenges"][split] != expected.audit:
        raise ValueError("challenge pairing/stratum/audit mismatch")
    return records


def generate_benchmarks(output: Path | str, config: BenchmarkConfig = BenchmarkConfig()) -> Path:
    """Bounded generation, no hidden root resampling; incomplete quotas visible."""
    destination = Path(output)
    if destination.exists():
        manifest = _manifest(destination)
        existing_config = {"constructed_additions": False, **manifest["generation_config"]}
        if existing_config != asdict(config):
            raise ValueError("existing benchmark configuration differs; use a new output")
        all_examples = {domain: [] for domain in Domain}
        for split in ("train","val","test"):
            for domain in Domain:
                all_examples[domain].extend(load_split(destination,domain,split))
            load_challenges(destination,split)
        reject_cross_split_duplicates(all_examples[Domain.MAZE])
        circuits.reject_duplicate_roots(all_examples[Domain.CIRCUIT])
        return destination
    assignments = assign_splits(config.roots, config.seed, config.train_fraction, config.val_fraction)
    examples: dict[Domain,list] = {domain: [] for domain in Domain}
    roots: dict[str,dict] = {domain.value: {} for domain in Domain}
    for base_id,split in sorted(assignments.items()):
        for domain in Domain:
            root = f"{domain.value}-{base_id}"
            seed = int.from_bytes(hashlib.sha256(f"{config.seed}:{root}".encode()).digest()[:8],"big")
            if domain == Domain.MAZE:
                maze = mazes.generate_maze(config.maze_height,config.maze_width,seed,config.extra_edge_probability)
                episode = [mazes.MazeExample(e.maze,e.root_id,e.frame_index,e.split,config.synthetic)
                           for e in mazes.generate_episode(maze,root,split,config.edits,seed^0x5EED)]
                family_hash = canonical_hash(maze)
            else:
                circuit = circuits.generate_circuit(config.circuit_nodes,seed,config.circuit_inputs)
                episode = circuits.generate_episode(circuit,root,split,config.edits,seed^0x5EED,synthetic=config.synthetic)
                family_hash = circuits.topology_fingerprint(circuit)
            examples[domain].extend(episode)
            roots[domain.value][root] = {"split":split,"family_hash":family_hash,"frames":len(episode)}
    reject_cross_split_duplicates(examples[Domain.MAZE])
    circuits.reject_duplicate_roots(examples[Domain.CIRCUIT])
    records: dict[str,list] = {}
    audits = {}
    maze_union = list(examples[Domain.MAZE])
    for split in ("train","val","test"):
        for domain in Domain:
            previous = {}
            records[f"{domain.value}/{split}.jsonl"] = []
            for example in examples[domain]:
                if example.split == split:
                    records[f"{domain.value}/{split}.jsonl"].append(_record(example,previous.get(example.root_id)))
                    previous[example.root_id] = example.maze if domain == Domain.MAZE else example.circuit
        base_roots = [e for e in examples[Domain.MAZE] if e.split == split and e.frame_index == 0]
        sample = _sample_for_split(base_roots, asdict(config), split)
        audits[split] = sample.audit
        records[f"maze/challenge-{split}.jsonl"] = _challenge_records(sample)
        for pair in sample.pairs:
            maze_union.append(pair.root)
            maze_union.extend(mazes.MazeExample(m,pair.root.root_id,1,split,config.synthetic) for m in (pair.low,pair.high))
    reject_cross_split_duplicates(maze_union)
    payloads = {name:b"".join(_json(row)+b"\n" for row in rows) for name,rows in records.items()}
    manifest = {"schema_version":SCHEMA_VERSION,"generation_config":asdict(config),"synthetic":config.synthetic,
        "status":"developmental", "challenge_protocol":CONSTRUCTED_CHALLENGE_PROTOCOL if config.constructed_additions else CHALLENGE_PROTOCOL, "challenges":audits,
        "challenge_complete":_challenge_complete(audits),"roots":roots,
        "files":{name:{"sha256":_hash(data),"records":len(records[name])} for name,data in payloads.items()},
        "duplicate_audit":{"checked_all_ordinary_and_challenge_descendants":True,"collisions":0,
                           "resampled_roots":0,"policy":"reject_entire_generation_on_first_collision"},
        "distributions":{
            "maze":"randomized Kruskal (not uniform spanning tree), independent extra passages, uniform potential-edge toggle; fixed start/goal",
            "circuit":"uniform operators AND/OR/XOR/NOT; uniformly sampled distinct earlier parents; random identity relabel and independent fixed presentation; 50/50 input-flip/binary-substitution types when binary gates exist, otherwise input flip",
            "challenge":"same-base same-operation paired low/high single toggles; optional constructed two-component addition roots; conditioned challenge is not ordinary prevalence"}}
    destination.mkdir(parents=True,exist_ok=False)
    for domain in Domain:
        (destination/domain.value).mkdir()
    for name,payload in payloads.items():
        with (destination/name).open("xb") as handle:
            handle.write(payload)
    with (destination/"manifest.json").open("x",encoding="utf-8") as handle:
        json.dump(manifest,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write("\n")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--roots",type=int,default=32)
    parser.add_argument("--edits",type=int,default=4)
    parser.add_argument("--synthetic",action="store_true",help="required for test-only generation")
    parser.add_argument("--legacy-challenges", action="store_true", help="reproduce historical unconditioned-root challenge sampling")
    args = parser.parse_args(argv)
    try:
        output = generate_benchmarks(args.output,BenchmarkConfig(seed=args.seed,roots=args.roots,edits=args.edits,synthetic=args.synthetic,
                                                               constructed_additions=not args.legacy_challenges))
        manifest = _manifest(output)
        print(json.dumps({"output":str(output),"synthetic":args.synthetic,
            "challenge_complete":manifest["challenge_complete"],"challenges":manifest["challenges"]},indent=2))
        return 0 if manifest["challenge_complete"] else 2
    except (ValueError,TypeError,FileNotFoundError,FileExistsError,KeyError) as error:
        print(json.dumps({"status":"failed","synthetic":args.synthetic,"error":str(error),
                          "resampled_roots":0,"collision_count":int("collision" in str(error) or "duplicate" in str(error))}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
