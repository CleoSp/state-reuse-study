from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from state_repair.data.benchmarks import (BenchmarkConfig, CHALLENGE_PROTOCOL, generate_benchmarks,
    load_challenges, load_split, main, _parse_record, _challenge_complete)
from state_repair.data.challenges import sample_challenges
from state_repair.data.maze import Maze, MazeExample, generate_maze, possible_edges
from state_repair.data.splits import reject_cross_split_duplicates
from state_repair.data.serialization import generate_dataset, load_split as load_static_split
from state_repair.config import ExperimentConfig, DatasetConfig
from state_repair.oracles.maze import impact_metadata, score_policy
from state_repair.types import Domain


@pytest.fixture(scope="module")
def benchmark_dir(tmp_path_factory):
    return generate_benchmarks(tmp_path_factory.mktemp("benchmarks")/"data",BenchmarkConfig(constructed_additions=False,roots=16,edits=2,synthetic=True))


def test_benchmark_roundtrip_manifests_and_no_overwrite(benchmark_dir):
    manifest = json.loads((benchmark_dir/"manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["challenge_protocol"] == CHALLENGE_PROTOCOL
    assert manifest["duplicate_audit"]["collisions"] == 0
    for domain in Domain:
        seen = set()
        for split,count in (("train",8),("val",4),("test",4)):
            examples = load_split(benchmark_dir,domain,split)
            assert len(examples) == count*3 and all(e.synthetic is True for e in examples)
            current = {e.root_id for e in examples}
            assert not seen.intersection(current)
            seen.update(current)
    assert generate_benchmarks(benchmark_dir,BenchmarkConfig(constructed_additions=False,roots=16,edits=2,synthetic=True)) == benchmark_dir
    with pytest.raises(ValueError,match="configuration differs"):
        generate_benchmarks(benchmark_dir,BenchmarkConfig(constructed_additions=False,roots=16,edits=3,synthetic=True))
    with pytest.raises(ValueError,match="Domain"):
        load_split(benchmark_dir,"maze","train")


def test_challenges_pair_base_edit_type_counts_and_lineage(benchmark_dir):
    manifest = json.loads((benchmark_dir/"manifest.json").read_text())
    for split in ("train","val","test"):
        records = load_challenges(benchmark_dir,split)
        ordinary = {e.root_id:e for e in load_split(benchmark_dir,Domain.MAZE,split) if e.frame_index == 0}
        assert len({r["root_id"] for r in records}) == len(records)
        for pair in records:
            base = ordinary[pair["root_id"]].maze
            assert pair["synthetic"] is True and pair["split"] == split
            assert len({b["branch_id"] for b in pair["branches"]}) == 2
            for branch in pair["branches"]:
                raw = branch["observation"]
                new = Maze(raw["height"],raw["width"],tuple(tuple(e) for e in raw["edges"]),raw["start"],raw["goal"])
                assert len(set(base.edges)^set(new.edges)) == 1
                assert (len(new.edges)>len(base.edges)) == (pair["edit_type"]=="addition")
                fraction = sum(impact_metadata(base,new)["action_set_changed"])/base.n
                if branch["oracle_metadata"]["stratum"] == "low": assert fraction <= .1
                else: assert fraction >= .4
        audit = manifest["challenges"][split]
        assert audit["ordinary_prevalence_estimate"] is False
        assert audit["roots_included"]+audit["roots_rejected"] == len(ordinary)
        for count in audit["counts"].values():
            assert count["low"]+count["middle"]+count["high"] == count["candidates"]
            assert count["included_candidates"]+count["rejected_candidates"] == count["candidates"]
            assert count["pairing_failures"]+count["eligible_roots"] == len(ordinary)
        assert sum(c["candidates"] for c in audit["counts"].values()) == len(ordinary)*len(possible_edges(8,8))


def test_unmet_challenge_quotas_and_determinism():
    root = MazeExample(Maze(1,2,((0,1),),0,1),"tiny",0,"test",True)
    sample = sample_challenges([root])
    assert not sample.audit["complete"] and not sample.pairs
    assert sample.audit["counts"]["addition"]["candidates"] == 0
    assert sample.audit["counts"]["removal"]["unmet_pair_quota"] == 1
    assert sample == sample_challenges([root])
    with pytest.raises(ValueError,match="frame zero"):
        sample_challenges([replace(root,frame_index=1)])
    with pytest.raises(ValueError): sample_challenges([root],low_max=.4,high_min=.1)


def test_disconnected_synthetic_base_has_matched_low_high_addition():
    maze = Maze(3,3,((0,1),(0,3),(1,2),(1,4),(2,5),(6,7),(7,8)),0,8)
    root = MazeExample(maze,"fixture-disconnected",0,"test",True)
    sample = sample_challenges([root])
    pair = next(p for p in sample.pairs if p.edit_type == "addition")
    assert pair.root == root
    assert sum(impact_metadata(maze,pair.low)["action_set_changed"]) == 0
    assert sum(impact_metadata(maze,pair.high)["action_set_changed"])/maze.n >= .4
    assert sample.audit["counts"]["addition"]["included_pairs"] == 1
    assert not sample.audit["complete"]


def test_complete_status_derived_from_constructed_paired_fixtures():
    addition = Maze(3,3,((0,1),(0,3),(1,2),(1,4),(2,5),(6,7),(7,8)),0,8)
    snake = [0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]
    removal = Maze(4,4,tuple(sorted(tuple(sorted((a,b))) for a,b in zip(snake,snake[1:]))),0,12)
    roots = [MazeExample(addition,"constructed-addition",0,"train",True),
             MazeExample(removal,"constructed-removal",0,"train",True)]
    sample = sample_challenges(roots)
    assert sample.audit["complete"] and {p.edit_type for p in sample.pairs} == {"addition","removal"}
    assert _challenge_complete({split:sample.audit for split in ("train","val","test")}) is True


def test_exact_duplicate_is_not_independent_root():
    root = MazeExample(generate_maze(4,5,21),"a",0,"train",True)
    with pytest.raises(ValueError,match="independent roots"):
        reject_cross_split_duplicates([root,replace(root,root_id="renamed")])


def test_split_reader_does_not_open_other_splits_and_corruption(tmp_path,benchmark_dir):
    (tmp_path/"circuit").mkdir()
    (tmp_path/"manifest.json").write_bytes((benchmark_dir/"manifest.json").read_bytes())
    payload = (benchmark_dir/"circuit/train.jsonl").read_bytes()
    selected = tmp_path/"circuit/train.jsonl"; selected.write_bytes(payload)
    assert load_split(tmp_path,Domain.CIRCUIT,"train")
    with pytest.raises(FileNotFoundError): load_split(tmp_path,Domain.CIRCUIT,"test")
    selected.write_bytes(payload+b"{}\n")
    with pytest.raises(ValueError,match="hash mismatch"): load_split(tmp_path,Domain.CIRCUIT,"train")
    records = [json.loads(line) for line in payload.splitlines()]
    node = records[0]["target"]["scored_mask"].index(True)
    records[0]["target"]["values"][node] ^= 1
    damaged = b"".join(json.dumps(r).encode()+b"\n" for r in records)
    selected.write_bytes(damaged)
    manifest = json.loads((tmp_path/"manifest.json").read_text())
    manifest["files"]["circuit/train.jsonl"]["sha256"] = hashlib.sha256(damaged).hexdigest()
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="target/metadata"): load_split(tmp_path,Domain.CIRCUIT,"train")


def test_old_maze_loader_rejects_changed_impact_even_with_updated_hash(tmp_path):
    directory = generate_dataset(ExperimentConfig(output_dir=str(tmp_path),dataset=DatasetConfig(height=5,width=5,roots=8,edits_per_episode=1)),synthetic=True)
    path = directory/"train.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    record = next(r for r in records if r["frame_index"]==1)
    record["oracle_metadata"]["distance_changed"][0] ^= True
    payload = "".join(json.dumps(r)+"\n" for r in records).encode(); path.write_bytes(payload)
    manifest = json.loads((directory/"manifest.json").read_text())
    manifest["splits"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
    (directory/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="impact metadata"): load_static_split(directory,"train")


def test_invalid_labels_and_test_fixture():
    maze = Maze(1,1,(),0,0)
    for actions in ([True],[4.0],[6]):
        with pytest.raises(ValueError): score_policy(maze,actions)
    records = json.loads((Path(__file__).parent/"fixtures/circuit_blocked.json").read_text())
    old = None
    for record in records:
        assert record["synthetic"] is True
        example = _parse_record(record,Domain.CIRCUIT,old)
        old = example.circuit


def test_distance_change_is_not_action_change_or_prediction_invalidation():
    old = Maze(2,3,((0,1),(1,2),(1,4),(2,5),(3,4),(4,5)),0,2)
    new = old.toggle(1,2)
    impact = impact_metadata(old,new,[1,1,4,1,1,0])
    assert impact["distance_changed"][0] is True
    assert impact["action_set_changed"][0] is False
    assert impact["old_prediction_now_invalid"][0] is False


def test_cli_errors_do_not_overwrite(tmp_path,capsys):
    assert main(["--output",str(tmp_path/"invalid"),"--roots","-1","--synthetic","--legacy-challenges"]) == 2
    assert not (tmp_path/"invalid").exists()
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    incomplete = tmp_path/"partial"; incomplete.mkdir(); (incomplete/"keep.txt").write_text("unrelated")
    assert main(["--output",str(incomplete),"--synthetic","--legacy-challenges"]) == 2
    assert (incomplete/"keep.txt").read_text() == "unrelated"


def test_completion_flag_tamper_cannot_turn_failed_generation_into_success(tmp_path,benchmark_dir,capsys):
    import shutil
    copy = tmp_path/"copy"
    shutil.copytree(benchmark_dir,copy)
    manifest = json.loads((copy/"manifest.json").read_text())
    assert manifest["challenge_complete"] is False
    assert main(["--output",str(copy),"--roots","16","--edits","2","--synthetic","--legacy-challenges"]) == 2
    assert json.loads(capsys.readouterr().out)["challenge_complete"] is False
    manifest["challenge_complete"] = True
    (copy/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="challenge_complete"):
        generate_benchmarks(copy,BenchmarkConfig(constructed_additions=False,roots=16,edits=2,synthetic=True))
    assert main(["--output",str(copy),"--roots","16","--edits","2","--synthetic","--legacy-challenges"]) == 2
    assert "challenge_complete" in json.loads(capsys.readouterr().out)["error"]
    for audit in manifest["challenges"].values():
        audit["complete"] = True
        for counts in audit["counts"].values(): counts["included_pairs"] = 1
    (copy/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="challenge pairing/stratum/audit"):
        generate_benchmarks(copy,BenchmarkConfig(constructed_additions=False,roots=16,edits=2,synthetic=True))


@pytest.mark.parametrize("status",[None,1,"true"])
def test_missing_or_nonboolean_completion_rejected(tmp_path,benchmark_dir,status):
    manifest = json.loads((benchmark_dir/"manifest.json").read_text())
    if status is None: manifest.pop("challenge_complete")
    else: manifest["challenge_complete"] = status
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="challenge_complete"):
        generate_benchmarks(tmp_path,BenchmarkConfig(constructed_additions=False,roots=16,edits=2,synthetic=True))
