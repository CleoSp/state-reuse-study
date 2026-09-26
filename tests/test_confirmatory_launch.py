"""Failure-path and end-to-end synthetic checks for the launch contract."""
from pathlib import Path
import json

import pytest
import torch

from state_repair.execution.durable import atomic_json, atomic_write, digest_json
from state_repair.eval.statistics import paired_cost_ratio, select_operating_points


def test_cost_ratio_keeps_crossed_pairs():
    b = [{"seed": s, "root_id": str(r), "amortized_ms": r+1., "synthetic": True} for s in range(3) for r in range(4)]
    a = [{**r, "amortized_ms": .7*r["amortized_ms"]} for r in b]
    result = paired_cost_ratio(a, b, repetitions=50, empirical=False)
    assert result["ratio"] == pytest.approx(.7)
    assert result["crossed_ci"] == pytest.approx([.7, .7])
    with pytest.raises(ValueError, match="same roots"):
        paired_cost_ratio(a[:-1], b, empirical=False)
    with pytest.raises(ValueError, match="positive"):
        paired_cost_ratio([{**r, "amortized_ms": 0} for r in a], b, empirical=False)


def test_operating_points_reject_test_and_preserve_negative_result():
    rows = [{"policy": p, "K": k, "post_accuracy": accuracy, "amortized_ms": cost, "split": "val", "synthetic": False}
            for p,k,accuracy,cost in [("restart",1,.9,2), ("restart",2,.95,4), ("answer_only",1,.93,1), ("answer_only",2,.92,2)]]
    result = select_operating_points(rows)
    assert result["reuse"]["K"] == 1 and not result["validation_noninferiority_feasible"]
    assert result["comparator"]["K"] == 2
    with pytest.raises(ValueError, match="validation"):
        select_operating_points([{**r,"split":"test"} for r in rows])


def test_production_resume_exact_step_records(tmp_path):
    from state_repair.execution.datasets import identities, prepare_development
    from state_repair.execution.driver import recover_output
    from state_repair.execution.jobs import build
    from state_repair.execution.training import Shutdown, run_training, load_resume
    data = tmp_path / "data.json"
    prepare_development({"family":"maze","size":3,"data_seed":19,
        "identities":{s:identities("fixture",s,2) for s in ("train","val")}}, data, synthetic=True)
    job = {"id":"stream","kind":"research_training","recipe":"stream","arm":"spatial_gate","family":"maze","size":3,
        "width":8,"heads":2,"inner_cycles":1,"context_width":4,"dataset":str(data),"device":"cpu","synthetic":True,
        "seed":19,"steps":4,"batch_size":2,"budgets":[1,2],"learning_rate":.001,"adapter_learning_rate":.001,
        "weight_decay":0.,"gradient_clip":1.,"edit_seed":17}
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full.mkdir(); resumed.mkdir()
    with Shutdown() as shutdown:
        run_training(build(job,tmp_path),job,full,60,shutdown,checkpoint_steps=1)
    (tmp_path / "STOP").touch()
    with Shutdown() as shutdown, pytest.raises(InterruptedError):
        run_training(build(job,tmp_path),job,resumed,60,shutdown,checkpoint_steps=1)
    (tmp_path / "STOP").unlink()
    recover_output(resumed,job,tmp_path / "audit.jsonl")
    with Shutdown() as shutdown:
        run_training(build(job,tmp_path),job,resumed,60,shutdown,checkpoint_steps=1)
    assert (full / "steps.jsonl").read_bytes() == (resumed / "steps.jsonl").read_bytes()
    a,b=(load_resume(p / "resume.pt",job) for p in (full,resumed))
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])


def test_dataset_binding_rejects_mutation(tmp_path):
    from state_repair.execution.datasets import verify_job_dataset
    from state_repair.provenance import file_hash
    path = tmp_path / "development.json"
    atomic_json(path,{"train":[],"val":[]})
    job = {"dataset":path.as_posix(),"kind":"research_training"}
    matrix = {"development_hashes":{path.as_posix():file_hash(path)}}
    verify_job_dataset(job,matrix,tmp_path)
    atomic_json(path,{"train":[1],"val":[]})
    with pytest.raises(ValueError,match="content changed"):
        verify_job_dataset(job,matrix,tmp_path)


def test_binding_requires_committed_source_and_exact_matrix(tmp_path, monkeypatch):
    from state_repair.execution import binding
    from state_repair.provenance import file_hash
    protocol=tmp_path / "FROZEN_PROTOCOL.md"
    source=tmp_path / "driver.py"
    source.write_text("# synthetic binding fixture\n")
    boot=tmp_path / "boot.json"
    atomic_json(boot,{"noninteractive_resume_verified":True,"strict_no_logon_passed":False})
    matrix={"launch_authorized":True,"status":"FROZEN","source_hashes":{"driver.py":file_hash(source)},"startup_acceptance":{"path":"boot.json","sha256":file_hash(boot),
        "user_accepted_session0_exception":True},"jobs":[{"id":"fixture","kind":"selection","synthetic":False,"device":"cpu"}],
        "authorization_gpu_hours":500,"projection":{"total_hours_with_interruptions":1}}
    protocol.write_text("<!-- matrix-sha256: "+digest_json(matrix)+" -->\n")
    committed={"FROZEN_PROTOCOL.md":protocol.read_bytes(),"driver.py":source.read_bytes()}
    def git(root,*args):
        return str(tmp_path).encode() if args[0]=="rev-parse" else committed[args[1].removeprefix("HEAD:")]
    monkeypatch.setattr(binding,"git_bytes",git)
    binding.validate_binding(matrix,protocol)
    source.write_text("# changed\n")
    with pytest.raises(ValueError,match="implementation changed"):
        binding.validate_binding(matrix,protocol)
    source.write_bytes(committed["driver.py"])
    matrix["projection"]["total_hours_with_interruptions"]=501
    with pytest.raises(ValueError,match="matrix hash"):
        binding.validate_binding(matrix,protocol)


def test_report_outputs_from_saved_records(tmp_path):
    from test_confirmatory_package import execute
    from state_repair.execution.datasets import prepare_development, identities
    from state_repair.eval.report import materialize_report
    data=tmp_path / "data.json"
    prepare_development({"family":"maze","size":3,"data_seed":17,
        "identities":{s:identities("report",s,2) for s in ("train","val")}},data,synthetic=True)
    base={"family":"maze","size":3,"width":8,"heads":2,"inner_cycles":1,"context_width":4,"device":"cpu","synthetic":True,
        "dataset":str(data),"seed":19,"batch_size":2,"steps":2,"edit_seed":17,"budgets":[1,2],"learning_rate":.001,
        "adapter_learning_rate":.001,"final_learning_rate":.001,"weight_decay":0.,"gradient_clip":1.,"suite":"fixture"}
    execute({**base,"id":"static","kind":"research_training","recipe":"static"},tmp_path)
    evaluations=[]
    for arm in ("restart","carry"):
        job={**base,"id":arm,"kind":"evaluation","source":"static","arm":arm,"split":"val","edits":2,"stream_seed":17}
        execute(job,tmp_path); evaluations.append(arm)
    job={"id":"report","kind":"report","device":"cpu","synthetic":True,"evaluations":evaluations,"bootstrap_repetitions":50}
    out=execute(job,tmp_path)
    materialize_report(out)
    assert (out / "accuracy-cost.svg").stat().st_size > 100
    result=json.loads((out / "results.json").read_text())
    assert len(result["curves"])==4 and len(result["contrasts"])==2
    assert result["checks"][0]["raw_actions_rescored"]==12


def test_generate_once_requires_commit_and_never_replaces_claim(tmp_path, monkeypatch):
    from state_repair.execution import binding, datasets
    from state_repair.provenance import file_hash
    spec={"family":"circuit","size":8,"inputs":2,"data_seed":891,"stream_seed":17,
        "identities":{s:datasets.identities("fixture",s,2) for s in ("train","val","test")}}
    spec["identity_hashes"]={s:digest_json(v) for s,v in spec["identities"].items()}
    matrix={"synthetic":True,"suites":{"fixture":spec}}
    data=tmp_path / "data"; data.mkdir()
    datasets.prepare_development(spec,data / "fixture-development.json",synthetic=True)
    protocol=tmp_path / "protocol.md"; protocol.write_text("synthetic fixture")
    def reject(*args): raise ValueError("uncommitted protocol")
    monkeypatch.setattr(binding,"validate_binding",reject)
    with pytest.raises(ValueError,match="uncommitted"):
        datasets.prepare_test_once(matrix,protocol,data)
    assert not (data / "fixture-test-claim.json").exists()
    monkeypatch.setattr(binding,"validate_binding",lambda *args:None)
    result=datasets.prepare_test_once(matrix,protocol,data)
    digest=file_hash(data / "fixture-test.json")
    assert result["complete"] and result["suites"]["fixture"]["actual_intervention_branches"]==2
    def no_regeneration(*args,**kwargs): raise AssertionError("regenerated test roots")
    monkeypatch.setattr(datasets,"generate_roots",no_regeneration)
    datasets.prepare_test_once(matrix,protocol,data)
    assert file_hash(data / "fixture-test.json")==digest
    (data / "test-manifest.json").unlink()
    with pytest.raises(ValueError,match="already claimed"):
        datasets.prepare_test_once(matrix,protocol,data)


def test_recovery_after_lossless_compression_before_seal(tmp_path):
    from state_repair.execution.records import compress_steps, step_bytes
    from state_repair.execution.driver import recover_output
    from state_repair.execution.training import SyntheticTrainer, run_training, Shutdown
    job={"id":"fixture","kind":"synthetic_training","device":"cpu","synthetic":True,"steps":3,"seed":17}
    out=tmp_path / "fixture"; out.mkdir()
    with Shutdown() as shutdown:
        run_training(SyntheticTrainer(job),job,out,60,shutdown)
    raw=step_bytes(out)
    compress_steps(out)
    assert not (out / "steps.jsonl").exists() and step_bytes(out)==raw
    recover_output(out,job,tmp_path / "audit.jsonl")
    assert (out / "steps.jsonl").read_bytes()==raw
    with Shutdown() as shutdown:
        result=run_training(SyntheticTrainer(job),job,out,60,shutdown)
    assert result["steps"]==3 and step_bytes(out)==raw
