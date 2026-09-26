"""Synthetic boundary/accounting fixtures; never empirical result records."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
import sqlite3

import pytest
import torch

from state_repair.accounting.ledger import AccountingLedger, RunManifest
from state_repair.cli import main
from state_repair.config import load_config
from state_repair import devices
from state_repair.types import (Domain, ObservationBatch, ObservedEdit,
                                OracleMetadata, RecurrentState, TargetBatch,
                                TrainingExample, TransitionInput)


def observation(frame=0, batch=2):
    return ObservationBatch(
        Domain.MAZE, torch.zeros(batch, 3, 4), torch.zeros(batch, 3, 3, dtype=torch.long),
        torch.tensor([[True, True, False]]).repeat(batch, 1),
        tuple(f"synthetic=true:{i}" for i in range(batch)), (frame,) * batch,
        torch.tensor([[0, 1, -1]]).repeat(batch, 1), ((1, 2),) * batch,
        torch.zeros(batch, dtype=torch.long), torch.ones(batch, dtype=torch.long),
    )


def state(obs, budget=0):
    return RecurrentState(torch.ones(*obs.valid_nodes.shape, 4, requires_grad=True),
                          torch.zeros(*obs.valid_nodes.shape, 4, requires_grad=True),
                          obs.episode_ids, obs.frame_indices,
                          obs.node_ids.clone(), obs.valid_nodes.clone(), budget)


@pytest.mark.parametrize("frame", [True, 1.5, -1, "1"])
def test_strict_frames(frame):
    with pytest.raises(ValueError, match="frame"):
        replace(observation(), frame_indices=(frame, frame))


@pytest.mark.parametrize("budget", [True, 1.5, -1])
def test_strict_state_budget(budget):
    with pytest.raises(ValueError, match="bookkeeping"):
        state(observation(), budget)


def test_edge_direction_reverse_and_permuted_padding():
    obs = observation()
    edges = obs.edge_types.clone()
    edges[:, 0, 1], edges[:, 1, 0] = 2, 4
    valid = replace(obs, edge_types=edges)
    order = torch.tensor([2, 1, 0])
    replace(valid, node_features=valid.node_features[:, order], node_ids=valid.node_ids[:, order],
            valid_nodes=valid.valid_nodes[:, order], edge_types=edges[:, order][:, :, order])
    one_way = edges.clone()
    one_way[:, 1, 0] = 0
    with pytest.raises(ValueError, match="reverse"):
        replace(obs, edge_types=one_way)
    wrong = edges.clone()
    wrong[:, 0, 1] = 3
    with pytest.raises(ValueError, match="direction"):
        replace(obs, edge_types=wrong)
    with pytest.raises(ValueError, match="feature count"):
        replace(obs, domain=Domain.CIRCUIT)


def test_partial_reset_storage_and_gradient_ownership():
    old = state(observation(), 2)
    fresh = state(replace(observation(), episode_ids=("new0", "new1")), 2)
    result = old.reset(torch.tensor([True, False]), fresh)
    assert result.episode_ids == ("new0", old.episode_ids[1])
    result.a.sum().backward()
    assert old.a.grad[0].count_nonzero() == fresh.a.grad[1].count_nonzero() == 0
    assert old.a.grad[1].sum() == fresh.a.grad[0].sum() == 12
    with torch.no_grad():
        result.a[0].fill_(42)
    assert torch.all(result.a[1] == 1) and torch.all(fresh.a == 1)
    different_budget = replace(fresh, budget=0)
    unchanged = old.reset(torch.tensor([False, False]), different_budget)
    assert unchanged.budget == 2 and unchanged.a.data_ptr() != old.a.data_ptr()
    assert old.reset(torch.tensor([True, True]), different_budget).budget == 0
    with pytest.raises(ValueError, match="mixed budgets"):
        old.reset(torch.tensor([True, False]), different_budget)
    with pytest.raises(ValueError, match="dtype"):
        old.reset(torch.tensor([True, True]), replace(fresh, a=fresh.a.double(), z=fresh.z.double()))


def test_state_metadata_finiteness_and_detach_isolation():
    original = state(observation())
    for kwargs in ({"a": original.a.double()}, {"a": torch.full_like(original.a, float("nan"))},
                   {"node_ids": torch.zeros_like(original.node_ids)}):
        with pytest.raises(ValueError):
            replace(original, **kwargs)
    for copy in (original.clone(), original.detach(), original.to("cpu")):
        copy.node_ids[0, 0] = 99
        assert original.node_ids[0, 0] == 0 and copy.node_ids[1, 0] == 0
    detached = original.detach()
    assert not detached.a.requires_grad and detached.a.grad_fn is None
    detached.a[0].fill_(3)
    assert original.a[0, 0, 0] == detached.a[1, 0, 0] == 1


def test_edit_shapes_and_transition_missing_inputs():
    old, new = observation(), observation(1)
    edit = ObservedEdit.between(old, new)
    TransitionInput(new, old, state(old), edit)
    for field in ("old", "old_state", "edit"):
        kwargs = dict(new=new, old=old, old_state=state(old), edit=edit)
        kwargs[field] = None
        with pytest.raises(ValueError, match="requires old"):
            TransitionInput(**kwargs)
    with pytest.raises(ValueError, match="bool"):
        replace(edit, edge_changed=edit.edge_changed.long())
    with pytest.raises(ValueError, match="dtype"):
        TransitionInput(new, old, state(old), replace(edit, node_features_delta=edit.node_features_delta.double()))
    with pytest.raises(TypeError):
        TransitionInput(new, old, {"state": state(old)}, edit)
    with pytest.raises(ValueError, match="mixed"):
        TransitionInput(replace(new, frame_indices=(0, 1)), old, state(old), edit)
    with pytest.raises(TypeError):
        TransitionInput(new, target=TargetBatch(torch.ones(2, 3, 6, dtype=torch.bool)))


def test_target_and_oracle_separation():
    obs = observation()
    labels = torch.zeros(2, 3, 6, dtype=torch.bool)
    labels[:, :2, 5] = True
    target = TargetBatch(labels)
    target.check_observation(obs)
    TrainingExample(TransitionInput(obs), target)
    with pytest.raises(ValueError, match="at least one"):
        TargetBatch(torch.zeros_like(labels)).check_observation(obs)
    with pytest.raises(ValueError, match="padded"):
        TargetBatch(torch.ones_like(labels)).check_observation(obs)
    with pytest.raises(ValueError, match="shape"):
        target.check_observation(observation(batch=1))
    metadata = OracleMetadata(torch.tensor([[0, -1, -2], [1, 0, -2]]), audit={"synthetic": True})
    with pytest.raises(ValueError, match="bool"):
        replace(metadata, distance_changed=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="sentinel"):
        replace(metadata, distances=torch.full((2, 3), -3, dtype=torch.long))
    with pytest.raises(TypeError):
        replace(obs, oracle=metadata)


@pytest.mark.parametrize("text", ["model: [", "seed: 1\nseed: 2", "model: {width: 16, width: 32}",
                                    "1: value", "training: null", "output_dir: ''"])
def test_malformed_configs_are_actionable(tmp_path, text, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)
    assert main(["generate", "--config", str(path)]) == 2
    assert "error:" in capsys.readouterr().err


def _reserve_in_process(args):
    path, index = args
    try:
        AccountingLedger(path, 10).reserve(f"synthetic:{index}", 7)
        return True
    except ValueError:
        return False


def test_independent_processes_share_one_allocation(tmp_path):
    path = str(tmp_path / "synthetic.sqlite")
    AccountingLedger(path, 10)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert sum(pool.map(_reserve_in_process, [(path, i) for i in range(2)])) == 1
    assert AccountingLedger(path, 10).snapshot()["reserved_usd"] == 7


def test_concurrent_reconciliation_and_cancel_release(tmp_path):
    ledger = AccountingLedger(tmp_path / "synthetic.sqlite", 10)
    ledger.reserve("synthetic:cancelled", 7)
    def reconcile(_):
        try:
            ledger.reconcile("synthetic:cancelled", 0)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reconcile, range(2))) == 1
    assert ledger.snapshot()["remaining_usd"] == 10
    with pytest.raises(ValueError):
        ledger.reconcile("unknown", 1)


@pytest.mark.parametrize("amount", [-1, True, float("nan"), float("inf"), 0.001, 1e100, "invalid"])
def test_invalid_money_is_rejected(tmp_path, amount):
    with pytest.raises(ValueError, match="USD"):
        AccountingLedger(tmp_path / "synthetic.sqlite", 10).reserve("synthetic", amount)


def test_journals_reject_mutation_and_manifest_concurrent_append(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    ledger = AccountingLedger(path, 10)
    ledger.reserve("synthetic:job", 2)
    manifest = RunManifest(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: manifest.append({"synthetic": True, "i": i}), range(16)))
    assert sorted(r["i"] for r in manifest.records()) == list(range(16))
    for table in ("events", "allocation", "manifest"):
        with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(f"DELETE FROM {table}")
    assert ledger.snapshot()["reserved_usd"] == 2
    with pytest.raises(ValueError):
        manifest.append({"synthetic": True, "loss": float("nan")})


def test_auto_backend_selection_and_failed_backend_exit(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(devices, "_nvidia_inventory", lambda: {"devices": []})
    calls = []
    def probe(device):
        calls.append(device)
        return {"synthetic": True, "status": "failed" if device.startswith("cuda") else "passed",
                "forward_backward": not device.startswith("cuda")}
    monkeypatch.setattr(devices, "_probe", probe)
    assert devices.doctor("cpu")["backends"]["cuda"]["reason"] == "not requested"
    assert calls == ["cpu"]
    assert main(["doctor", "--device", "auto"]) == 2
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["selected_device"] == "cuda" and report["backends"]["cuda"]["status"] == "failed"
    assert report["backends"]["mps"]["status"] == "not tested"
    assert "diagnostic failed" in captured.err


@pytest.mark.parametrize("command", ["evaluate", "report"])
def test_empty_existing_run_cannot_succeed(tmp_path, capsys, command):
    args = [command, "--run", str(tmp_path)]
    if command == "evaluate":
        args += ["--suite", "smoke"]
    assert main(args) == 2
    assert "error:" in capsys.readouterr().err and not list(tmp_path.iterdir())


def test_training_missing_dataset_creates_nothing(tmp_path, capsys):
    output = tmp_path / "absent-run"
    config = tmp_path / "config.yaml"
    config.write_text(f"output_dir: {output.as_posix()}\n")
    assert main(["train", "--config", str(config)]) == 2
    assert "dataset is missing" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("workflow", ["repair", "confirmatory"])
def test_draft_workflows_never_create_artifacts_or_reserve(tmp_path, monkeypatch, capsys, workflow):
    from state_repair.data.serialization import generate_dataset
    from state_repair.train.loop import train_static
    config_file = tmp_path / "draft.yaml"
    output = tmp_path / "output"
    config_file.write_text(f"workflow: {workflow}\noutput_dir: {output.as_posix()}\n")
    config = load_config(config_file)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported workflow attempted a reservation")
    monkeypatch.setattr(AccountingLedger, "reserve", forbidden)
    for function in (generate_dataset, train_static):
        with pytest.raises(NotImplementedError, match="not implemented"):
            function(config)
    for command in ("generate", "train"):
        assert main([command, "--config", str(config_file)]) == 2
        assert "not implemented" in capsys.readouterr().err
    assert not output.exists()


def test_workflow_defaults_errors_and_static_dispatch(tmp_path, monkeypatch, capsys):
    from state_repair.data import serialization
    config_file = tmp_path / "static.yaml"
    config_file.write_text("schema_version: 1\n")
    assert load_config(config_file).workflow == "static"
    calls = []
    def generate(config):
        calls.append(config.workflow)
        return tmp_path / "synthetic-dispatch-only"
    monkeypatch.setattr(serialization, "generate_dataset", generate)
    assert main(["generate", "--config", str(config_file)]) == 0
    assert calls == ["static"]
    capsys.readouterr()
    config_file.write_text("workflow: typo\n")
    with pytest.raises(ValueError, match="workflow"):
        load_config(config_file)


@pytest.mark.parametrize("filename,workflow", [("smoke", "static"), ("static_pilot", "static"),
                                               ("repair_pilot", "repair"), ("confirmatory_draft", "confirmatory")])
def test_versioned_repository_configs(filename, workflow):
    assert load_config(f"configs/{filename}.yaml").workflow == workflow
