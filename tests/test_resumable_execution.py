"""Real killed CPU subprocesses and fault injection; every fixture is synthetic."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from state_repair.execution.budget import AUTHORIZED_SECONDS, Budget
from state_repair.execution.driver import recover_output, run, validate_matrix, verify
from state_repair.execution.durable import DriverLock, append_jsonl, atomic_json, atomic_write, read_json, repair_jsonl
from state_repair.execution.training import load_resume

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_confirmatory.py"


def configuration(tmp_path: Path, *, steps: int = 35, delay: float = .015) -> tuple[Path, dict]:
    job = {"id": "fixture", "kind": "synthetic_training", "synthetic": True, "device": "cpu",
           "steps": steps, "step_delay": delay, "seed": 19, "seconds": 90,
           "checkpoint_steps": 5, "checkpoint_seconds": 2}
    config = tmp_path / "config.json"
    atomic_json(config, {"schema": 1, "authorization_gpu_hours": 150, "jobs": [job]})
    return config, job


def launch(config: Path, output: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(SCRIPT.parent.parent / "src")}
    return subprocess.Popen([sys.executable, str(SCRIPT), "--config", str(config), "--output", str(output)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def wait_step(process: subprocess.Popen, root: Path, step: int) -> None:
    deadline = time.monotonic() + 30
    path = root / "fixture" / "steps.jsonl"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.communicate())
        if path.exists() and len(path.read_bytes().splitlines()) >= step:
            return
        time.sleep(.01)
    process.kill()
    raise AssertionError("subprocess did not reach the requested step")


def finish(process: subprocess.Popen) -> None:
    stdout, stderr = process.communicate(timeout=40)
    assert process.returncode == 0, (stdout, stderr)


def assert_tensor_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tensor_tree_equal(a, b)
    else:
        assert left == right


def test_kill_resume_exact_and_noop(tmp_path):
    config, job = configuration(tmp_path, steps=60, delay=.025)
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    finish(launch(config, baseline))
    child = launch(config, resumed)
    try:
        wait_step(child, resumed, 13)
        child.kill()
        child.communicate(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    with (resumed / "fixture" / "steps.jsonl").open("ab") as handle:
        handle.write(b'{"partial":')
    finish(launch(config, resumed))
    first = load_resume(baseline / "fixture" / "checkpoint.pt", job)
    second = load_resume(resumed / "fixture" / "checkpoint.pt", job)
    for key in ("model", "optimizer", "trainer", "schedule", "remaining_batch_order", "forward_calls", "step"):
        assert_tensor_tree_equal(first[key], second[key])
    assert (baseline / "fixture" / "steps.jsonl").read_bytes() == (resumed / "fixture" / "steps.jsonl").read_bytes()
    assert read_json(baseline / "fixture" / "summary.json")["final_loss"] == read_json(resumed / "fixture" / "summary.json")["final_loss"]
    checked = verify(resumed / "fixture", job)
    assert checked["verified"] and checked["resume_events"][0]["step"] > 0
    quarantines = list(resumed.glob("fixture.incomplete-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "steps.jsonl").read_bytes().endswith(b'{"partial":')
    events = Budget(resumed).events()
    actual = [r for r in events if r["kind"] == "actual"]
    assert len(actual) == 2 and actual[0]["aborted"] is True
    assert actual[0]["wall_seconds"] < 90
    assert Budget(resumed).ledger.snapshot()["reserved_usd"] == 0
    before = (resumed / "fixture" / "seal.json").read_bytes()
    finish(launch(config, resumed))
    assert before == (resumed / "fixture" / "seal.json").read_bytes()
    assert events == Budget(resumed).events()


def test_graceful_stop_checkpoints_and_resumes(tmp_path):
    config, job = configuration(tmp_path, steps=40, delay=.025)
    output = tmp_path / "run"
    child = launch(config, output)
    wait_step(child, output, 8)
    (output / "STOP").touch()
    stdout, stderr = child.communicate(timeout=20)
    assert child.returncode != 0 and "shutdown" in stderr
    checkpoint = load_resume(output / "fixture" / "resume.pt", job)
    assert checkpoint["step"] == len((output / "fixture" / "steps.jsonl").read_text().splitlines())
    (output / "STOP").unlink()
    finish(launch(config, output))
    assert verify(output / "fixture", job)["resume_events"]


@pytest.mark.parametrize("fault", ["receipt", "ledger", "journal", "actual_receipt", "actual_ledger", "actual_journal"])
def test_reconcile_each_two_store_crash_window(tmp_path, fault):
    budget = Budget(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    receipt = {"job_id": "p06-fixture", "matrix_job": "fixture", "started_unix": time.time()-4,
               "output": str(output), "reserved_seconds": 60, "segment_cap_seconds": 60,
               "synthetic": True, "device": "cuda"}
    atomic_json(budget.receipts / "p06-fixture.json", receipt)
    if fault != "receipt":
        budget.ledger.reserve("p06-fixture", 0, "prompt06_local")
    if fault not in ("receipt", "ledger"):
        append_jsonl(budget.journal, {**receipt, "kind": "reserve", "seconds": 60})
    if fault.startswith("actual"):
        result = {**receipt, "wall_seconds": 3, "aborted": False, "time_basis": "monotonic", "error": None}
        atomic_json(budget.receipts / "p06-fixture.actual.json", result)
        if fault in ("actual_ledger", "actual_journal"):
            budget.ledger.reconcile("p06-fixture", 0)
        if fault == "actual_journal":
            append_jsonl(budget.journal, {**result, "kind": "actual", "seconds": 3})
    budget.reconcile()
    budget.reconcile()
    assert len(budget.events()) == 2
    actual = budget.events()[-1]
    assert actual["kind"] == "actual" and actual["seconds"] < 60
    assert actual["aborted"] is (not fault.startswith("actual"))
    assert budget.ledger.snapshot()["reserved_usd"] == 0


def test_unsealed_without_checkpoint_is_preserved_and_rerun(tmp_path):
    config, job = configuration(tmp_path, steps=2, delay=0)
    output = tmp_path / "run"
    (output / "fixture").mkdir(parents=True)
    (output / "fixture" / "evidence.txt").write_text("preserve me")
    run(config, output)
    assert next(output.glob("fixture.incomplete-*/evidence.txt")).read_text() == "preserve me"
    assert verify(output / "fixture", job)["verified"]


def test_two_drivers_cannot_hold_lock(tmp_path):
    config, _ = configuration(tmp_path)
    root = tmp_path / "run"
    with DriverLock(root / "driver.lock"):
        process = launch(config, root)
        _, stderr = process.communicate(timeout=20)
        assert process.returncode != 0 and "another driver" in stderr
    assert not (root / "attempts").exists()


def test_atomic_failed_write_preserves_previous_file(tmp_path):
    path = tmp_path / "summary.json"
    atomic_json(path, {"synthetic": True, "complete": False})
    before = path.read_bytes()

    def fail(handle):
        handle.write(b"half")
        raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError):
        atomic_write(path, fail)
    assert path.read_bytes() == before


def test_partial_line_is_preserved_in_audit_and_full_corruption_rejected(tmp_path):
    path, audit = tmp_path / "rows.jsonl", tmp_path / "audit.jsonl"
    path.write_bytes(b'{"synthetic":true}\n{"torn"')
    repair_jsonl(path, audit)
    assert path.read_bytes() == b'{"synthetic":true}\n'
    assert json.loads(audit.read_text())["bytes_hex"] == b'{"torn"'.hex()
    path.write_bytes(b'bad full line\n')
    with pytest.raises(ValueError):
        repair_jsonl(path, audit)


def test_caps_count_interrupted_attempts(tmp_path):
    budget = Budget(tmp_path)
    job = {"id": "fixture", "seconds": 5, "device": "cuda", "synthetic": True}
    receipt = budget.begin(job, tmp_path / "out")
    budget.finish(receipt, 3, aborted=True)
    second = budget.begin(job, tmp_path / "out")
    assert second["segment_cap_seconds"] == 2
    budget.finish(second, 2, aborted=False)
    with pytest.raises(TimeoutError):
        budget.begin(job, tmp_path / "out")
    assert AUTHORIZED_SECONDS == 500 * 3600
    with pytest.raises(ValueError, match="four hours"):
        budget.begin({**job, "seconds": 14401}, tmp_path / "out")


def test_empirical_and_cloud_dispatch_fail_closed(tmp_path):
    config, job = configuration(tmp_path)
    value = read_json(config)
    value["jobs"][0]["synthetic"] = False
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_matrix(value, None)
    value["jobs"][0]["device"] = "cloud"
    with pytest.raises(ValueError, match="only local"):
        validate_matrix(value, None)


def test_checker_rejects_changed_sealed_artifact(tmp_path):
    config, job = configuration(tmp_path, steps=2, delay=0)
    output = tmp_path / "run"
    run(config, output)
    (output / "fixture" / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        run(config, output)
    assert not list(output.glob("fixture.incomplete-*"))


def test_checkpoint_fallback_after_interrupted_copy(tmp_path):
    config, job = configuration(tmp_path, steps=2, delay=0)
    output = tmp_path / "run"
    run(config, output)
    original = output / "fixture"
    original.rename(output / "fixture.incomplete-100")
    original.mkdir()
    (original / "resume.pt").write_bytes(b"broken checkpoint")
    recover_output(original, job, output / "recovery.jsonl")
    assert load_resume(original / "resume.pt", job)["step"] == 2
    assert len(list(output.glob("fixture.incomplete-*"))) == 2


def test_unknown_accounting_reservation_fails_closed(tmp_path):
    budget = Budget(tmp_path)
    budget.ledger.reserve("unknown", 0)
    with pytest.raises(ValueError, match="without write-ahead receipt"):
        budget.reconcile()


def test_startup_task_template_requires_boot_password_and_no_desktop():
    import xml.etree.ElementTree as ET
    path = SCRIPT.parent.parent / "reports/foundation/confirmatory-startup.requested.xml"
    tree = ET.fromstring(path.read_text(encoding="utf-16"))
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    assert tree.find("t:Triggers/t:BootTrigger", ns) is not None
    assert tree.find("t:Principals/t:Principal/t:LogonType", ns).text == "Password"
    assert tree.find("t:Settings/t:ExecutionTimeLimit", ns).text == "PT0S"
    args = tree.find("t:Actions/t:Exec/t:Arguments", ns).text
    assert "-NonInteractive" in args and "-WindowStyle Hidden" in args


@pytest.mark.parametrize("family", ["maze", "circuit"])
@pytest.mark.parametrize("recipe", ["static", "stream", "one_edit"])
def test_real_solver_and_adapter_resume(tmp_path, family, recipe):
    from dataclasses import replace
    from state_repair.execution.research_training import ResearchTrainer
    from state_repair.execution.training import Shutdown, run_training
    from state_repair.models.adapters import make_adapter
    from state_repair.models.recursive import RecursiveSolver
    from state_repair.types import Domain

    config = {"synthetic": True, "recipe": recipe, "seed": 31, "steps": 6,
              "budgets": [1, 2], "learning_rate": .001, "final_learning_rate": .0003,
              "adapter_learning_rate": .001, "weight_decay": 0, "gradient_clip": 1}
    frame_count = {"static": 1, "stream": 5, "one_edit": 2}[recipe]
    if family == "maze":
        from state_repair.data.maze import Maze, MazeExample, collate
        maze = Maze(2, 2, ((0, 1), (0, 2), (1, 3)), 0, 3)
        frames = [collate([MazeExample(maze if f % 2 == 0 else maze.toggle(2, 3),
                                      "synthetic-resume", f, "train", True)])[:2] for f in range(frame_count)]
    else:
        from state_repair.data.circuit import CircuitExample, collate, generate_circuit, generate_episode
        circuit = generate_circuit(8, 11)
        episodes = generate_episode(circuit, "synthetic-resume", "train", frame_count-1, 17, synthetic=True)
        frames = [collate([replace(e, node_order=tuple(range(8)))],
                          previous=None if f == 0 else [episodes[f-1].circuit])[:2]
                  for f, e in enumerate(episodes)]
    def create():
        torch.manual_seed(31)
        solver = RecursiveSolver(width=8, heads=2, inner_cycles=1, domain=Domain(family))
        adapter = None if recipe == "static" else make_adapter("spatial_gate", width=8, context_width=4, domain=Domain(family))
        return ResearchTrainer(solver, adapter, [frames], config)
    baseline, resumed = tmp_path / "baseline" / "job", tmp_path / "resumed" / "job"
    baseline.mkdir(parents=True)
    resumed.mkdir(parents=True)
    with Shutdown() as shutdown:
        run_training(create(), config, baseline, 60, shutdown, checkpoint_steps=2)
    interrupted = create()
    original_step = interrupted.step
    with Shutdown() as shutdown:
        def stop_after_step(index, batch, k):
            row = original_step(index, batch, k)
            if index == 2:
                shutdown.requested.set()
            return row
        interrupted.step = stop_after_step
        with pytest.raises(InterruptedError):
            run_training(interrupted, config, resumed, 60, shutdown, checkpoint_steps=2)
    recover_output(resumed, config, tmp_path / "audit.jsonl")
    with Shutdown() as shutdown:
        run_training(create(), config, resumed, 60, shutdown, checkpoint_steps=2)
    before, after = load_resume(baseline / "resume.pt", config), load_resume(resumed / "resume.pt", config)
    for key in ("model", "optimizer", "trainer", "forward_calls", "schedule"):
        assert_tensor_tree_equal(before[key], after[key])
    assert (baseline / "steps.jsonl").read_bytes() == (resumed / "steps.jsonl").read_bytes()


def test_boot_checker_synthetic_event_fixture(tmp_path):
    from datetime import datetime, timezone
    from state_repair.execution.boot import check_boot
    config, job = configuration(tmp_path, steps=2, delay=0)
    root = tmp_path / "run"
    run(config, root)
    (root / "fixture").rename(root / "fixture.incomplete-100")
    boot = time.time()
    append_jsonl(root / "startup.jsonl", {"event": "startup_wrapper", "synthetic": True,
                                         "boot_utc": datetime.fromtimestamp(boot, timezone.utc).isoformat()})
    run(config, root)
    audit = {"query_succeeded": True, "source": "System/Microsoft-Windows-Winlogon/7001",
             "boot_unix": boot, "queried_through_unix": time.time(), "interactive_logon_unix": []}
    result = check_boot(root, audit)
    assert result["verified"] and result["synthetic"]
    assert result["boot_to_completion_seconds"] >= result["boot_to_first_driver_log_seconds"] >= 0
    with pytest.raises(ValueError, match="logged on"):
        check_boot(root, {**audit, "interactive_logon_unix": [boot]})
    with pytest.raises(ValueError, match="audit ends"):
        check_boot(root, {**audit, "queried_through_unix": boot})
    with pytest.raises(ValueError, match="audit required"):
        check_boot(root, {**audit, "query_succeeded": False})
    log = root / "driver.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines()]
    for row in records:
        if row["event"] == "startup":
            row["windows_session_id"] = 0
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    observed = check_boot(root, {**audit, "interactive_logon_unix": [boot]}, observe_automatic_logon=True)
    assert observed["noninteractive_resume_verified"] and not observed["verified"]
    assert not observed["strict_no_logon_passed"]


def test_boot_fixture_holds_until_new_boot_and_resumes_exactly(tmp_path, monkeypatch):
    from state_repair.execution.training import Shutdown, SyntheticTrainer, run_training
    config, job = configuration(tmp_path, steps=8, delay=0)
    job["hold_before_step"] = 3
    monkeypatch.setenv("PAPER1_BOOT_ID", "synthetic-boot-1")
    held = SyntheticTrainer(job)
    assert held.ready(2) and not held.ready(3)
    original = held.step
    output = tmp_path / "job"
    output.mkdir()
    with Shutdown() as shutdown:
        def stop(index, batch, k):
            row = original(index, batch, k)
            if index == 2:
                shutdown.requested.set()
            return row
        held.step = stop
        with pytest.raises(InterruptedError):
            run_training(held, job, output, 30, shutdown)
    checkpoint = load_resume(output / "resume.pt", job)
    assert checkpoint["step"] == 3 and checkpoint["trainer"]["initial_boot_id"] == "synthetic-boot-1"
    recover_output(output, job, tmp_path / "audit.jsonl")
    monkeypatch.setenv("PAPER1_BOOT_ID", "synthetic-boot-2")
    with Shutdown() as shutdown:
        run_training(SyntheticTrainer(job), job, output, 30, shutdown)
    assert load_resume(output / "resume.pt", job)["step"] == 8


def test_preserve_lower_matrix_cap_and_reject_unapproved_extension(tmp_path):
    with pytest.raises(ValueError, match="500"):
        Budget(tmp_path, 501 * 3600)
    budget = Budget(tmp_path, 150 * 3600)
    assert budget.authorized_seconds == 150 * 3600
    config, _ = configuration(tmp_path)
    matrix = read_json(config)
    matrix["authorization_gpu_hours"] = 501
    with pytest.raises(ValueError, match="authorized"):
        validate_matrix(matrix, None)
