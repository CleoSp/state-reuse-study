from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from state_repair.accounting.ledger import AccountingLedger, RunManifest
from state_repair.config import load_config
from state_repair.devices import doctor


def test_config_and_errors(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model.width == 64
    assert cfg.training.eval_budgets == (0, 1, 2, 4)
    assert cfg.digest() == load_config("configs/smoke.yaml").digest()
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent")
    for text in ("bogus: 1", "model: {width: 63}", "training: {steps: true}", "seed: -1", "training: {learning_rate: .nan}", "training: {eval_budgets: [1, 1]}", "schema_version: true"):
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)


def test_ledger_reserve_reconcile(tmp_path):
    ledger = AccountingLedger(tmp_path / "cost.sqlite", 10)
    ledger.reserve("job", 7)
    assert ledger.snapshot()["remaining_usd"] == 3
    with pytest.raises(ValueError):
        ledger.reserve("other", 4)
    with pytest.raises(ValueError):
        ledger.reserve("job", 0)
    ledger.reconcile("job", 5)
    assert ledger.snapshot() == {"allocated_usd": 10, "actual_usd": 5, "reserved_usd": 0, "remaining_usd": 5}
    with pytest.raises(ValueError):
        ledger.reconcile("job", 5)
    with pytest.raises(ValueError):
        AccountingLedger(tmp_path / "cost.sqlite", 11)


def test_concurrent_reservation_cannot_double_spend(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    AccountingLedger(path, 10)
    def reserve(index):
        try:
            AccountingLedger(path, 10).reserve(str(index), 7)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(reserve, range(2))) == 1
    assert AccountingLedger(path, 10).snapshot()["reserved_usd"] == 7


def test_overrun_is_recorded(tmp_path):
    ledger = AccountingLedger(tmp_path / "cost.sqlite", 10)
    ledger.reserve("failed", 10)
    assert ledger.reconcile("failed", 11)["remaining_usd"] == -1
    with pytest.raises(ValueError):
        ledger.reserve("later", 0)


def test_manifest_requires_synthetic_marker_and_preserves_events(tmp_path):
    manifest = RunManifest(tmp_path / "manifest.sqlite")
    with pytest.raises(ValueError):
        manifest.append({"loss": 1})
    manifest.append({"synthetic": True, "event": "fixture"})
    path = tmp_path / "manifest.jsonl"
    manifest.export_jsonl(path)
    assert json.loads(path.read_text())["synthetic"] is True
    with pytest.raises(FileExistsError):
        manifest.export_jsonl(path)


def test_cpu_doctor(monkeypatch):
    import torch
    result = doctor("cpu")
    assert result["backends"]["cpu"]["forward_backward"]
    assert result["backends"]["cuda"]["status"] == "not tested"
    assert result["backends"]["mps"]["status"] == "not tested"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="not tested"):
        doctor("cuda")
