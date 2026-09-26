import pytest

from state_repair.cli import main


def test_missing_run_does_not_create_artifacts(tmp_path,capsys):
    path=tmp_path/"absent"
    assert main(["evaluate","--run",str(path),"--suite","smoke"])==2
    assert "does not exist" in capsys.readouterr().err
    assert not path.exists()


@pytest.mark.parametrize("cap",["21","0","-1","nan","inf"])
def test_smoke_rejects_unapproved_runtime(cap,capsys):
    assert main(["train","--config","configs/smoke.yaml","--max-minutes",cap])==2
    assert "<= 20" in capsys.readouterr().err


def test_accelerator_is_not_falsely_passed(capsys, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["doctor","--device","cuda"])==2
    assert "not tested" in capsys.readouterr().err
