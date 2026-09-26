"""The added paired contrasts must preserve root/seed alignment."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_stream_addenda import paired


def records(delta=0.):
    return [{"seed": s, "root_id": f"synthetic-{i}", "post32": (i % 7)/10+delta,
             "synthetic": True} for s in (29,43,71) for i in range(256)]


def test_paired_contrast_ignores_record_order_and_retains_episode_difference():
    config = {"seeds": [29,43,71], "bootstrap_seed": 23, "bootstrap_repetitions": 100}
    result = paired(records(.125), list(reversed(records())), config)
    assert result["delta"] == pytest.approx(.125)
    assert result["ci"] == pytest.approx([.125,.125])
    assert result["per_seed_delta"] == pytest.approx([.125]*3)


def test_paired_contrast_rejects_a_missing_root_in_any_seed():
    config = {"seeds": [29,43,71], "bootstrap_seed": 23, "bootstrap_repetitions": 100}
    with pytest.raises(ValueError, match="unpaired"):
        paired(records(), records()[:-1], config)
