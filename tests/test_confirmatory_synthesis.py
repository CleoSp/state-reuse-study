"""Synthesis-script rules on synthetic fixtures; no empirical numbers here."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthesize_confirmatory as syn  # noqa: E402


def operating_point(delta, root_ci, crossed_ci, ratio, cost_ci):
    return {"suite": "synthetic", "frozen_point": {"reuse": {"policy": "answer_only", "K": 1}, "comparator": {"policy": "restart", "K": 2}},
            "accuracy": {"delta": delta, "root_ci": root_ci, "crossed_ci": crossed_ci, "per_seed_delta": [delta], "training_seed_sd": None,
                         "root_sd": 0.01, "roots": 4, "seeds": [1], "synthetic": True},
            "cost": {"ratio": ratio, "root_ci": cost_ci, "crossed_ci": cost_ci, "per_seed_ratio": [ratio], "roots": 4, "seeds": [1], "synthetic": True}}


def test_frozen_decision_rule_uses_interval_bounds_not_point_estimates():
    met = syn.decide(operating_point(-0.002, [-0.009, 0.004], [-0.012, 0.006], 0.70, [0.68, 0.74]))
    assert met["noninferior_root"] and not met["noninferior_crossed"]
    assert met["measured_saving"] and met["target_saving_25pct"] and met["claim_met"]
    lower = syn.decide(operating_point(-0.004, [-0.011, 0.003], [-0.011, 0.003], 0.70, [0.68, 0.74]))
    assert not lower["noninferior_root"] and not lower["claim_met"]
    cost = syn.decide(operating_point(0.0, [-0.001, 0.001], [-0.001, 0.001], 0.74, [0.72, 0.76]))
    assert cost["noninferior_root"] and not cost["target_saving_25pct"] and cost["measured_saving"]
    none = syn.decide(operating_point(0.0, [-0.001, 0.001], [-0.001, 0.001], 0.99, [0.98, 1.0]))
    assert not none["measured_saving"] and not none["claim_met"]


def test_chart_writers_emit_well_formed_svg(tmp_path):
    line = tmp_path / "line.svg"
    syn.line_chart(line, [{"label": "a", "color": "#000", "points": [(1, .2), (2, .4), (4, .8)]},
                          {"label": "ref", "color": "#111", "points": [(0.5, 1.0)], "marker_only": True}],
                   "synthetic title", "synthetic subtitle", "x", "y", xlog=True)
    stacked = tmp_path / "stacked.svg"
    syn.stacked_bars(stacked, [{"label": "g", "values": {"s1": 1.0, "s2": 2.5}}], ["s1", "s2"], "t", "s", "y")
    grouped = tmp_path / "grouped.svg"
    syn.grouped_bars(grouped, ["low", "high"], [{"label": "p", "color": "#222", "values": {"low": .5, "high": .25}}], "t", "s", "y")
    for path in (line, stacked, grouped):
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg") and len(list(root)) > 5
    before = line.read_bytes()
    syn.line_chart(line, [{"label": "a", "color": "#000", "points": [(1, .2), (2, .4), (4, .8)]},
                          {"label": "ref", "color": "#111", "points": [(0.5, 1.0)], "marker_only": True}],
                   "synthetic title", "synthetic subtitle", "x", "y", xlog=True)
    assert line.read_bytes() == before


def test_ticks_and_formatting_are_stable():
    assert syn._ticks(0, 1) == pytest.approx([0, .2, .4, .6, .8, 1.0])
    assert syn._fmt(0.5) == "0.5" and syn._fmt(2.0) == "2"
    assert syn.pp(0.01234) == "+1.23" and syn.pct(0.5) == "50.00"
    assert syn.ci([-0.011, 0.003]) == "[-1.10, +0.30]" and syn.ci([0.9, 1.1], 1, False) == "[0.900, 1.100]"
