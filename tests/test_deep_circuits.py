"""Synthetic tests for observed circuit-depth shift; no training evidence."""
from state_repair.data.circuit import Operator, observation
from state_repair.data.deep_circuits import generate_deep_circuit
from state_repair.oracles.circuit import evaluate, recursive_evaluate


def test_deep_circuit_has_declared_depth_without_identity_order():
    for seed in range(6):
        circuit = generate_deep_circuit(48, seed, inputs=8)
        depths = {}
        def depth(u):
            if u not in depths:
                depths[u] = 0 if not circuit.parents[u] else 1+max(depth(p) for p in circuit.parents[u])
            return depths[u]
        assert max(depth(u) for u in range(circuit.n)) == 40
        assert sum(op == Operator.INPUT for op in circuit.operators) == 8
        assert any(parent > u for u in range(circuit.n) for parent in circuit.parents[u])
        assert evaluate(circuit) == recursive_evaluate(circuit)
        assert circuit == generate_deep_circuit(48, seed, inputs=8)
        observed = observation(circuit, "synthetic")
        assert observed.node_features.shape[-1] == 6
        assert all(observed.node_features[0, u, 5].item() == 0 for u in range(circuit.n) if circuit.operators[u] != Operator.INPUT)


def test_static_circuit_score_respects_order_and_excludes_input_copies(monkeypatch):
    from pathlib import Path
    from state_repair.data.circuit import CircuitExample, generate_circuit
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from train_static_circuit import score
    circuit = generate_circuit(12, 73)
    order = tuple(reversed(range(circuit.n)))
    example = CircuitExample(circuit, "synthetic", 0, "val", order, True)
    predictions = [evaluate(circuit)[u] for u in order]
    for pos, u in enumerate(order):
        if circuit.operators[u] == Operator.INPUT:
            predictions[pos] ^= 1
    assert score(example, predictions)["exact_correct"]
    assert score(example, predictions)["node_accuracy"] == 1
    for pos, u in enumerate(order):
        if circuit.operators[u] != Operator.INPUT:
            predictions[pos] ^= 1
            break
    assert not score(example, predictions)["exact_correct"]
    assert score(example, predictions)["node_accuracy"] == 1-1/score(example, predictions)["scored_nodes"]


def test_circuit_static_generation_and_gate(monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from train_static_circuit import gate, graph_depth, make_data
    config = json.loads(Path("configs/static_circuit_v1.json").read_text())
    config.update(train_roots=8, val_roots=2, synthetic=True)
    data = make_data(config)
    assert data == make_data(config)
    assert len(data["train"]) == 8 and len(data["ordinary"]) == len(data["depth48"]) == 2
    assert all(e.synthetic for examples in data.values() for e in examples)
    assert all(graph_depth(e.circuit) == 40 for e in data["depth48"])
    rows = [{"suite": "ordinary", "K": k, "exact_accuracy": a} for k, a in zip(config["budgets"], [.1,.3,.7,.85,.9])]
    assert gate(rows, 6000, config)["backbone_eligible"]
    rows[-1]["exact_accuracy"] = .84
    assert not gate(rows, 6000, config)["backbone_eligible"]
    rows[-1]["exact_accuracy"] = .9
    assert not gate(rows, 5999, config)["backbone_eligible"]
