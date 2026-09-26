"""Exact classical Boolean evaluators, strictly outside learned inference."""
from __future__ import annotations

from collections import deque
import heapq
from typing import Sequence

from state_repair.data.circuit import Circuit, Operator


def _gate(circuit: Circuit, node: int, values: Sequence[int]) -> int:
    op, parents = circuit.operators[node], circuit.parents[node]
    if op == Operator.INPUT:
        return circuit.input_bits[node]
    if op == Operator.NOT:
        return 1 - values[parents[0]]
    a, b = (values[p] for p in parents)
    if op == Operator.AND:
        return a & b
    if op == Operator.OR:
        return a | b
    return a ^ b


def _topology(circuit: Circuit) -> tuple[list[int], list[list[int]]]:
    children: list[list[int]] = [[] for _ in range(circuit.n)]
    pending = [len(p) for p in circuit.parents]
    for u, parents in enumerate(circuit.parents):
        for p in parents:
            children[p].append(u)
    queue = deque(u for u in range(circuit.n) if pending[u] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in children[u]:
            pending[v] -= 1
            if pending[v] == 0:
                queue.append(v)
    if len(order) != circuit.n:
        raise ValueError("circuit contains a cycle")
    return order, children


def evaluate(circuit: Circuit) -> list[int]:
    order, _ = _topology(circuit)
    values = [0] * circuit.n
    for u in order:
        values[u] = _gate(circuit, u, values)
    return values


def recursive_evaluate(circuit: Circuit) -> list[int]:
    """Independent memoized recursion; no topology, gate or adjacency helper."""
    answers: dict[int, int] = {}
    active: set[int] = set()

    def visit(u: int) -> int:
        if u in answers:
            return answers[u]
        if u in active:
            raise ValueError("cycle in recursive evaluation")
        active.add(u)
        args = [visit(p) for p in circuit.parents[u]]
        name = circuit.operators[u].name
        if name == "INPUT":
            answer = circuit.input_bits[u]
        elif name == "AND":
            answer = int(sum(args) == 2)
        elif name == "OR":
            answer = int(sum(args) > 0)
        elif name == "XOR":
            answer = sum(args) % 2
        else:
            answer = int(args[0] == 0)
        active.remove(u)
        answers[u] = answer
        return answer

    return [visit(u) for u in range(circuit.n)]


def edited_node(old: Circuit, new: Circuit) -> int:
    if old.n != new.n or old.parents != new.parents:
        raise ValueError("circuit updates require stable identities and wiring")
    changed = [u for u in range(old.n) if (old.operators[u], old.input_bits[u]) != (new.operators[u], new.input_bits[u])]
    if len(changed) != 1:
        raise ValueError("circuit update requires exactly one input flip or binary substitution")
    u = changed[0]
    if old.operators[u] == new.operators[u] == Operator.INPUT:
        return u
    if old.operators[u] in (Operator.AND, Operator.OR, Operator.XOR) and new.operators[u] in (Operator.AND, Operator.OR, Operator.XOR):
        return u
    raise ValueError("circuit edit must preserve arity")


def descendant_mask(circuit: Circuit, node: int) -> list[bool]:
    if type(node) is not int or not 0 <= node < circuit.n:
        raise ValueError("descendant root outside circuit")
    _, children = _topology(circuit)
    mask = [False] * circuit.n
    pending = list(children[node])
    while pending:
        u = pending.pop()
        if not mask[u]:
            mask[u] = True
            pending.extend(children[u])
    return mask


def impact_metadata(old: Circuit, new: Circuit) -> dict[str, list[bool]]:
    node = edited_node(old, new)
    return {"descendant_mask": descendant_mask(new, node),
            "changed_value_mask": [a != b for a, b in zip(evaluate(old), evaluate(new))]}


class EventDrivenEvaluator:
    """Classical reference owning exact values. Never a deployable neural adapter.

    Initial full evaluation and topology construction are required setup cost.
    Every update validates/scans observed descriptions to find the edited node.
    Gate counters alone are not total latency: include validation, scans, queue,
    cache copying and initial solve when timing this reference.
    """
    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        order, self._children = _topology(circuit)
        self._rank = {u: i for i, u in enumerate(order)}
        self._values = evaluate(circuit)
        self.initial_gate_evaluations = sum(op != Operator.INPUT for op in circuit.operators)
        self.last_gate_evaluations = 0
        self.last_input_updates = 0

    @property
    def values(self) -> list[int]:
        return self._values.copy()

    def update(self, new: Circuit) -> list[int]:
        edited = edited_node(self.circuit, new)
        queue = [(self._rank[edited], edited)]
        queued = {edited}
        self.last_gate_evaluations = 0
        self.last_input_updates = 0
        while queue:
            _, u = heapq.heappop(queue)
            queued.remove(u)
            answer = _gate(new, u, self._values)
            self.last_gate_evaluations += int(new.operators[u] != Operator.INPUT)
            self.last_input_updates += int(new.operators[u] == Operator.INPUT)
            if answer != self._values[u]:
                self._values[u] = answer
                for v in self._children[u]:
                    if v not in queued:
                        heapq.heappush(queue, (self._rank[v], v))
                        queued.add(v)
        self.circuit = new
        return self.values


def score_circuit(circuit: Circuit, predictions: Sequence[int]) -> dict[str, bool | float | int]:
    """Full-gate correctness excludes observed input copies from every metric."""
    if len(predictions) != circuit.n or any(type(v) is not int or v not in (0,1) for v in predictions):
        raise ValueError("circuit predictions must be N binary integers in stable identity order")
    values = evaluate(circuit)
    correct = [predictions[u] == values[u] for u in range(circuit.n) if circuit.operators[u] != Operator.INPUT]
    if not correct:
        raise ValueError("cannot score a circuit without non-input gates")
    return {"circuit_correct": all(correct), "node_accuracy": sum(correct) / len(correct), "scored_nodes": len(correct)}
