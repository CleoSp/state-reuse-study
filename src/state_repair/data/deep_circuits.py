"""Observed chain-depth shift for a separately declared circuit validation suite."""
from __future__ import annotations

import random

from state_repair.data.circuit import Circuit, Operator


def generate_deep_circuit(nodes: int, seed: int, inputs: int = 8) -> Circuit:
    """Guarantee a gate chain using wiring alone, then randomly relabel nodes.

    Every gate after the first has the preceding gate as one parent. Binary
    gates have one further distinct earlier parent; operator and input values
    are unconditioned random draws. Graph depth is nodes-inputs, regardless of
    whether those structural dependencies transmit a particular input change.
    No truth values or oracle-selected rejection is used in construction.
    """
    if type(nodes) is not int or type(inputs) is not int or not 2 <= inputs < nodes:
        raise ValueError("require integers 2 <= inputs < nodes")
    rng = random.Random(seed)
    operators = [Operator.INPUT]*inputs
    parents: list[tuple[int, ...]] = [()]*inputs
    bits = [rng.randrange(2) for _ in range(inputs)]
    for u in range(inputs, nodes):
        op = rng.choice((Operator.AND, Operator.OR, Operator.XOR, Operator.NOT))
        first = u-1 if u > inputs else rng.randrange(inputs)
        if op == Operator.NOT:
            sources = (first,)
        else:
            candidates = [v for v in range(u) if v != first]
            sources = tuple(sorted((first, rng.choice(candidates))))
        operators.append(op)
        parents.append(sources)
        bits.append(0)
    order = list(range(nodes))
    rng.shuffle(order)
    return Circuit(tuple(operators), tuple(parents), tuple(bits)).relabel(order)
