"""Observed Boolean DAGs. No evaluated internal values enter observation()."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import hashlib
import json
import random
from typing import Any, Sequence

import torch
from torch import Tensor

from state_repair.types import Domain, ObservationBatch, TargetBatch


class Operator(IntEnum):
    INPUT = 0
    AND = 1
    OR = 2
    XOR = 3
    NOT = 4


@dataclass(frozen=True)
class Circuit:
    operators: tuple[Operator, ...]
    parents: tuple[tuple[int, ...], ...]
    input_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        if not all(isinstance(v, tuple) for v in (self.operators, self.parents, self.input_bits)):
            raise ValueError("circuit fields must be immutable tuples")
        if not self.n or len(self.parents) != self.n or len(self.input_bits) != self.n:
            raise ValueError("circuit fields need matching nonempty lengths")
        if any(not isinstance(op, Operator) for op in self.operators):
            raise ValueError("operators must be Operator enum values")
        if all(op == Operator.INPUT for op in self.operators):
            raise ValueError("circuit needs at least one scored gate")
        for u, (op, parents, bit) in enumerate(zip(self.operators, self.parents, self.input_bits)):
            if not isinstance(parents, tuple) or any(type(p) is not int or not 0 <= p < self.n or p == u for p in parents):
                raise ValueError("parents must be valid distinct non-self node identities")
            if len(parents) != (0,2,2,2,1)[op] or len(set(parents)) != len(parents):
                raise ValueError("invalid gate arity or duplicate parents")
            if type(bit) is not int or bit not in (0,1) or (op != Operator.INPUT and bit != 0):
                raise ValueError("only external inputs may have observed binary bits")
        remaining = set(range(self.n))
        while remaining:
            ready = {u for u in remaining if not remaining.intersection(self.parents[u])}
            if not ready:
                raise ValueError("circuit contains a cycle")
            remaining -= ready

    @property
    def n(self) -> int:
        return len(self.operators)

    def flip_input(self, node: int) -> Circuit:
        if type(node) is not int or not 0 <= node < self.n or self.operators[node] != Operator.INPUT:
            raise ValueError("input flip requires an input node")
        bits = list(self.input_bits)
        bits[node] ^= 1
        return replace(self, input_bits=tuple(bits))

    def substitute(self, node: int, operator: Operator) -> Circuit:
        if type(node) is not int or not 0 <= node < self.n or self.operators[node] not in (Operator.AND, Operator.OR, Operator.XOR):
            raise ValueError("substitution requires a binary gate")
        if not isinstance(operator, Operator) or operator not in (Operator.AND, Operator.OR, Operator.XOR) or operator == self.operators[node]:
            raise ValueError("substitution requires a different binary operator")
        operators = list(self.operators)
        operators[node] = operator
        return replace(self, operators=tuple(operators))

    def relabel(self, new_to_old: Sequence[int]) -> Circuit:
        """Change identities jointly; apply the same mapping to every frame."""
        order = list(new_to_old)
        if any(type(v) is not int for v in order) or sorted(order) != list(range(self.n)):
            raise ValueError("relabeling must be a permutation")
        inverse = {old: new for new, old in enumerate(order)}
        return Circuit(tuple(self.operators[u] for u in order),
                       tuple(tuple(sorted(inverse[p] for p in self.parents[u])) for u in order),
                       tuple(self.input_bits[u] for u in order))


@dataclass(frozen=True)
class CircuitExample:
    circuit: Circuit
    root_id: str
    frame_index: int
    split: str
    node_order: tuple[int, ...]
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, Circuit) or not isinstance(self.root_id, str) or not self.root_id:
            raise ValueError("example needs circuit and nonempty root")
        if type(self.frame_index) is not int or self.frame_index < 0 or self.split not in ("train", "val", "test"):
            raise ValueError("invalid frame or split")
        if type(self.synthetic) is not bool or not isinstance(self.node_order, tuple) or any(type(v) is not int for v in self.node_order) or sorted(self.node_order) != list(range(self.circuit.n)):
            raise ValueError("require Boolean synthetic flag and presentation permutation")


@dataclass(frozen=True)
class CircuitOracleMetadata:
    """Bool[B,N] masks in presentation order, false at padding/frame zero.

    Descendants exclude the edited node itself; changed_value_mask includes any
    changed external input as well as internal gates. Targets score only gates.
    """
    descendant_mask: Tensor
    changed_value_mask: Tensor
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        a, b = self.descendant_mask, self.changed_value_mask
        if not isinstance(a, Tensor) or not isinstance(b, Tensor) or a.ndim != 2 or a.dtype != torch.bool or b.dtype != torch.bool or a.shape != b.shape or a.device != b.device:
            raise ValueError("circuit oracle masks require matching Bool[B,N] tensors")
        if min(a.shape) < 1 or not isinstance(self.audit, dict):
            raise ValueError("circuit oracle metadata requires nonempty masks and audit mapping")


def generate_circuit(nodes: int, seed: int, inputs: int | None = None) -> Circuit:
    if type(nodes) is not int or nodes < 3:
        raise ValueError("circuit generator requires at least three nodes")
    inputs = max(2, nodes // 4) if inputs is None else inputs
    if type(inputs) is not int or not 2 <= inputs < nodes:
        raise ValueError("require 2 <= inputs < nodes")
    rng = random.Random(seed)
    ops = [Operator.INPUT] * inputs
    parents: list[tuple[int, ...]] = [()] * inputs
    bits = [rng.randrange(2) for _ in range(inputs)]
    for u in range(inputs, nodes):
        op = rng.choice([Operator.AND, Operator.OR, Operator.XOR, Operator.NOT])
        ops.append(op)
        parents.append(tuple(sorted(rng.sample(range(u), 1 if op == Operator.NOT else 2))))
        bits.append(0)
    order = list(range(nodes))
    rng.shuffle(order)
    return Circuit(tuple(ops), tuple(parents), tuple(bits)).relabel(order)


def generate_episode(circuit: Circuit, root_id: str, split: str, edits: int, seed: int,
                     *, synthetic: bool = False) -> list[CircuitExample]:
    if type(edits) is not int or edits < 0:
        raise ValueError("edit count must be a nonnegative integer")
    rng = random.Random(seed)
    order = list(range(circuit.n))
    rng.shuffle(order)
    examples = [CircuitExample(circuit, root_id, 0, split, tuple(order), synthetic)]
    for frame in range(1, edits + 1):
        inputs = [u for u, op in enumerate(circuit.operators) if op == Operator.INPUT]
        binary = [u for u, op in enumerate(circuit.operators) if op in (Operator.AND, Operator.OR, Operator.XOR)]
        if not binary or rng.randrange(2) == 0:
            circuit = circuit.flip_input(rng.choice(inputs))
        else:
            node = rng.choice(binary)
            circuit = circuit.substitute(node, rng.choice([op for op in (Operator.AND, Operator.OR, Operator.XOR) if op != circuit.operators[node]]))
        examples.append(CircuitExample(circuit, root_id, frame, split, tuple(order), synthetic))
    return examples


def observation(circuit: Circuit, episode_id: str = "observed", frame_index: int = 0,
                node_order: Sequence[int] | None = None) -> ObservationBatch:
    order = list(range(circuit.n)) if node_order is None else list(node_order)
    if any(type(v) is not int for v in order) or sorted(order) != list(range(circuit.n)):
        raise ValueError("node_order must be a permutation")
    inverse = {u: i for i, u in enumerate(order)}
    features = torch.zeros((1, circuit.n, 6))
    relations = torch.zeros((1, circuit.n, circuit.n), dtype=torch.long)
    for pos, u in enumerate(order):
        features[0, pos, circuit.operators[u]] = 1
        features[0, pos, 5] = circuit.input_bits[u]
        for parent in circuit.parents[u]:
            relations[0, inverse[parent], pos] = 1
            relations[0, pos, inverse[parent]] = 2
    return ObservationBatch(Domain.CIRCUIT, features, relations,
                            torch.ones((1, circuit.n), dtype=torch.bool), (episode_id,), (frame_index,),
                            torch.tensor([order]))


def collate(examples: Sequence[CircuitExample], device: str | torch.device = "cpu",
            *, previous: Sequence[Circuit | None] | None = None) -> tuple[ObservationBatch, TargetBatch, CircuitOracleMetadata]:
    from state_repair.oracles.circuit import evaluate, impact_metadata
    if not examples:
        raise ValueError("cannot collate empty circuits")
    if previous is None:
        if any(e.frame_index != 0 for e in examples):
            raise ValueError("edited circuit metadata requires previous observations")
        previous = [None] * len(examples)
    if len(previous) != len(examples):
        raise ValueError("previous circuits batch mismatch")
    b, n = len(examples), max(e.circuit.n for e in examples)
    x = torch.zeros((b,n,6)); edges = torch.zeros((b,n,n), dtype=torch.long)
    valid = torch.zeros((b,n), dtype=torch.bool); ids = torch.full((b,n), -1, dtype=torch.long)
    values = torch.full((b,n), -1, dtype=torch.long); scored = torch.zeros((b,n), dtype=torch.bool)
    descendants = torch.zeros((b,n), dtype=torch.bool); changed = torch.zeros((b,n), dtype=torch.bool)
    for i, (example, old) in enumerate(zip(examples, previous)):
        if (example.frame_index == 0) != (old is None):
            raise ValueError("frame-zero requires no previous circuit; edited frames require one")
        circuit, order = example.circuit, list(example.node_order)
        obs = observation(circuit, example.root_id, example.frame_index, order)
        count = circuit.n
        x[i,:count] = obs.node_features[0]; edges[i,:count,:count] = obs.edge_types[0]
        valid[i,:count] = True; ids[i,:count] = obs.node_ids[0]
        answer = evaluate(circuit)
        for pos,u in enumerate(order):
            if circuit.operators[u] != Operator.INPUT:
                scored[i,pos] = True; values[i,pos] = answer[u]
        if old is not None:
            meta = impact_metadata(old, circuit)
            descendants[i,:count] = torch.tensor(meta["descendant_mask"])[order]
            changed[i,:count] = torch.tensor(meta["changed_value_mask"])[order]
    obs = ObservationBatch(Domain.CIRCUIT, x, edges, valid, tuple(e.root_id for e in examples),
                           tuple(e.frame_index for e in examples), ids).to(device)
    target = TargetBatch(values=values, scored_mask=scored).to(device)
    target.check_observation(obs)
    return obs, target, CircuitOracleMetadata(descendants.to(device), changed.to(device),
        {"synthetic": [e.synthetic for e in examples], "splits": [e.split for e in examples]})


def topology_fingerprint(circuit: Circuit) -> str:
    """Conservative directed color-refinement invariant, not exact isomorphism.

    Ignores input bits and binary operator kind; retains INPUT/unary/binary type,
    edge direction and multiplicity. Color refinement can conflate nonisomorphic
    graphs, causing visible false-positive rejection, never silent resampling.
    """
    children = [[] for _ in range(circuit.n)]
    for u, parents in enumerate(circuit.parents):
        for p in parents:
            children[p].append(u)
    colors = [str(len(p)) for p in circuit.parents]
    for _ in range(circuit.n):
        descriptions = [(colors[u], sorted(colors[p] for p in circuit.parents[u]), sorted(colors[v] for v in children[u])) for u in range(circuit.n)]
        colors = [hashlib.sha256(json.dumps(d, separators=(",", ":")).encode()).hexdigest() for d in descriptions]
    return hashlib.sha256(json.dumps(sorted(colors)).encode()).hexdigest()


def reject_duplicate_roots(examples: Sequence[CircuitExample]) -> None:
    roots: dict[str, tuple[str, str]] = {}
    hashes: dict[str, str] = {}
    for example in examples:
        key = topology_fingerprint(example.circuit)
        identity = (example.split, key)
        if example.root_id in roots and roots[example.root_id] != identity:
            raise ValueError("circuit root crosses splits or changes topology")
        if key in hashes and hashes[key] != example.root_id:
            raise ValueError("circuit topology fingerprint collision across independent roots; no resampling")
        roots[example.root_id] = identity
        hashes[key] = example.root_id
