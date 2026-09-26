"""Observation-only frozen-state interventions, deliberately separate from streams."""
from __future__ import annotations

from dataclasses import dataclass

from state_repair.models.adapters import AdapterResult, StateAdapter, make_adapter
from state_repair.models.recursive import RecursiveSolver, SolverResult
from state_repair.types import ObservationBatch, ObservedEdit, PredictionBatch, RecurrentState, TransitionInput


def frozen_prediction(solver: RecursiveSolver, old: ObservationBatch,
                      new: ObservationBatch, prior: RecurrentState,
                      policy: str, k: int) -> SolverResult:
    """Fork model-produced history; no labels or evaluator metadata are accepted.

    This helper preserves autograd through the current rollout but detaches the
    previous frame before initialization. Evaluation callers disable gradients.
    Source budget is intentionally independent of k, unlike a deployed stream.
    """
    if policy not in ("restart", "carry"):
        raise ValueError("frozen crossover accepts only restart/carry")
    solver._check_state(old, prior)
    transition = TransitionInput(new, old, prior.detach(), ObservedEdit.between(old, new))
    initialized = make_adapter(policy)(transition, solver.fresh_state(new)).state
    return solver(new, k, initialized)


@dataclass(frozen=True)
class FrozenAdapterPrediction:
    solver: SolverResult
    adapter: AdapterResult
    source_budget: int
    record_kind: str = "frozen_state_intervention"


def frozen_adapter_prediction(solver: RecursiveSolver, adapter: StateAdapter,
                              old: ObservationBatch, new: ObservationBatch,
                              prior: RecurrentState, k: int, *,
                              previous_prediction: PredictionBatch | None = None) -> FrozenAdapterPrediction:
    """Fork one saved model-produced state through a loaded deployable adapter.

    Callers verify the frozen checkpoint/state provenance and supply the same
    prior to compared arms. This function never accepts targets or impact masks.
    Its deliberately different source/current budgets cannot enter a stream
    through this API; stream evaluation must use FixedBudgetPolicy instead.
    """
    if not isinstance(adapter, StateAdapter):
        raise TypeError("frozen intervention requires a loaded StateAdapter")
    if type(k) is not int or k < 0:
        raise ValueError("frozen intervention requires nonnegative integer K")
    if not isinstance(old, ObservationBatch) or not isinstance(new, ObservationBatch):
        raise TypeError("frozen intervention accepts only typed observations")
    solver._check_state(old, prior)
    transition = TransitionInput(new, old, prior.detach(), ObservedEdit.between(old, new))
    fresh = solver.fresh_state(new)
    if adapter.requires_previous_prediction:
        if not isinstance(previous_prediction, PredictionBatch):
            raise TypeError("output-only intervention requires the previous model prediction")
        prediction = PredictionBatch(previous_prediction.logits.detach().clone())
        adapted = adapter(transition, fresh, previous_prediction=prediction)
    else:
        if previous_prediction is not None:
            raise ValueError("this adapter does not consume previous predictions")
        adapted = adapter(transition, fresh)
    return FrozenAdapterPrediction(solver(new, k, adapted.state), adapted, prior.budget)
