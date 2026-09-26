"""Content interventions on a policy's own previous prediction, never labels."""
from __future__ import annotations

import torch
from state_repair.models.adapters import AnswerOnlyAdapter
from state_repair.types import PredictionBatch


class AnswerLesion(AnswerOnlyAdapter):
    def __init__(self, source: AnswerOnlyAdapter, lesion: str):
        super().__init__(source.width, source.domain)
        if lesion not in ("uniform", "shuffled_nodes"):
            raise ValueError("unknown answer-content lesion")
        self.load_state_dict(source.state_dict())
        self.lesion, self.key = lesion, "answer_"+lesion

    def initialize_from_prediction(self, transition, fresh, prediction):
        if prediction is not None:
            logits = prediction.logits.clone()
            if self.lesion == "uniform":
                logits.zero_()
            else:
                for row, valid in enumerate(fresh.valid_nodes):
                    nodes = valid.nonzero().flatten()
                    logits[row, nodes] = logits[row, nodes[torch.randperm(len(nodes), device=nodes.device)]]
            prediction = PredictionBatch(logits)
        return super().initialize_from_prediction(transition, fresh, prediction)
