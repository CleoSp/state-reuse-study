"""Read-only model evaluation plus explicitly labeled input lesions on training roots."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from diagnose_static import DATA, ROOT, fixtures, metrics, write
from state_repair.accounting.resources import memory_snapshot
from state_repair.data.maze import collate
from state_repair.data.serialization import load_split
from state_repair.eval.smoke import evaluate_examples
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.losses import maze_valid_set_loss


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    start = time.perf_counter()
    out = ROOT / args.run
    destination = out / "inspection.json"
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(out / "checkpoint.pt", weights_only=True)
    config = checkpoint["config"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = RecursiveSolver(width=config["width"], heads=2, inner_cycles=1, attention_mode="dense")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    examples = fixtures(config["count"])
    obs, target, _ = collate(examples)
    rows = {}
    with torch.no_grad():
        for name, altered in [
            ("intact", obs),
            ("no_edges", replace(obs, edge_types=torch.zeros_like(obs.edge_types))),
            ("no_goal_flag", replace(obs, node_features=obs.node_features * torch.tensor([1., 1., 1., 0.]))),
        ]:
            logits = model(altered, config["depth"]).prediction.logits
            actions = logits.argmax(-1)
            valid = target.valid_actions.gather(-1, actions[..., None]).squeeze(-1)
            rows[name] = {"record_kind": "input_lesion_diagnostic" if name != "intact" else "static_overfit_diagnostic",
                "loss": maze_valid_set_loss(logits, target, obs.valid_nodes).item(), **metrics(examples, logits),
                "goal_action_counts": dict(Counter(actions[torch.arange(len(examples)), obs.goals].tolist())),
                "invalid_special_actions": int(((actions >= 4) & ~valid).sum()),
                "wrong_nodes": int((~valid).sum())}

    validation = [e for e in load_split(DATA, "val") if e.frame_index == 0]
    val_obs, val_targets, _ = collate(validation)
    with torch.no_grad():
        val_logits = model(val_obs, config["depth"]).prediction.logits
        validation_loss = maze_valid_set_loss(val_logits, val_targets, val_obs.valid_nodes).item()
    records = evaluate_examples(model, validation, sorted(set([1, 2, config["depth"], 4, 8])))
    write(destination, {"input_lesions": rows, "validation_loss_at_training_depth": validation_loss,
                        "validation_evaluation": records, "wall_s": time.perf_counter() - start,
                        "memory": memory_snapshot(), "synthetic": False})
    print(json.dumps({"file": str(destination), "wall_s": time.perf_counter() - start,
                      "lesions": {k: {m: v[m] for m in ("loss", "acceptable_action_accuracy", "complete_route_success", "goal_action_counts", "invalid_special_actions")} for k, v in rows.items()},
                      "validation_loss": validation_loss}), flush=True)


if __name__ == "__main__":
    main()
