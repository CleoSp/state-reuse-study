"""Synthetic diagnosis of repeated validated transfers versus immutable GPU batches."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.circuit import CircuitExample, collate, generate_circuit
from state_repair.models.recursive import RecursiveSolver
from state_repair.provenance import file_hash
from state_repair.train.losses import circuit_value_loss
from state_repair.types import Domain
from train_scaled_maze import write


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path("runs/circuit_transfer_profile_v2")
    with GPUJob(out, 300, "prompt05_circuit_transfer_diagnosis", synthetic=True):
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        examples = [CircuitExample(generate_circuit(32, i, 8), f"synthetic-{i}", 0,
                                   "train", tuple(range(32)), True) for i in range(128)]
        cpu_obs, cpu_target, _ = collate(examples)
        transfer_seconds = []
        for _ in range(3):
            torch.cuda.synchronize()
            started = time.perf_counter()
            cached_obs, cached_target = cpu_obs.to("cuda"), cpu_target.to("cuda")
            cached_target.check_observation(cached_obs)
            torch.cuda.synchronize()
            transfer_seconds.append(time.perf_counter()-started)
        for name in ("node_features", "edge_types", "valid_nodes", "node_ids"):
            torch.testing.assert_close(getattr(cached_obs, name).cpu(), getattr(cpu_obs, name), atol=0, rtol=0)
        torch.testing.assert_close(cached_target.values.cpu(), cpu_target.values, atol=0, rtol=0)
        before = cached_obs.node_features.clone(), cached_obs.edge_types.clone(), cached_target.values.clone()
        reference = RecursiveSolver(width=128, heads=4, inner_cycles=2, domain=Domain.CIRCUIT)
        initial = reference.state_dict()
        outputs, states, timings = {}, {}, {}
        for mode in ("transfer_each_step", "preloaded_batch"):
            model = RecursiveSolver(width=128, heads=4, inner_cycles=2, domain=Domain.CIRCUIT).cuda()
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
            outputs[mode], timings[mode] = [], []
            for k in (1, 2, 4):
                torch.cuda.synchronize()
                started = time.perf_counter()
                obs, target = ((cpu_obs.to("cuda"), cpu_target.to("cuda")) if mode == "transfer_each_step"
                               else (cached_obs, cached_target))
                target.check_observation(obs)
                optimizer.zero_grad(set_to_none=True)
                result = model(obs, k)
                outputs[mode].append(result.prediction.logits.detach().cpu())
                loss = circuit_value_loss(result.prediction.logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                timings[mode].append(time.perf_counter()-started)
                del result, loss
            states[mode] = {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}
            del optimizer, model
        torch.save({"synthetic": True, "predictions": outputs, "parameters": states}, out / "parity.pt")
        write(out / "timings.json", {"synthetic": True, "transfer_seconds": transfer_seconds, "step_seconds": timings})
        for got, want in zip(outputs["transfer_each_step"], outputs["preloaded_batch"]):
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
            assert torch.equal(got.argmax(-1), want.argmax(-1))
        for name in states["transfer_each_step"]:
            torch.testing.assert_close(states["transfer_each_step"][name], states["preloaded_batch"][name], atol=1e-5, rtol=1e-5)
        assert torch.equal(before[0], cached_obs.node_features) and torch.equal(before[1], cached_obs.edge_types)
        assert torch.equal(before[2], cached_target.values)
        record = {"synthetic": True, "batch_size": 128, "dtype": "float32", "device": "cuda",
                  "transfer_and_validation_seconds": transfer_seconds, "step_seconds": timings,
                  "prediction_and_parameter_tolerance_parity": True, "atol": 1e-5, "rtol": 1e-5,
                  "predicted_labels_identical": True, "cached_inputs_unchanged": True,
                  "max_prediction_absolute_difference": max(float((a-b).abs().max()) for a, b in zip(outputs["transfer_each_step"], outputs["preloaded_batch"])),
                  "max_parameter_absolute_difference": max(float((states["transfer_each_step"][n]-states["preloaded_batch"][n]).abs().max()) for n in states["transfer_each_step"]),
                  "script_sha256": file_hash(__file__)}
        write(out / "profile.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
