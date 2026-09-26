"""Repeat discrepant streams in original CUDA runtime and compare CPU drift."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from state_repair.accounting.gpu_job import GPUJob
from state_repair.data.maze import collate
from state_repair.models.policy import FixedBudgetPolicy
from state_repair.provenance import file_hash
from state_repair.train.pilot import decode_example
from run_adapter_pilot import json_rows, load_selected, read, seal, write


def main() -> None:
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path("runs/adapter_pilot_v1")
    config = read(root / "prepared/config.json")
    source_rows = list(json_rows(Path("runs/prompt05-pilot-cpu-replay.jsonl")))
    cases = sorted({(r["track"], r["seed"], r["arm"], r["K"]) for r in source_rows if r["action_mismatches"]})
    previous_cpu = {(r["track"], r["seed"], r["arm"], r["K"], r["frame"], r["root_id"]): r for r in source_rows}
    out = Path("runs/pilot_replay_diagnostic")
    with GPUJob(out, 600, "prompt05_original_runtime_replay_diagnostic", synthetic=False) as job:
        payload = read(root / "prepared/dataset.json")
        selection = read(root / "selection.json")
        weights = torch.load(config["source_checkpoint"], map_location="cpu", weights_only=False)["model"]
        frames = [[decode_example(e) for e in frame[:64]] for frame in payload["streams"]]
        gpu_inputs = [collate(frame)[0] for frame in frames]
        cpu_inputs = [collate(frame[:2])[0] for frame in frames]
        identities = {e.root_id for e in frames[0]}
        all_records = []
        for track, seed, arm, k in cases:
            expected = {(r["frame"], r["root_id"]): r for r in json_rows(
                root / "streams" / f"{track}-{arm}-seed{seed}" / "predictions.jsonl.gz")
                if r["K"] == k and r["root_id"] in identities}
            gpu_model, gpu_adapter, identity = load_selected(root, config, selection, weights, track, arm, seed, "cuda")
            cpu_model, cpu_adapter, _ = load_selected(root, config, selection, weights, track, arm, seed, "cpu")
            gpu_policy = FixedBudgetPolicy(gpu_model.eval(), gpu_adapter.eval(), k)
            cpu_policy = FixedBudgetPolicy(cpu_model.eval(), cpu_adapter.eval(), k)
            records = []
            with torch.no_grad():
                for frame in range(33):
                    job.check_limit()
                    gpu_result = gpu_policy(gpu_inputs[frame].cuda() if hasattr(gpu_inputs[frame], "cuda") else gpu_inputs[frame].to("cuda"))
                    cpu_result = cpu_policy(cpu_inputs[frame])
                    gpu_logits = gpu_result.prediction.logits.cpu()
                    gpu_actions, cpu_actions = gpu_logits.argmax(-1).tolist(), cpu_result.prediction.logits.argmax(-1).tolist()
                    record = {"track": track, "seed": seed, "arm": arm, "K": k, "frame": frame,
                        "checkpoint_sha256": identity["checkpoint_sha256"], "synthetic": False,
                        "cuda_replay_action_mismatches": sum(a != b for e, actions in zip(frames[frame], gpu_actions)
                            for a, b in zip(actions, expected[frame, e.root_id]["actions"])),
                        "cpu211_vs_cuda211_max_abs_logit": float((cpu_result.prediction.logits-gpu_logits[:2]).abs().max()),
                        "cpu211_vs_cuda211_action_mismatches": sum(a != b for one, two in zip(cpu_actions, gpu_actions[:2]) for a, b in zip(one, two)),
                        "cpu211_vs_cpu214_action_mismatches": sum(a != b for e, actions in zip(frames[frame][:2], cpu_actions)
                            for a, b in zip(actions, previous_cpu[track, seed, arm, k, frame, e.root_id]["cpu_actions"]))}
                    records.append(record)
                    del gpu_result, cpu_result
            all_records.extend(records)
            write(out / "records.json", all_records)
            print(json.dumps({"case": [track, seed, arm, k], "cuda_mismatches": sum(r["cuda_replay_action_mismatches"] for r in records),
                "initial_max_abs_logit": records[0]["cpu211_vs_cuda211_max_abs_logit"],
                "last_max_abs_logit": records[-1]["cpu211_vs_cuda211_max_abs_logit"]}), flush=True)
            del gpu_policy, cpu_policy, gpu_model, gpu_adapter, cpu_model, cpu_adapter
        write(out / "summary.json", {"cases": [list(c) for c in cases], "frames_per_case": 33, "gpu_batch_size": 64,
            "cpu_batch_size": 2, "cuda_predictions_replayed": len(cases)*33*64,
            "cuda_replay_action_mismatches": sum(r["cuda_replay_action_mismatches"] for r in all_records),
            "cpu211_vs_cuda211_action_mismatches": sum(r["cpu211_vs_cuda211_action_mismatches"] for r in all_records),
            "cpu211_vs_cpu214_action_mismatches": sum(r["cpu211_vs_cpu214_action_mismatches"] for r in all_records),
            "maximum_initial_logit_difference": max(r["cpu211_vs_cuda211_max_abs_logit"] for r in all_records if r["frame"] == 0),
            "maximum_final_logit_difference": max(r["cpu211_vs_cuda211_max_abs_logit"] for r in all_records if r["frame"] == 32),
            "torch": str(torch.__version__), "synthetic": False, "script_sha256": file_hash(__file__),
            "source_cpu_replay_sha256": file_hash("runs/prompt05-pilot-cpu-replay.jsonl")})
        (out / "script.py").write_bytes(Path(__file__).read_bytes())
    seal(out)


if __name__ == "__main__":
    main()
