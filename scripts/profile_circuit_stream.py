"""Synthetic capacity and parity for the exact five-frame circuit path."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time

import torch

import run_circuit_stream as runner
from state_repair.data.circuit import CircuitExample, generate_circuit
from state_repair.models.recursive import RecursiveSolver
from state_repair.train.stream_step import stream_backward
from state_repair.types import Domain


def main() -> None:
    config = runner.read("configs/circuit_stream_v2.json")
    out = Path(config["capacity_run"])
    torch.set_num_threads(config["threads"]); torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(73)
    roots = [CircuitExample(generate_circuit(32,i),f"synthetic-{i}",0,"train",tuple(range(32)),True) for i in range(64)]
    frames = runner.stream_frames(roots,4,config["edit_seed"])
    batch = runner.batches([[asdict(e) for e in f] for f in frames],config,4)[0]
    weights = RecursiveSolver(width=128,heads=4,inner_cycles=2,domain=Domain.CIRCUIT).state_dict()
    rows = []
    with runner.gpu_job(config,out,300,"prompt05b_circuit_stream_capacity",synthetic=True) as job:
        (out / "profile_script.py").write_bytes(Path(__file__).read_bytes())
        for arm in config["arms"]:
            model,adapter = runner.new_model(config,weights,"cpu"),runner.adapter_for(arm,config)
            if arm == "spatial_gate":
                ref,ref_adapter = deepcopy(model),deepcopy(adapter)
                cpu = stream_backward(ref,ref_adapter,[o for _,o,t in batch],[t for _,o,t in batch],1,track="joint")
            model,adapter = model.cuda(),adapter.cuda()
            data = [(o.to("cuda"),t.to("cuda")) for _,o,t in batch]
            params = [*model.parameters(),*adapter.parameters()]
            opt = torch.optim.AdamW(params,lr=0.,weight_decay=0.)
            for k in config["budgets"]:
                times = []
                for repeat in range(2):
                    job.check_limit(); opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(); started = time.perf_counter()
                    records = stream_backward(model,adapter,[o for o,t in data],[t for o,t in data],k,track="joint")
                    torch.cuda.synchronize()
                    backward_seconds = time.perf_counter()-started
                    parity_started = time.perf_counter()
                    if arm == "spatial_gate" and k == 1 and repeat == 0:
                        torch.testing.assert_close(torch.tensor([r.loss for r in cpu]),torch.tensor([r.loss for r in records]),atol=2e-5,rtol=2e-4)
                        for a,b in zip(params,[*ref.parameters(),*ref_adapter.parameters()]):
                            if a.grad is not None:
                                torch.testing.assert_close(a.grad.cpu(),b.grad,atol=3e-5,rtol=5e-3)
                    parity_seconds = time.perf_counter()-parity_started
                    started = time.perf_counter()
                    if list(adapter.parameters()) and not sum(float(p.grad.square().sum()) for p in adapter.parameters() if p.grad is not None) > 0:
                        raise ValueError("missing circuit adapter gradient")
                    torch.nn.utils.clip_grad_norm_(params,config["gradient_clip"],error_if_nonfinite=True)
                    opt.step(); torch.cuda.synchronize()
                    times.append(backward_seconds+time.perf_counter()-started)
                    print({"arm":arm,"K":k,"repeat":repeat,"training_step_seconds":times[-1],
                        "one_time_parity_check_seconds":parity_seconds},flush=True)
                rows.append({"arm":arm,"K":k,"step_seconds":times,"synthetic":True})
                runner.write(out / "steps.json",rows)
            del model,adapter,params,opt,data
            torch.cuda.empty_cache()
        train = sum(max(r["step_seconds"])*config["steps"]/len(config["budgets"])*2*len(config["seeds"]) for r in rows)
        projection = (train+1200)*1.25


        component_limit = 7200*1.25
        runner.write(out / "profile.json",{"passed":projection <= component_limit,"synthetic":True,"batch_size":64,"frames_per_step":5,
            "cpu_cuda_stream_gradient_parity":True,"rows":rows,"training_projection_seconds":train,
            "replication_projection_seconds_with_contingency":projection,
            "component_limit_seconds_with_contingency":component_limit,
            "timing_scope":"training forward/backward, gradient checks, clipping and optimizer update; one-time CPU/CUDA parity comparisons excluded from step extrapolation but included in job accounting"})
        if projection > component_limit:
            raise ValueError("circuit estimate exceeds the recorded component allowance; replan before launch")
    runner.seal(out)


if __name__ == "__main__":
    main()
