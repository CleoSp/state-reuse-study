"""Production job construction. Dependencies are sealed before consumption."""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from state_repair.models.adapters import make_adapter
from state_repair.models.recursive import RecursiveSolver
from state_repair.types import Domain
from .datasets import collate, decode, frames
from .durable import read_json
from .research_training import ResearchTrainer


def environment(job: dict) -> None:
    torch.set_num_threads(job.get("threads", 2))
    seed = job.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if job["device"] == "cuda":
        if job.get("required_torch_version") and torch.__version__ != job["required_torch_version"]:
            raise ValueError("CUDA runtime differs from frozen PyTorch version")
        if not torch.cuda.is_available() or "RTX 5070" not in torch.cuda.get_device_name(0):
            raise ValueError("production CUDA requires the authorized local RTX 5070")
        torch.cuda.set_per_process_memory_fraction(10 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()


def model_for(job: dict) -> RecursiveSolver:
    return RecursiveSolver(width=job["width"], heads=job["heads"], inner_cycles=job["inner_cycles"],
        attention_mode="masked_neighbor", domain=Domain(job["family"])).to(job["device"])


def adapter_for(arm: str, job: dict, spatial=None):
    if arm.startswith("local_reset_"):
        return make_adapter("local_reset", radius=int(arm[-1]))
    if arm == "random_reset":
        return make_adapter(arm, reset_rate=job.get("reset_rate", .5))
    if arm == "noisy_carry":
        return make_adapter(arm, noise_scale=.01)
    if arm == "shuffled_gate":
        return make_adapter(arm, spatial_gate=spatial)
    kwargs = {} if arm in ("restart", "carry") else {"width": job["width"], "domain": Domain(job["family"])}
    if arm not in ("restart", "carry", "answer_only"):
        kwargs["context_width"] = job.get("context_width", 16)
    return make_adapter(arm, **kwargs)


def load_weights(model, adapter, root: Path, source: str, *, backbone_only: bool = False) -> None:
    payload = torch.load(root / source / "checkpoint.pt", map_location="cpu", weights_only=False)
    weights = payload["model"]
    model.load_state_dict({k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")})
    if adapter is not None and not backbone_only:
        adapter.load_state_dict({k.removeprefix("adapter."): v for k, v in weights.items() if k.startswith("adapter.")})


def build_training(job: dict, root: Path):
    payload = read_json(Path(job["dataset"]))
    roots = [decode(e) for e in payload["train"]]
    if not roots or any(e.split != "train" or e.synthetic != job["synthetic"] for e in roots):
        raise ValueError("training accepts only correctly labeled training roots")
    length = {"static": 0, "stream": 4, "one_edit": 1}[job["recipe"]]
    stream = frames(roots, length, job["edit_seed"]) if length else [roots]
    if job["recipe"] == "one_edit" and job["family"] == "maze":
        from state_repair.train.pilot import training_views
        views = training_views(roots, {**job, "training_views": 4})
        batches = [[collate(roots[i:i+job["batch_size"]])[:2], collate(view[i:i+job["batch_size"]])[:2]]
                   for view in views for i in range(0, len(roots), job["batch_size"])]
    else:
        batches = []
        for offset in range(0, len(roots), job["batch_size"]):
            batches.append([collate(f[offset:offset+job["batch_size"]],
                None if index == 0 else stream[index-1][offset:offset+job["batch_size"]])[:2]
                for index, f in enumerate(stream)])
    model = model_for(job)
    adapter = None if job["recipe"] == "static" else adapter_for(job["arm"], job).to(job["device"])
    if job.get("source"):
        load_weights(model, adapter, root, job["source"], backbone_only=True)
    return ResearchTrainer(model, adapter, batches, job)


def build(job: dict, root: Path):
    environment(job)
    if job["kind"] == "research_training":
        return build_training(job, root)
    from state_repair.eval.jobs import build_evaluation_job
    return build_evaluation_job(job, root)
