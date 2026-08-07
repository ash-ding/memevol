"""Dataset registry — maps a dataset name to its (workflow, env module,
recorder class). THE single registry, shared by common.evaluate (the unified
evaluate_memo), every baseline (via the baselines/registry.py re-export), and
forge's in-container launch.py. Lives in datasets/ because it is pure
dataset-layer wiring (imports only datasets/*), importable from common/ without
violating the "common never imports a method" dependency direction.

Add a new benchmark by adding one line here.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict, List, Tuple

from benchmarks.dynamicmem import env as dm_env
from benchmarks.dynamicmem.env import DynamicMemRecorder
from benchmarks.dynamicmem.workflow import DynamicMemWorkflow
from benchmarks.locomo import env as locomo_env
from benchmarks.locomo.env import LoCoMoRecorder
from benchmarks.locomo.workflow import LoCoMoWorkflow
from benchmarks.longmemeval import env as lme_env
from benchmarks.longmemeval.env import LongMemEvalRecorder
from benchmarks.longmemeval.workflow import LongMemEvalSWorkflow, LongMemEvalMWorkflow

# dataset → (workflow_cls, env_module, recorder_cls).
# env_module must expose get_task_list(status, eval_n_samples).
REGISTRY: Dict[str, Tuple[type, ModuleType, type]] = {
    "dynamicmem":    (DynamicMemWorkflow,   dm_env,     DynamicMemRecorder),
    "locomo":        (LoCoMoWorkflow,       locomo_env, LoCoMoRecorder),
    "longmemeval_s": (LongMemEvalSWorkflow, lme_env,    LongMemEvalRecorder),
    "longmemeval_m": (LongMemEvalMWorkflow, lme_env,    LongMemEvalRecorder),
}

DATASETS: List[str] = sorted(REGISTRY)


def resolve(dataset: str) -> Tuple[type, ModuleType, type]:
    """Return (workflow_cls, env_module, recorder_cls) for `dataset`."""
    if dataset not in REGISTRY:
        raise ValueError(
            f"unknown dataset {dataset!r}; supported datasets: {DATASETS}."
        )
    return REGISTRY[dataset]
