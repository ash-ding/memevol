"""Dataset registry for ALMA — maps a dataset name to its workflow, env
module, and recorder class. Mirrors forge/launch.py::WORKFLOWS (which ALMA
must NOT import — baselines are standalone), extended with the recorder class
because ALMA's code-generation prompt introspects it via get_metadata_dict.

ALMA targets exactly ONE dataset per run (selected with --dataset); this is
not multi-dataset-at-once. Add a new benchmark by adding one line here + one
block in dataset_info.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from datasets.dynamicmem import env as dm_env
from datasets.dynamicmem.env import DynamicMemRecorder
from datasets.dynamicmem.workflow import DynamicMemWorkflow
from datasets.locomo import env as locomo_env
from datasets.locomo.env import LoCoMoRecorder
from datasets.locomo.workflow import LoCoMoWorkflow
from datasets.longmemeval import env as lme_env
from datasets.longmemeval.env import LongMemEvalRecorder
from datasets.longmemeval.workflow import LongMemEvalSWorkflow, LongMemEvalMWorkflow

# dataset → (workflow_cls, env_module, recorder_cls).
# env_module must expose get_task_list(status, eval_n_samples).
REGISTRY: Dict[str, Tuple] = {
    "dynamicmem":    (DynamicMemWorkflow,   dm_env,     DynamicMemRecorder),
    "locomo":        (LoCoMoWorkflow,       locomo_env, LoCoMoRecorder),
    "longmemeval_s": (LongMemEvalSWorkflow, lme_env,    LongMemEvalRecorder),
    "longmemeval_m": (LongMemEvalMWorkflow, lme_env,    LongMemEvalRecorder),
}

DATASETS: List[str] = sorted(REGISTRY)


def resolve(dataset: str) -> Tuple:
    """Return (workflow_cls, env_module, recorder_cls) for `dataset`."""
    if dataset not in REGISTRY:
        raise ValueError(
            f"unknown dataset {dataset!r}; ALMA supports {DATASETS}. "
            f"(ALMA runs one dataset per run — pick one.)"
        )
    return REGISTRY[dataset]
