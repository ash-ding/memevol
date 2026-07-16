"""Shared runner for the cc + hipporag2 baselines: run a MemoStructure through
the main method's per-dataset workflow on a split, with the SAME data path the
main method's container uses (forge/launch.py). This is the single place the
'identical to the main method' guarantee lives.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from common.harness_base import MemoStructure
from baselines.registry import resolve


def parse_stage_spec(s: Optional[str]) -> Dict[str, Any]:
    """Raw USER overrides only ({} when omitted). The family-full base is
    applied by effective_stage_spec — NOT here."""
    return json.loads(s) if s else {}


# Mirrors forge.orchestrator._FAMILY_FIELDS (baselines never import forge).
_FAMILY_FIELDS = {
    "dynamicmem": ("n_users", ("n_checkpoints", "n_task_a", "n_task_c")),
    "locomo": ("n_conversations", ("n_qa",)),
    "longmemeval": ("n_questions", ()),
}


def _family(dataset: str) -> str:
    family = "longmemeval" if dataset.startswith("longmemeval") else dataset
    if family not in _FAMILY_FIELDS:
        raise ValueError(
            f"unknown dataset {dataset!r}; supported datasets: {sorted(_FAMILY_FIELDS.keys())}"
        )
    return family


def family_full_spec(dataset: str) -> Dict[str, Any]:
    """Byte-identical to forge.orchestrator.full_wire_spec: {"n_samples": None,
    ...family extras: None}. The n_checkpoints KEY (dynamicmem) MUST be present
    so run_single_user takes the full TCE path, not the legacy flat path."""
    _sample_field, extras = _FAMILY_FIELDS[_family(dataset)]
    spec: Dict[str, Any] = {"n_samples": None}
    for f in extras:
        spec[f] = None
    return spec


def effective_stage_spec(dataset: str, user_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**family_full_spec(dataset), **(user_spec or {})}


def make_memo_class(base_cls: Type[MemoStructure], **cfg) -> Type[MemoStructure]:
    """Workflow instantiates memo_class() with NO args (common/workflow.py:465),
    so per-run config travels as a class attribute the instance reads via
    self._cfg."""
    return type(f"Configured{base_cls.__name__}", (base_cls,), {"_cfg": cfg})


def resolve_task_list(dataset: str, split: str, stage_spec: Dict[str, Any]) -> List[str]:
    """EXACTLY forge/launch.py:185-189 — same env.get_task_list, same n_samples
    key. This makes the baseline split byte-identical to the main method."""
    _wf, env_module, _rec = resolve(dataset)
    n_samples = stage_spec.get("n_samples")
    return env_module.get_task_list(
        status=split, eval_n_samples=None if n_samples is None else int(n_samples)
    )


async def run_baseline(
    *,
    dataset: str,
    split: str,
    user_stage_spec: Optional[Dict[str, Any]],
    memo_class: Type[MemoStructure],
    qa_model: str,
    judge_model: str,
    out_dir: Path,
    max_sample_concurrent: int = 3,
    update_type: str = "all_at_once",
) -> Dict[str, Any]:
    from common.tokens import init_global_tracker
    stage_spec = effective_stage_spec(dataset, user_stage_spec)   # family-full base + user overrides
    workflow_cls, env_module, _rec = resolve(dataset)
    task_list = resolve_task_list(dataset, split, stage_spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = init_global_tracker()
    workflow = workflow_cls(
        memo_class=memo_class, model=qa_model, judge_model=judge_model,
        update_type=update_type,
    )
    workflow.status = split
    workflow.output_run_dir = out_dir
    records, n = await workflow.run_all_users(
        task_list, stage="full", stage_spec=stage_spec,
        max_sample_concurrent=max_sample_concurrent,
    )
    workflow.save_full_traces(records[:n])
    # Score summary: reuse alma's builder shape (mean per-user reward).
    from baselines.alma.launch import _build_score_json
    score = _build_score_json(records[:n])
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    with (out_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)
    return score
