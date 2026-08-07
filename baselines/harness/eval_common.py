"""Shared entry for the harness baselines (cc, hipporag2, amem, lightmem,
simplemem, zep, mem0, memoryos): adapt a fixed MemoStructure into the shared,
execution-independent `common.evaluate.evaluate_memo` — the SAME function
forge's container and alma's subprocess run — so a baseline's score is
identical-by-construction to the main method's data path, not merely
comparable. What lives here is only the baseline-side glue: make_memo_class
(config injection + picklability) and the thin run_baseline wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from common.harness_base import MemoStructure
from baselines.registry import resolve


def make_memo_class(base_cls: Type[MemoStructure], **cfg) -> Type[MemoStructure]:
    """Workflow instantiates memo_class() with NO args (common/workflow.py:465),
    so per-run config travels as a class attribute the instance reads via
    self._cfg.

    The generated class is made PICKLABLE: the cross-stage memory cache
    (common/memory_cache.py) pickles the built memo, and a bare `type(...)`
    subclass of an ABC gets `__module__ == "abc"` and is unresolvable by
    pickle (attribute lookup fails) — so the pickle would silently fail and
    every memcache save/load would degrade to a rebuild. Anchoring the class in
    the base's module (so `getattr(module, qualname) is cls`) fixes that.
    NOTE: called once per run (run.py's main), so the single stable name is
    safe; a second call with the same base rebinds the module attribute."""
    name = f"Configured{base_cls.__name__}"
    cls = type(name, (base_cls,), {"_cfg": cfg})
    cls.__module__ = base_cls.__module__
    cls.__qualname__ = name
    mod = sys.modules.get(base_cls.__module__)
    if mod is not None:
        setattr(mod, name, cls)
    return cls


def resolve_task_list(dataset: str, split: str, stage_spec: Dict[str, Any]) -> List[str]:
    """EXACTLY forge/launch.py:185-189 — same env.get_task_list, same n_samples
    key. This makes the baseline split byte-identical to the main method.

    `stage_spec["sample_seed"]` (progressive gauntlet only) reaches
    env.get_task_list as `seed`, driving common.sampling.shuffle_prefix's
    nested random subset selection when a stage caps n. Absent (single-stage
    path / whole split) it is None → the historical deterministic prefix, so
    default whole-split baseline results are unchanged."""
    _wf, env_module, _rec = resolve(dataset)
    n_samples = stage_spec.get("n_samples")
    return env_module.get_task_list(
        status=split,
        eval_n_samples=None if n_samples is None else int(n_samples),
        seed=stage_spec.get("sample_seed"),
    )


async def run_baseline(
    *,
    dataset: str,
    split: str,
    single_stage: Optional[Dict[str, Any]] = None,
    memo_class: Type[MemoStructure],
    qa_model: str,
    judge_model: str,
    out_dir: Path,
    max_sample_concurrent: int = 3,
    progressive: bool = False,
    sampling_seed: int = 42,
    stages: Optional[Dict[str, Any]] = None,
    memory_cache: bool = True,
) -> Dict[str, Any]:
    """Evaluate one fixed memory system on a split.

    Sizing is config-driven (no sizing CLI flags — config file only):

    progressive=False (default): ONE single-stage pass sized by the REQUIRED
    `single_stage` block (a null/omitted field = the whole split; absent block
    raises ValueError — no silent whole-split). progressive=True: the staged
    stage1→2→3 gauntlet with promotion thresholds (`stages` overrides the family
    DEFAULT_STAGES). Both are one call into common.evaluate.evaluate_memo — the
    shared, execution-independent evaluator (see its docstring for the artifact
    layout: out_dir/<stage>/ + stages.json + reached-stage root copies). The
    per-run sample_seed is the fixed step-0 derivation from `sampling_seed`
    (no search steps here); a no-op at whole-split n=None.

    Returns {dataset: metrics} where metrics is evaluate_memo's dict
    {raw_score, score_max, per_user_stddev, tokens, stage, eliminated}."""
    # One call into the shared, execution-independent evaluate_memo (which runs
    # the whole gauntlet — or the single pass — right here in-process). The
    # per-run sample seed is the fixed step-0 derivation (no search steps here).
    from common.evaluate import evaluate_memo
    from common.sampling import derive_sample_seed

    metrics = await evaluate_memo(
        memo_class=memo_class, dataset=dataset, split=split,
        progressive=progressive, out_dir=out_dir,
        qa_model=qa_model, judge_model=judge_model,
        stages=stages, single_stage=single_stage,
        max_sample_concurrent=max_sample_concurrent,
        sample_seed=derive_sample_seed(sampling_seed, 0, dataset),
        memory_cache=memory_cache,
    )
    return {dataset: metrics}


def print_result(dataset: str, progressive: bool, result: Dict[str, Any], out_dir: Path) -> None:
    """One-line run summary. run_baseline returns {dataset: metrics} for both
    the progressive gauntlet and the single-stage pass — read it uniformly."""
    m = result.get(dataset, {})
    label = "gauntlet" if progressive else "single"
    print(
        f"[{dataset}] {label} stage={m.get('stage')} "
        f"raw_score={m.get('raw_score')} eliminated={m.get('eliminated')} → {out_dir}"
    )
