"""Shared entry for the harness baselines (hipporag2, amem, lightmem,
simplemem, zep, mem0, memoryos): adapt a fixed MemoClass into the shared,
execution-independent `common.evaluate.evaluate_memo` — the SAME function
forge's container and alma's subprocess run — so a baseline's score is
identical-by-construction to the main method's data path, not merely
comparable. Config reaches the per-user memo instances through the
constructor (`memo_config` → `memo_class(config=...)`, see
common/memo_class.py) — the old make_memo_class dynamic-subclass injection
was removed 2026-08-06.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from common.memo_class import MemoClass


async def run_baseline(
    *,
    dataset: str,
    split: str,
    single_stage: Optional[Dict[str, Any]] = None,
    memo_class: Type[MemoClass],
    memo_config: Optional[Dict[str, Any]] = None,
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
        memo_class=memo_class, memo_config=memo_config,
        dataset=dataset, split=split,
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
