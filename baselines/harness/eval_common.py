"""Shared runner for the cc + hipporag2 baselines: run a MemoStructure through
the main method's per-dataset workflow on a split, with the SAME data path the
main method's container uses (forge/launch.py). This is the single place the
'identical to the main method' guarantee lives.
"""
from __future__ import annotations

import copy
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from common.harness_base import MemoStructure
from baselines.registry import resolve


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


def _memo_source_dir(memo_class: Type[MemoStructure]) -> Optional[Path]:
    """Directory holding the memo module — fingerprinted (common.memory_cache.
    harness_fingerprint) so a change to the memo's source (e.g. the vendored
    amem code) invalidates the cross-stage cache. `memo_class` may be a
    make_memo_class wrapper whose source file is THIS module; walk the MRO to
    the first real MemoStructure subclass defined in its own file."""
    this_file = Path(__file__).resolve()
    for cls in memo_class.__mro__:
        if cls is MemoStructure or not issubclass(cls, MemoStructure):
            continue
        try:
            src = inspect.getsourcefile(cls)
        except TypeError:
            src = None
        if not src:
            continue
        srcp = Path(src).resolve()
        if srcp == this_file:
            continue  # the make_memo_class wrapper lives in eval_common
        return srcp.parent
    return None


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
    `single_stage` block (via common.evaluate.single_stage_wire_spec; a null /
    omitted field = the whole split for that dimension). Raises ValueError when
    `single_stage` is absent — no silent whole-split. The pass runs as
    `run_all_users(stage="single", ...)`; the "single" stage label is purely
    informational (the memory cache is gated by the workflow's memory_cache_dir,
    not the stage name). A per-run sample_seed is derived (fixed step 0, honoring
    `sampling_seed`) and threaded into the spec so it reaches BOTH the task-list
    cap and the per-user QA sampling; it is a no-op at whole-split n=None.
    Returns the _build_score_json dict.

    progressive=True: run the SAME memory system through the shared staged
    gauntlet (common.evaluate.run_gauntlet, coverage="sample"): stage1 →
    stage2 → stage3 with promotion thresholds, per-stage artifacts under
    out_dir/<stage>/, cross-stage Phase-1 memory reuse via out_dir/memory_cache/,
    and a stages.json at the out_dir root. `stages` overrides the family
    DEFAULT_STAGES; `sampling_seed` seeds the (fixed step=0) nested subset
    selection; `memory_cache=False` disables the cache. Returns run_gauntlet's
    per-dataset metrics dict {dataset: {..., stage, eliminated}}."""
    if progressive:
        return await _run_baseline_progressive(
            dataset=dataset, split=split, memo_class=memo_class,
            qa_model=qa_model, judge_model=judge_model, out_dir=out_dir,
            max_sample_concurrent=max_sample_concurrent,
            sampling_seed=sampling_seed, stages=stages, memory_cache=memory_cache,
        )

    # progressive=False: single pass sized by the REQUIRED single_stage block.
    # The shared resolver presence-checks (absent → ValueError, never silent
    # whole-split; an empty {} counts as present = all-null = whole), validates
    # (rejects unknown fields — a typo like `n_user` for `n_users` would
    # otherwise be silently ignored → whole split — and a stray threshold),
    # normalizes null/"full"/"all" → None, and returns the wire spec.
    from common.tokens import init_global_tracker
    from common.sampling import derive_sample_seed
    from common.evaluate import resolve_single_stage_spec

    # Same fixed step-0 seed derivation as the gauntlet (no search steps here);
    # a no-op at whole-split n=None, only selecting a subset when a field caps.
    spec = {
        **resolve_single_stage_spec(dataset, single_stage),
        "sample_seed": derive_sample_seed(sampling_seed, 0, dataset),
    }
    workflow_cls, _env, _rec = resolve(dataset)
    task_list = resolve_task_list(dataset, split, spec)   # honors n_samples + sample_seed

    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = init_global_tracker()
    workflow = workflow_cls(
        memo_class=memo_class, model=qa_model, judge_model=judge_model,
    )
    workflow.status = split
    workflow.output_run_dir = out_dir
    records, n = await workflow.run_all_users(
        task_list, stage="single", stage_spec=spec,
        max_sample_concurrent=max_sample_concurrent,
    )
    workflow.save_full_traces(records[:n])
    # Score summary: reuse alma's builder shape (mean per-user reward).
    from baselines.evolve.alma.launch import _build_score_json
    score = _build_score_json(records[:n])
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    with (out_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)
    return score


def print_result(dataset: str, progressive: bool, result: Dict[str, Any], out_dir: Path) -> None:
    """One-line run summary handling BOTH run_baseline return shapes:
    progressive=False returns the _build_score_json dict; progressive=True
    returns run_gauntlet's per-dataset metrics dict {dataset: {...}}."""
    if progressive:
        m = result.get(dataset, {})
        print(
            f"[{dataset}] gauntlet stage={m.get('stage')} "
            f"raw_score={m.get('raw_score')} eliminated={m.get('eliminated')} → {out_dir}"
        )
    else:
        print("overall:", result["benchmark_eval_score"]["benchmark_overall_eval_score"], "→", out_dir)


def _total_tokens(summary: Dict[str, Any]) -> int:
    """Sum total_tokens across all models in a TokenTracker.summary()."""
    return sum(
        int(m.get("total_tokens", 0))
        for m in summary.values()
        if isinstance(m, dict)
    )


async def _run_baseline_progressive(
    *,
    dataset: str,
    split: str,
    memo_class: Type[MemoStructure],
    qa_model: str,
    judge_model: str,
    out_dir: Path,
    max_sample_concurrent: int,
    sampling_seed: int,
    stages: Optional[Dict[str, Any]],
    memory_cache: bool,
) -> Dict[str, Dict[str, Any]]:
    """progressive=True branch of run_baseline. Drives the fixed memory system
    through common.evaluate.run_gauntlet with an IN-PROCESS stage runner
    (contrast forge, whose runner is a Singularity exec). The promotion /
    elimination / cost-accounting / stages.json logic lives entirely in
    run_gauntlet (identical to forge); only stage EXECUTION + artifact layout +
    memcache mounting are this closure's concern."""
    from common.tokens import init_global_tracker
    from common.sampling import derive_sample_seed
    from common.evaluate import (
        DEFAULT_STAGES, _benchmark_family, _resolve_dataset_stages, run_gauntlet,
    )
    from common.memory_cache import harness_fingerprint
    from baselines.evolve.alma.launch import _build_score_json

    family = _benchmark_family(dataset)
    # deepcopy so DEFAULT_STAGES (module global) is never mutated in place by
    # _resolve_dataset_stages (which fills defaults + validates the block).
    params: Dict[str, Any] = {"stages": copy.deepcopy(stages or DEFAULT_STAGES[family])}
    _resolve_dataset_stages(dataset, params)
    datasets_config = {dataset: params}

    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = init_global_tracker()
    workflow_cls, _env, _rec = resolve(dataset)

    # Cross-stage memcache: one dir SHARED across all stages (so stage2/stage3
    # reuse stage1's Phase-1 memory). Fingerprint the memo module's directory so
    # a change to the memo source invalidates the cache.
    memcache_dir: Optional[Path] = None
    fingerprint = ""
    if memory_cache:
        src_dir = _memo_source_dir(memo_class)
        fingerprint = harness_fingerprint(src_dir) if src_dir is not None else ""
        memcache_dir = out_dir / "memory_cache"
        memcache_dir.mkdir(parents=True, exist_ok=True)

    # Per-stage metrics captured by the runner, read back by read_metrics_fn.
    # Tokens are per-stage DELTAS of the shared global tracker (each stage
    # accumulates into the SAME in-process tracker), so run_gauntlet's
    # cross-stage token summation matches forge's per-container accounting.
    stage_metrics: Dict[str, Dict[str, Any]] = {}

    async def _run_stage_fn(ds: str, stage_name: str,
                            spec: Dict[str, Any]) -> Optional[Exception]:
        stage_dir = out_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        task_list = resolve_task_list(ds, split, spec)   # honors n_samples + sample_seed

        workflow = workflow_cls(
            memo_class=memo_class, model=qa_model, judge_model=judge_model,
        )
        workflow.status = split
        workflow.output_run_dir = stage_dir
        # Activate the shared cross-stage memory cache on this fresh workflow.
        if memcache_dir is not None:
            workflow.memory_cache_dir = memcache_dir
            workflow.harness_fingerprint = fingerprint

        tokens_before = _total_tokens(tracker.summary())
        raw_score = 0.0
        crashed: Optional[Exception] = None
        try:
            records, n = await workflow.run_all_users(
                task_list, stage=stage_name, stage_spec=spec,
                max_sample_concurrent=max_sample_concurrent,
            )
            workflow.save_full_traces(records[:n])
            score = _build_score_json(records[:n])
            raw_score = float(
                score["benchmark_eval_score"]["benchmark_overall_eval_score"]
            )
            with (stage_dir / "score.json").open("w", encoding="utf-8") as f:
                json.dump(score, f, indent=2, ensure_ascii=False)
            with (stage_dir / "token_usage.json").open("w", encoding="utf-8") as f:
                json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)
        except Exception as exc:  # a crashed stage eliminates + stops (run_gauntlet)
            crashed = exc
        tokens_after = _total_tokens(tracker.summary())

        stage_metrics[stage_name] = {
            "raw_score": raw_score,
            # All current benchmarks judge on a 0-1 scale (judge_score_max=1);
            # score.json omits score_max, so read it off the workflow directly.
            "score_max": workflow.judge_score_max,
            "tokens": tokens_after - tokens_before,   # this stage's delta only
        }
        return crashed

    def _read_metrics_fn(ds: str, stage_name: str) -> Dict[str, Any]:
        # Copy — run_gauntlet mutates m["tokens"] in place (cross-stage sum).
        return dict(stage_metrics[stage_name])

    def _stages_writer(ds: str, summary: Dict[str, Any]) -> None:
        reached = summary["reached"]
        # Final (highest-reached) stage artifacts to the out_dir root (mirrors
        # forge's dataset-root copy + the single-stage path's layout).
        for fname in ("score.json", "token_usage.json"):
            src = out_dir / reached / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        with (out_dir / "stages.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"reached": reached, "eliminated": summary["eliminated"],
                 "stages": summary["stages"]},
                f, indent=2, ensure_ascii=False,
            )

    # Harness baselines have NO search steps → the per-run seed is fixed at
    # step 0 (reproducible). A no-op at whole-split n=None (shuffle_prefix
    # returns the whole pool); it only selects a subset when a stage caps n.
    def _sample_seed_for(ds: str) -> Optional[str]:
        return derive_sample_seed(sampling_seed, 0, ds)

    return await run_gauntlet(
        datasets_config=datasets_config,
        coverage="sample",
        smoke=False,
        sample_seed_for=_sample_seed_for,
        run_stage_fn=_run_stage_fn,
        read_metrics_fn=_read_metrics_fn,
        stages_writer=_stages_writer,
    )
