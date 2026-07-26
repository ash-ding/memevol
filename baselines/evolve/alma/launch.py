"""
Alma subprocess entry point.

Invoked by baselines/evolve/alma/eval_runner.py. Steps:
  1. Dynamically load the MemoStructure subclass from the staged memo file.
  2. Run the dataset's workflow (resolved from the registry) across all users.
  3. Write score.json, full traces (per-user), and token usage
     to the caller-supplied output_run_dir.

No QA sampling is done here — alma's sampling.py reads the full traces
in the main process when constructing the meta-agent analysis input.

Output layout under <output_run_dir>:
  score.json                 — overall + per_user + invalid_users
  traces/<user_id>.json      — full QA trajectory for each user
  token_usage.json           — per-model token totals
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.harness_base import MemoStructure
from baselines.registry import resolve, DATASETS
from common.tokens import init_global_tracker
from common.logger import get_logger

log = get_logger("main", level_styles={
    "INFO": {"icon": "🚀", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})


def find_subclass_in_file(file_path: str, base_class: type):
    """Dynamically load a Python file and return the first subclass of base_class."""
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for file {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_module"] = module
    spec.loader.exec_module(module)

    subclasses = [
        obj for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base_class) and obj is not base_class
    ]
    if not subclasses:
        raise ValueError(f"No class in {file_path} inherits from {base_class.__name__}")
    return subclasses[0]


def _build_score_json(recorder_list: List[Any]) -> dict:
    """Summarize the recorder list into a score dict."""
    rewards = []
    per_user = {}
    invalid_users = []

    for rec in recorder_list:
        if isinstance(rec, Exception):
            uid = getattr(rec, "user_id", "unknown")
            invalid_users.append({"user_id": uid, "error": repr(rec)})
            continue
        try:
            uid = getattr(rec, "user_id", "") or "unknown"
            reward = float(getattr(rec, "reward", 0.0))
            steps = getattr(rec, "steps", [])
            fi = getattr(rec, "failure_info", None)
            rewards.append(reward)
            per_user[uid] = {
                "reward": reward,
                "n_qa": len(steps),
                "failure_info": fi,
            }
        except Exception as exc:
            invalid_users.append({"user_id": "unknown", "error": f"record access error: {exc!r}"})

    overall_avg = float(np.mean(rewards)) if rewards else 0.0
    overall_se = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else 0.0

    return {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": overall_avg,
            "benchmark_overall_eval_standard_deviation": overall_se,
        },
        "per_user": per_user,
        "invalid_users": invalid_users,
    }


def _write_error_score(output_run_dir: Path, error_info: str) -> None:
    """Write a minimal score.json when the memo file fails to load at all."""
    output_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": error_info}],
    }
    with (output_run_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _total_tokens(summary: Dict[str, Any]) -> int:
    """Sum total_tokens across all models in a TokenTracker.summary()."""
    return sum(
        int(m.get("total_tokens", 0))
        for m in summary.values()
        if isinstance(m, dict)
    )


def _module_fingerprint(module_path: Path) -> str:
    """sha256[:16] over the SINGLE staged memo file (name + content). alma
    stages every memo into the shared memo_test/ dir, so the dir-scoped
    common.memory_cache.harness_fingerprint would churn as sibling memos land;
    a per-file fingerprint is stable within one launch.py run and distinct per
    memo — exactly what the cross-stage cache needs to gate reuse."""
    h = hashlib.sha256()
    p = Path(module_path)
    try:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()[:16]


def _task_list(env_module, status: str, spec: Dict[str, Any]) -> List[str]:
    """env.get_task_list honoring n_samples + the per-step sample_seed (mirrors
    forge/launch.py). seed absent → historical deterministic prefix."""
    n = spec.get("n_samples")
    return env_module.get_task_list(
        status=status,
        eval_n_samples=None if n is None else int(n),
        seed=spec.get("sample_seed"),
    )


async def _run_single_stage(
    *,
    workflow_cls,
    env_module,
    memo_class,
    memo_sha: str,
    status: str,
    out_dir: Path,
    stage: str,
    spec: Dict[str, Any],
    model: str,
    max_logs: Optional[int],
    judge_model: str,
    max_sample_concurrent: int,
    tracker,
) -> dict:
    """One workflow pass at a single tier (no gauntlet). Writes score.json /
    traces/ / token_usage.json under out_dir and returns the score dict. Used
    for mode=check (sanity gate) and progressive=False eval."""
    out_dir.mkdir(parents=True, exist_ok=True)
    task_list = _task_list(env_module, status, spec)
    log.info(f"Task list ({status}, stage={stage}, size={len(task_list)}): "
             f"{[t[-15:] for t in task_list]}")

    workflow = workflow_cls(
        memo_class=memo_class, model=model, max_logs=max_logs, judge_model=judge_model,
    )
    workflow.memo_sha = memo_sha
    workflow.status = status
    workflow.output_run_dir = out_dir

    records, rlen = await workflow.run_all_users(
        task_list=task_list, stage=stage, stage_spec=spec,
        max_sample_concurrent=max_sample_concurrent,
    )
    score = _build_score_json(records[:rlen])
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)
    workflow.save_full_traces(records[:rlen])
    with (out_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)
    log.info(f"score.json written: overall="
             f"{score['benchmark_eval_score']['benchmark_overall_eval_score']:.3f}")
    return score


async def _run_progressive(
    *,
    workflow_cls,
    env_module,
    dataset: str,
    datasets_config: Dict[str, Dict[str, Any]],
    memo_class,
    module_path: str,
    memo_sha: str,
    status: str,
    out_dir: Path,
    sample_seed: Optional[str],
    model: str,
    max_logs: Optional[int],
    judge_model: str,
    max_sample_concurrent: int,
    use_memcache: bool,
    tracker,
) -> Dict[str, Dict[str, Any]]:
    """Drive one candidate through the shared staged gauntlet
    (common.staged_eval.run_gauntlet, coverage="sample") with an IN-PROCESS
    stage runner — the same pattern as baselines/harness/eval_common.py, but
    alma's per-STEP seed (already folded into `sample_seed`) rides every stage
    spec. Per-stage artifacts under out_dir/<stage>/, cross-stage Phase-1 memory
    reuse via out_dir/memory_cache/, and the highest-reached stage's
    score.json/token_usage.json/traces/ copied to the out_dir root (where alma's
    memo_manager + sampling read the reward + examples). stages.json at the
    root."""
    from common.staged_eval import run_gauntlet

    out_dir.mkdir(parents=True, exist_ok=True)

    # Cross-stage memcache: one dir shared across stage1..3 (stage2/3 reuse
    # stage1's Phase-1 memory). Fingerprint the single staged memo file.
    memcache_dir: Optional[Path] = None
    fingerprint = ""
    if use_memcache:
        fingerprint = _module_fingerprint(Path(module_path))
        memcache_dir = out_dir / "memory_cache"
        memcache_dir.mkdir(parents=True, exist_ok=True)

    stage_metrics: Dict[str, Dict[str, Any]] = {}

    async def _run_stage_fn(ds: str, stage_name: str,
                            spec: Dict[str, Any]) -> Optional[Exception]:
        stage_dir = out_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        task_list = _task_list(env_module, status, spec)   # honors n_samples + sample_seed

        workflow = workflow_cls(
            memo_class=memo_class, model=model, max_logs=max_logs, judge_model=judge_model,
        )
        workflow.memo_sha = memo_sha
        workflow.status = status
        workflow.output_run_dir = stage_dir
        if memcache_dir is not None:
            workflow.memory_cache_dir = memcache_dir
            workflow.harness_fingerprint = fingerprint

        tokens_before = _total_tokens(tracker.summary())
        raw_score = 0.0
        crashed: Optional[Exception] = None
        try:
            records, n = await workflow.run_all_users(
                task_list=task_list, stage=stage_name, stage_spec=spec,
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
            log.warning(f"[{stage_name}] stage crashed: {exc!r}")
        tokens_after = _total_tokens(tracker.summary())

        stage_metrics[stage_name] = {
            "raw_score": raw_score,
            "score_max": workflow.judge_score_max,
            "tokens": tokens_after - tokens_before,
        }
        return crashed

    def _read_metrics_fn(ds: str, stage_name: str) -> Dict[str, Any]:
        return dict(stage_metrics[stage_name])   # copy: run_gauntlet mutates tokens

    def _stages_writer(ds: str, summary: Dict[str, Any]) -> None:
        reached = summary["reached"]
        for fname in ("score.json", "token_usage.json"):
            src = out_dir / reached / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)
        # alma reads per-user traces from out_dir/traces — mirror the final
        # (highest-reached) stage's traces up to the root.
        src_traces = out_dir / reached / "traces"
        dst_traces = out_dir / "traces"
        if src_traces.is_dir():
            if dst_traces.exists():
                shutil.rmtree(dst_traces)
            shutil.copytree(src_traces, dst_traces)
        with (out_dir / "stages.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"reached": reached, "eliminated": summary["eliminated"],
                 "stages": summary["stages"]},
                f, indent=2, ensure_ascii=False,
            )

    def _sample_seed_for(ds: str) -> Optional[str]:
        return sample_seed   # per-step seed already derived (None when random_sample off)

    metrics = await run_gauntlet(
        datasets_config=datasets_config,
        coverage="sample",
        smoke=False,
        sample_seed_for=_sample_seed_for,
        run_stage_fn=_run_stage_fn,
        read_metrics_fn=_read_metrics_fn,
        stages_writer=_stages_writer,
    )
    # Guarantee a root score.json even if the first stage crashed before writing
    # one, so alma's memo_manager never hard-fails on a missing file (it degrades
    # to a 0.0-reward examination failure, matching the pre-gauntlet contract).
    if not (out_dir / "score.json").exists():
        _write_error_score(out_dir, "progressive gauntlet produced no score "
                                     "(first stage crashed before writing one)")
    return metrics


async def main(
    module_path: str,
    memory_id: str,
    output_run_dir: str,
    dataset: str = "dynamicmem",
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    status: str = "search",
    max_sample_concurrent: int = 6,
    mode: str = "eval",
    judge_model: str = "gpt-5-mini",
    progressive: bool = True,
    random_sample: bool = False,
    sampling_seed: int = 42,
    step_index: int = 0,
    stages: Optional[dict] = None,
    single_stage: Optional[dict] = None,
    memory_cache: bool = True,
):
    from common.staged_eval import (
        DEFAULT_STAGES, _FAMILY_FIELDS, _benchmark_family, _resolve_dataset_stages,
        single_stage_wire_spec, stage_wire_spec,
    )
    from common.sampling import derive_sample_seed

    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MemoStructure class
    try:
        memo_class = find_subclass_in_file(module_path, MemoStructure)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        log.warning(f"Failed to load memo structure {memory_id}: {exc}")
        _write_error_score(run_dir, err)
        return

    log.info(f"Start evaluation for Memory Structure: {memory_id} → {run_dir} "
             f"(mode={mode}, progressive={progressive}, random_sample={random_sample}, "
             f"step={step_index})")

    # 2. Init token tracker (Judge picks it up via GLOBAL_TOKEN_TRACKER)
    tracker = init_global_tracker()

    # 3. Resolve workflow/env + the stages block (user override or family default).
    workflow_cls, env_module, _recorder_cls = resolve(dataset)
    family = _benchmark_family(dataset)
    ds_params: Dict[str, Any] = {
        "stages": copy.deepcopy(stages) if stages else copy.deepcopy(DEFAULT_STAGES[family])
    }
    _resolve_dataset_stages(dataset, ds_params)   # fill defaults + validate, in place
    stages_block = ds_params["stages"]

    # Per-STEP seed: differs across search steps (design decision 2) so each
    # step samples a different subset; None when random_sample is off (historical
    # deterministic prefix / nesting).
    sample_seed = derive_sample_seed(sampling_seed, step_index, dataset) if random_sample else None

    # 4. Route the eval.
    #  - mode=check  → cheap single sanity-size run (alma's sanity gate; sizes
    #                  from `sanity_check`, ignores `single_stage`).
    #  - mode=eval + progressive → the shared stage1→2→3 gauntlet.
    #  - mode=eval + not progressive → a single pass sized by the REQUIRED
    #    `single_stage` block (same size fields as a stage; a null/omitted field
    #    = the whole split for that dimension). No stage3 fallback — sizing is
    #    config-driven and required (mirrors baselines/harness/eval_common.py).
    if mode == "eval" and progressive:
        await _run_progressive(
            workflow_cls=workflow_cls, env_module=env_module, dataset=dataset,
            datasets_config={dataset: ds_params}, memo_class=memo_class,
            module_path=module_path, memo_sha=memory_id, status=status,
            out_dir=run_dir, sample_seed=sample_seed, model=model, max_logs=max_logs,
            judge_model=judge_model, max_sample_concurrent=max_sample_concurrent,
            use_memcache=memory_cache, tracker=tracker,
        )
    else:
        if mode == "check":
            stage, spec = "sanity", stage_wire_spec(dataset, stages_block["sanity_check"])
        else:
            if not single_stage:
                sample_field = _FAMILY_FIELDS[_benchmark_family(dataset)][0]
                raise ValueError(
                    f"{dataset}: progressive=false requires a `single_stage` block "
                    f"(same size fields as a stage; use all-null for the whole split), "
                    f"e.g. single_stage: {{{sample_field}: null}}"
                )
            # Validate + normalize `single_stage` through the SAME resolver the
            # gauntlet path uses (rejects unknown fields / a stray threshold,
            # normalizes null/full/all → None) on a throwaway copy.
            _ss_params = {"single_stage": copy.deepcopy(single_stage)}
            _resolve_dataset_stages(dataset, _ss_params)
            stage, spec = "single", single_stage_wire_spec(dataset, _ss_params["single_stage"])
        if sample_seed is not None:
            spec["sample_seed"] = sample_seed
        await _run_single_stage(
            workflow_cls=workflow_cls, env_module=env_module, memo_class=memo_class,
            memo_sha=memory_id, status=status, out_dir=run_dir, stage=stage, spec=spec,
            model=model, max_logs=max_logs, judge_model=judge_model,
            max_sample_concurrent=max_sample_concurrent, tracker=tracker,
        )

    log.info(f"Evaluation complete: results under {run_dir}")

    # Bypass asyncio/GC/atexit shutdown — langchain_chroma + httpx connection
    # pools + Chroma telemetry hooks occasionally leave non-daemon threads
    # alive that prevent the interpreter from exiting for several minutes,
    # causing the parent's process.communicate() (with pidfd+epoll) to miss
    # the child-exit event. All on-disk artifacts (score.json, traces/,
    # token_usage.json) have already been flushed above, so
    # abrupt termination is safe.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", required=True)
    parser.add_argument("--memory_id", required=True)
    parser.add_argument("--output_run_dir", required=True,
                        help="Absolute path to the per-run output directory")
    parser.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--judge_model", default="gpt-5-mini")
    # Progressive gauntlet + per-step deterministic sampling (Task 9).
    parser.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=True,
                        help="mode=eval runs the stage1→2→3 gauntlet (default). "
                             "--no-progressive = a single terminal-size pass.")
    parser.add_argument("--random_sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Seed per-step subset selection (different subset each search step).")
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--step_index", type=int, default=0,
                        help="Search-step index; folds into the per-step sample seed.")
    parser.add_argument("--stages", type=str, default=None,
                        help="JSON stages-block override (sanity_check/stage1..3); "
                             "default = family DEFAULT_STAGES.")
    parser.add_argument("--single_stage", type=str, default=None,
                        help="JSON single-pass size block (progressive=false only; "
                             "same size fields as a stage, no threshold). REQUIRED "
                             "when --no-progressive; a null field = whole split.")
    parser.add_argument("--memory_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-stage Phase-1 memory reuse in the gauntlet.")

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        output_run_dir=args.output_run_dir,
        dataset=args.dataset,
        max_logs=args.max_logs,
        model=args.model,
        status=args.status,
        max_sample_concurrent=args.max_sample_concurrent,
        mode=args.mode,
        judge_model=args.judge_model,
        progressive=args.progressive,
        random_sample=args.random_sample,
        sampling_seed=args.sampling_seed,
        step_index=args.step_index,
        stages=json.loads(args.stages) if args.stages else None,
        single_stage=json.loads(args.single_stage) if args.single_stage else None,
        memory_cache=args.memory_cache,
    ))

    # Force immediate process termination. Python's normal shutdown runs
    # atexit handlers + GC, which can hang for minutes on
    # langchain_chroma / httpx background threads & connection-pool cleanup.
    # All artifacts (score.json, traces/, token_usage.json)
    # are already flushed to disk by this point, so bypassing graceful
    # shutdown is safe and prevents the parent from waiting on a zombie
    # that asyncio's pidfd watcher occasionally fails to observe.
    os._exit(0)
