"""Forge in-container runner.

Invoked by forge/evaluator.py inside Singularity:

    python /app/forge/launch.py --harness-dir /harness --out-dir /out ...

Steps:
  1. Dynamically load MemoStructure subclass from /harness/harness.py.
  2. Run datasets.dynamicmem.workflow.DynamicMemWorkflow across task users.
     (The workflow is methodology-neutral per-user execution — two phases,
     timeouts, trace capture, token tracker — and is reused directly.)
  3. Write score.json, traces/, memory_dumps/, token_usage.json → /out.

The harness_dir is bind-mounted read-only; /out is bind-mounted read-write.
The project root is bind-mounted at /app so `datasets.dynamicmem.env` and
`baselines.alma.*` are importable.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Type

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.harness_base import MemoStructure
from common.tokens import init_global_tracker

# Workflow registry — add new benchmarks here. The env module must expose
# `get_task_list(status, eval_n_samples)`. (Token tracker is now lazy-read
# from `common.tokens.GLOBAL_TOKEN_TRACKER` by `Judge`/`Agent` directly —
# no per-module injection needed.)
from datasets.dynamicmem import env as dm_env
from datasets.dynamicmem.workflow import DynamicMemWorkflow
from datasets.locomo import env as locomo_env
from datasets.locomo.workflow import LoCoMoWorkflow
from datasets.longmemeval import env as lme_env
from datasets.longmemeval.workflow import LongMemEvalSWorkflow, LongMemEvalMWorkflow

WORKFLOWS = {
    "dynamicmem":    (DynamicMemWorkflow,   dm_env),
    "locomo":        (LoCoMoWorkflow,       locomo_env),
    "longmemeval_s": (LongMemEvalSWorkflow, lme_env),
    "longmemeval_m": (LongMemEvalMWorkflow, lme_env),
}


def _load_harness_class(harness_dir: Path) -> Type[MemoStructure]:
    """Import harness.py and return the MemoStructure subclass.

    On import failure, raise ImportError with an actionable message. The
    error string is propagated up to score.json::invalid_users[0].error and
    becomes the trace shown to CC during sanity-retry — so write it for an
    LLM reader.
    """
    harness_py = harness_dir / "harness.py"
    if not harness_py.exists():
        raise ImportError(
            f"harness.py missing at {harness_py}. The proposer must write a "
            f"harness.py file inside its target dir."
        )
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    spec = importlib.util.spec_from_file_location("forge_harness_mod", str(harness_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {harness_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_harness_mod"] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"harness.py imports a package not in the container: {exc.name!r}. "
            f"Either (a) declare it in `requirements.txt` (will trigger a delta "
            f"image build), or (b) switch to a package already in the base image "
            f"(see PROPOSER_SYSTEM for the list). Original: {exc}"
        ) from exc
    except Exception as exc:
        raise ImportError(
            f"harness.py raised at import time: [{type(exc).__name__}] {exc}"
        ) from exc
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, MemoStructure) and obj is not MemoStructure:
            return obj
    raise ImportError(
        f"No MemoStructure subclass found in {harness_py}. Define a class "
        f"that inherits from `common.harness_base.MemoStructure` and "
        f"implements both `general_update` and `general_retrieve`."
    )


def _build_score(records: List[Any], score_max: int = 10) -> dict:
    rewards: List[float] = []
    per_user: dict = {}
    invalid: list = []
    for rec in records:
        if isinstance(rec, Exception):
            invalid.append({
                "user_id": getattr(rec, "user_id", "unknown"),
                "error": repr(rec),
            })
            continue
        uid = getattr(rec, "user_id", "unknown")
        r = float(getattr(rec, "reward", 0.0))
        steps = getattr(rec, "steps", [])
        fi = getattr(rec, "failure_info", None)
        rewards.append(r)
        per_user[uid] = {"reward": r, "n_qa": len(steps), "failure_info": fi}
    overall = float(np.mean(rewards)) if rewards else 0.0
    se = (
        float(np.std(rewards, ddof=1) / np.sqrt(len(rewards)))
        if len(rewards) > 1 else 0.0
    )
    return {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": overall,
            "benchmark_overall_eval_standard_deviation": se,
            # Maximum score the judge can produce — orchestrator uses this to
            # normalize each benchmark to [0, 1] before mean (so DynamicMem 0-10
            # and LoCoMo/LongMemEval 0-1 carry equal weight).
            "score_max": int(score_max),
        },
        "per_user": per_user,
        "invalid_users": invalid,
    }


def _write_error(out_dir: Path, err: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": err}],
    }
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def _async_main(args: argparse.Namespace) -> None:
    harness_dir = Path(args.harness_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset not in WORKFLOWS:
        _write_error(
            out_dir,
            f"unknown dataset '{args.dataset}'; known: {sorted(WORKFLOWS)}",
        )
        return
    workflow_cls, env_module = WORKFLOWS[args.dataset]

    try:
        memo_class = _load_harness_class(harness_dir)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        _write_error(out_dir, err)
        return

    tracker = init_global_tracker()

    # Task-list size: for mode=check, cap at check_n_samples; for mode=eval,
    # cap at eval_n_samples. The env module's parameter name is `eval_n_samples`
    # but its semantic is "how many items to return" — we reuse it for both.
    task_list_size = args.check_n_samples if args.mode == "check" else args.eval_n_samples
    task_list = env_module.get_task_list(status=args.status, eval_n_samples=int(task_list_size))

    workflow = workflow_cls(
        memo_class=memo_class,
        model=args.model,
        update_type=args.update_type,
        n_chunks=args.n_chunks,
        max_logs=args.max_logs,
        eval_n_qa=args.eval_n_qa,
        judge_model=args.judge_model,
        memory_dumps=args.memory_dumps,
    )
    workflow.memo_sha = harness_dir.name
    workflow.status = args.status
    workflow.output_run_dir = out_dir

    records, rlen = await workflow.run_all_users(
        task_list=task_list,
        mode=args.mode,
        max_sample_concurrent=args.max_sample_concurrent,
        check_n_samples=args.check_n_samples,
        check_n_qa=args.check_n_qa,
    )

    score = _build_score(records[:rlen], score_max=workflow.judge_score_max)
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2, ensure_ascii=False)

    workflow.save_full_traces(records[:rlen])

    with (out_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(tracker.summary(), f, indent=2, ensure_ascii=False)

    # chroma / httpx background threads can delay normal interpreter
    # shutdown. Artifacts are flushed; exit abruptly.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="dynamicmem", choices=sorted(WORKFLOWS))
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--eval-n-samples", type=int, default=6,
                        help="Task-list cap when mode=eval")
    parser.add_argument("--eval-n-qa", type=int, default=None,
                        help="Per-sample QA cap when mode=eval (None = all available)")
    parser.add_argument("--check-n-samples", type=int, default=1,
                        help="Task-list cap when mode=check (sanity run)")
    parser.add_argument("--check-n-qa", type=int, default=3,
                        help="Per-sample QA cap when mode=check")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--update-type", default="all_at_once",
                        choices=["all_at_once", "chunked", "sequential"])
    parser.add_argument("--n-chunks", type=int, default=5)
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-sample-concurrent", type=int, default=3)
    parser.add_argument("--memory-dumps", default="full",
                        choices=["full", "stats", "none"],
                        help="After Phase 1, dump memo state: full / stats / none")
    args = parser.parse_args()
    asyncio.run(_async_main(args))
    os._exit(0)
