"""Forge in-container runner.

Invoked by forge/evaluator.py inside Singularity:

    python /app/forge/launch.py --harness-dir /harness --out-dir /out ...

Steps:
  1. Dynamically load MemoStructure subclass from /harness/harness.py.
  2. Run datasets.dynamicmem.workflow.DynamicMemWorkflow across task users.
     (The workflow is methodology-neutral per-user execution — two phases,
     timeouts, trace capture, token tracker — and is reused directly.)
  3. Write score.json, traces/, token_usage.json → /out.

The harness_dir is bind-mounted read-only; /out is bind-mounted read-write.
Binds are SELECTIVE (v10): only common/, datasets/, forge/{__init__,launch,
harness_base}.py are mounted under /app — the rest of forge/ (host outer
loop) and baselines/ are deliberately NOT visible in-container. See
forge/evaluator.py's module docstring for the authoritative bind list.
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
        # select the class defined in this harness file; imported bases like
        # forge.harness_base.MemoStructure are no longer abstract, so an
        # isabstract filter would wrongly match them.
        if issubclass(obj, MemoStructure) and obj.__module__ == module.__name__:
            return obj
    raise ImportError(
        f"No MemoStructure subclass found in {harness_py}. Define a class "
        f"that inherits from `forge.harness_base.MemoStructure` and "
        f"implements both `build_memory_from_data` and `retrieve_memory_for_query`."
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
            # normalize each benchmark to [0, 1] before mean. All three current
            # benchmarks report 1 (DynamicMem TCE holistic 0.0-1.0 partial
            # credit; LoCoMo/LongMemEval 0/1 binary).
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

    # Stage spec: benchmark-native size fields normalized by the orchestrator
    # (see forge/orchestrator.py::stage_wire_spec). `n_samples` caps the task
    # list — a deterministic PREFIX of the split, so a smaller stage's task
    # list is always a prefix of a larger one (staged nesting).
    stage_spec = json.loads(args.stage_spec)
    # n_samples None = no cap (a null single_stage/stage field = the whole split).
    n_samples = stage_spec.get("n_samples")
    task_list = env_module.get_task_list(
        status=args.split,
        eval_n_samples=None if n_samples is None else int(n_samples),
        seed=stage_spec.get("sample_seed"),
    )

    workflow = workflow_cls(
        memo_class=memo_class,
        model=args.model,
        max_logs=args.max_logs,
        judge_model=args.judge_model,
    )
    workflow.memo_sha = harness_dir.name
    workflow.status = args.split
    workflow.output_run_dir = out_dir

    # Cross-stage memory cache: active for the gauntlet tiers (stage1..3) and
    # the progressive=false single pass ("single"; "full" kept for back-compat
    # — no caller emits it anymore); sanity/dev runs never read or write it
    # (harness code can still change during the sanity-fix retry loop).
    if args.memcache_dir and args.stage in ("stage1", "stage2", "stage3", "full", "single"):
        from common.memory_cache import harness_fingerprint
        workflow.memory_cache_dir = Path(args.memcache_dir)
        workflow.harness_fingerprint = harness_fingerprint(harness_dir)

    records, rlen = await workflow.run_all_users(
        task_list=task_list,
        stage=args.stage,
        stage_spec=stage_spec,
        max_sample_concurrent=args.max_sample_concurrent,
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
    parser.add_argument("--split", default="search", choices=["search", "test"],
                        help="Benchmark split (the user-facing `mode` maps search/test/dev onto this)")
    parser.add_argument("--stage", default="stage3",
                        choices=["sanity", "stage1", "stage2", "stage3", "full", "single"],
                        help="Staged-evaluation tier this run belongs to "
                             "('single' = the progressive=false one-pass tier, "
                             "sized by the dataset's single_stage config block; "
                             "'full' is a legacy alias, no longer emitted). "
                             "Gates the cross-stage memory cache (active for "
                             "stage1..3 + single/full); sizes come from --stage-spec.")
    parser.add_argument("--stage-spec", required=True,
                        help='JSON stage spec, e.g. \'{"n_samples": 2, "n_qa": 20}\' '
                             '(dynamicmem: n_samples/n_checkpoints/n_task_a/n_task_c; '
                             'locomo: n_samples/n_qa; longmemeval: n_samples).')
    parser.add_argument("--memcache-dir", default=None,
                        help="RW dir for cross-stage memory snapshots "
                             "(omit to disable caching)")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-sample-concurrent", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(_async_main(args))
    os._exit(0)
