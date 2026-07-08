"""
Alma subprocess entry point.

Invoked by baselines/alma/eval_runner.py. Steps:
  1. Dynamically load the MemoStructure subclass from the staged memo file.
  2. Run DynamicMemWorkflow across all users.
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
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.harness_base import MemoStructure
from datasets.dynamicmem.workflow import DynamicMemWorkflow
from common.tokens import init_global_tracker
from common.logger import get_logger
from datasets.dynamicmem.env import get_task_list

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


async def main(
    module_path: str,
    memory_id: str,
    output_run_dir: str,
    update_type: str = "all_at_once",
    n_chunks: int = 5,
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    status: str = "search",
    eval_n_qa: Optional[int] = None,
    max_sample_concurrent: int = 6,
    mode: str = "eval",
    judge_model: str = "gpt-5-mini",
    check_n_samples: int = 6,
    check_n_qa: int = 3,
):
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

    log.info(f"Start evaluation for Memory Structure: {memory_id} → {run_dir}")

    # 2. Init token tracker (Judge picks it up via GLOBAL_TOKEN_TRACKER)
    tracker = init_global_tracker()

    # 3. Get task list — alma's mode=check keeps its old semantics (small
    # task list + small QA count) by capping the list itself; the shared
    # workflow no longer resamples.
    task_list_size = check_n_samples if mode == "check" else eval_n_samples
    task_list = get_task_list(status=status, eval_n_samples=int(task_list_size))
    log.info(f"Task list ({status}, size={len(task_list)}): {[t[-15:] for t in task_list]}")

    # 4. Build workflow and run. alma maps its legacy (mode, *_n_qa) knobs
    # onto the shared workflow's staged API: check → the cheap "sanity" tier
    # (DynamicMem: first checkpoint only), eval → the terminal "stage3" tier.
    # The spec's n_qa drives the legacy total-count
    # sampling path in DynamicMemWorkflow.
    workflow = DynamicMemWorkflow(
        memo_class=memo_class,
        model=model,
        update_type=update_type,
        n_chunks=n_chunks,
        max_logs=max_logs,
        eval_n_qa=eval_n_qa,
        judge_model=judge_model,
    )
    workflow.memo_sha = memory_id
    workflow.status = status
    workflow.output_run_dir = run_dir

    stage = "sanity" if mode == "check" else "stage3"
    stage_spec = {"n_samples": int(task_list_size)}
    if mode == "check":
        stage_spec["n_qa"] = int(check_n_qa)
    elif eval_n_qa is not None:
        stage_spec["n_qa"] = int(eval_n_qa)

    records, record_len = await workflow.run_all_users(
        task_list=task_list,
        stage=stage,
        stage_spec=stage_spec,
        max_sample_concurrent=max_sample_concurrent,
    )

    # 5. Persist outputs
    score_payload = _build_score_json(records[:record_len])
    with (run_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)
    log.info(f"score.json written: overall={score_payload['benchmark_eval_score']['benchmark_overall_eval_score']:.3f}")

    workflow.save_full_traces(records[:record_len])

    token_payload = tracker.summary()
    with (run_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(token_payload, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--update_type", default="all_at_once",
                        choices=["all_at_once", "chunked", "sequential"])
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--eval_n_qa", type=int, default=None)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--judge_model", default="gpt-5-mini")
    parser.add_argument("--check_n_samples", type=int, default=6)
    parser.add_argument("--check_n_qa", type=int, default=3)

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        output_run_dir=args.output_run_dir,
        update_type=args.update_type,
        n_chunks=args.n_chunks,
        max_logs=args.max_logs,
        model=args.model,
        eval_n_samples=args.eval_n_samples,
        status=args.status,
        eval_n_qa=args.eval_n_qa,
        max_sample_concurrent=args.max_sample_concurrent,
        mode=args.mode,
        judge_model=args.judge_model,
        check_n_samples=args.check_n_samples,
        check_n_qa=args.check_n_qa,
    ))

    # Force immediate process termination. Python's normal shutdown runs
    # atexit handlers + GC, which can hang for minutes on
    # langchain_chroma / httpx background threads & connection-pool cleanup.
    # All artifacts (score.json, traces/, token_usage.json)
    # are already flushed to disk by this point, so bypassing graceful
    # shutdown is safe and prevents the parent from waiting on a zombie
    # that asyncio's pidfd watcher occasionally fails to observe.
    os._exit(0)
