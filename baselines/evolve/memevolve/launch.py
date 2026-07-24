"""MemEvolve subprocess entry point (pattern copied from evolve/alma/launch.py
per the baselines README convention — copy, don't import).

Loads the assembled genotype file (a `MemoStructure` subclass), runs the
dataset workflow over the requested split, writes score.json / traces /
token_usage.json to the caller-supplied output_run_dir.

mode=check runs the cheap sanity tier (stage="sanity"), mode=eval the full
stage3 pass — same mapping as alma.
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

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.harness_base import MemoStructure
from baselines.registry import resolve, DATASETS
from common.tokens import init_global_tracker
from common.logger import get_logger

log = get_logger("main", level_styles={
    "INFO": {"icon": "🧫", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})


def find_subclass_in_file(file_path: str, base_class: type):
    spec = importlib.util.spec_from_file_location("memevolve_genotype", file_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for file {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["memevolve_genotype"] = module
    spec.loader.exec_module(module)
    subclasses = [
        obj for _name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base_class) and obj is not base_class
    ]
    if not subclasses:
        raise ValueError(f"No class in {file_path} inherits from {base_class.__name__}")
    return subclasses[0]


def _build_score_json(recorder_list: List[Any]) -> dict:
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
            per_user[uid] = {"reward": reward, "n_qa": len(steps), "failure_info": fi}
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
    dataset: str = "dynamicmem",
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    status: str = "search",
    eval_n_qa: Optional[int] = None,
    max_sample_concurrent: int = 6,
    mode: str = "eval",
    judge_model: str = "gpt-5-mini",
    check_n_samples: int = 2,
    check_n_qa: int = 3,
):
    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        memo_class = find_subclass_in_file(module_path, MemoStructure)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        log.warning(f"Failed to load genotype {memory_id}: {exc}")
        _write_error_score(run_dir, err)
        return

    log.info(f"Start evaluation for genotype {memory_id} → {run_dir}")
    tracker = init_global_tracker()

    task_list_size = check_n_samples if mode == "check" else eval_n_samples

    workflow_cls, env_module, _recorder_cls = resolve(dataset)
    task_list = env_module.get_task_list(status=status, eval_n_samples=int(task_list_size))
    log.info(f"Task list ({status}, size={len(task_list)}): {[t[-15:] for t in task_list]}")

    workflow = workflow_cls(
        memo_class=memo_class,
        model=model,
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

    score_payload = _build_score_json(records[:record_len])
    with (run_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)
    log.info(f"score.json written: overall={score_payload['benchmark_eval_score']['benchmark_overall_eval_score']:.3f}")

    workflow.save_full_traces(records[:record_len])

    token_payload = tracker.summary()
    with (run_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(token_payload, f, indent=2, ensure_ascii=False)

    log.info(f"Evaluation complete: results under {run_dir}")
    # Same rationale as alma/launch.py: skip interpreter shutdown.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", required=True)
    parser.add_argument("--memory_id", required=True)
    parser.add_argument("--output_run_dir", required=True)
    parser.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--eval_n_qa", type=int, default=None)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--judge_model", default="gpt-5-mini")
    parser.add_argument("--check_n_samples", type=int, default=2)
    parser.add_argument("--check_n_qa", type=int, default=3)

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        output_run_dir=args.output_run_dir,
        dataset=args.dataset,
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
    os._exit(0)
