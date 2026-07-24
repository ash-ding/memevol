"""EvolveMem subprocess entry point (pattern copied from evolve/alma/launch.py
per the baselines README convention — copy, don't import).

Invoked by baselines/evolve/evolvemem/eval_runner.py. Steps:
  1. Load EvolveMemMemo from memo_evolvemem.py (θ comes from the JSON file
     named by $EVOLVEMEM_CONFIG, staged by eval_runner).
  2. Run the dataset's workflow (resolved from the shared registry) across
     all users of the requested split.
  3. Write score.json, full traces (per-user), and token usage to the
     caller-supplied output_run_dir.

Output layout under <output_run_dir>:
  score.json                 — overall + per_user + invalid_users
  traces/<user_id>.json      — full QA trajectory for each user
  token_usage.json           — per-model token totals
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from baselines.registry import resolve, DATASETS
from common.tokens import init_global_tracker
from common.logger import get_logger

log = get_logger("main", level_styles={
    "INFO": {"icon": "🧬", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})


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
    memory_id: str,
    output_run_dir: str,
    dataset: str = "dynamicmem",
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    status: str = "search",
    eval_n_qa: Optional[int] = None,
    max_sample_concurrent: int = 6,
    judge_model: str = "gpt-5-mini",
):
    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the fixed memo class (θ arrives via $EVOLVEMEM_CONFIG).
    try:
        from baselines.evolve.evolvemem.memo_evolvemem import EvolveMemMemo
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        log.warning(f"Failed to load EvolveMemMemo: {exc}")
        _write_error_score(run_dir, err)
        return

    cfg_path = os.environ.get("EVOLVEMEM_CONFIG", "<defaults>")
    log.info(f"Start evaluation: config={cfg_path} id={memory_id} → {run_dir}")

    # 2. Token tracker (Judge picks it up via GLOBAL_TOKEN_TRACKER).
    tracker = init_global_tracker()

    # 3. Resolve workflow + task list from the shared registry.
    workflow_cls, env_module, _recorder_cls = resolve(dataset)
    task_list = env_module.get_task_list(status=status, eval_n_samples=int(eval_n_samples))
    log.info(f"Task list ({status}, size={len(task_list)}): {[t[-15:] for t in task_list]}")

    workflow = workflow_cls(
        memo_class=EvolveMemMemo,
        model=model,
        max_logs=max_logs,
        eval_n_qa=eval_n_qa,
        judge_model=judge_model,
    )
    workflow.memo_sha = memory_id
    workflow.status = status
    workflow.output_run_dir = run_dir

    stage_spec = {"n_samples": int(eval_n_samples)}
    if eval_n_qa is not None:
        stage_spec["n_qa"] = int(eval_n_qa)

    records, record_len = await workflow.run_all_users(
        task_list=task_list,
        stage="stage3",
        stage_spec=stage_spec,
        max_sample_concurrent=max_sample_concurrent,
    )

    # 4. Persist outputs.
    score_payload = _build_score_json(records[:record_len])
    with (run_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(score_payload, f, indent=2, ensure_ascii=False)
    log.info(f"score.json written: overall={score_payload['benchmark_eval_score']['benchmark_overall_eval_score']:.3f}")

    workflow.save_full_traces(records[:record_len])

    token_payload = tracker.summary()
    with (run_dir / "token_usage.json").open("w", encoding="utf-8") as f:
        json.dump(token_payload, f, indent=2, ensure_ascii=False)

    log.info(f"Evaluation complete: results under {run_dir}")

    # Same rationale as alma/launch.py: bypass interpreter shutdown so
    # lingering httpx/chroma threads can't wedge the parent's wait.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_id", required=True)
    parser.add_argument("--output_run_dir", required=True)
    parser.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--eval_n_qa", type=int, default=None)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--judge_model", default="gpt-5-mini")

    args = parser.parse_args()
    asyncio.run(main(
        memory_id=args.memory_id,
        output_run_dir=args.output_run_dir,
        dataset=args.dataset,
        max_logs=args.max_logs,
        model=args.model,
        eval_n_samples=args.eval_n_samples,
        status=args.status,
        eval_n_qa=args.eval_n_qa,
        max_sample_concurrent=args.max_sample_concurrent,
        judge_model=args.judge_model,
    ))
    os._exit(0)
