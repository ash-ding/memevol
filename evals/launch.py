"""
Main evaluation entry point for memevol (DynamicMem only).

Simplified from ALMA's launch.py:
  - No Docker, no ENVS dict, DynamicMem only
  - Uses DynamicMem_Workflow for two-phase per-user execution
  - Saves results to evals/logs/dynamicmem/<memory_id>_<mode>.json
"""

from pathlib import Path
import asyncio
import os
import sys
import json
import random
import importlib
import inspect
import traceback
from collections import defaultdict
from typing import Dict, List, Optional

# Ensure project root is on sys.path so `envs` package can be imported
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from agents.memo_structure import MemoStructure
from workflows.dynamicmem_workflow import DynamicMem_Workflow
from eval_envs.base_envs import Basic_Recorder
from agents.base import init_global_tracker
from logger import get_logger

log = get_logger("main", level_styles={
    "INFO": {"icon": "🚀", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})

LOG_DIR = Path(os.environ.get("EVALS_LOG_DIR", str(Path(__file__).resolve().parent / "logs")))


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def _sample_steps_from_bins(steps: list, n_score_bins: int, samples_per_bin: int) -> list:
    """Bin steps by score and sample up to samples_per_bin from each bin."""
    bin_width = 10.0 / n_score_bins
    binned: dict = {i: [] for i in range(n_score_bins)}
    for step in steps:
        s = step["score"]
        idx = min(int(s / bin_width), n_score_bins - 1)
        idx = max(0, idx)
        binned[idx].append(step)

    sampled = []
    for i in range(n_score_bins):
        recs = binned[i]
        k = min(samples_per_bin, len(recs))
        if k > 0:
            sampled.extend(random.sample(recs, k))
    return sampled


def get_meta_eval_info(
    recorder_list: list,
    record_len: int,
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
):
    """Process a list of DynamicMemRecorder objects with single-user sampling.

    The reward (``benchmark_overall_eval_score``) is averaged across **all**
    valid users so it still reflects the memo's full performance. For analysis
    trajectories, however, only **one** user is randomly chosen and its QA
    steps are binned by score (``n_score_bins`` equal-width bins over [0, 10])
    with up to ``samples_per_bin`` steps drawn from each bin. This keeps the
    analysis prompt small enough to fit in the meta-model's context window.

    Each sampled step is tagged with its ``user_id``. Each fresh subprocess
    has an independent random seed, so the chosen user naturally varies
    across memo evaluations.

    Returns (eval_meta_info_dict, sampled_steps, invalid_sample_records).
    """
    invalid_records_info = []

    for rec in recorder_list[record_len:]:
        if isinstance(rec, Exception):
            invalid_records_info.append(f"[Update] Task failed: {repr(rec)}")

    # Compute overall reward across ALL valid users (sampling restriction below
    # only affects which user's QA steps end up in the analysis prompt — not
    # how the benchmark score is computed).
    rewards = []
    for rec in recorder_list[:record_len]:
        if not isinstance(rec, Exception):
            try:
                rewards.append(rec.reward)
            except Exception:
                pass
    overall_avg = float(np.mean(rewards)) if rewards else 0.0
    overall_se = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else 0.0

    # Collect per-user QA pools from valid recorders. We tag steps with user_id
    # so any later sampling preserves the source user.
    valid_users = []  # list of (user_id, user_steps)
    total_steps = 0
    for rec in recorder_list[:record_len]:
        if isinstance(rec, Exception):
            failed_user_id = getattr(rec, 'user_id', 'unknown')
            invalid_records_info.append(f"User {failed_user_id} failed: {repr(rec)}")
            continue
        try:
            user_steps = rec.steps
            user_id = getattr(rec, 'user_id', '') or 'unknown'
        except Exception as exc:
            invalid_records_info.append(f"Record access error: {repr(exc)}")
            continue

        # Surface partial-user failures so LLM reflection can see them even
        # though the recorder itself is "successful" (the partial QAs above
        # are valid data).
        failure_info = getattr(rec, 'failure_info', None)
        if failure_info:
            invalid_records_info.append(f"User {user_id} partially failed: {failure_info}")

        total_steps += len(user_steps)
        for step in user_steps:
            step["user_id"] = user_id
        valid_users.append((user_id, user_steps))

    # Pick exactly one valid user and sample its bins. This compresses the
    # examples block from (n_users * samples_per_bin * n_score_bins) down to
    # (samples_per_bin * n_score_bins) — typically 9 — to stay under the
    # meta-model's input-token limit.
    sampled_steps = []
    chosen_user_id = None
    if valid_users:
        chosen_user_id, chosen_user_steps = random.choice(valid_users)
        sampled_steps = _sample_steps_from_bins(chosen_user_steps, n_score_bins, samples_per_bin)

    for user_id, user_steps in valid_users:
        marker = "[selected]" if user_id == chosen_user_id else ""
        sampled_n = len(sampled_steps) if user_id == chosen_user_id else 0
        log.info(f"[User {user_id}] {len(user_steps)} QAs, sampled {sampled_n} {marker}".rstrip())

    if invalid_records_info:
        invalid_sample_records = random.sample(invalid_records_info, min(3, len(invalid_records_info)))
    else:
        invalid_sample_records = []

    log.info(
        f"--- AVG Reward: {overall_avg:.3f} | SE: {overall_se:.3f} | "
        f"Valid users: {len(rewards)} | Total QA steps: {total_steps} | "
        f"Sampled {len(sampled_steps)} QAs from user={chosen_user_id} "
        f"({n_score_bins} bins × {samples_per_bin} samples/bin) ---"
    )

    eval_meta_info = {
        "benchmark_overall_eval_score": overall_avg,
        "benchmark_overall_eval_standard_deviation": overall_se,
    }
    return eval_meta_info, sampled_steps, invalid_sample_records


def get_json(
    env: str,
    file_name: str,
    sampled_steps: list,
    avg_reward: Dict,
    invalid_sample_records: List[str],
    token_usg: Dict = {},
):
    env_dir = LOG_DIR / env
    env_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for step in sampled_steps:
        examples.append({
            "user_id": step.get("user_id", ""),
            "query": step["query"],
            "retrieved_memory": step.get("retrieved_memory", {}),
            "predicted": step["predicted"],
            "reference": step["reference"],
            "score": step["score"],
            "judge_reason": step.get("judge_reason", ""),
            "relevant_app_logs": step.get("relevant_app_logs", []),
        })

    for error in invalid_sample_records:
        examples.append({"error_info": error, "score": 0.0})

    output = {
        "benchmark_eval_score": avg_reward,
        "examples": examples,
        "token_usage": token_usg,
    }

    file_path = env_dir / f"{file_name}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info(f"Benchmark evaluation result saved to: {file_path}")


def find_subclass_in_file(file_path: str, base_class: type):
    """Dynamically load a Python file and return the first subclass of base_class."""
    import importlib.util
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    module_path: str,
    memory_id: str,
    update_type: str = "all_at_once",
    n_chunks: int = 5,
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    eval_n_users: int = 6,
    status: str = "search",
    eval_n_qa: Optional[int] = None,
    max_user_concurrent: int = 6,
    mode: str = "eval",
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
    judge_model: str = "gpt-5-mini",
    check_n_users: int = 3,
    check_n_qa: int = 10,
):
    # 1. Load MemoStructure class from memo file
    try:
        memo_class = find_subclass_in_file(module_path, MemoStructure)
    except Exception as exc:
        log.warning(f"Failed to load memo structure {memory_id}: {exc}")
        file_name = f"{memory_id}_{status}_{mode}"
        error_info = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        get_json("dynamicmem", file_name, [], {"benchmark_overall_eval_score": 0.0, "benchmark_overall_eval_standard_deviation": 0.0}, [error_info])
        return

    log.info(f"Start evaluation for Memory Structure: {memory_id}")

    tracker = init_global_tracker()

    # 2. Get task list
    from envs.dynamicmem_env import get_task_list
    task_list = get_task_list(status=status, eval_n_users=int(eval_n_users))
    log.info(f"Task list ({status}, size={len(task_list)}): {[t[-15:] for t in task_list]}")

    # 3. Build workflow and run
    workflow = DynamicMem_Workflow(
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

    records, record_len = await workflow.run_all_users(
        task_list=task_list,
        mode=mode,
        max_user_concurrent=max_user_concurrent,
        check_n_users=check_n_users,
        check_n_qa=check_n_qa,
    )

    # 4. Process results
    avg_reward, sampled_records, invalid_sample_records = get_meta_eval_info(records, record_len, n_score_bins, samples_per_bin)

    token_tracker = tracker.summary()
    file_name = f"{memory_id}_{status}_{mode}"
    get_json("dynamicmem", file_name, sampled_records, avg_reward, invalid_sample_records, token_tracker)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", required=True)
    parser.add_argument("--memory_id", required=True)
    parser.add_argument("--update_type", default="all_at_once",
                        choices=["all_at_once", "chunked", "sequential"])
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--eval_n_users", type=int, default=6)
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--eval_n_qa", type=int, default=None,
                        help="QA pairs per user (None = use all, recommended for eval)")
    parser.add_argument("--max_user_concurrent", type=int, default=3)
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--n_score_bins", type=int, default=3,
                        help="Number of equal-width bins over score range 1–10")
    parser.add_argument("--samples_per_bin", type=int, default=3,
                        help="Max QA trajectories sampled per score bin")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Model used by the LLM judge for scoring QA answers")
    parser.add_argument("--check_n_users", type=int, default=3,
                        help="Number of users sampled during sanity check (mode=check)")
    parser.add_argument("--check_n_qa", type=int, default=10,
                        help="Number of QA pairs per user during sanity check (mode=check)")

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        update_type=args.update_type,
        n_chunks=args.n_chunks,
        max_logs=args.max_logs,
        model=args.model,
        eval_n_users=args.eval_n_users,
        status=args.status,
        eval_n_qa=args.eval_n_qa,
        max_user_concurrent=args.max_user_concurrent,
        mode=args.mode,
        n_score_bins=args.n_score_bins,
        samples_per_bin=args.samples_per_bin,
        judge_model=args.judge_model,
        check_n_users=args.check_n_users,
        check_n_qa=args.check_n_qa,
    ))
