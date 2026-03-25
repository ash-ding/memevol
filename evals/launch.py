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

def get_meta_eval_info(
    recorder_list: list,
    record_len: int,
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
):
    """Process a list of DynamicMemRecorder objects.

    Scores are 1–10 (LLM judge). We divide the 1–10 range into
    ``n_score_bins`` equal-width bins, then sample up to
    ``samples_per_bin`` QA steps from each bin.

    Each sampled step includes retrieved_memory, relevant_app_logs,
    and judge_reason.

    Returns (eval_meta_info_dict, sampled_steps, invalid_sample_records).
    """
    all_steps = []
    invalid_records_info = []

    for rec in recorder_list[:record_len]:
        if isinstance(rec, Exception):
            invalid_records_info.append(f"Task failed: {repr(rec)}")
        else:
            try:
                all_steps.extend(rec.steps)
            except Exception as exc:
                invalid_records_info.append(f"Record access error: {repr(exc)}")

    for rec in recorder_list[record_len:]:
        if isinstance(rec, Exception):
            invalid_records_info.append(f"[Update] Task failed: {repr(rec)}")

    # Compute overall reward across users (already normalized to 0–1)
    rewards = []
    for rec in recorder_list[:record_len]:
        if not isinstance(rec, Exception):
            try:
                rewards.append(rec.reward)
            except Exception:
                pass
    overall_avg = float(np.mean(rewards)) if rewards else 0.0
    overall_se = float(np.std(rewards, ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else 0.0

    # Build equal-width score bins over [0, 10]
    bin_width = 10.0 / n_score_bins  # score range is 0–10
    bins = []
    for i in range(n_score_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        bins.append((low, high))

    # Assign steps to bins
    binned: dict = {i: [] for i in range(n_score_bins)}
    for step in all_steps:
        s = step["score"]
        # Last bin includes the upper boundary (score == 10)
        idx = min(int(s / bin_width), n_score_bins - 1)
        idx = max(0, idx)
        binned[idx].append(step)

    # Sample from each bin
    sampled_steps = []
    for i in range(n_score_bins):
        recs = binned[i]
        k = min(samples_per_bin, len(recs))
        if k > 0:
            sampled = random.sample(recs, k)
            sampled_steps.extend(sampled)
        low, high = bins[i]
        log.info(f"[Score bin {i} ({low:.1f}–{high:.1f})] {len(recs)} QAs, sampled {k}")

    if invalid_records_info:
        invalid_sample_records = random.sample(invalid_records_info, min(3, len(invalid_records_info)))
    else:
        invalid_sample_records = []

    log.info(
        f"--- AVG Reward: {overall_avg:.3f} | SE: {overall_se:.3f} | "
        f"Valid users: {len(rewards)} | Total QA steps: {len(all_steps)} | "
        f"Sampled {len(sampled_steps)} QAs across {n_score_bins} bins ---"
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
    train_size: int = 6,
    status: str = "train",
    qa_sample_size: Optional[int] = None,
    max_concurrent: int = 5,
    mode: str = "eval",
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
    judge_model: str = "gpt-5-mini",
):
    # 1. Load MemoStructure class from memo file
    try:
        memo_class = find_subclass_in_file(module_path, MemoStructure)
    except Exception as exc:
        log.warning(f"Failed to load memo structure {memory_id}: {exc}")
        file_name = f"{memory_id}_{mode}"
        error_info = {"error_type": type(exc).__name__, "message": str(exc), "trace": traceback.format_exc()}
        get_json("dynamicmem", file_name, [], {"benchmark_overall_eval_score": 0.0, "benchmark_overall_eval_standard_deviation": 0.0}, [error_info])
        return

    log.info(f"Start evaluation for Memory Structure: {memory_id}")

    tracker = init_global_tracker()

    # 2. Get task list
    from envs.dynamicmem_env import get_task_list
    task_list = get_task_list(status=status, train_size=int(train_size))
    log.info(f"Task list ({status}, size={len(task_list)}): {[t[-15:] for t in task_list]}")

    # 3. Build workflow and run
    workflow = DynamicMem_Workflow(
        memo_class=memo_class,
        model=model,
        update_type=update_type,
        n_chunks=n_chunks,
        max_logs=max_logs,
        qa_sample_size=qa_sample_size,
        judge_model=judge_model,
    )

    records, record_len = await workflow.run_all_users(
        task_list=task_list,
        mode=mode,
        max_concurrent=max_concurrent,
    )

    # 4. Process results
    avg_reward, sampled_records, invalid_sample_records = get_meta_eval_info(records, record_len, n_score_bins, samples_per_bin)

    token_tracker = tracker.summary()
    file_name = f"{memory_id}_{mode}"
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
    parser.add_argument("--train_size", type=int, default=6)
    parser.add_argument("--status", default="train", choices=["train", "eval"])
    parser.add_argument("--qa_sample_size", type=int, default=None,
                        help="QA pairs per user (None = use all, recommended for eval)")
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--mode", default="eval", choices=["eval", "test"])
    parser.add_argument("--n_score_bins", type=int, default=3,
                        help="Number of equal-width bins over score range 1–10")
    parser.add_argument("--samples_per_bin", type=int, default=3,
                        help="Max QA trajectories sampled per score bin")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Model used by the LLM judge for scoring QA answers")

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        update_type=args.update_type,
        n_chunks=args.n_chunks,
        max_logs=args.max_logs,
        model=args.model,
        train_size=args.train_size,
        status=args.status,
        qa_sample_size=args.qa_sample_size,
        max_concurrent=args.max_concurrent,
        mode=args.mode,
        n_score_bins=args.n_score_bins,
        samples_per_bin=args.samples_per_bin,
        judge_model=args.judge_model,
    ))
