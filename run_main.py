import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Add evals folder to sys.path so submodule imports work
evals_path = Path(__file__).resolve().parent / "evals"
sys.path.insert(0, str(evals_path))

# Add project root to sys.path so eval_runner and envs can be found
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import gc
gc.collect()

from core.meta_agent import MetaAgent


async def main(args):
    meta_agent = MetaAgent(
        meta_model=args.meta_model,
        execution_model=args.execution_model,
        status=args.status,
        history_ckpt_path=args.history_ckpt_path,
    )

    if args.status == 'train':
        await meta_agent.forward(
            steps=args.steps,
            max_concurrent=args.max_container_concurrent,
            result_dir=args.result_dir,
            train_size=args.train_size,
            update_type=args.update_type,
            n_chunks=args.n_chunks,
            max_logs=args.max_logs,
            qa_sample_size=args.qa_sample_size,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
        )
    else:
        await meta_agent.run_single_memo(
            memo_SHA=args.memo_SHA,
            status=args.status,
            train_size=args.train_size,
            update_type=args.update_type,
            n_chunks=args.n_chunks,
            max_logs=args.max_logs,
            qa_sample_size=args.qa_sample_size,
            max_concurrent=args.max_container_concurrent,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="memevol — DynamicMem memory evolution")

    # Meta-learning
    parser.add_argument("--meta_model", type=str, default="gpt-5",
                        help="Model used by MetaAgent for analysis and code generation")
    parser.add_argument("--execution_model", type=str, default="gpt-5-mini",
                        help="Model used by the downstream QA agent")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of meta-learning iterations")
    parser.add_argument("--max_container_concurrent", type=int, default=5,
                        help="Max parallel memo evaluations in meta-learning loop")
    parser.add_argument("--result_dir", type=str, default="check",
                        help="Prefix for checkpoint JSON filenames in logs/")

    # Evaluation settings
    parser.add_argument("--status", type=str, default='train',
                        choices=['train', 'eval'],
                        help="Data split: 'train' uses users 001-006 (6 users), 'eval' uses users 007-010 (4 users)")
    parser.add_argument("--train_size", type=int, default=6,
                        help="Number of train users per evaluation round (max 6)")
    parser.add_argument("--memo_SHA", type=str, default=None,
                        help="SHA of a specific memo structure to evaluate (eval mode only)")
    parser.add_argument("--history_ckpt_path", type=str, default=None,
                        help="Checkpoint JSON file in logs/ to resume from")

    # Update-type hyperparameters
    parser.add_argument("--update_type", type=str, default="all_at_once",
                        choices=["all_at_once", "chunked", "sequential"],
                        help="How app_logs are batched for general_update calls")
    parser.add_argument("--n_chunks", type=int, default=5,
                        help="Number of chunks (used when update_type=chunked)")
    parser.add_argument("--max_logs", type=int, default=None,
                        help="Max app_log entries (truncate oldest; used when update_type=all_at_once)")

    # QA evaluation settings
    parser.add_argument("--qa_sample_size", type=int, default=None,
                        help="QA pairs per user (default None = use all; set e.g. 20 during training to control cost)")
    parser.add_argument("--max_concurrent", type=int, default=5,
                        help="Max parallel users within a single evaluation subprocess")
    parser.add_argument("--n_score_bins", type=int, default=3,
                        help="Number of equal-width bins over score range 1–10 for trajectory sampling")
    parser.add_argument("--samples_per_bin", type=int, default=3,
                        help="Max QA trajectories sampled per score bin for meta-agent analysis")
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini",
                        help="Model used by the LLM judge for scoring QA answers")

    args = parser.parse_args()

    # Ensure logs directory exists
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    asyncio.run(main(args))
