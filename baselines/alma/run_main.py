"""
Alma CLI entry point.

Invoke from the project root:

    python baselines/alma/run_main.py --status search --steps 10

Routes to `MetaAgent.forward()` for --status search (meta-learning loop), or
`MetaAgent.run_single_memo()` for --status test (held-out evaluation).
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def parse_args():
    parser = argparse.ArgumentParser(description="alma — DynamicMem memory evolution (baseline). DynamicMem-ONLY: no multi-dataset support (that is a forge-only capability).")

    parser.add_argument("--meta_model", type=str, default="gpt-5")
    parser.add_argument("--execution_model", type=str, default="gpt-5-mini")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_memo_concurrent", type=int, default=2)
    parser.add_argument("--result_dir", type=str, default="check")

    parser.add_argument("--status", type=str, default='search', choices=['search', 'test'])
    parser.add_argument("--eval_n_samples", type=int, default=6)
    parser.add_argument("--memo_SHA", type=str, default=None)
    parser.add_argument("--history_ckpt_path", type=str, default=None)

    parser.add_argument("--update_type", type=str, default="all_at_once",
                        choices=["all_at_once", "chunked", "sequential"])
    parser.add_argument("--n_chunks", type=int, default=5)
    parser.add_argument("--max_logs", type=int, default=None)

    # Default: 20 QA per user (deterministically seeded by user_dir) — ~6x faster than
    # the full 178 QA, trades a bit of reward stability for throughput during search.
    # Pass an explicit large value (or a flag like `--eval_n_qa 178`) for full eval.
    parser.add_argument("--eval_n_qa", type=int, default=20)
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--n_score_bins", type=int, default=3)
    parser.add_argument("--samples_per_bin", type=int, default=3)
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")
    # Default: 6 users × 3 QA — full user coverage for Phase 1 bugs + minimal Phase 2 probe
    parser.add_argument("--check_n_samples", type=int, default=6)
    parser.add_argument("--check_n_qa", type=int, default=3)

    return parser.parse_args()


async def main(args):
    from baselines.alma.meta_agent import MetaAgent

    meta_agent = MetaAgent(
        meta_model=args.meta_model,
        execution_model=args.execution_model,
        status=args.status,
        history_ckpt_path=args.history_ckpt_path,
    )

    if args.status == 'search':
        await meta_agent.forward(
            steps=args.steps,
            max_memo_concurrent=args.max_memo_concurrent,
            max_sample_concurrent=args.max_sample_concurrent,
            result_dir=args.result_dir,
            eval_n_samples=args.eval_n_samples,
            update_type=args.update_type,
            n_chunks=args.n_chunks,
            max_logs=args.max_logs,
            eval_n_qa=args.eval_n_qa,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
            check_n_samples=args.check_n_samples,
            check_n_qa=args.check_n_qa,
        )
    else:
        await meta_agent.run_single_memo(
            memo_SHA=args.memo_SHA,
            status=args.status,
            eval_n_samples=args.eval_n_samples,
            update_type=args.update_type,
            n_chunks=args.n_chunks,
            max_logs=args.max_logs,
            eval_n_qa=args.eval_n_qa,
            max_sample_concurrent=args.max_sample_concurrent,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    args = parse_args()

    logs_dir = PROJECT_ROOT / "baselines" / "alma" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_tag = f"{args.status}_{args.update_type}"
    if args.update_type == "chunked":
        log_tag += f"_{args.n_chunks}"
    log_tag += f"_{timestamp}"
    os.environ["MEMEVOL_LOG_FILE"] = f"{log_tag}.log"
    # The alma logger honours EVALS_LOG_DIR for RotatingFileHandler output.
    os.environ.setdefault("EVALS_LOG_DIR", str(logs_dir))

    gc.collect()
    asyncio.run(main(args))
