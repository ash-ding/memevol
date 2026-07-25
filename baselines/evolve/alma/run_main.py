"""
Alma CLI entry point.

Invoke from the project root:

    python baselines/evolve/alma/run_main.py --status search --steps 10

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def parse_args():
    from baselines.evolve.alma.registry import DATASETS
    parser = argparse.ArgumentParser(description="alma — memory-structure evolution (baseline); one benchmark per run via --dataset.")

    parser.add_argument("--meta_model", type=str, default="gpt-5")
    parser.add_argument("--execution_model", type=str, default="gpt-5-mini")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_memo_concurrent", type=int, default=2)
    parser.add_argument("--result_dir", type=str, default="check")

    parser.add_argument("--status", type=str, default='search', choices=['search', 'test'])
    parser.add_argument("--dataset", type=str, default="dynamicmem", choices=DATASETS,
                        help="Which benchmark to evolve on (one per run).")
    parser.add_argument("--memo_SHA", type=str, default=None)
    parser.add_argument("--history_ckpt_path", type=str, default=None)

    parser.add_argument("--max_logs", type=int, default=None)

    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--n_score_bins", type=int, default=3)
    parser.add_argument("--samples_per_bin", type=int, default=3)
    parser.add_argument("--judge_model", type=str, default="gpt-5-mini")

    # --- Progressive gauntlet + per-step deterministic sampling (Task 9) ---
    # These REPLACE the removed flat eval_n_samples/eval_n_qa/check_n_samples/
    # check_n_qa knobs: evaluation sizes now live in the shared `stages` schema
    # (common.staged_eval.DEFAULT_STAGES), and each candidate is scored through
    # the same stage1→2→3 gauntlet forge uses.
    parser.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=True,
                        help="Evaluate each candidate through the stage1→2→3 gauntlet "
                             "(default). --no-progressive = a single terminal-size pass.")
    parser.add_argument("--random_sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Draw a DIFFERENT deterministic task subset each search step "
                             "(seeded by --sampling_seed + step). Off = fixed prefix.")
    parser.add_argument("--sampling_seed", type=int, default=42,
                        help="Base seed for --random_sample per-step subset selection.")
    parser.add_argument("--stages", type=str, default=None,
                        help="JSON override of the stages block "
                             "(sanity_check/stage1..3); default = family DEFAULT_STAGES.")
    parser.add_argument("--memory_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-stage Phase-1 memory reuse inside the gauntlet.")

    return parser.parse_args()


async def main(args):
    import json
    from baselines.evolve.alma.meta_agent import MetaAgent

    stages = json.loads(args.stages) if args.stages else None

    meta_agent = MetaAgent(
        meta_model=args.meta_model,
        execution_model=args.execution_model,
        status=args.status,
        dataset=args.dataset,
        history_ckpt_path=args.history_ckpt_path,
    )

    if args.status == 'search':
        await meta_agent.forward(
            steps=args.steps,
            max_memo_concurrent=args.max_memo_concurrent,
            max_sample_concurrent=args.max_sample_concurrent,
            result_dir=args.result_dir,
            max_logs=args.max_logs,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
            progressive=args.progressive,
            random_sample=args.random_sample,
            sampling_seed=args.sampling_seed,
            stages=stages,
            memory_cache=args.memory_cache,
        )
    else:
        await meta_agent.run_single_memo(
            memo_SHA=args.memo_SHA,
            status=args.status,
            max_logs=args.max_logs,
            max_sample_concurrent=args.max_sample_concurrent,
            n_score_bins=args.n_score_bins,
            samples_per_bin=args.samples_per_bin,
            judge_model=args.judge_model,
            progressive=args.progressive,
            random_sample=args.random_sample,
            sampling_seed=args.sampling_seed,
            stages=stages,
            memory_cache=args.memory_cache,
        )


if __name__ == "__main__":
    args = parse_args()

    logs_dir = PROJECT_ROOT / "baselines" / "evolve" / "alma" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_tag = f"{args.status}_{args.dataset}_{timestamp}"
    os.environ["MEMEVOL_LOG_FILE"] = f"{log_tag}.log"
    # The alma logger honours EVALS_LOG_DIR for RotatingFileHandler output.
    os.environ.setdefault("EVALS_LOG_DIR", str(logs_dir))

    gc.collect()
    asyncio.run(main(args))
