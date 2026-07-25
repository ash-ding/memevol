"""
Alma CLI entry point.

Invoke from the project root:

    python baselines/evolve/alma/run.py --status search --steps 10
    python baselines/evolve/alma/run.py --config baselines/evolve/alma/config.example.yaml

Routes to `MetaAgent.forward()` for --status search (meta-learning loop), or
`MetaAgent.run_single_memo()` for --status test (held-out evaluation).

Config precedence (lowest → highest): DEFAULT_CONFIG < --config YAML < CLI flags.
Every CLI flag defaults to None (a sentinel meaning "not given"); the real
defaults live in DEFAULT_CONFIG and are applied by common.config.resolve_config.
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

from common.config import resolve_config

DEFAULT_CONFIG = {
    "meta_model": "gpt-5", "execution_model": "gpt-5-mini", "steps": 10,
    "max_memo_concurrent": 2, "result_dir": "check", "status": "search",
    "dataset": "dynamicmem", "memo_SHA": None, "history_ckpt_path": None,
    "max_logs": None, "max_sample_concurrent": 3, "n_score_bins": 3,
    "samples_per_bin": 3, "judge_model": "gpt-5-mini",
    "progressive": True, "random_sample": False, "sampling_seed": 42,
    "stages": None, "memory_cache": True,
}


def parse_args():
    from baselines.evolve.alma.registry import DATASETS
    parser = argparse.ArgumentParser(description="alma — memory-structure evolution (baseline); one benchmark per run via --dataset.")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (CLI flags override it).")

    parser.add_argument("--meta_model", type=str, default=None)
    parser.add_argument("--execution_model", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_memo_concurrent", type=int, default=None)
    parser.add_argument("--result_dir", type=str, default=None)

    parser.add_argument("--status", type=str, default=None, choices=['search', 'test'])
    parser.add_argument("--dataset", type=str, default=None, choices=DATASETS,
                        help="Which benchmark to evolve on (one per run).")
    parser.add_argument("--memo_SHA", type=str, default=None)
    parser.add_argument("--history_ckpt_path", type=str, default=None)

    parser.add_argument("--max_logs", type=int, default=None)

    parser.add_argument("--max_sample_concurrent", type=int, default=None)
    parser.add_argument("--n_score_bins", type=int, default=None)
    parser.add_argument("--samples_per_bin", type=int, default=None)
    parser.add_argument("--judge_model", type=str, default=None)

    # --- Progressive gauntlet + per-step deterministic sampling (Task 9) ---
    # These REPLACE the removed flat eval_n_samples/eval_n_qa/check_n_samples/
    # check_n_qa knobs: evaluation sizes now live in the shared `stages` schema
    # (common.staged_eval.DEFAULT_STAGES), and each candidate is scored through
    # the same stage1→2→3 gauntlet forge uses.
    # NOTE (Task 4): default=None is the resolve_config sentinel — the real
    # defaults (progressive=True / random_sample=False / memory_cache=True) live
    # in DEFAULT_CONFIG; BooleanOptionalAction still yields True/False when the
    # flag (or its --no- form) is given.
    parser.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=None,
                        help="Evaluate each candidate through the stage1→2→3 gauntlet "
                             "(default). --no-progressive = a single terminal-size pass.")
    parser.add_argument("--random_sample", action=argparse.BooleanOptionalAction, default=None,
                        help="Draw a DIFFERENT deterministic task subset each search step "
                             "(seeded by --sampling_seed + step). Off = fixed prefix.")
    parser.add_argument("--sampling_seed", type=int, default=None,
                        help="Base seed for --random_sample per-step subset selection.")
    parser.add_argument("--stages", type=str, default=None,
                        help="JSON override of the stages block "
                             "(sanity_check/stage1..3); default = family DEFAULT_STAGES.")
    parser.add_argument("--memory_cache", action=argparse.BooleanOptionalAction, default=None,
                        help="Cross-stage Phase-1 memory reuse inside the gauntlet.")

    return parser.parse_args()


def build_cfg(args):
    """Resolve the effective config: DEFAULT_CONFIG < --config YAML < CLI flags.

    Every CLI flag defaults to None ("not given"); only non-None CLI values
    override. --stages is a JSON string on the CLI but a native dict in YAML /
    DEFAULT_CONFIG — parse it here so cfg["stages"] is always a dict-or-None.
    """
    import json

    def _json_or_none(s):
        return json.loads(s) if s else None

    cli = {k: getattr(args, k) for k in DEFAULT_CONFIG if k != "stages"}
    cli["stages"] = _json_or_none(getattr(args, "stages"))
    return resolve_config(DEFAULT_CONFIG, args.config, cli)


async def main(cfg):
    from baselines.evolve.alma.meta_agent import MetaAgent

    stages = cfg["stages"]

    meta_agent = MetaAgent(
        meta_model=cfg["meta_model"],
        execution_model=cfg["execution_model"],
        status=cfg["status"],
        dataset=cfg["dataset"],
        history_ckpt_path=cfg["history_ckpt_path"],
    )

    if cfg["status"] == 'search':
        await meta_agent.forward(
            steps=cfg["steps"],
            max_memo_concurrent=cfg["max_memo_concurrent"],
            max_sample_concurrent=cfg["max_sample_concurrent"],
            result_dir=cfg["result_dir"],
            max_logs=cfg["max_logs"],
            n_score_bins=cfg["n_score_bins"],
            samples_per_bin=cfg["samples_per_bin"],
            judge_model=cfg["judge_model"],
            progressive=cfg["progressive"],
            random_sample=cfg["random_sample"],
            sampling_seed=cfg["sampling_seed"],
            stages=stages,
            memory_cache=cfg["memory_cache"],
        )
    else:
        await meta_agent.run_single_memo(
            memo_SHA=cfg["memo_SHA"],
            status=cfg["status"],
            max_logs=cfg["max_logs"],
            max_sample_concurrent=cfg["max_sample_concurrent"],
            n_score_bins=cfg["n_score_bins"],
            samples_per_bin=cfg["samples_per_bin"],
            judge_model=cfg["judge_model"],
            progressive=cfg["progressive"],
            random_sample=cfg["random_sample"],
            sampling_seed=cfg["sampling_seed"],
            stages=stages,
            memory_cache=cfg["memory_cache"],
        )


if __name__ == "__main__":
    args = parse_args()
    cfg = build_cfg(args)

    logs_dir = PROJECT_ROOT / "baselines" / "evolve" / "alma" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_tag = f"{cfg['status']}_{cfg['dataset']}_{timestamp}"
    os.environ["MEMEVOL_LOG_FILE"] = f"{log_tag}.log"
    # The alma logger honours EVALS_LOG_DIR for RotatingFileHandler output.
    os.environ.setdefault("EVALS_LOG_DIR", str(logs_dir))

    gc.collect()
    asyncio.run(main(cfg))
