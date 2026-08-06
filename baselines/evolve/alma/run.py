"""
Alma CLI entry point.

Invoke from this directory (baselines/evolve/alma/):

    cd baselines/evolve/alma && uv run python run.py --status search --steps 10
    uv run python run.py --config config.example.yaml

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
    # Evaluation SIZES are config-file-only (no CLI flag): `stages` overrides the
    # progressive gauntlet's family DEFAULT_STAGES; `single_stage` sizes the
    # progressive=false single pass (REQUIRED when progressive=false).
    "stages": None, "single_stage": None, "memory_cache": True,
    "strict_config": True,
}


def parse_args():
    from baselines.evolve.alma.registry import DATASETS
    parser = argparse.ArgumentParser(description="alma — memory-structure evolution (baseline); one benchmark per run via --dataset.")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (CLI flags override it).")
    parser.add_argument("--strict-config", dest="strict_config",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Require the config to list every parameter (default on when --config is given). "
                             "--no-strict-config to disable.")

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
    # (common.evaluate.DEFAULT_STAGES), and each candidate is scored through
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
    # Evaluation SIZES (`stages` / `single_stage`) are config-file-only — set them
    # in the --config YAML, not on the CLI. `stages` overrides the gauntlet's
    # family DEFAULT_STAGES; `single_stage` sizes the progressive=false single
    # pass (required when --no-progressive). The old `--stages` CLI flag is gone.
    parser.add_argument("--memory_cache", action=argparse.BooleanOptionalAction, default=None,
                        help="Cross-stage Phase-1 memory reuse inside the gauntlet.")

    return parser.parse_args()


def build_cfg(args):
    """Resolve the effective config: DEFAULT_CONFIG < --config YAML < CLI flags.

    Every CLI flag defaults to None ("not given"); only non-None CLI values
    override. `stages` / `single_stage` are config-file-only (native YAML dicts,
    no CLI flag) — kept out of the CLI overlay so their YAML value survives.
    """
    _config_only = {"stages", "single_stage"}
    cli = {k: getattr(args, k) for k in DEFAULT_CONFIG if k not in _config_only}
    cli.update({k: None for k in _config_only})
    cfg = resolve_config(DEFAULT_CONFIG, args.config, cli)

    from common.config import strict_on, load_config_file, provided_keys, require_present_keys, ConfigCompletenessError
    from common.evaluate import missing_sizing_config
    if strict_on(args.config, cfg):
        _fc = load_config_file(args.config)
        require_present_keys(provided_keys(_fc, cli),
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="alma config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"alma config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    return cfg


async def main(cfg):
    from baselines.evolve.alma.meta_agent import MetaAgent

    stages = cfg["stages"]
    single_stage = cfg["single_stage"]

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
            single_stage=single_stage,
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
            single_stage=single_stage,
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
