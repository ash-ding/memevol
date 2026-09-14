"""Meta-Harness CLI entry point.

    cd baselines/evolve/meta-harness
    uv run python run.py --config config.example.yaml
    uv run python run.py --config config.example.yaml --status test --run-name <run>

Config precedence, lowest to highest: DEFAULT_CONFIG < --config YAML < CLI
flags. Every CLI flag defaults to None (the "not given" sentinel), so an unset
flag never clobbers a YAML value. With --config, strict-config is on by
default: the file must list every key in DEFAULT_CONFIG (a null value counts as
listed) and the active sizing block down to its leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[2]
for _path in (str(PROJECT_ROOT), str(BASELINE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from benchmarks.registry import DATASETS
from common.config import (
    ConfigCompletenessError, load_config_file, provided_keys, require_present_keys,
    resolve_config, strict_on,
)
from common.evaluate import missing_sizing_config

from proposer import AGENTS

DEFAULT_CONFIG = {
    # --- run ---
    "status": "search",             # search = evolve; test = finalize one run
    "dataset": "dynamicmem",
    "run_name": None,               # null = timestamp
    "iterations": 10,
    "n_candidates": 3,
    "baselines": ["no_memory", "full_context"],
    "skip_baselines": False,
    # Which systems --status test scores, ALWAYS alongside the baselines.
    # "pareto" is the paper's rule (a final test evaluation on the Pareto
    # frontier); "best" scores only the top-scoring system, which is cheaper
    # but reports no accuracy/context trade-off curve.
    "finalize_systems": "pareto",   # pareto | best
    # --- proposer (the coding agent that writes harnesses) ---
    "agent": "claude_code",         # claude_code | codex
    "agent_model": "opus",
    "agent_effort": None,      # null = paper default per agent (see proposer.DEFAULT_EFFORT)
    "agent_auth": "subscription",   # subscription | api_key (claude_code only)
    "propose_timeout": 2400,
    # --- evaluation (shared evaluator, same knobs as alma) ---
    "execution_model": "gpt-5-mini",
    "judge_model": "gpt-5-mini",
    "max_sample_concurrent": 3,
    "max_eval_concurrent": 2,
    "eval_timeout": 50400,
    "max_logs": None,
    "progressive": True,
    "random_sample": False,
    "sampling_seed": 42,
    "stages": None,
    "single_stage": None,
    "memory_cache": True,
    "strict_config": True,
}

# Sizing blocks are config-file only — no CLI flag anywhere in this repo.
CONFIG_ONLY = {"stages", "single_stage"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Meta-Harness — coding-agent search over memory harnesses (baseline)."
    )
    p.add_argument("--config", default=None, help="YAML config path (CLI flags override it).")
    p.add_argument("--strict-config", dest="strict_config",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Require the config to list every parameter (default on with --config).")

    p.add_argument("--status", choices=["search", "test"], default=None)
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="Which benchmark to evolve on (one per run).")
    p.add_argument("--run-name", dest="run_name", default=None,
                   help="Isolated output dirs under logs/<run_name>/. Required for --status test.")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--agent", choices=list(AGENTS), default=None)
    p.add_argument("--agent-model", dest="agent_model", default=None)
    p.add_argument("--finalize-systems", dest="finalize_systems",
                   choices=["pareto", "best"], default=None,
                   help="Which systems --status test scores, besides the baselines.")
    p.add_argument("--skip-baselines", dest="skip_baselines",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Skip the phase-0 baseline evaluations.")

    # Action flag, not a parameter: never part of the config surface.
    p.add_argument("--fresh", action="store_true",
                   help="Delete this run's logs and every proposed harness before starting.")
    return p.parse_args()


def build_cfg(args: argparse.Namespace) -> dict:
    cli = {k: getattr(args, k, None) for k in DEFAULT_CONFIG if k not in CONFIG_ONLY}
    cli.update({k: None for k in CONFIG_ONLY})
    cfg = resolve_config(DEFAULT_CONFIG, args.config, cli)

    if strict_on(args.config, cfg):
        file_cfg = load_config_file(args.config)
        require_present_keys(provided_keys(file_cfg, cli),
                             set(DEFAULT_CONFIG) - {"strict_config"},
                             context="meta-harness config")
        missing = missing_sizing_config(cfg["dataset"], file_cfg, cfg["progressive"],
                                        path_prefix="")
        if missing:
            raise ConfigCompletenessError(
                f"meta-harness config: missing sizing leaf(s): {sorted(missing)} "
                f"(strict-config mode; set strict_config: false to disable)"
            )

    if cfg["agent"] not in AGENTS:
        raise ValueError(f"unknown agent {cfg['agent']!r}; valid: {list(AGENTS)}")
    if cfg["finalize_systems"] not in ("pareto", "best"):
        raise ValueError(
            f"finalize_systems must be 'pareto' or 'best', got {cfg['finalize_systems']!r}")
    if cfg["status"] == "test" and not cfg["run_name"]:
        raise ValueError("--status test needs the run_name of the run to finalize")
    return cfg


def main() -> None:
    import loop

    args = parse_args()
    cfg = build_cfg(args)
    run_name = cfg["run_name"] or datetime.now().strftime("%Y%m%d_%H%M%S")
    asyncio.run(loop.main(cfg, run_name=run_name, fresh=args.fresh))


if __name__ == "__main__":
    main()
