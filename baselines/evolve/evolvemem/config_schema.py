"""Config surface for the evolvemem baseline — one schema, two entry points.

`evolve.py` (search) and `run.py` (scoring) read the SAME config file and the
same key set, so a theta can never be evolved under one set of assumptions and
scored under another.

Validation is `common.config.validate_exact_config`, the repo-wide strict check:
a missing key OR an unknown key aborts before anything executes, and a `null`
value counts as listed. Sizing lives in `single_stage:` / `stages:` (config-file
only), matching every other baseline. Per `baselines/README.md`, an evolve
baseline may also carry genuine runtime knobs on the CLI — here that is
`--max-rounds` (evolve.py) and `--theta` (run.py), and nothing else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

REQUIRED_KEYS = frozenset({
    # what / where
    "dataset",
    "split",
    "progressive",
    "single_stage",
    "stages",
    "sampling_seed",
    "memory_cache",
    # the search loop (evolve.py)
    "max_rounds",
    "initial_config",
    "convergence_threshold",
    "evolve_llm_model",
    "extraction_window_size",
    "extraction_overlap",
    # the artifact (memo.py)
    "theta_path",
    "embedding_model",
    "honor_answer_policy",
    "benchmark_adapter",
    # shared eval (baseline convention)
    "llm_model",
    "judge_model",
    "max_sample_concurrent",
})


def load_and_validate(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and strict-validate it against REQUIRED_KEYS."""
    from common.config import load_config_file, validate_exact_config

    cfg = load_config_file(str(path)) or {}
    return validate_exact_config(cfg, REQUIRED_KEYS, context="evolvemem config")


def memo_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The subset `EvolveMemMemo` reads. Keeping this in one place is what stops
    run.py and evolve.py from drifting apart."""
    return {
        "dataset": cfg["dataset"],
        "theta_path": cfg["theta_path"],
        "initial_config": cfg["initial_config"],
        "embedding_model": cfg["embedding_model"],
        "evolve_llm_model": cfg["evolve_llm_model"],
        "honor_answer_policy": cfg["honor_answer_policy"],
        "benchmark_adapter": cfg["benchmark_adapter"],
        "extraction_window_size": cfg["extraction_window_size"],
        "extraction_overlap": cfg["extraction_overlap"],
    }
