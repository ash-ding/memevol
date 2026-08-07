"""Mem0 baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    uv run --project baselines/harness/mem0 python baselines/harness/mem0/run.py --config baselines/harness/mem0/config.example.yaml
    uv run --project baselines/harness/mem0 python baselines/harness/mem0/run.py --config baselines/harness/mem0/config.example.yaml --progressive
"""
from __future__ import annotations
import argparse, asyncio, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
from baselines.registry import DATASETS
from baselines.harness.eval_utility import run_baseline, print_result
from baselines.harness.mem0.memo import Mem0Memo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "locomo",           # Mem0 reports on LoCoMo in its own paper
    "split": "test",
    "progressive": False,
    "sampling_seed": 42,
    "single_stage": None,          # native YAML dict; REQUIRED when progressive: false
    "stages": None,                # native YAML dict; overrides DEFAULT_STAGES when progressive: true
    "memory_cache": True,
    # --- Mem0 internal knobs (its own defaults) ---
    "mem0_llm_model": "gpt-4o-mini",   # Mem0's extractor/updater LLM (decides ADD/UPDATE/DELETE per fact)
    "embedding_model": "text-embedding-3-small",   # Mem0's embedder (OpenAI provider)
    "base_url": None,              # OpenAI-compatible base URL for Mem0's internal LLM (None = OpenAI)
    "add_batch_size": 20,          # messages per Memory.add() extraction call
    "infer": True,                 # True = LLM fact extraction (Mem0's contribution); False = store raw
    "top_k": 10,                   # Memory.search hits/query. Paper: s=10 (library default is 20)
    "threshold": 0.0,              # min similarity for a hit (0 = keep all top_k; Mem0 default 0.1)
    # --- shared eval (baseline convention) ---
    "llm_model": "gpt-5-mini",     # shared QA agent (answers from Mem0's retrieved units)
    "judge_model": "gpt-5-mini",   # LLM-as-judge
    "max_sample_concurrent": 3,
    "strict_config": True,
}


def main():
    p = argparse.ArgumentParser(description="Mem0 baseline — multi-dataset")
    p.add_argument("--config", default=None, help="YAML config path (CLI flags override it)")
    p.add_argument("--strict-config", dest="strict_config",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Require the config to list every parameter (default on when --config is given). "
                        "--no-strict-config to disable.")
    p.add_argument("--dataset", default=None, choices=DATASETS)
    p.add_argument("--split", default=None, choices=["test", "search"])
    p.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sampling-seed", dest="sampling_seed", type=int, default=None)
    p.add_argument("--memory-cache", dest="memory_cache", action=argparse.BooleanOptionalAction, default=None,
                   help="Cross-stage Phase-1 memory reuse (default on). --no-memory-cache to disable.")
    p.add_argument("--mem0_llm_model", default=None,
                   help="Mem0's internal extractor/updater LLM.")
    p.add_argument("--embedding_model", default=None)
    p.add_argument("--base_url", default=None, help="OpenAI-compatible base URL for Mem0's internal LLM")
    p.add_argument("--add_batch_size", type=int, default=None)
    p.add_argument("--infer", action=argparse.BooleanOptionalAction, default=None,
                   help="LLM fact extraction on add (default on). --no-infer stores messages verbatim.")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--llm_model", default=None)    # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "mem0_llm_model": a.mem0_llm_model, "embedding_model": a.embedding_model,
        "base_url": a.base_url, "add_batch_size": a.add_batch_size, "infer": a.infer,
        "top_k": a.top_k, "threshold": a.threshold,
        "llm_model": a.llm_model, "judge_model": a.judge_model,
        "max_sample_concurrent": a.max_sample_concurrent,
        "strict_config": a.strict_config,
    }
    cfg = resolve_config(DEFAULT_CONFIG, a.config, cli)

    from common.config import strict_on, load_config_file, provided_keys, require_present_keys, ConfigCompletenessError
    from common.evaluate import missing_sizing_config
    if strict_on(a.config, cfg):
        _fc = load_config_file(a.config)
        require_present_keys(provided_keys(_fc, cli),
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="mem0 config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"mem0 config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    memo_config = dict(
        mem0_llm_model=cfg["mem0_llm_model"], embedding_model=cfg["embedding_model"],
        base_url=cfg["base_url"], add_batch_size=cfg["add_batch_size"], infer=cfg["infer"],
        top_k=cfg["top_k"], threshold=cfg["threshold"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"],
        single_stage=cfg["single_stage"],
        memo_class=Mem0Memo, memo_config=memo_config,
        qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir,
        max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"],
        sampling_seed=cfg["sampling_seed"],
        stages=cfg["stages"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
