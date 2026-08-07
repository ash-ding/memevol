"""SimpleMem baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    cd baselines/harness/simplemem && uv run python run.py --config config.example.yaml
    uv run python run.py --config config.example.yaml --progressive
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
from baselines.harness.eval_common import run_baseline, print_result
from baselines.harness.simplemem.memo import SimpleMemMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "locomo",           # SimpleMem is a LoCoMo-native system
    "split": "test",
    "progressive": False,
    "sampling_seed": 42,
    "single_stage": None,          # native YAML dict; REQUIRED when progressive: false
    "stages": None,                # native YAML dict; overrides DEFAULT_STAGES when progressive: true
    "memory_cache": True,
    # --- SimpleMem internal knobs (its own defaults @ db80b6a) ---
    "simplemem_llm_model": "gpt-4.1-mini",   # SimpleMem's compression + retrieval-planning LLM.
                                             # 4-series ONLY — its LLMClient sends temperature=0.1..0.3,
                                             # which the gpt-5 family rejects.
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",   # faithful default (local sentence-transformer)
    "base_url": None,              # OpenAI-compatible base URL for SimpleMem's internal LLM (None = OpenAI)
    "window_size": 40,             # dialogues per compression window (SimpleMem WINDOW_SIZE)
    "overlap_size": 2,             # window overlap (SimpleMem OVERLAP_SIZE)
    "semantic_top_k": 25,          # vector-search hits (SimpleMem SEMANTIC_TOP_K)
    "keyword_top_k": 5,            # keyword-search hits (SimpleMem KEYWORD_TOP_K)
    "structured_top_k": 5,         # structured metadata hits (SimpleMem STRUCTURED_TOP_K)
    "enable_planning": True,       # multi-query intent-aware planning
    "enable_reflection": True,     # reflection-based additional retrieval
    "max_reflection_rounds": 2,    # reflection rounds when enabled
    "enable_parallel_processing": True,   # SimpleMem default — parallel window compression (faithful + fast)
    "max_parallel_workers": 16,    # threads for parallel build (SimpleMem MAX_PARALLEL_WORKERS)
    "enable_parallel_retrieval": True,    # SimpleMem default — parallel multi-view retrieval
    "max_retrieval_workers": 8,    # threads for parallel retrieval (SimpleMem MAX_RETRIEVAL_WORKERS)
    # --- shared eval (baseline convention) ---
    "llm_model": "gpt-5-mini",      # shared QA agent (answers from SimpleMem's retrieved units)
    "judge_model": "gpt-5-mini",   # LLM-as-judge
    "max_sample_concurrent": 3,
    "strict_config": True,
}


def main():
    p = argparse.ArgumentParser(description="SimpleMem baseline — multi-dataset")
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
    p.add_argument("--simplemem_llm_model", default=None,
                   help="SimpleMem's internal LLM (compression + retrieval planning). 4-series ONLY — "
                        "its LLMClient sends a non-default temperature the gpt-5 family rejects.")
    p.add_argument("--embedding_model", default=None,
                   help="SimpleMem embedder (sentence-transformers). Faithful default "
                        "Qwen/Qwen3-Embedding-0.6B; pass all-MiniLM-L6-v2 for the light fallback.")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible base URL for SimpleMem's internal LLM")
    p.add_argument("--window_size", type=int, default=None)
    p.add_argument("--overlap_size", type=int, default=None)
    p.add_argument("--semantic_top_k", type=int, default=None)
    p.add_argument("--keyword_top_k", type=int, default=None)
    p.add_argument("--structured_top_k", type=int, default=None)
    p.add_argument("--enable_planning", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--enable_reflection", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max_reflection_rounds", type=int, default=None)
    p.add_argument("--enable_parallel_processing", action=argparse.BooleanOptionalAction, default=None,
                   help="SimpleMem's parallel window compression (default on). --no-enable_parallel_processing "
                        "to force the serial path (NOT equivalent — serial threads previous-window dedup context).")
    p.add_argument("--max_parallel_workers", type=int, default=None)
    p.add_argument("--enable_parallel_retrieval", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max_retrieval_workers", type=int, default=None)
    p.add_argument("--llm_model", default=None)    # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "simplemem_llm_model": a.simplemem_llm_model, "embedding_model": a.embedding_model,
        "base_url": a.base_url, "window_size": a.window_size, "overlap_size": a.overlap_size,
        "semantic_top_k": a.semantic_top_k, "keyword_top_k": a.keyword_top_k,
        "structured_top_k": a.structured_top_k, "enable_planning": a.enable_planning,
        "enable_reflection": a.enable_reflection, "max_reflection_rounds": a.max_reflection_rounds,
        "enable_parallel_processing": a.enable_parallel_processing, "max_parallel_workers": a.max_parallel_workers,
        "enable_parallel_retrieval": a.enable_parallel_retrieval, "max_retrieval_workers": a.max_retrieval_workers,
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
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="simplemem config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"simplemem config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    memo_config = dict(
        simplemem_llm_model=cfg["simplemem_llm_model"], embedding_model=cfg["embedding_model"],
        base_url=cfg["base_url"], window_size=cfg["window_size"], overlap_size=cfg["overlap_size"],
        semantic_top_k=cfg["semantic_top_k"], keyword_top_k=cfg["keyword_top_k"],
        structured_top_k=cfg["structured_top_k"], enable_planning=cfg["enable_planning"],
        enable_reflection=cfg["enable_reflection"], max_reflection_rounds=cfg["max_reflection_rounds"],
        enable_parallel_processing=cfg["enable_parallel_processing"], max_parallel_workers=cfg["max_parallel_workers"],
        enable_parallel_retrieval=cfg["enable_parallel_retrieval"], max_retrieval_workers=cfg["max_retrieval_workers"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"],
        single_stage=cfg["single_stage"], stages=cfg["stages"],
        memo_class=SimpleMemMemo, memo_config=memo_config, qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir, max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
