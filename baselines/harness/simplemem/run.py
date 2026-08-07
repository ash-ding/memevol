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
from baselines.harness.eval_utility import run_baseline, print_result
from baselines.harness.simplemem.memo import SimpleMemMemo
from common.config import load_config_file, validate_exact_config

# The config file must list EXACTLY these keys (a null value counts as
# listed; sizing leaves are checked separately) — no CLI overrides, no
# built-in defaults. Copy config.example.yaml and edit.
REQUIRED_KEYS = frozenset({
    "dataset",
    "split",
    "progressive",
    "sampling_seed",
    "single_stage",
    "stages",
    "memory_cache",
    "simplemem_llm_model",
    "embedding_model",
    "base_url",
    "window_size",
    "overlap_size",
    "semantic_top_k",
    "keyword_top_k",
    "structured_top_k",
    "enable_planning",
    "enable_reflection",
    "max_reflection_rounds",
    "enable_parallel_processing",
    "max_parallel_workers",
    "enable_parallel_retrieval",
    "max_retrieval_workers",
    "llm_model",
    "judge_model",
    "max_sample_concurrent",
})


def main():
    p = argparse.ArgumentParser(description="SimpleMem baseline — multi-dataset")
    p.add_argument("--config", required=True,
                   help="YAML config file — the ONLY parameter surface "
                        "(no CLI overrides). Copy config.example.yaml and edit.")
    a = p.parse_args()

    cfg = validate_exact_config(load_config_file(a.config) or {},
                                REQUIRED_KEYS, context="simplemem config")

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
