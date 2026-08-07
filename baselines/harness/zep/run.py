"""Zep (Graphiti temporal knowledge graph) baseline — evaluate on one benchmark's
split, comparable to the main method (same split/judge/scoring via the per-dataset
workflow). Backend: embedded FalkorDB Lite (no server). Embedder/reranker default
to paper-faithful BGE-m3.

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    cd baselines/harness/zep && uv run python run.py --config config.example.yaml
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
from baselines.harness.zep.memo import ZepMemo
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
    "retrieve_k",
    "embedder",
    "embedder_model",
    "reranker",
    "reranker_model",
    "device",
    "db_root",
    "graph_llm_model",
    "graph_llm_small_model",
    "llm_model",
    "judge_model",
    "max_sample_concurrent",
})


def main():
    p = argparse.ArgumentParser(description="Zep (Graphiti) baseline — multi-dataset")
    p.add_argument("--config", required=True,
                   help="YAML config file — the ONLY parameter surface "
                        "(no CLI overrides). Copy config.example.yaml and edit.")
    a = p.parse_args()

    cfg = validate_exact_config(load_config_file(a.config) or {},
                                REQUIRED_KEYS, context="zep config")

    memo_config = dict(
        retrieve_k=cfg["retrieve_k"],
        embedder=cfg["embedder"], embedder_model=cfg["embedder_model"],
        reranker=cfg["reranker"], reranker_model=cfg["reranker_model"],
        device=cfg["device"], db_root=cfg["db_root"],
        graph_llm_model=cfg["graph_llm_model"], graph_llm_small_model=cfg["graph_llm_small_model"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"],
        single_stage=cfg["single_stage"], stages=cfg["stages"],
        memo_class=ZepMemo, memo_config=memo_config, qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir, max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
