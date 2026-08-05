"""Zep (Graphiti temporal knowledge graph) baseline — evaluate on one benchmark's
split, comparable to the main method (same split/judge/scoring via the per-dataset
workflow). Backend: embedded FalkorDB Lite (no server). Embedder/reranker default
to paper-faithful BGE-m3.

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py --config baselines/harness/zep/config.example.yaml
    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py --config baselines/harness/zep/config.example.yaml --progressive
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
from baselines.harness.eval_common import make_memo_class, run_baseline, print_result
from baselines.harness.zep.memo import ZepMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "locomo",
    "split": "test",
    "progressive": False,
    "sampling_seed": 42,
    "single_stage": None,        # native YAML dict; REQUIRED when progressive: false
    "stages": None,              # native YAML dict; overrides DEFAULT_STAGES when progressive: true
    "memory_cache": True,
    # Zep/Graphiti knobs
    "retrieve_k": 20,            # paper: top-20 edges (facts) + entity nodes (summaries)
    "embedder": "bge-m3",        # bge-m3 (paper-faithful, local) | openai
    "embedder_model": "BAAI/bge-m3",
    "reranker": "bge",           # bge (paper-faithful cross-encoder) | openai
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "device": None,              # sentence-transformers device: cuda | cpu | null(auto)
    "db_root": None,             # dir for the embedded FalkorDB Lite store; null → system temp.
                                 # MUST be a native POSIX FS (redislite unix socket) — NOT /mnt/c under WSL.
    "graph_llm_model": "gpt-4o-mini",       # Graphiti graph-construction LLM (paper: gpt-4o-mini; 4-series)
    "graph_llm_small_model": None,          # Graphiti "small" LLM; null → same as graph_llm_model
    # Shared eval knobs
    "llm_model": "gpt-5-mini",              # shared QA agent model
    "judge_model": "gpt-5-mini",            # LLM-as-judge model
    "max_sample_concurrent": 3,
    "strict_config": True,
}


def main():
    p = argparse.ArgumentParser(description="Zep (Graphiti) baseline — multi-dataset")
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
    p.add_argument("--retrieve_k", type=int, default=None)
    p.add_argument("--embedder", default=None, choices=["bge-m3", "openai"])
    p.add_argument("--embedder_model", default=None)
    p.add_argument("--reranker", default=None, choices=["bge", "openai"])
    p.add_argument("--reranker_model", default=None)
    p.add_argument("--device", default=None, help="sentence-transformers device: cuda | cpu (default auto)")
    p.add_argument("--db_root", default=None,
                   help="dir for the embedded FalkorDB Lite store (native POSIX FS; NOT /mnt/c under WSL). Default: system temp.")
    p.add_argument("--graph_llm_model", default=None,
                   help="Graphiti graph-construction LLM (paper: gpt-4o-mini; keep a 4-series model)")
    p.add_argument("--graph_llm_small_model", default=None)
    p.add_argument("--llm_model", default=None)     # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "retrieve_k": a.retrieve_k,
        "embedder": a.embedder, "embedder_model": a.embedder_model,
        "reranker": a.reranker, "reranker_model": a.reranker_model,
        "device": a.device, "db_root": a.db_root,
        "graph_llm_model": a.graph_llm_model, "graph_llm_small_model": a.graph_llm_small_model,
        "llm_model": a.llm_model, "judge_model": a.judge_model,
        "max_sample_concurrent": a.max_sample_concurrent,
        "strict_config": a.strict_config,
    }
    cfg = resolve_config(DEFAULT_CONFIG, a.config, cli)

    from common.config import strict_on, load_config_file, provided_keys, require_present_keys, ConfigCompletenessError
    from common.staged_eval import missing_sizing_config
    if strict_on(a.config, cfg):
        _fc = load_config_file(a.config)
        require_present_keys(provided_keys(_fc, cli),
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="zep config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"zep config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    memo_class = make_memo_class(
        ZepMemo,
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
        memo_class=memo_class, qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir, max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
        memory_cache=cfg["memory_cache"],
    ))
    print_result(cfg["dataset"], cfg["progressive"], result, out_dir)


if __name__ == "__main__":
    main()
