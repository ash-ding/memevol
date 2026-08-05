"""LightMem baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    baselines/venv/bin/python baselines/harness/lightmem/run.py --config baselines/harness/lightmem/config.example.yaml
    baselines/venv/bin/python baselines/harness/lightmem/run.py --config baselines/harness/lightmem/config.example.yaml --progressive
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
from baselines.harness.lightmem.memo import LightMemMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "locomo",           # LightMem's headline benchmark
    "split": "test",
    "progressive": False,
    "sampling_seed": 42,
    "single_stage": None,          # native YAML dict; REQUIRED when progressive: false
    "stages": None,                # native YAML dict; overrides DEFAULT_STAGES when progressive: true
    "memory_cache": True,
    # --- LightMem internal knobs (its own experiment defaults @ 34410f4) ---
    "pre_compress": True,          # LLMlingua-2 token pre-compression (a core LightMem stage)
    "topic_segment": True,         # attention-based topic segmentation (shares the LLMlingua-2 model)
    "llmlingua_model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "llmlingua_device": "cuda",    # device_map for the LLMlingua-2 model (set "cpu" if no GPU)
    "compress_rate": 0.6,          # LLMlingua-2 target compression rate (LoCoMo experiment value)
    "messages_use": "user_only",   # which turns feed extraction (user_only | assistant_only | hybrid)
    "extract_threshold": 0.1,      # segmentation/extraction trigger threshold (LoCoMo experiment value)
    "extraction_mode": "flat",     # flat (factual entries) | event (factual + relational)
    "lightmem_llm_model": "gpt-4o-mini",   # LightMem's internal extraction/update LLM.
                                           # 4-series ONLY — it sends temperature=0.1, which the gpt-5 family rejects.
    "manager_max_tokens": 16000,   # max_tokens for the internal LLM (LoCoMo experiment value)
    "base_url": None,              # OpenAI-compatible base URL for LightMem's internal LLM (None = OpenAI)
    "embedding_model": "all-MiniLM-L6-v2",   # LightMem's HF sentence-transformer embedder (LongMemEval default)
    "embedding_dims": 384,         # embedding dimension (must match the embedder)
    "embedding_device": "cuda",    # device for the embedder (set "cpu" if no GPU)
    "offline_update": True,        # run the offline-update refinement phase after build (LoCoMo-paper pipeline)
    "update_sim_threshold": 0.9,   # score_threshold for offline_update_all_entries (LoCoMo experiment value)
    "retrieve_limit": 20,          # top-k memories LightMemory.retrieve returns (LongMemEval driver value)
    # --- shared eval (baseline convention) ---
    "llm_model": "gpt-5-mini",     # shared QA agent (answers from LightMem's retrieved memories)
    "judge_model": "gpt-5-mini",   # LLM-as-judge
    "max_sample_concurrent": 3,
    "strict_config": True,
}


def main():
    p = argparse.ArgumentParser(description="LightMem baseline — multi-dataset")
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
    p.add_argument("--pre_compress", action=argparse.BooleanOptionalAction, default=None,
                   help="LLMlingua-2 pre-compression (default on). --no-pre_compress to disable.")
    p.add_argument("--topic_segment", action=argparse.BooleanOptionalAction, default=None,
                   help="Topic segmentation (default on; requires pre_compress — shared LLMlingua-2 model).")
    p.add_argument("--llmlingua_model", default=None,
                   help="LLMlingua-2 model (HF hub id or local path). Only used when pre_compress is on.")
    p.add_argument("--llmlingua_device", default=None, help="device_map for LLMlingua-2 (cuda | cpu)")
    p.add_argument("--compress_rate", type=float, default=None)
    p.add_argument("--messages_use", default=None, choices=["user_only", "assistant_only", "hybrid"])
    p.add_argument("--extract_threshold", type=float, default=None)
    p.add_argument("--extraction_mode", default=None, choices=["flat", "event"])
    p.add_argument("--lightmem_llm_model", default=None,
                   help="LightMem's internal LLM (extraction + offline update). 4-series ONLY — "
                        "it sends temperature=0.1 which the gpt-5 family rejects.")
    p.add_argument("--manager_max_tokens", type=int, default=None)
    p.add_argument("--base_url", default=None, help="OpenAI-compatible base URL for LightMem's internal LLM")
    p.add_argument("--embedding_model", default=None,
                   help="LightMem embedder (sentence-transformers). Default all-MiniLM-L6-v2 (384-dim).")
    p.add_argument("--embedding_dims", type=int, default=None)
    p.add_argument("--embedding_device", default=None, help="device for the embedder (cuda | cpu)")
    p.add_argument("--offline_update", action=argparse.BooleanOptionalAction, default=None,
                   help="Offline-update refinement phase after build (default on). --no-offline_update to skip.")
    p.add_argument("--update_sim_threshold", type=float, default=None)
    p.add_argument("--retrieve_limit", type=int, default=None)
    p.add_argument("--llm_model", default=None)    # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "pre_compress": a.pre_compress, "topic_segment": a.topic_segment,
        "llmlingua_model": a.llmlingua_model, "llmlingua_device": a.llmlingua_device,
        "compress_rate": a.compress_rate, "messages_use": a.messages_use,
        "extract_threshold": a.extract_threshold, "extraction_mode": a.extraction_mode,
        "lightmem_llm_model": a.lightmem_llm_model, "manager_max_tokens": a.manager_max_tokens,
        "base_url": a.base_url, "embedding_model": a.embedding_model,
        "embedding_dims": a.embedding_dims, "embedding_device": a.embedding_device,
        "offline_update": a.offline_update, "update_sim_threshold": a.update_sim_threshold,
        "retrieve_limit": a.retrieve_limit,
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
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="lightmem config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"lightmem config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    memo_class = make_memo_class(
        LightMemMemo,
        pre_compress=cfg["pre_compress"], topic_segment=cfg["topic_segment"],
        llmlingua_model=cfg["llmlingua_model"], llmlingua_device=cfg["llmlingua_device"],
        compress_rate=cfg["compress_rate"], messages_use=cfg["messages_use"],
        extract_threshold=cfg["extract_threshold"], extraction_mode=cfg["extraction_mode"],
        lightmem_llm_model=cfg["lightmem_llm_model"], manager_max_tokens=cfg["manager_max_tokens"],
        base_url=cfg["base_url"], embedding_model=cfg["embedding_model"],
        embedding_dims=cfg["embedding_dims"], embedding_device=cfg["embedding_device"],
        offline_update=cfg["offline_update"], update_sim_threshold=cfg["update_sim_threshold"],
        retrieve_limit=cfg["retrieve_limit"],
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
