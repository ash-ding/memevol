"""MemoryOS baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    baselines/harness/memoryos/venv/bin/python baselines/harness/memoryos/run.py --config baselines/harness/memoryos/config.example.yaml
    baselines/harness/memoryos/venv/bin/python baselines/harness/memoryos/run.py --config baselines/harness/memoryos/config.example.yaml --progressive
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
from baselines.harness.memoryos.memo import MemoryOSMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "locomo",           # MemoryOS reports on LoCoMo (+ GVD, not in this repo)
    "split": "test",
    "progressive": False,
    "sampling_seed": 42,
    "single_stage": None,          # native YAML dict; REQUIRED when progressive: false
    "stages": None,                # native YAML dict; overrides DEFAULT_STAGES when progressive: true
    "memory_cache": True,
    # --- MemoryOS internal knobs (vendored defaults; paper values in comments) ---
    "memoryos_llm_model": "gpt-4o-mini",   # MemoryOS's own LLM: page/segment summarisation,
                                           # keyword extraction, persona + knowledge distillation.
                                           # The paper's headline numbers are on gpt-4o-mini.
    "base_url": None,              # OpenAI-compatible base URL for MemoryOS's internal LLM (None = OpenAI)
    "short_term_capacity": 7,      # STM dialogue-page queue length (paper: 7; vendored default: 10)
    "mid_term_capacity": 2000,     # max MTM segments (vendored default; paper says 200)
    "mid_term_heat_threshold": 5.0,        # tau — Heat above which a segment is distilled into LPM (paper: 5)
    "mid_term_similarity_threshold": 0.6,  # theta in F_score > theta for page->segment merge (paper: 0.6)
    "retrieval_queue_capacity": 7,         # retrieved MTM pages handed back per query
    "long_term_knowledge_capacity": 100,   # FIFO capacity of User KB / Assistant Traits (paper: 100)
    # --- shared eval (baseline convention) ---
    "llm_model": "gpt-5-mini",     # shared QA agent (answers from MemoryOS's retrieved units)
    "judge_model": "gpt-5-mini",   # LLM-as-judge
    "max_sample_concurrent": 3,
    "strict_config": True,
}


def main():
    p = argparse.ArgumentParser(description="MemoryOS baseline — multi-dataset")
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
    p.add_argument("--memoryos_llm_model", default=None,
                   help="MemoryOS's internal LLM (summarisation + persona/knowledge distillation).")
    p.add_argument("--base_url", default=None, help="OpenAI-compatible base URL for MemoryOS's internal LLM")
    p.add_argument("--short_term_capacity", type=int, default=None)
    p.add_argument("--mid_term_capacity", type=int, default=None)
    p.add_argument("--mid_term_heat_threshold", type=float, default=None)
    p.add_argument("--mid_term_similarity_threshold", type=float, default=None)
    p.add_argument("--retrieval_queue_capacity", type=int, default=None)
    p.add_argument("--long_term_knowledge_capacity", type=int, default=None)
    p.add_argument("--llm_model", default=None)    # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "memoryos_llm_model": a.memoryos_llm_model, "base_url": a.base_url,
        "short_term_capacity": a.short_term_capacity,
        "mid_term_capacity": a.mid_term_capacity,
        "mid_term_heat_threshold": a.mid_term_heat_threshold,
        "mid_term_similarity_threshold": a.mid_term_similarity_threshold,
        "retrieval_queue_capacity": a.retrieval_queue_capacity,
        "long_term_knowledge_capacity": a.long_term_knowledge_capacity,
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
                             set(DEFAULT_CONFIG) - {"strict_config"}, context="memoryos config")
        _miss = missing_sizing_config(cfg["dataset"], _fc, cfg["progressive"], path_prefix="")
        if _miss:
            raise ConfigCompletenessError(f"memoryos config: missing sizing leaf(s): {sorted(_miss)} "
                                          f"(strict-config mode; set strict_config: false to disable)")

    memo_class = make_memo_class(
        MemoryOSMemo,
        memoryos_llm_model=cfg["memoryos_llm_model"], base_url=cfg["base_url"],
        short_term_capacity=cfg["short_term_capacity"],
        mid_term_capacity=cfg["mid_term_capacity"],
        mid_term_heat_threshold=cfg["mid_term_heat_threshold"],
        mid_term_similarity_threshold=cfg["mid_term_similarity_threshold"],
        retrieval_queue_capacity=cfg["retrieval_queue_capacity"],
        long_term_knowledge_capacity=cfg["long_term_knowledge_capacity"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    result = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"],
        single_stage=cfg["single_stage"],
        memo_class=memo_class,
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
