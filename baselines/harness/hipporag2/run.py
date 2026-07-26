"""HippoRAG2 baseline — evaluate on one benchmark's split, comparable to the
main method (same split/judge/scoring via the per-dataset workflow).

Sizing is config-file only (no sizing CLI flags): `single_stage` (progressive:
false) or `stages` (progressive: true) — see config.example.yaml.

    python baselines/harness/hipporag2/run.py --config baselines/harness/hipporag2/config.example.yaml
    python baselines/harness/hipporag2/run.py --config baselines/harness/hipporag2/config.example.yaml --progressive
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
from baselines.harness.hipporag2.memo import HippoRAGMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    # Sizing lives in the config file only (native YAML dicts), never on the CLI:
    #   single_stage: {...}  (progressive: false) — REQUIRED for the single pass
    #   stages: {...}        (progressive: true)  — overrides family DEFAULT_STAGES
    "dataset": "dynamicmem", "split": "test",
    "progressive": False, "sampling_seed": 42,
    "single_stage": None, "stages": None, "memory_cache": True,
    "embedding": "text-embedding-3-small", "llm_model": "gpt-5-mini", "judge_model": "gpt-5-mini",
    "embedding_batch_size": None, "embedding_dtype": None, "max_sample_concurrent": 3,
}


def main():
    p = argparse.ArgumentParser(description="HippoRAG2 baseline — multi-dataset")
    p.add_argument("--config", default=None, help="YAML config path (CLI flags override it)")
    p.add_argument("--dataset", default=None, choices=DATASETS)
    p.add_argument("--split", default=None, choices=["test", "search"])
    p.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sampling-seed", dest="sampling_seed", type=int, default=None)
    p.add_argument("--memory-cache", dest="memory_cache", action=argparse.BooleanOptionalAction, default=None,
                   help="Cross-stage Phase-1 memory reuse (default on). --no-memory-cache to disable.")
    p.add_argument("--embedding", default=None)
    p.add_argument("--llm_model", default=None)   # QA agent + HippoRAG internal LLM
    p.add_argument("--judge_model", default=None)
    p.add_argument("--embedding_batch_size", type=int, default=None)
    p.add_argument("--embedding_dtype", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    cli = {
        "dataset": a.dataset, "split": a.split,
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "memory_cache": a.memory_cache,
        "embedding": a.embedding, "llm_model": a.llm_model, "judge_model": a.judge_model,
        "embedding_batch_size": a.embedding_batch_size, "embedding_dtype": a.embedding_dtype,
        "max_sample_concurrent": a.max_sample_concurrent,
    }
    cfg = resolve_config(DEFAULT_CONFIG, a.config, cli)

    memo_class = make_memo_class(
        HippoRAGMemo, embedding=cfg["embedding"], llm_model=cfg["llm_model"],
        judge_model=cfg["judge_model"], embedding_batch_size=cfg["embedding_batch_size"],
        embedding_dtype=cfg["embedding_dtype"],
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
