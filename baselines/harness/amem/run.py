"""A-mem baseline — evaluate on one benchmark's split, comparable to the main
method (same split/judge/scoring via the per-dataset workflow).

    baselines/venv/bin/python baselines/harness/amem/run.py --dataset locomo
    baselines/venv/bin/python baselines/harness/amem/run.py --dataset dynamicmem \
        --split search --stage-spec '{"n_samples": 1, "n_checkpoints": 1, "n_task_a": 1, "n_task_c": 1}'
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
from baselines.harness.eval_common import make_memo_class, run_baseline
from baselines.harness.amem.memo import AMemMemo
from common.config import resolve_config

DEFAULT_CONFIG = {
    "dataset": "locomo",
    "split": "test",
    "stage_spec": None,          # native dict in YAML; JSON string on CLI
    "progressive": False,
    "sampling_seed": 42,
    "stages": None,              # native dict in YAML; JSON string on CLI
    "memory_cache": True,
    "amem_llm_model": "gpt-4o-mini",
    "retrieve_k": 10,
    "llm_model": "gpt-5-mini",
    "judge_model": "gpt-5-mini",
    "max_sample_concurrent": 3,
}


def main():
    p = argparse.ArgumentParser(description="A-mem baseline — multi-dataset")
    p.add_argument("--config", default=None, help="YAML config path (CLI flags override it)")
    p.add_argument("--dataset", default=None, choices=DATASETS)
    p.add_argument("--split", default=None, choices=["test", "search"])
    p.add_argument("--stage-spec", dest="stage_spec", default=None)
    p.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sampling-seed", dest="sampling_seed", type=int, default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--memory-cache", dest="memory_cache", action=argparse.BooleanOptionalAction, default=None,
                   help="Cross-stage Phase-1 memory reuse (default on). --no-memory-cache to disable.")
    p.add_argument("--amem_llm_model", default=None,
                   help="A-mem internal LLM (note analysis / evolution / keywords "
                        "rewrite). Its OpenAIController hardcodes temperature+"
                        "max_tokens, which the gpt-5 family rejects — keep a "
                        "4-series model here.")
    p.add_argument("--retrieve_k", type=int, default=None)   # upstream default
    p.add_argument("--llm_model", default=None)    # shared QA agent
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    def _json_or_none(s):
        import json
        return json.loads(s) if s else None

    cli = {
        "dataset": a.dataset, "split": a.split, "stage_spec": _json_or_none(a.stage_spec),
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "stages": _json_or_none(a.stages), "memory_cache": a.memory_cache,
        "amem_llm_model": a.amem_llm_model, "retrieve_k": a.retrieve_k,
        "llm_model": a.llm_model, "judge_model": a.judge_model,
        "max_sample_concurrent": a.max_sample_concurrent,
    }
    cfg = resolve_config(DEFAULT_CONFIG, a.config, cli)

    memo_class = make_memo_class(
        AMemMemo, amem_llm_model=cfg["amem_llm_model"], retrieve_k=cfg["retrieve_k"],
    )
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    score = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"], user_stage_spec=cfg["stage_spec"] or {},
        memo_class=memo_class, qa_model=cfg["llm_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir, max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
        stages=cfg["stages"], memory_cache=cfg["memory_cache"],
    ))
    print("overall:", score["benchmark_eval_score"]["benchmark_overall_eval_score"], "→", out_dir)


if __name__ == "__main__":
    main()
