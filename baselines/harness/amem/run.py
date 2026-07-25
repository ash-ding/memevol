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
from baselines.harness.eval_common import make_memo_class, run_baseline, parse_stage_spec
from baselines.harness.amem.memo import AMemMemo


def main():
    p = argparse.ArgumentParser(description="A-mem baseline — multi-dataset")
    p.add_argument("--dataset", default="locomo", choices=DATASETS)
    p.add_argument("--split", default="test", choices=["test", "search"])
    p.add_argument("--stage-spec", default=None)
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--stages", type=str, default=None)
    p.add_argument("--no-memory-cache", dest="memory_cache", action="store_false", default=True)
    p.add_argument("--amem_llm_model", default="gpt-4o-mini",
                   help="A-mem internal LLM (note analysis / evolution / keywords "
                        "rewrite). Its OpenAIController hardcodes temperature+"
                        "max_tokens, which the gpt-5 family rejects — keep a "
                        "4-series model here.")
    p.add_argument("--retrieve_k", type=int, default=10)   # upstream default
    p.add_argument("--llm_model", default="gpt-5-mini")    # shared QA agent
    p.add_argument("--judge_model", default="gpt-5-mini")
    p.add_argument("--max_sample_concurrent", type=int, default=3)
    a = p.parse_args()

    memo_class = make_memo_class(
        AMemMemo, amem_llm_model=a.amem_llm_model, retrieve_k=a.retrieve_k,
    )
    out_dir = Path(__file__).resolve().parent / "results" / a.dataset / a.split
    score = asyncio.run(run_baseline(
        dataset=a.dataset, split=a.split, user_stage_spec=parse_stage_spec(a.stage_spec),
        memo_class=memo_class, qa_model=a.llm_model, judge_model=a.judge_model,
        out_dir=out_dir, max_sample_concurrent=a.max_sample_concurrent,
        progressive=a.progressive, sampling_seed=a.sampling_seed,
        stages=parse_stage_spec(a.stages), memory_cache=a.memory_cache,
    ))
    print("overall:", score["benchmark_eval_score"]["benchmark_overall_eval_score"], "→", out_dir)


if __name__ == "__main__":
    main()
