"""HippoRAG2 baseline — evaluate on one benchmark's split, comparable to the
main method (same split/judge/scoring via the per-dataset workflow).

    python baselines/harness/hipporag2/run.py --dataset locomo
    python baselines/harness/hipporag2/run.py --dataset dynamicmem --stage-spec '{"n_samples": 2}'
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
from baselines.harness.hipporag2.memo import HippoRAGMemo


def main():
    p = argparse.ArgumentParser(description="HippoRAG2 baseline — multi-dataset")
    p.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    p.add_argument("--split", default="test", choices=["test", "search"])
    p.add_argument("--stage-spec", default=None)
    p.add_argument("--progressive", action="store_true")
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--stages", type=str, default=None)
    p.add_argument("--no-memory-cache", dest="memory_cache", action="store_false", default=True)
    p.add_argument("--embedding", default="text-embedding-3-small")
    p.add_argument("--llm_model", default="gpt-5-mini")   # QA agent + HippoRAG internal LLM
    p.add_argument("--judge_model", default="gpt-5-mini")
    p.add_argument("--embedding_batch_size", type=int, default=None)
    p.add_argument("--embedding_dtype", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=3)
    a = p.parse_args()

    memo_class = make_memo_class(
        HippoRAGMemo, embedding=a.embedding, llm_model=a.llm_model,
        judge_model=a.judge_model, embedding_batch_size=a.embedding_batch_size,
        embedding_dtype=a.embedding_dtype,
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
