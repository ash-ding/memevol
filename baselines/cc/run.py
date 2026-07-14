"""cc (Claude Code) baseline — evaluate on one benchmark's split, comparable
to the main method (same split/judge/scoring via the per-dataset workflow).
cc is a NATIVE-answer baseline: `CCPassThroughMixin` bypasses the shared QA
agent so the workflow judges cc's own tool-using answer verbatim.

    python baselines/cc/run.py --dataset locomo
    python baselines/cc/run.py --dataset dynamicmem --stage-spec '{"n_samples": 2}'
"""
from __future__ import annotations
import argparse, asyncio, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass
from baselines.registry import DATASETS
from baselines.eval_common import make_memo_class, run_baseline, parse_stage_spec
from baselines.cc.memo import CCMemo, CCPassThroughMixin, MODEL_ALIASES


def main():
    p = argparse.ArgumentParser(description="cc (Claude Code) baseline — multi-dataset")
    p.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    p.add_argument("--split", default="test", choices=["test", "search"])
    p.add_argument("--stage-spec", default=None)
    p.add_argument("--model", default="sonnet",
                   help="Model: claude-sonnet-4-20250514, claude-opus-4-20250514, sonnet, or opus")
    p.add_argument("--max_turns", type=int, default=30, help="Max tool-use turns per QA question")
    p.add_argument("--judge_model", default="gpt-5-mini")
    p.add_argument("--max_sample_concurrent", type=int, default=3)
    a = p.parse_args()

    model = MODEL_ALIASES.get(a.model, a.model)
    memo_class = make_memo_class(CCMemo, model=model, max_turns=a.max_turns, judge_model=a.judge_model)
    out_dir = Path(__file__).resolve().parent / "results" / a.dataset / a.split
    # qa_model=a.judge_model: the shared QA agent is bypassed by
    # CCPassThroughMixin (cc's own answer is judged verbatim), so its model
    # choice is irrelevant to scoring — but BaseWorkflow's constructor still
    # requires one.
    score = asyncio.run(run_baseline(
        dataset=a.dataset, split=a.split, user_stage_spec=parse_stage_spec(a.stage_spec),
        memo_class=memo_class, workflow_overrides=(CCPassThroughMixin,),
        qa_model=a.judge_model, judge_model=a.judge_model,
        out_dir=out_dir, max_sample_concurrent=a.max_sample_concurrent,
    ))
    print("overall:", score["benchmark_eval_score"]["benchmark_overall_eval_score"], "→", out_dir)


if __name__ == "__main__":
    main()
