"""cc (Claude Code) baseline — evaluate on one benchmark's split, comparable
to the main method (same split/judge/scoring via the per-dataset workflow).
cc is a NATIVE-answer baseline: `CCMemo.use_memory_to_answer` bypasses the shared
QA agent so the workflow judges cc's own tool-using answer verbatim.

    python baselines/harness/cc/run.py --dataset locomo
    python baselines/harness/cc/run.py --dataset dynamicmem --stage-spec '{"n_samples": 2}'
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
from baselines.harness.cc.memo import CCMemo, MODEL_ALIASES
from common.config import resolve_config

DEFAULT_CONFIG = {
    "dataset": "dynamicmem", "split": "test", "stage_spec": None,
    "progressive": False, "sampling_seed": 42, "stages": None, "memory_cache": True,
    "model": "sonnet", "max_turns": 30, "judge_model": "gpt-5-mini",
    "max_sample_concurrent": 3,
}


def main():
    p = argparse.ArgumentParser(description="cc (Claude Code) baseline — multi-dataset")
    p.add_argument("--config", default=None, help="YAML config path (CLI flags override it)")
    p.add_argument("--dataset", default=None, choices=DATASETS)
    p.add_argument("--split", default=None, choices=["test", "search"])
    p.add_argument("--stage-spec", dest="stage_spec", default=None)
    p.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sampling-seed", dest="sampling_seed", type=int, default=None)
    p.add_argument("--stages", default=None)
    p.add_argument("--memory-cache", dest="memory_cache", action=argparse.BooleanOptionalAction, default=None,
                   help="Cross-stage Phase-1 memory reuse (default on). --no-memory-cache to disable.")
    p.add_argument("--model", default=None,
                   help="Model: claude-sonnet-4-20250514, claude-opus-4-20250514, sonnet, or opus")
    p.add_argument("--max_turns", type=int, default=None, help="Max tool-use turns per QA question")
    p.add_argument("--judge_model", default=None)
    p.add_argument("--max_sample_concurrent", type=int, default=None)
    a = p.parse_args()

    def _json_or_none(s):
        import json
        return json.loads(s) if s is not None else None

    cli = {
        "dataset": a.dataset, "split": a.split, "stage_spec": _json_or_none(a.stage_spec),
        "progressive": a.progressive, "sampling_seed": a.sampling_seed,
        "stages": _json_or_none(a.stages), "memory_cache": a.memory_cache,
        "model": a.model, "max_turns": a.max_turns, "judge_model": a.judge_model,
        "max_sample_concurrent": a.max_sample_concurrent,
    }
    cfg = resolve_config(DEFAULT_CONFIG, a.config, cli)

    model = MODEL_ALIASES.get(cfg["model"], cfg["model"])
    memo_class = make_memo_class(CCMemo, model=model, max_turns=cfg["max_turns"], judge_model=cfg["judge_model"])
    out_dir = Path(__file__).resolve().parent / "results" / cfg["dataset"] / cfg["split"]
    # qa_model=cfg["judge_model"]: the shared QA agent is bypassed by
    # CCMemo.use_memory_to_answer (cc's own answer is judged verbatim), so its
    # model choice is irrelevant to scoring — but BaseWorkflow's constructor
    # still requires one.
    score = asyncio.run(run_baseline(
        dataset=cfg["dataset"], split=cfg["split"], user_stage_spec=cfg["stage_spec"] or {},
        memo_class=memo_class,
        qa_model=cfg["judge_model"], judge_model=cfg["judge_model"],
        out_dir=out_dir, max_sample_concurrent=cfg["max_sample_concurrent"],
        progressive=cfg["progressive"], sampling_seed=cfg["sampling_seed"],
        stages=cfg["stages"], memory_cache=cfg["memory_cache"],
    ))
    print("overall:", score["benchmark_eval_score"]["benchmark_overall_eval_score"], "→", out_dir)


if __name__ == "__main__":
    main()
