"""
Alma subprocess entry point.

Invoked by baselines/evolve/alma/eval_runner.py. Steps:
  1. Dynamically load the MemoStructure subclass from the staged memo file.
  2. One call into the shared, execution-independent
     common.evaluate.evaluate_memo (gauntlet / single pass / check-mode smoke),
     which writes score.json, per-stage artifacts, traces and token usage
     under the caller-supplied output_run_dir.

No QA sampling is done here — alma's sampling.py reads the full traces
in the main process when constructing the meta-agent analysis input.

Output layout under <output_run_dir>:
  score.json                 — overall + per_user + invalid_users
  traces/<user_id>.json      — full QA trajectory for each user
  token_usage.json           — per-model token totals
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.harness_base import MemoStructure
from baselines.registry import DATASETS
from common.logger import get_logger

log = get_logger("main", level_styles={
    "INFO": {"icon": "🚀", "color": "green"},
    "ERROR": {"icon": "💥", "color": "red"},
})


def find_subclass_in_file(file_path: str, base_class: type):
    """Dynamically load a Python file and return the first subclass of base_class."""
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for file {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_module"] = module
    spec.loader.exec_module(module)

    subclasses = [
        obj for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base_class) and obj is not base_class
    ]
    if not subclasses:
        raise ValueError(f"No class in {file_path} inherits from {base_class.__name__}")
    return subclasses[0]


def _write_error_score(output_run_dir: Path, error_info: str) -> None:
    """Write a minimal score.json when the memo file fails to load at all."""
    output_run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": error_info}],
    }
    with (output_run_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _module_fingerprint(module_path: Path) -> str:
    """sha256[:16] over the SINGLE staged memo file (name + content). alma
    stages every memo into the shared memo_test/ dir, so the dir-scoped
    common.memory_cache.harness_fingerprint would churn as sibling memos land;
    a per-file fingerprint is stable within one launch.py run and distinct per
    memo — exactly what the cross-stage cache needs to gate reuse."""
    h = hashlib.sha256()
    p = Path(module_path)
    try:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()[:16]


async def main(
    module_path: str,
    memory_id: str,
    output_run_dir: str,
    dataset: str = "dynamicmem",
    max_logs: Optional[int] = None,
    model: str = "gpt-5-mini",
    status: str = "search",
    max_sample_concurrent: int = 6,
    mode: str = "eval",
    judge_model: str = "gpt-5-mini",
    progressive: bool = True,
    random_sample: bool = False,
    sampling_seed: int = 42,
    step_index: int = 0,
    stages: Optional[dict] = None,
    single_stage: Optional[dict] = None,
    memory_cache: bool = True,
):
    from common.evaluate import evaluate_memo
    from common.sampling import derive_sample_seed

    run_dir = Path(output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load MemoStructure class
    try:
        memo_class = find_subclass_in_file(module_path, MemoStructure)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        log.warning(f"Failed to load memo structure {memory_id}: {exc}")
        _write_error_score(run_dir, err)
        return

    log.info(f"Start evaluation for Memory Structure: {memory_id} → {run_dir} "
             f"(mode={mode}, progressive={progressive}, random_sample={random_sample}, "
             f"step={step_index})")

    # Per-STEP seed: differs across search steps (design decision 2) so each
    # step samples a different subset; None when random_sample is off (historical
    # deterministic prefix / nesting).
    sample_seed = derive_sample_seed(sampling_seed, step_index, dataset) if random_sample else None

    # 4. One call into the shared, execution-independent evaluate_memo — the
    # SAME function forge's container runs. mode=check → smoke=True (ONE
    # sanity_check-sized pass, artifacts at the run_dir root, no gauntlet);
    # mode=eval → the staged gauntlet (progressive) or the REQUIRED-single_stage
    # single pass. evaluate_memo owns sizing resolution, the token tracker,
    # per-stage artifacts, stages.json, root copies (incl. traces — alma's
    # memo_manager/sampling read them from the run_dir root), and the
    # crashed-run root score.json guarantee.
    await evaluate_memo(
        memo_class=memo_class, dataset=dataset, split=status,
        progressive=progressive, out_dir=run_dir,
        qa_model=model, judge_model=judge_model,
        stages=stages, single_stage=single_stage,
        max_sample_concurrent=max_sample_concurrent,
        sample_seed=sample_seed,
        memory_cache=memory_cache,
        memcache_fingerprint=_module_fingerprint(Path(module_path)),
        smoke=(mode == "check"),
        max_logs=max_logs, memo_sha=memory_id,
    )

    log.info(f"Evaluation complete: results under {run_dir}")

    # Bypass asyncio/GC/atexit shutdown — langchain_chroma + httpx connection
    # pools + Chroma telemetry hooks occasionally leave non-daemon threads
    # alive that prevent the interpreter from exiting for several minutes,
    # causing the parent's process.communicate() (with pidfd+epoll) to miss
    # the child-exit event. All on-disk artifacts (score.json, traces/,
    # token_usage.json) have already been flushed above, so
    # abrupt termination is safe.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", required=True)
    parser.add_argument("--memory_id", required=True)
    parser.add_argument("--output_run_dir", required=True,
                        help="Absolute path to the per-run output directory")
    parser.add_argument("--dataset", default="dynamicmem", choices=DATASETS)
    parser.add_argument("--max_logs", type=int, default=None)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--status", default="search", choices=["search", "test"])
    parser.add_argument("--max_sample_concurrent", type=int, default=3)
    parser.add_argument("--mode", default="eval", choices=["eval", "check"])
    parser.add_argument("--judge_model", default="gpt-5-mini")
    # Progressive gauntlet + per-step deterministic sampling (Task 9).
    parser.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=True,
                        help="mode=eval runs the stage1→2→3 gauntlet (default). "
                             "--no-progressive = a single terminal-size pass.")
    parser.add_argument("--random_sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Seed per-step subset selection (different subset each search step).")
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--step_index", type=int, default=0,
                        help="Search-step index; folds into the per-step sample seed.")
    parser.add_argument("--stages", type=str, default=None,
                        help="JSON stages-block override (sanity_check/stage1..3); "
                             "default = family DEFAULT_STAGES.")
    parser.add_argument("--single_stage", type=str, default=None,
                        help="JSON single-pass size block (progressive=false only; "
                             "same size fields as a stage, no threshold). REQUIRED "
                             "when --no-progressive; a null field = whole split.")
    parser.add_argument("--memory_cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Cross-stage Phase-1 memory reuse in the gauntlet.")

    args = parser.parse_args()
    asyncio.run(main(
        module_path=args.module_path,
        memory_id=args.memory_id,
        output_run_dir=args.output_run_dir,
        dataset=args.dataset,
        max_logs=args.max_logs,
        model=args.model,
        status=args.status,
        max_sample_concurrent=args.max_sample_concurrent,
        mode=args.mode,
        judge_model=args.judge_model,
        progressive=args.progressive,
        random_sample=args.random_sample,
        sampling_seed=args.sampling_seed,
        step_index=args.step_index,
        stages=json.loads(args.stages) if args.stages else None,
        single_stage=json.loads(args.single_stage) if args.single_stage else None,
        memory_cache=args.memory_cache,
    ))

    # Force immediate process termination. Python's normal shutdown runs
    # atexit handlers + GC, which can hang for minutes on
    # langchain_chroma / httpx background threads & connection-pool cleanup.
    # All artifacts (score.json, traces/, token_usage.json)
    # are already flushed to disk by this point, so bypassing graceful
    # shutdown is safe and prevents the parent from waiting on a zombie
    # that asyncio's pidfd watcher occasionally fails to observe.
    os._exit(0)
