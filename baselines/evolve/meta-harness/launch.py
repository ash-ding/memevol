"""Subprocess entry point — evaluate ONE candidate harness.

Spawned by `evaluator.py`. Loads the `MemoClass` subclass out of a candidate
file and makes a single call into the shared, execution-independent
`common.evaluate.evaluate_memo` — the same function forge runs in-container and
alma runs in its own subprocess. Everything scored here is therefore on the
same axis as any forge-evolved harness's number.

Artifacts under <output_run_dir>:
  score.json        overall + per_user + invalid_users
  metrics.json      evaluate_memo's returned metrics (score, cost, stage)
  stages.json       per-stage gauntlet summary (progressive runs)
  traces/<user>.json  full QA trajectory — the proposer's main feedback channel
  token_usage.json  per-(model, phase) totals
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.memo_class import MemoClass
from benchmarks.registry import DATASETS


def load_harness_class(file_path: str) -> type:
    """Import a candidate file and return its first MemoClass subclass."""
    spec = importlib.util.spec_from_file_location("mh_candidate", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mh_candidate"] = module
    spec.loader.exec_module(module)

    subclasses = [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, MemoClass) and obj is not MemoClass
    ]
    if not subclasses:
        raise ValueError(f"no MemoClass subclass in {file_path}")
    return subclasses[0]


def _write_load_error(out_dir: Path, error: str) -> None:
    """Minimal score.json so the parent always has something to read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score.json").write_text(json.dumps({
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": error}],
    }, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({
        "raw_score": 0.0, "score_max": 1, "eliminated": True, "load_error": error,
    }, indent=2), encoding="utf-8")


def _file_fingerprint(path: Path) -> str:
    """sha256[:16] over the single candidate file — gates cross-stage memory
    cache reuse per candidate (the dir-scoped fingerprint would churn as
    sibling candidates land in harnesses/)."""
    h = hashlib.sha256()
    try:
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    except OSError:
        return ""
    return h.hexdigest()[:16]


async def main(args: argparse.Namespace) -> None:
    from common.evaluate import evaluate_memo
    from common.sampling import derive_sample_seed

    out_dir = Path(args.output_run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        harness_cls = load_harness_class(args.harness_file)
    except Exception as exc:
        _write_load_error(out_dir, f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}")
        return

    sample_seed = (
        derive_sample_seed(args.sampling_seed, args.step_index, args.dataset)
        if args.random_sample else None
    )

    metrics = await evaluate_memo(
        memo_class=harness_cls, dataset=args.dataset, split=args.split,
        progressive=args.progressive, out_dir=out_dir,
        qa_model=args.execution_model, judge_model=args.judge_model,
        stages=json.loads(args.stages) if args.stages else None,
        single_stage=json.loads(args.single_stage) if args.single_stage else None,
        max_sample_concurrent=args.max_sample_concurrent,
        sample_seed=sample_seed,
        memory_cache=args.memory_cache,
        memcache_fingerprint=_file_fingerprint(Path(args.harness_file)),
        max_logs=args.max_logs, memo_sha=args.name,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate one meta-harness candidate")
    p.add_argument("--harness-file", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-run-dir", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--split", required=True, choices=["search", "test"])
    p.add_argument("--execution-model", required=True)
    p.add_argument("--judge-model", required=True)
    p.add_argument("--max-sample-concurrent", type=int, default=3)
    p.add_argument("--max-logs", type=int, default=None)
    p.add_argument("--progressive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--random-sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--step-index", type=int, default=0)
    p.add_argument("--stages", default=None, help="JSON stages block")
    p.add_argument("--single-stage", default=None, help="JSON single_stage block")
    p.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
    # Artifacts are flushed; skip interpreter shutdown. Vector-store and httpx
    # background threads left behind by candidate code can otherwise keep the
    # process alive for minutes and stall the parent's wait().
    os._exit(0)
