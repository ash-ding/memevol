"""Forge in-container runner.

Invoked by forge/evaluator.py inside Singularity — ONE container per
(harness, dataset) runs the WHOLE evaluation plan:

    python /app/forge/launch.py --harness-dir /harness --out-dir /out \
        --dataset locomo --split search --plan-json '{"progressive": true, ...}'

Steps:
  1. Dynamically load the MemoClass subclass from /harness/memo.py.
  2. One call into the shared, execution-independent
     common.evaluate.evaluate_memo — the staged stage1→2→3 gauntlet
     (progressive), the single_stage single pass, or the sanity-size smoke
     run, all inside THIS container (promotion/elimination decisions
     included; the host no longer orchestrates stages).
  3. evaluate_memo writes /out/<stage>/{score.json,token_usage.json,run_record.json,traces/},
     /out/stages.json and the reached-stage root copies; this runner
     additionally writes /out/metrics.json (evaluate_memo's return dict —
     raw_score/score_max/per_user_stddev/tokens/stage/eliminated) for the
     host to read back.

The harness_dir is bind-mounted read-only; /out is bind-mounted read-write.
Binds are SELECTIVE (v10): only common/, datasets/, forge/{__init__,launch,
memo_class}.py are mounted under /app — the rest of forge/ (host outer
loop) and baselines/ are deliberately NOT visible in-container. See
forge/evaluator.py's module docstring for the authoritative bind list.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Type

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from common.memo_class import MemoClass
from benchmarks.registry import DATASETS


#: Kept in step with forge.paths — the container gets only launch.py,
#: memo_class.py and __init__.py from forge, so it cannot import that module.
ENTRY_FILE = "memo.py"


def _load_harness_class(harness_dir: Path) -> Type[MemoClass]:
    """Import the harness's interface file and return its MemoClass subclass.

    On import failure, raise ImportError with an actionable message. The
    error string is propagated up to score.json::invalid_users[0].error and
    becomes the trace shown to CC during sanity-retry — so write it for an
    LLM reader.
    """
    harness_py = harness_dir / ENTRY_FILE
    if not harness_py.exists():
        raise ImportError(
            f"{ENTRY_FILE} missing at {harness_dir / ENTRY_FILE}. The proposer "
            f"must write a {ENTRY_FILE} defining its MemoClass subclass; any "
            f"further source goes under {harness_dir.name}/src/."
        )
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))
    spec = importlib.util.spec_from_file_location("forge_harness_mod", str(harness_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {harness_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["forge_harness_mod"] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{harness_py.name} imports a package not in the container: {exc.name!r}. "
            f"Either (a) declare it in `requirements.txt` (will trigger a delta "
            f"image build), or (b) switch to a package already in the base image "
            f"(see PROPOSER_SYSTEM for the list). Original: {exc}"
        ) from exc
    except Exception as exc:
        raise ImportError(
            f"harness.py raised at import time: [{type(exc).__name__}] {exc}"
        ) from exc
    from common.memo_select import select_memo_class
    # Only classes DEFINED in this harness file (not imported bases), concrete first.
    defined = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
               if obj.__module__ == module.__name__]
    try:
        return select_memo_class(defined, str(harness_py))
    except TypeError as exc:
        raise ImportError(
            f"{exc} Define a class that inherits from `forge.memo_class.MemoClass`."
        ) from exc


def _write_error(out_dir: Path, err: str) -> None:
    """Load-failure artifacts: an error score.json + a matching metrics.json so
    the host always has both to read back."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "load_failed", "error": err}],
    }
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "raw_score": 0.0, "score_max": 1, "per_user_stddev": None,
            "tokens": 0, "stage": 0.0, "eliminated": True, "error": err,
        }, f, indent=2, ensure_ascii=False)


async def _async_main(args: argparse.Namespace) -> None:
    from common.evaluate import evaluate_memo

    harness_dir = Path(args.harness_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        memo_class = _load_harness_class(harness_dir)
    except Exception as exc:
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        _write_error(out_dir, err)
        return

    # The evaluation plan, serialized by the host (forge/evaluator.py):
    #   progressive: bool — gauntlet vs single_stage single pass
    #   smoke: bool       — ONE sanity_check-sized pass (sanity gate / smoke_test)
    #   stages: dict|null — per-stage sizes (resolved by the host config layer)
    #   single_stage: dict|null — single-pass sizes (progressive=false)
    #   sample_seed: str|null   — per-run nested-sampling seed
    plan = json.loads(args.plan_json)

    try:
        metrics = await evaluate_memo(
            memo_class=memo_class,
            dataset=args.dataset,
            split=args.split,
            progressive=bool(plan.get("progressive", True)),
            out_dir=out_dir,
            qa_model=args.model,
            judge_model=args.judge_model,
            stages=plan.get("stages"),
            single_stage=plan.get("single_stage"),
            max_sample_concurrent=args.max_sample_concurrent,
            sample_seed=plan.get("sample_seed"),
            smoke=bool(plan.get("smoke", False)),
            max_logs=args.max_logs,
            memo_sha=harness_dir.name,
        )
    except Exception as exc:
        # Config/plan errors (e.g. a missing single_stage block that slipped
        # past the host) or unexpected evaluator failures: leave readable
        # artifacts rather than a bare traceback + empty /out.
        err = f"[{type(exc).__name__}] {exc}\n{traceback.format_exc()}"
        _write_error(out_dir, err)
        return

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # chroma / httpx background threads can delay normal interpreter
    # shutdown. Artifacts are flushed; exit abruptly.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="dynamicmem", choices=sorted(DATASETS))
    parser.add_argument("--split", default="search", choices=["search", "test"])
    parser.add_argument("--plan-json", required=True,
                        help='JSON evaluation plan, e.g. \'{"progressive": true, '
                             '"smoke": false, "stages": {...}, "single_stage": null, '
                             '"sample_seed": null}\'. The whole gauntlet (or single '
                             'pass) runs inside THIS container via evaluate_memo.')
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--max-logs", type=int, default=None)
    parser.add_argument("--max-sample-concurrent", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(_async_main(args))
    os._exit(0)
