"""Candidate evaluation — one `launch.py` subprocess per harness.

A candidate is arbitrary proposer-written code, so it is evaluated out of
process: a crash, a hang, or a leaked thread pool costs one candidate, never
the search loop. Scoring itself is `common.evaluate.evaluate_memo` inside that
subprocess (see launch.py) — the shared evaluator, unmodified.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[2]


def read_metrics(run_dir: Path) -> Dict[str, Any]:
    """Metrics for a finished eval; a dict with raw_score 0 when it produced
    nothing (crashed subprocess, killed on timeout)."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    score_path = run_dir / "score.json"
    if score_path.exists():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        raw = score.get("benchmark_eval_score", {}).get("benchmark_overall_eval_score", 0.0)
        return {"raw_score": float(raw), "score_max": 1, "eliminated": False}
    return {"raw_score": 0.0, "score_max": 1, "eliminated": True, "error": "no result written"}


def normalized_score(metrics: Dict[str, Any]) -> float:
    """raw_score on a 0-1 scale — the same number every other baseline reports."""
    score_max = int(metrics.get("score_max") or 1) or 1
    return float(metrics.get("raw_score", 0.0)) / score_max


async def evaluate_candidate(
    *,
    name: str,
    harness_file: Path,
    out_dir: Path,
    dataset: str,
    split: str,
    cfg: Dict[str, Any],
    step_index: int = 0,
    timeout_s: int = 14 * 3600,
) -> Dict[str, Any]:
    """Run one candidate through the shared evaluator. Returns its metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("metrics.json", "score.json"):
        (out_dir / stale).unlink(missing_ok=True)

    cmd = [
        sys.executable, "-u", str(BASELINE_ROOT / "launch.py"),
        "--harness-file", str(harness_file),
        "--name", name,
        "--output-run-dir", str(out_dir),
        "--dataset", dataset,
        "--split", split,
        "--execution-model", cfg["execution_model"],
        "--judge-model", cfg["judge_model"],
        "--max-sample-concurrent", str(cfg["max_sample_concurrent"]),
        "--sampling-seed", str(cfg["sampling_seed"]),
        "--step-index", str(step_index),
        "--progressive" if cfg["progressive"] else "--no-progressive",
        "--random-sample" if cfg["random_sample"] else "--no-random-sample",
        "--memory-cache" if cfg["memory_cache"] else "--no-memory-cache",
    ]
    if cfg["max_logs"] is not None:
        cmd += ["--max-logs", str(cfg["max_logs"])]
    if cfg["stages"] is not None:
        cmd += ["--stages", json.dumps(cfg["stages"])]
    if cfg["single_stage"] is not None:
        cmd += ["--single-stage", json.dumps(cfg["single_stage"])]

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "EVALS_LOG_DIR": str(out_dir),
        "MEMEVOL_LOG_FILE": "subprocess.log",
    }

    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(PROJECT_ROOT), env=env,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        _kill(proc)
        await proc.wait()
        return {"raw_score": 0.0, "score_max": 1, "eliminated": True,
                "error": f"timed out after {timeout_s}s"}

    metrics = read_metrics(out_dir)
    if proc.returncode not in (0, None) and "error" not in metrics:
        tail = (stderr or b"").decode(errors="replace").strip()[-2000:]
        metrics["error"] = f"exit {proc.returncode}: {tail}"
    return metrics


def _kill(proc: "asyncio.subprocess.Process") -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def import_check(harness_file: Path, timeout_s: int = 60) -> Optional[str]:
    """Load the candidate in a throwaway process. Returns None when it imports
    and exposes a MemoClass subclass, else the error text."""
    import subprocess

    code = (
        "import sys;"
        f"sys.path.insert(0, {str(BASELINE_ROOT)!r});"
        "from launch import load_harness_class;"
        f"load_harness_class({str(harness_file)!r});"
        "print('OK')"
    )
    env = {**os.environ,
           "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    try:
        result = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT),
                                env=env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return f"import check timed out after {timeout_s}s"
    if result.returncode == 0 and "OK" in result.stdout:
        return None
    return (result.stderr or result.stdout or "unknown import failure").strip()[-800:]
