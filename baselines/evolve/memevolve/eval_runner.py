"""MemEvolve eval runner — spawns launch.py on an assembled genotype file
(pattern copied from evolve/alma/eval_runner.py per the baselines README
convention). Adds the inner-loop feedback vector on top of alma's runner:

    F = (perf, cost, delay)
      perf  = score.json benchmark_overall_eval_score
      cost  = total tokens across models (token_usage.json)
      delay = wall-clock seconds of the eval subprocess (timing.json)

Per-run output directory:
    baselines/evolve/memevolve/results/<dataset>/<sha>_<status>_<mode>/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import dotenv_values
except ImportError:
    def dotenv_values(path):  # type: ignore
        result = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        result[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
        return result

from common.logger import get_logger

log = get_logger("main")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEMEVOLVE_ROOT = PROJECT_ROOT / "baselines" / "evolve" / "memevolve"

SUBPROCESS_TIMEOUT = {
    "check": 2 * 3600,
    "eval": 8 * 3600,
}


def get_output_run_dir(sha: str, status: str, mode: str, dataset: str) -> Path:
    return MEMEVOLVE_ROOT / "results" / dataset / f"{sha}_{status}_{mode}"


async def run_evaluation(
    sha: str,
    module_path: Path,
    dataset: str,
    mode: str = "eval",
    status: str = "search",
    model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    eval_n_qa: Optional[int] = None,
    max_logs: Optional[int] = None,
    max_sample_concurrent: int = 3,
    check_n_samples: int = 2,
    check_n_qa: int = 3,
    timeout_s: Optional[float] = None,
) -> Path:
    """Run one eval subprocess for an assembled genotype. Returns run dir.

    `timeout_s` overrides the mode-default subprocess timeout — the cost
    guard uses it to hard-abort an over-budget check instead of letting it
    burn LLM spend up to the 2h mode timeout."""
    output_run_dir = get_output_run_dir(sha, status, mode, dataset)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    stale_score = output_run_dir / "score.json"
    if stale_score.exists():
        try:
            stale_score.unlink()
        except OSError as exc:
            log.warning(f"Could not remove stale score.json at {stale_score}: {exc}")

    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    api_key = env_vals.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    launch_args = [
        sys.executable, "-u",
        str(MEMEVOLVE_ROOT / "launch.py"),
        "--module_path", str(module_path),
        "--memory_id", sha,
        "--output_run_dir", str(output_run_dir),
        "--dataset", dataset,
        "--model", model,
        "--eval_n_samples", str(eval_n_samples),
        "--status", status,
        "--max_sample_concurrent", str(max_sample_concurrent),
        "--mode", mode,
        "--judge_model", judge_model,
        "--check_n_samples", str(check_n_samples),
        "--check_n_qa", str(check_n_qa),
    ]
    if max_logs is not None:
        launch_args += ["--max_logs", str(max_logs)]
    if eval_n_qa is not None:
        launch_args += ["--eval_n_qa", str(eval_n_qa)]

    env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        "EVALS_LOG_DIR": str(output_run_dir),
        "MEMEVOL_LOG_FILE": "subprocess.log",
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    log.info(f"Running evaluation: sha={sha} dataset={dataset} mode={mode}")
    t_start = time.monotonic()

    process = await asyncio.create_subprocess_exec(
        *launch_args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = timeout_s or SUBPROCESS_TIMEOUT.get(mode, 8 * 3600)
    score_path = output_run_dir / "score.json"

    # Dual-signal wait carried over from alma — see that file for the
    # Python 3.12 pidfd rationale.
    GRACE_AFTER_SCORE = 60
    SCORE_POLL_INTERVAL = 5

    async def _poll_for_score() -> bool:
        while not score_path.exists():
            await asyncio.sleep(SCORE_POLL_INTERVAL)
        return True

    communicate_task = asyncio.create_task(process.communicate())
    score_task = asyncio.create_task(_poll_for_score())

    done, pending = await asyncio.wait(
        {communicate_task, score_task},
        timeout=timeout_s,
        return_when=asyncio.FIRST_COMPLETED,
    )

    forced_kill_after_score = False
    stderr_bytes = b""

    if not done:
        for t in pending:
            t.cancel()
        log.error(f"Evaluation subprocess HUNG for sha={sha} after {timeout_s}s — killing")
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"Failed to reap subprocess for sha={sha} within 5s")
        raise RuntimeError(f"Subprocess for sha={sha} timed out past {timeout_s}s and was killed")

    if communicate_task in done:
        _, stderr_bytes = communicate_task.result()
        score_task.cancel()
        try:
            await score_task
        except asyncio.CancelledError:
            pass
    else:
        log.warning(f"score.json written for sha={sha} but subprocess has not exited; "
                    f"awaiting up to {GRACE_AFTER_SCORE}s for clean shutdown")
        try:
            _, stderr_bytes = await asyncio.wait_for(communicate_task, timeout=GRACE_AFTER_SCORE)
        except asyncio.TimeoutError:
            forced_kill_after_score = True
            log.warning(f"asyncio exit notification stuck for sha={sha} — force-killing")
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.error(f"Failed to reap sha={sha} within 5s after SIGKILL")
            stderr_bytes = b""

    wall_clock_s = time.monotonic() - t_start
    try:
        (output_run_dir / "timing.json").write_text(
            json.dumps({"wall_clock_s": round(wall_clock_s, 2)}), encoding="utf-8")
    except OSError as exc:
        log.warning(f"Could not write timing.json: {exc}")

    if forced_kill_after_score:
        log.info(f"Evaluation completed via score.json (reaped by SIGKILL after grace): "
                 f"sha={sha} → {output_run_dir}")
    elif process.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(f"Evaluation subprocess exited with code {process.returncode} for sha={sha}"
                  + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else ""))
    else:
        log.info(f"Evaluation finished: sha={sha} → {output_run_dir} ({wall_clock_s:.0f}s)")

    return output_run_dir


def read_feedback(output_run_dir: Path) -> Dict[str, Any]:
    """Assemble the inner-loop feedback vector F from a finished run dir."""
    output_run_dir = Path(output_run_dir)
    score_path = output_run_dir / "score.json"
    if not score_path.exists():
        raise FileNotFoundError(f"score.json missing under {output_run_dir}")
    with score_path.open(encoding="utf-8") as f:
        score = json.load(f)

    tokens_total = 0
    tk_path = output_run_dir / "token_usage.json"
    if tk_path.exists():
        try:
            with tk_path.open(encoding="utf-8") as f:
                usage = json.load(f)
            for model_usage in usage.values():
                if isinstance(model_usage, dict):
                    tokens_total += sum(v for v in model_usage.values()
                                        if isinstance(v, (int, float)))
        except Exception as exc:
            log.warning(f"Could not read token_usage.json: {exc}")

    wall = None
    tm_path = output_run_dir / "timing.json"
    if tm_path.exists():
        try:
            wall = json.loads(tm_path.read_text(encoding="utf-8")).get("wall_clock_s")
        except Exception:
            wall = None

    return {
        "perf": float(score.get("benchmark_eval_score", {}).get("benchmark_overall_eval_score", 0.0)),
        "cost": float(tokens_total),
        "delay": float(wall) if wall is not None else 0.0,
        "invalid_users": score.get("invalid_users", []),
        "per_user": score.get("per_user", {}),
        "run_dir": str(output_run_dir),
    }
