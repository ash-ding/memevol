"""EvolveMem eval runner — stages the round's θ as a config.json, then spawns
baselines/evolve/evolvemem/launch.py in a subprocess (pattern copied from
evolve/alma/eval_runner.py per the baselines README convention).

Per-run output directory:
    baselines/evolve/evolvemem/results/<dataset>/<run_id>_<status>/

The dual-signal wait (clean exit OR score.json appearing OR wall-clock
timeout) is carried over from alma verbatim — see that file's comments for
the Python 3.12 pidfd rationale.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
EVOLVEMEM_ROOT = PROJECT_ROOT / "baselines" / "evolve" / "evolvemem"

SUBPROCESS_TIMEOUT_S = 8 * 3600


def get_output_run_dir(run_id: str, status: str, dataset: str) -> Path:
    return EVOLVEMEM_ROOT / "results" / dataset / f"{run_id}_{status}"


async def run_evaluation(
    run_id: str,
    config: Dict[str, Any],
    dataset: str,
    status: str = "search",
    model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    eval_n_qa: Optional[int] = None,
    max_logs: Optional[int] = None,
    max_sample_concurrent: int = 3,
    output_run_dir: Optional[Path] = None,
    substrate: str = "native",
) -> Path:
    """Stage `config` and run one full evaluation pass. Returns the run dir."""
    if output_run_dir is None:
        output_run_dir = get_output_run_dir(run_id, status, dataset)
    output_run_dir = Path(output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

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
        str(EVOLVEMEM_ROOT / "launch.py"),
        "--memory_id", run_id,
        "--output_run_dir", str(output_run_dir),
        "--dataset", dataset,
        "--model", model,
        "--eval_n_samples", str(eval_n_samples),
        "--status", status,
        "--max_sample_concurrent", str(max_sample_concurrent),
        "--judge_model", judge_model,
        "--substrate", substrate,
    ]
    if max_logs is not None:
        launch_args += ["--max_logs", str(max_logs)]
    if eval_n_qa is not None:
        launch_args += ["--eval_n_qa", str(eval_n_qa)]

    env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        "EVOLVEMEM_CONFIG": str(config_path),
        "EVALS_LOG_DIR": str(output_run_dir),
        "MEMEVOL_LOG_FILE": "subprocess.log",
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    log.info(f"Running evaluation: run_id={run_id} dataset={dataset} status={status}")

    process = await asyncio.create_subprocess_exec(
        *launch_args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stderr=asyncio.subprocess.PIPE,
    )

    score_path = output_run_dir / "score.json"
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
        timeout=SUBPROCESS_TIMEOUT_S,
        return_when=asyncio.FIRST_COMPLETED,
    )

    forced_kill_after_score = False
    stderr_bytes = b""

    if not done:
        for t in pending:
            t.cancel()
        log.error(
            f"Evaluation subprocess HUNG for run_id={run_id} after "
            f"{SUBPROCESS_TIMEOUT_S}s (neither clean exit nor score.json) — killing"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"Failed to reap subprocess for run_id={run_id} within 5s")
        raise RuntimeError(
            f"Subprocess for run_id={run_id} timed out past {SUBPROCESS_TIMEOUT_S}s and was killed"
        )

    if communicate_task in done:
        _, stderr_bytes = communicate_task.result()
        score_task.cancel()
        try:
            await score_task
        except asyncio.CancelledError:
            pass
    else:
        log.warning(
            f"score.json written for run_id={run_id} but subprocess has not exited; "
            f"awaiting up to {GRACE_AFTER_SCORE}s for clean shutdown"
        )
        try:
            _, stderr_bytes = await asyncio.wait_for(communicate_task, timeout=GRACE_AFTER_SCORE)
        except asyncio.TimeoutError:
            forced_kill_after_score = True
            log.warning(
                f"asyncio exit notification stuck for run_id={run_id} despite completed work — "
                f"force-killing to unblock pipeline"
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.error(f"Failed to reap run_id={run_id} within 5s after SIGKILL")
            stderr_bytes = b""

    if forced_kill_after_score:
        log.info(
            f"Evaluation completed via score.json (subprocess reaped by SIGKILL after grace): "
            f"run_id={run_id} → {output_run_dir}"
        )
    elif process.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(
            f"Evaluation subprocess exited with code {process.returncode} for run_id={run_id}"
            + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else "")
        )
    else:
        log.info(f"Evaluation finished: run_id={run_id} → {output_run_dir}")

    return output_run_dir


def read_score(output_run_dir: Path) -> float:
    """Read the overall score from a finished run dir (raises if missing)."""
    score_path = Path(output_run_dir) / "score.json"
    if not score_path.exists():
        raise FileNotFoundError(f"score.json missing under {output_run_dir}")
    with score_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("benchmark_eval_score", {}).get("benchmark_overall_eval_score", 0.0))
