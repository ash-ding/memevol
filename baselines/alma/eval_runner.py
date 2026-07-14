"""
Alma eval runner — spawns baselines/alma/launch.py in a subprocess.

Produces a per-run output directory at:
    baselines/alma/results/<dataset>/<SHA>_<status>_<mode>/

Wall-clock subprocess timeout is enforced per mode (check=2h, eval=8h) with a
hard kill + RuntimeError when exceeded.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALMA_ROOT = PROJECT_ROOT / "baselines" / "alma"

SUBPROCESS_TIMEOUT = {
    "check": 2 * 3600,   # 2 hours
    "eval":  8 * 3600,   # 8 hours
}


def get_output_run_dir(memory_SHA: str, status: str, mode: str, dataset: str = "dynamicmem") -> Path:
    """Canonical per-run output directory."""
    return ALMA_ROOT / "results" / dataset / f"{memory_SHA}_{status}_{mode}"


async def run_evaluation(
    memory_SHA: str,
    mode: str = "eval",
    model: str = "gpt-5-mini",
    eval_n_samples: int = 6,
    status: str = "search",
    update_type: str = "all_at_once",
    n_chunks: int = 5,
    max_logs: Optional[int] = None,
    eval_n_qa: Optional[int] = None,
    max_sample_concurrent: int = 6,
    judge_model: str = "gpt-5-mini",
    check_n_samples: int = 6,
    check_n_qa: int = 3,
    source_path: Optional[Path] = None,
    output_run_dir: Optional[Path] = None,
    dataset: str = "dynamicmem",
) -> Path:
    """Copy memo code to alma/memo_test/ and run launch.py in a subprocess.

    Returns the per-run output directory path.

    By default the source .py file is looked up at
    ``alma/memo_archive/{dynamicmem,baseline}/memo_structure_<SHA>.py`` and the
    output directory is ``alma/results/dynamicmem/<SHA>_<status>_<mode>/``.
    Both defaults can be overridden via ``source_path`` / ``output_run_dir`` so
    callers outside alma (e.g. baselines/meta-harness) can reuse this runner
    without touching alma's archive layout.
    """

    memo_test_dir = ALMA_ROOT / "memo_test"
    memo_test_dir.mkdir(parents=True, exist_ok=True)

    if source_path is not None:
        memo_path = Path(source_path)
        if not memo_path.exists():
            raise FileNotFoundError(f"source_path does not exist: {memo_path}")
    else:
        memo_archive = ALMA_ROOT / "memo_archive"
        memo_path = memo_archive / dataset / f"memo_structure_{memory_SHA}.py"
        if not memo_path.exists():
            memo_path = memo_archive / "baseline" / f"memo_structure_{memory_SHA}.py"
        if not memo_path.exists():
            raise FileNotFoundError(f"Memo structure file not found for SHA: {memory_SHA}")

    # Per-SHA staging filename — previous `memo_test.py` shared slot caused a
    # race when `max_memo_concurrent > 1`: two concurrent subprocesses would
    # `shutil.copy` into the same path and then the later importer would read
    # the OTHER memo's code, producing wrong-SHA reward attribution.
    dest = memo_test_dir / f"memo_test_{memory_SHA}.py"
    shutil.copy(str(memo_path), str(dest))

    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    api_key = env_vals.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    if output_run_dir is None:
        output_run_dir = get_output_run_dir(memory_SHA, status, mode, dataset)
    else:
        output_run_dir = Path(output_run_dir)
    # Send the subprocess's own rotating log to its per-run output dir so every
    # memo's subprocess log is isolated from the main-process log and from
    # other concurrent subprocesses. Pre-create the dir so the logger's
    # mkdir(exist_ok=True) at import time does not race with launch.py's
    # run_dir.mkdir(parents=True).
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # Remove any stale score.json from a prior run with the same (SHA, status,
    # mode) triple. The dual-signal wait below treats score.json as the "work
    # done" side-channel, and a leftover file from an earlier run would fire
    # that signal on startup — the parent would then force-kill the subprocess
    # before it actually did anything. Deleting here makes score.json's
    # appearance unambiguously mean "this subprocess finished".
    stale_score = output_run_dir / "score.json"
    if stale_score.exists():
        try:
            stale_score.unlink()
        except OSError as exc:
            log.warning(f"Could not remove stale score.json at {stale_score}: {exc}")

    launch_args = [
        sys.executable, "-u",
        str(ALMA_ROOT / "launch.py"),
        "--module_path", str(dest),
        "--memory_id", memory_SHA,
        "--output_run_dir", str(output_run_dir),
        "--update_type", update_type,
        "--n_chunks", str(n_chunks),
        "--model", model,
        "--eval_n_samples", str(eval_n_samples),
        "--status", status,
        "--max_sample_concurrent", str(max_sample_concurrent),
        "--mode", mode,
        "--judge_model", judge_model,
        "--check_n_samples", str(check_n_samples),
        "--check_n_qa", str(check_n_qa),
        "--dataset", dataset,
    ]
    if max_logs is not None:
        launch_args += ["--max_logs", str(max_logs)]
    if eval_n_qa is not None:
        launch_args += ["--eval_n_qa", str(eval_n_qa)]

    env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        # Subprocess logger writes to <output_run_dir>/subprocess.log, NOT to
        # the parent-process log file. Override both variables explicitly so
        # inherited values from the main process do not leak through.
        "EVALS_LOG_DIR": str(output_run_dir),
        "MEMEVOL_LOG_FILE": "subprocess.log",
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    log.info(f"Running evaluation: SHA={memory_SHA} mode={mode} update_type={update_type}")

    process = await asyncio.create_subprocess_exec(
        *launch_args,
        cwd=str(PROJECT_ROOT),
        env=env,
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = SUBPROCESS_TIMEOUT.get(mode, 8 * 3600)
    score_path = output_run_dir / "score.json"

    # Dual-signal wait — race between:
    #   (a) process.communicate() returning cleanly (asyncio's pidfd/epoll
    #       notification works),
    #   (b) score.json appearing on disk (subprocess finished its real work
    #       but may be stuck in interpreter shutdown / asyncio may have
    #       missed the pidfd event — a sporadic Python 3.12 issue observed
    #       to keep subprocesses as unreaped zombies for hours),
    #   (c) wall-clock timeout (SUBPROCESS_TIMEOUT[mode]).
    # Whichever wins, we proceed. If (b) wins we give the process a short
    # grace window to exit on its own, then force-kill to unblock the
    # pipeline rather than wait for wall-clock.
    GRACE_AFTER_SCORE = 60   # seconds to wait for clean exit once score.json exists
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
        # Wall-clock timeout with neither signal firing → true hang.
        for t in pending:
            t.cancel()
        log.error(
            f"Evaluation subprocess HUNG for SHA={memory_SHA} mode={mode} "
            f"after {timeout_s}s (neither clean exit nor score.json observed) — killing"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"Failed to reap subprocess for SHA={memory_SHA} within 5s")
        raise RuntimeError(
            f"Subprocess for SHA={memory_SHA} mode={mode} timed out "
            f"past {timeout_s}s and was killed"
        )

    if communicate_task in done:
        # Happy path: asyncio noticed the exit promptly.
        _, stderr_bytes = communicate_task.result()
        score_task.cancel()
        try:
            await score_task
        except asyncio.CancelledError:
            pass
    else:
        # score.json landed but communicate() still pending — likely asyncio
        # pidfd notification race. Give the subprocess a grace window to
        # clean up on its own; if still not exiting, force-kill.
        log.warning(
            f"score.json written for SHA={memory_SHA} but subprocess has not exited; "
            f"awaiting up to {GRACE_AFTER_SCORE}s for clean shutdown"
        )
        try:
            _, stderr_bytes = await asyncio.wait_for(communicate_task, timeout=GRACE_AFTER_SCORE)
        except asyncio.TimeoutError:
            forced_kill_after_score = True
            log.warning(
                f"asyncio exit notification stuck for SHA={memory_SHA} despite completed work — "
                f"force-killing to unblock pipeline"
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            # communicate_task was already cancelled by the failed wait_for
            # above — DO NOT re-await it (it would raise CancelledError, which
            # is a BaseException and slips through `except Exception`). Reap
            # the killed subprocess via a direct process.wait() instead. stderr
            # cannot be recovered cleanly from the cancelled task.
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.error(f"Failed to reap SHA={memory_SHA} within 5s after SIGKILL")
            stderr_bytes = b""

    if forced_kill_after_score:
        log.info(
            f"Evaluation completed via score.json (subprocess reaped by SIGKILL after grace): "
            f"SHA={memory_SHA} → {output_run_dir}"
        )
    elif process.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(
            f"Evaluation subprocess exited with code {process.returncode} for SHA={memory_SHA}"
            + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else "")
        )
    else:
        log.info(f"Evaluation finished: SHA={memory_SHA} → {output_run_dir}")

    return output_run_dir
