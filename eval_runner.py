"""
Local evaluation runner for memevol (no Docker).

Replaces ALMA's eval_in_container.py. Runs evals/launch.py in a subprocess
using the current Python interpreter (alma conda env) with the necessary
environment variables set.
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

try:
    from evals.logger import get_logger
    log = get_logger("main")
except Exception:
    import logging
    log = logging.getLogger("main")

PROJECT_ROOT = Path(__file__).resolve().parent

# Wall-clock ceiling for the launch.py subprocess, by mode. Past this we assume
# the subprocess is hung (e.g. langchain_chroma cleanup deadlock, generated code
# leaving background tasks alive) and force-kill it. Numbers are deliberately
# loose: the longest healthy run we have observed is ~78 min for sanity check
# and ~6.4 h for full eval, so 2 h / 8 h give ~1.5x and ~1.25x headroom.
SUBPROCESS_TIMEOUT = {
    "check": 2 * 3600,   # 2 hours for sanity check
    "eval":  8 * 3600,   # 8 hours for full eval
}


async def run_evaluation(
    memory_SHA: str,
    mode: str = "eval",
    model: str = "gpt-5-mini",
    eval_n_users: int = 6,
    status: str = "search",
    update_type: str = "all_at_once",
    n_chunks: int = 5,
    max_logs: Optional[int] = None,
    eval_n_qa: Optional[int] = None,
    max_user_concurrent: int = 6,
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
    judge_model: str = "gpt-5-mini",
    check_n_users: int = 3,
    check_n_qa: int = 10,
):
    """Copy memo code to evals/memo_test/ and run launch.py in a subprocess."""

    evals_dir = PROJECT_ROOT / "evals"
    memo_test_dir = evals_dir / "memo_test"
    memo_test_dir.mkdir(parents=True, exist_ok=True)

    # Resolve memo code path (check domain dir first, then baseline)
    memo_archive = PROJECT_ROOT / "memo_archive"
    memo_path = memo_archive / "dynamicmem" / f"memo_structure_{memory_SHA}.py"
    if not memo_path.exists():
        memo_path = memo_archive / "baseline" / f"memo_structure_{memory_SHA}.py"
    if not memo_path.exists():
        raise FileNotFoundError(f"Memo structure file not found for SHA: {memory_SHA}")

    # Copy to evals/memo_test/memo_test.py (used by launch.py via --module_path)
    dest = memo_test_dir / "memo_test.py"
    shutil.copy(str(memo_path), str(dest))

    # Read API key
    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    api_key = env_vals.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    evals_log_dir = str(evals_dir / "logs")
    dynamicmem_data = str(PROJECT_ROOT / "dynamicmem")

    # Build launch.py command
    launch_args = [
        sys.executable, "-u", "-m", "launch",
        "--module_path", "memo_test/memo_test.py",
        "--memory_id", memory_SHA,
        "--update_type", update_type,
        "--n_chunks", str(n_chunks),
        "--model", model,
        "--eval_n_users", str(eval_n_users),
        "--status", status,
        "--max_user_concurrent", str(max_user_concurrent),
        "--mode", mode,
        "--n_score_bins", str(n_score_bins),
        "--samples_per_bin", str(samples_per_bin),
        "--judge_model", judge_model,
        "--check_n_users", str(check_n_users),
        "--check_n_qa", str(check_n_qa),
    ]
    if max_logs is not None:
        launch_args += ["--max_logs", str(max_logs)]
    if eval_n_qa is not None:
        launch_args += ["--eval_n_qa", str(eval_n_qa)]

    env = {
        **os.environ,
        "OPENAI_API_KEY": api_key,
        "EVALS_LOG_DIR": evals_log_dir,
        "DYNAMICMEM_DATA": dynamicmem_data,
        "PYTHONPATH": str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }

    log.info(f"Running evaluation: SHA={memory_SHA} mode={mode} update_type={update_type}")

    process = await asyncio.create_subprocess_exec(
        *launch_args,
        cwd=str(evals_dir),
        env=env,
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = SUBPROCESS_TIMEOUT.get(mode, 8 * 3600)
    try:
        _, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        log.error(
            f"Evaluation subprocess HUNG for SHA={memory_SHA} mode={mode} "
            f"after {timeout_s}s — killing"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass  # already dead, communicate just hadn't observed it
        # Give the kernel a few seconds to reap the killed process so we don't
        # leak a zombie. If wait() also hangs, log and move on.
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"Failed to reap subprocess for SHA={memory_SHA} within 5s")
        raise RuntimeError(
            f"Subprocess for SHA={memory_SHA} mode={mode} timed out "
            f"past {timeout_s}s and was killed"
        )

    if process.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(
            f"Evaluation subprocess exited with code {process.returncode} for SHA={memory_SHA}"
            + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else "")
        )
    else:
        log.info(f"Evaluation finished: SHA={memory_SHA}")
