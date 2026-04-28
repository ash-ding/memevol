"""Host-side launcher for the Singularity-sandboxed proposer.

This module spawns `forge/propose_in_container.py` inside `proposer-base.sif`
with a SELECTIVE bind list — CC's reachable surface inside the container is
exactly the bind set built by `_build_singularity_cmd` (see that function's
docstring for the per-bind rationale). Nothing else from the host is visible.

History:
  v6 (2026-04-25)  proposer moved off the host into Singularity (~60 MB SIF
                   inheriting from python:3.12-slim).
  v8 (2026-04-26)  proposer-base rebuilt from eval-base.sif (~3.4 GB) so
                   CC has the full ML stack for `python -c` self-validation;
                   default disallowed_tools relaxed to ["mcp__*"] (was
                   [Bash, WebFetch, WebSearch, mcp__*]); jq + tree added.
  v10 (2026-04-27) /app:ro whole-root bind replaced with selective bind to
                   common/{harness_base,llm,logger,__init__}.py + datasets/
                   + forge/{__init__,prompts,propose_in_container}.py;
                   /seeds bind dropped (selected seed already copied into
                   /workspace/harnesses/0/ at startup). PROPOSER_SYSTEM made
                   sanity-conditional via build_proposer_system(...).

In-container runtime (set by propose_in_container.py from CLI args):
  cwd                        /workspace = workspace/<run_id>/
  PYTHONPATH                 /app  (so common.* / datasets.* import cleanly)
  allowed_tools              not set (= SDK default = full toolset)
  disallowed_tools           configurable; default ["mcp__*"]. See
                             cfg["proposer"]["disallowed_tools"] in
                             orchestrator.py / DEFAULT_CONFIG.
  system_prompt              build_proposer_system(sanity_enabled), where
                             sanity_enabled reflects this run's actual
                             sanity-layer behavior.
  max_turns                  caps the agent loop
  timeout_s                  wall-clock cap; enforced via subprocess.wait
                             + SIGTERM/SIGKILL escalation on the singularity
                             exec process group.

The proposer ONLY writes files (into /workspace, RW). It never executes the
harness end-to-end — evaluation happens later in a separate Singularity
container (forge/evaluator.py). It MAY do `python -c "..."` for import-level
self-validation; this is encouraged by PROPOSER_SYSTEM.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from forge.paths import (
    PROJECT_ROOT,
    PROPOSER_BASE_SIF,
    paths,
)

log = logging.getLogger("forge.proposer")


# ---------------------------------------------------------------------------
# Bind / launch helpers
# ---------------------------------------------------------------------------

# Where the host's `claude` CLI install lives. CC is the new-style native
# binary distribution (a 100-MB-ish ELF + a versions/ subtree). The Python
# claude_code_sdk searches PATH for `claude`, so we just need to put the
# binary at /usr/local/bin/claude inside the container.
_HOME = Path.home()
_HOST_CLAUDE_BIN = _HOME / ".local" / "bin" / "claude"
_HOST_CLAUDE_SHARE = _HOME / ".local" / "share" / "claude"
_HOST_CLAUDE_CREDS = _HOME / ".claude" / ".credentials.json"


class ProposerLaunchError(RuntimeError):
    """Raised when the proposer subprocess cannot be launched or exits abnormally."""


def _check_environment() -> None:
    """Pre-flight checks. Raise with actionable messages so the user knows
    exactly what to install or configure."""
    if not PROPOSER_BASE_SIF.exists():
        raise ProposerLaunchError(
            f"proposer-base.sif missing at {PROPOSER_BASE_SIF}. "
            f"Build it once with:\n"
            f"  PATH=$HOME/.local/bin:$PATH singularity build "
            f"{PROPOSER_BASE_SIF} containers/proposer-base.def"
        )
    if not _HOST_CLAUDE_BIN.exists():
        raise ProposerLaunchError(
            f"host claude CLI not found at {_HOST_CLAUDE_BIN}. "
            f"Install Claude Code (subscription login) and ensure `which claude` resolves to it."
        )
    if not _HOST_CLAUDE_CREDS.exists():
        raise ProposerLaunchError(
            f"host claude credentials missing at {_HOST_CLAUDE_CREDS}. "
            f"Run `claude login` first."
        )


def _prepare_claude_home() -> Path:
    """Stage a writable per-run HOME for the in-container `claude` CLI.

    `claude` is a Bun-compiled binary that appendFileSync's to log files,
    refresh tokens, and cache state under $HOME/.claude/. If we only bind
    .credentials.json (RO single file), CC crashes silently with ENOENT on
    its first log write.

    Workaround: create `<workspace>/.proposer_home/.claude/`, copy the
    host's credentials into it, and bind the whole `.proposer_home` as
    /root inside the container. CC then writes its logs/sessions there
    (per-run, ephemeral — not synced back to the host's ~/.claude/, which
    is the right isolation behavior).
    """
    proposer_home = paths.workspace / ".proposer_home"
    claude_dir = proposer_home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    creds_dst = claude_dir / ".credentials.json"
    # Always refresh from host so an expired/refreshed host token reaches
    # the container next launch. Host token mtime check would be brittle.
    shutil.copy2(_HOST_CLAUDE_CREDS, creds_dst)
    creds_dst.chmod(0o600)
    return proposer_home


def _build_singularity_cmd(
    *,
    propose_args: List[str],
    proposer_home: Path,
) -> List[str]:
    """Build the `singularity exec ...` argv with the selective bind list.

    Bind list rationale:
      - LAYER-1 wrapper script (propose_in_container.py + its forge.prompts
        import) needs to be reachable via /app/forge/. CC never reads these.
      - CC's reference materials live under /app/{common,datasets}/. Only the
        files that contribute to writing a correct harness are bound:
          common/harness_base.py   the MemoStructure ABC contract
          common/llm.py            Agent + Embedding helpers (token-tracked)
          common/logger.py         transitive dep of common.llm
          common/__init__.py       package marker
          datasets/                full directory: env.py, workflow.py,
                                   prompts.py per benchmark + raw data files.
                                   No selective filtering — see PROGRESS for
                                   the cheat-via-trace-traces note.
      - workspace/ is CC's cwd (RW). Per-run isolation: only THIS run is bound,
        sibling runs unreachable.
      - claude binary + scratch HOME unchanged from v6.
      - NOT bound (vs the previous /app:ro whole-root bind):
          /app/{forge/orchestrator.py, forge/proposer.py, forge/...}.py
                                   host-side outer-loop, irrelevant
          /app/common/{workflow,judge,tokens}.py    internal infrastructure
          /app/baselines, /app/docs, /app/configs, /app/seeds, /app/venv
          /app/workspace           cross-run isolation
          /app/.env, /app/.git     not needed; OAuth via /root binds
    """
    binds = [
        # LAYER-1 wrapper script imports
        f"{PROJECT_ROOT}/forge/__init__.py:/app/forge/__init__.py:ro",
        f"{PROJECT_ROOT}/forge/prompts.py:/app/forge/prompts.py:ro",
        f"{PROJECT_ROOT}/forge/propose_in_container.py:/app/forge/propose_in_container.py:ro",

        # CC's reference materials
        f"{PROJECT_ROOT}/common/__init__.py:/app/common/__init__.py:ro",
        f"{PROJECT_ROOT}/common/harness_base.py:/app/common/harness_base.py:ro",
        f"{PROJECT_ROOT}/common/llm.py:/app/common/llm.py:ro",
        f"{PROJECT_ROOT}/common/logger.py:/app/common/logger.py:ro",
        f"{PROJECT_ROOT}/datasets:/app/datasets:ro",

        # Workspace (cwd) + claude CLI + scratch HOME
        f"{paths.workspace}:/workspace:rw",
        f"{_HOST_CLAUDE_BIN}:/usr/local/bin/claude:ro",
        f"{_HOST_CLAUDE_SHARE}:/usr/local/share/claude:ro",
        f"{proposer_home}:/root:rw",
    ]
    cmd = [
        "singularity", "exec",
        "--containall",
    ]
    for b in binds:
        cmd += ["--bind", b]
    cmd += [
        # SINGULARITYENV_HOME=/root is implied by --containall; no need to set.
        "--env", "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING=true",
        # PYTHONPATH so any `python -c "from common.harness_base import ..."`
        # CC runs for self-validation finds the bound /app/common package.
        "--env", "PYTHONPATH=/app",
        str(PROPOSER_BASE_SIF),
        "python", "/app/forge/propose_in_container.py",
    ] + propose_args
    return cmd


async def _stream_subprocess(cmd: List[str], timeout_s: int, label: str) -> int:
    """Run `cmd` and forward stdout line-by-line to the host log.

    Returns the subprocess exit code. Raises TimeoutError on wall-clock
    timeout (after escalating SIGTERM → SIGKILL on the singularity exec
    process group).
    """
    log.info(f"{label}: launching ({len(cmd)} argv tokens, image={PROPOSER_BASE_SIF.name})")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Put singularity in its own process group so we can kill it cleanly.
        preexec_fn=os.setsid,
    )

    async def _pump_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            # Lines from propose_in_container.py are pre-tagged
            # ("proposer·tool: ...", "proposer·text: ...", etc).
            # Map error tag → log.error, info → log.info, everything else → debug.
            if line.startswith("proposer·error"):
                log.error(line)
            elif line.startswith("proposer·info"):
                log.info(line)
            elif line.startswith("proposer·tool"):
                log.info(line)
            elif line.startswith("proposer·text"):
                log.info(line)
            else:
                log.debug(line)

    pump_task = asyncio.create_task(_pump_stdout())
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        log.error(f"{label}: HUNG past {timeout_s}s — killing process group")
        try:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (ProcessLookupError, asyncio.TimeoutError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                await proc.wait()
            except ProcessLookupError:
                pass
        raise TimeoutError(f"proposer subprocess timed out after {timeout_s}s")
    finally:
        # Drain any remaining stdout so log lines aren't lost.
        try:
            await asyncio.wait_for(pump_task, timeout=5)
        except asyncio.TimeoutError:
            pump_task.cancel()

    return proc.returncode if proc.returncode is not None else -1


# ---------------------------------------------------------------------------
# Public API — same signatures as v5 host-side proposer; orchestrator unchanged.
# ---------------------------------------------------------------------------

async def propose(
    new_id: str,
    *,
    model: str = "claude-opus-4-7",
    max_turns: int = 80,
    timeout_s: int = 25 * 60,
    disallowed_tools: Optional[List[str]] = None,
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
) -> Path:
    """Run a sandboxed proposer; return the new harness directory on success.

    No parent_id passed — CC explores `harnesses/` itself and decides which
    prior(s) to draw from. CC records its choices in `meta.json::parent_ids`.

    Raises:
      ProposerLaunchError  pre-flight (SIF/binary/creds missing).
      TimeoutError         the SDK call exceeds timeout_s.
      RuntimeError         the proposer exits without PROPOSAL_READY, or
                           the subprocess returns a non-zero exit code.
    """
    _check_environment()

    new_dir = paths.harnesses_dir / new_id
    # mkdir on host so the bind sees it; container has it RW via /workspace.
    new_dir.mkdir(parents=True, exist_ok=True)

    proposer_home = _prepare_claude_home()
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "propose",
        "--model", model,
        "--max-turns", str(max_turns),
        "--timeout-s", str(timeout_s),
        "--disallowed-tools", ",".join(disallowed_tools or []),
        "--sanity-enabled", "true" if sanity_enabled else "false",
        "--active-datasets", ",".join(active_datasets or []),
    ]
    cmd = _build_singularity_cmd(propose_args=propose_args, proposer_home=proposer_home)

    rc = await _stream_subprocess(cmd, timeout_s=timeout_s + 60, label=f"proposer[{new_id}]")
    if rc != 0:
        raise RuntimeError(
            f"proposer subprocess exited with rc={rc} for {new_id}; "
            f"see orchestrator log for the streamed CC output"
        )

    sentinel = new_dir / "PROPOSAL_READY"
    if not sentinel.exists():
        raise RuntimeError(
            f"proposer finished but PROPOSAL_READY missing at {new_dir}"
        )
    return new_dir


async def propose_with_fix(
    new_id: str,
    error_trace: str,
    *,
    model: str = "claude-opus-4-7",
    max_turns: int = 80,
    timeout_s: int = 25 * 60,
    disallowed_tools: Optional[List[str]] = None,
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
) -> Path:
    """Ask CC to Read + Edit the existing harness to fix a sanity-check failure.

    The error trace is staged to a host-side file inside the harness dir so
    the container can read it via the workspace bind. The file is removed
    after the subprocess returns (success or failure).
    """
    _check_environment()

    new_dir = paths.harnesses_dir / new_id
    if not (new_dir / "harness.py").exists():
        raise RuntimeError(
            f"propose_with_fix called but {new_dir}/harness.py does not exist"
        )

    # Stage the error trace inside the harness dir so it's reachable via the
    # /workspace bind. Hidden filename so CC doesn't accidentally read it as
    # part of normal exploration.
    trace_host = new_dir / ".fix_error_trace.txt"
    trace_host.write_text(error_trace, encoding="utf-8")
    trace_in_container = f"/workspace/harnesses/{new_id}/.fix_error_trace.txt"

    proposer_home = _prepare_claude_home()
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "fix",
        "--error-trace-file", trace_in_container,
        "--model", model,
        "--max-turns", str(max_turns),
        "--timeout-s", str(timeout_s),
        "--disallowed-tools", ",".join(disallowed_tools or []),
        "--sanity-enabled", "true" if sanity_enabled else "false",
        "--active-datasets", ",".join(active_datasets or []),
    ]
    cmd = _build_singularity_cmd(propose_args=propose_args, proposer_home=proposer_home)
    try:
        rc = await _stream_subprocess(cmd, timeout_s=timeout_s + 60, label=f"propose_fix[{new_id}]")
    finally:
        try:
            trace_host.unlink()
        except OSError:
            pass

    if rc != 0:
        raise RuntimeError(
            f"propose_with_fix subprocess exited with rc={rc} for {new_id}"
        )
    return new_dir
