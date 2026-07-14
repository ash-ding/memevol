"""Singularity-sandboxed evaluator.

Runs `forge/launch.py` inside the harness's resolved image with a selective
bind list — NOT a whole-project /app:ro bind. Only the directories actually
needed by launch.py + harness imports are exposed:

  singularity exec \\
      --containall \\
      --bind <project_root>/common:/app/common:ro          # full common/ pkg
      --bind <project_root>/datasets:/app/datasets:ro      # full datasets/ pkg (incl. raw data)
      --bind <project_root>/forge/__init__.py:/app/forge/__init__.py:ro
      --bind <project_root>/forge/launch.py:/app/forge/launch.py:ro
      --bind <project_root>/forge/harness_base.py:/app/forge/harness_base.py:ro
      --bind <harness_dir>:/harness:ro \\
      --bind <out_dir>:/out:rw \\
      --env OPENAI_API_KEY=... ANTHROPIC_API_KEY=... \\
      --env PYTHONPATH=/app EVALS_LOG_DIR=/out ...
      <image.sif> \\
      python /app/forge/launch.py --harness-dir /harness --out-dir /out ...

What's NOT bound (vs the previous /app:ro bind):
  - /app/workspace/     (cross-run isolation; this run's workspace is reachable
                         only via /out for the output dir)
  - /app/.env           (keys already injected via --env; load_dotenv() no-ops)
  - /app/.git, /app/baselines, /app/docs, /app/CLAUDE.md, /app/configs, etc.
  - /app/forge/{orchestrator,proposer,evaluator,...}.py (host-side outer-loop)
  - /app/seeds          (evaluator never reads seeds)
  - /app/venv           (container has its own python env)

Uses alma's dual-signal wait: race process.communicate() against score.json
appearance, kill on wall-clock timeout.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import dotenv_values
except ImportError:
    def dotenv_values(path):  # type: ignore
        out = {}
        try:
            with open(path) as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, _, v = s.partition("=")
                        out[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
        return out

from forge.paths import PROJECT_ROOT

log = logging.getLogger("forge.evaluator")

# Wall-clock caps per staged-evaluation tier. Small tiers fail fast; the
# terminal tier keeps the historical 8h budget.
SUBPROCESS_TIMEOUT = {
    "sanity": 2 * 3600,
    "stage1": 2 * 3600,
    "stage2": 4 * 3600,
    "stage3": 8 * 3600,
}

GRACE_AFTER_SCORE = 60
SCORE_POLL_INTERVAL = 5


async def run_evaluation(
    harness_dir: Path,
    image_path: Path,
    out_dir: Path,
    *,
    dataset: str = "dynamicmem",
    split: str = "search",   # search | test — benchmark split
    stage: str = "stage3",
    stage_spec: Dict[str, Any],
    model: str = "gpt-5-mini",
    judge_model: str = "gpt-5-mini",
    update_type: str = "all_at_once",
    max_sample_concurrent: int = 3,
    memcache_dir: Optional[Path] = None,
    gpu: bool = False,
    anthropic_transport: str = "api",
    vertex_cfg: Optional[Dict[str, Any]] = None,
) -> Path:
    """Evaluate a harness inside Singularity.

    `anthropic_transport` / `vertex_cfg` (from cfg.llm.*) configure how any
    claude-* model inside the container (QA agent / judge via common.llm)
    reaches Anthropic: "api" needs nothing beyond the ANTHROPIC_API_KEY env
    already injected below; "vertex" additionally injects
    MEMEVOL_ANTHROPIC_TRANSPORT + the official Vertex env vars and RO-binds
    the GCP credentials json at /gcp/credentials.json. NOTE: in-container
    vertex requires google-auth in the image — eval-base images built before
    2026-07-08 lack it (see containers/base_requirements_cpu.txt).

    Returns out_dir on success (or on score.json-observed completion); raises
    RuntimeError if the subprocess hangs past the wall-clock cap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale score.json so its appearance unambiguously means "this run
    # finished". Without this, a leftover from a prior run would fire the
    # dual-signal wait immediately and cause a premature kill.
    stale = out_dir / "score.json"
    if stale.exists():
        try:
            stale.unlink()
        except OSError as exc:
            log.warning(f"evaluator: could not remove stale score.json: {exc}")

    env_vals = dotenv_values(str(PROJECT_ROOT / ".env"))
    openai_key = env_vals.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    anthropic_key = env_vals.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

    cmd = [
        "singularity", "exec",
        "--containall",
    ]
    if gpu:
        # Bind-mounts host NVIDIA driver libs (libcuda, libnvidia-ml, ...) so
        # the container's torch+cu126 wheel can talk to the driver. Required
        # for torch.cuda.is_available() to return True.
        cmd.append("--nv")
    cmd += [
        # Selective binds — only what launch.py + harness actually need.
        # See module docstring for what's intentionally NOT bound.
        "--bind", f"{PROJECT_ROOT}/common:/app/common:ro",
        "--bind", f"{PROJECT_ROOT}/datasets:/app/datasets:ro",
        "--bind", f"{PROJECT_ROOT}/forge/__init__.py:/app/forge/__init__.py:ro",
        "--bind", f"{PROJECT_ROOT}/forge/launch.py:/app/forge/launch.py:ro",
        # Harnesses inherit forge.harness_base.MemoStructure (subclass of the
        # common ABC) — the module must be importable inside the container.
        "--bind", f"{PROJECT_ROOT}/forge/harness_base.py:/app/forge/harness_base.py:ro",
        "--bind", f"{harness_dir}:/harness:ro",
        "--bind", f"{out_dir}:/out:rw",
    ]
    if memcache_dir is not None:
        # Cross-stage memory snapshots (RW; shared across this harness's
        # stage execs for one dataset).
        cmd += ["--bind", f"{memcache_dir}:/memcache:rw"]

    # Anthropic transport for claude-* models inside the container. "api" is
    # common.llm's default — inject nothing so the env stays minimal.
    if anthropic_transport == "vertex":
        vc = vertex_cfg or {}
        # Same resolution chain as the proposer's vertex auth (explicit path →
        # $GOOGLE_APPLICATION_CREDENTIALS → gcloud ADC default).
        from forge.proposer import _resolve_gcp_credentials
        gcp_creds = _resolve_gcp_credentials(vc)
        cmd += [
            "--bind", f"{gcp_creds}:/gcp/credentials.json:ro",
            "--env", "MEMEVOL_ANTHROPIC_TRANSPORT=vertex",
            "--env", f"ANTHROPIC_VERTEX_PROJECT_ID={vc.get('project_id', '')}",
            "--env", f"CLOUD_ML_REGION={vc.get('region', '')}",
            "--env", "GOOGLE_APPLICATION_CREDENTIALS=/gcp/credentials.json",
        ]

    cmd += [
        "--env", f"OPENAI_API_KEY={openai_key}",
        "--env", f"ANTHROPIC_API_KEY={anthropic_key}",
        "--env", "PYTHONPATH=/app",
        "--env", "PYTHONUNBUFFERED=1",
        # Alma's logger writes rotating logs to EVALS_LOG_DIR. Send them into
        # the per-run out dir so they're writable and colocated with artifacts.
        "--env", "EVALS_LOG_DIR=/out",
        "--env", "MEMEVOL_LOG_FILE=subprocess.log",
        str(image_path),
        "python", "/app/forge/launch.py",
        "--harness-dir", "/harness",
        "--out-dir", "/out",
        "--dataset", dataset,
        "--split", split,
        "--stage", stage,
        "--stage-spec", json.dumps(stage_spec),
        "--model", model,
        "--judge-model", judge_model,
        "--update-type", update_type,
        "--max-sample-concurrent", str(max_sample_concurrent),
    ]
    if memcache_dir is not None:
        cmd += ["--memcache-dir", "/memcache"]

    log.info(
        f"evaluator: launching {harness_dir.name} [{dataset}/{stage}] with {image_path.name} "
        f"(spec={json.dumps(stage_spec)}, gpu={'on' if gpu else 'off'})"
    )

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(PROJECT_ROOT),
        stderr=asyncio.subprocess.PIPE,
    )

    timeout_s = SUBPROCESS_TIMEOUT.get(stage, 8 * 3600)
    score_path = out_dir / "score.json"

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

    stderr_bytes = b""
    forced_kill = False

    if not done:
        for t in pending:
            t.cancel()
        log.error(
            f"evaluator: HUNG past {timeout_s}s for {harness_dir.name} — killing"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            log.error(f"evaluator: failed to reap {harness_dir.name} within 5s")
        raise RuntimeError(
            f"Subprocess for {harness_dir.name} timed out past {timeout_s}s"
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
            f"evaluator: score.json landed for {harness_dir.name} but "
            f"communicate() still pending; grace {GRACE_AFTER_SCORE}s"
        )
        try:
            _, stderr_bytes = await asyncio.wait_for(
                communicate_task, timeout=GRACE_AFTER_SCORE
            )
        except asyncio.TimeoutError:
            forced_kill = True
            log.warning(
                f"evaluator: asyncio exit notification stuck for "
                f"{harness_dir.name}; force-killing"
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                log.error(f"evaluator: failed to reap {harness_dir.name} after SIGKILL")
            stderr_bytes = b""

    if not forced_kill and process.returncode not in (0, None):
        stderr_text = stderr_bytes.decode(errors="replace").strip() if stderr_bytes else ""
        log.error(
            f"evaluator: subprocess exited rc={process.returncode} for {harness_dir.name} [{dataset}]"
            + (f"\nstderr:\n{stderr_text[-2000:]}" if stderr_text else "")
        )

    log.info(f"evaluator: done {harness_dir.name} [{dataset}] → {out_dir}")
    return out_dir
