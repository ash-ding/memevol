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
  2026-07-16       common/recorder.py added to the bind list (Basic_Recorder
                   moved out of harness_base — the in-container
                   `import common.harness_base` chain needs it).

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
  max_turns                  soft budget only — NOT enforced (kept for
                             backward compat; not plumbed to the agent CLI —
                             see propose_in_container.py --max-turns help)
  timeout_s                  wall-clock cap; enforced via subprocess.wait
                             + SIGTERM/SIGKILL escalation on the singularity
                             exec process group. This is the hard limit.

The proposer ONLY writes files (into /workspace, RW). It never executes the
harness end-to-end — evaluation happens later in a separate Singularity
container (forge/evaluator.py). It MAY do `python -c "..."` for import-level
self-validation; this is encouraged by PROPOSER_SYSTEM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from forge.paths import (
    PROJECT_ROOT,
    PROPOSER_BASE_SIF,
    paths,
)
from forge.prompts import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)

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

log = logging.getLogger("forge.proposer")


# ---------------------------------------------------------------------------
# Bind / launch helpers
# ---------------------------------------------------------------------------

# Host paths to the supported agent CLIs.
#
# claude_code: 100-MB-ish native binary + a versions/ subtree under
#   ~/.local/share/claude/. The in-container script invokes `claude -p
#   --output-format stream-json …` directly (no SDK dependency).
#
# codex: ~/.local/bin/codex is a symlink into ~/.codex/packages/standalone/.
#   The in-container script invokes `codex exec --json …`. Codex uses
#   OPENAI_API_KEY for auth (plumbed via SINGULARITYENV_OPENAI_API_KEY).
_HOME = Path.home()
_HOST_CLAUDE_BIN = _HOME / ".local" / "bin" / "claude"
_HOST_CLAUDE_SHARE = _HOME / ".local" / "share" / "claude"
_HOST_CLAUDE_CREDS = _HOME / ".claude" / ".credentials.json"

_HOST_CODEX_BIN = _HOME / ".local" / "bin" / "codex"
# codex's binary is a symlink — resolve it so the bind targets the real file
# (singularity can bind symlinks but it's cleaner to bind the resolved path).
_HOST_CODEX_BIN_RESOLVED = _HOST_CODEX_BIN.resolve() if _HOST_CODEX_BIN.exists() else _HOST_CODEX_BIN

# claude_code auth modes (cfg.proposer.claude_code.auth):
#   subscription — `claude login` OAuth: oauth_token env injection when
#                  ~/.claude/oauth_token exists, else credentials.json copy.
#   api_key      — ANTHROPIC_API_KEY (host env, .env fallback) injected as
#                  SINGULARITYENV_ANTHROPIC_API_KEY. Official auth precedence
#                  puts ANTHROPIC_API_KEY above OAuth tokens, so a leftover
#                  credentials.json in the scratch HOME can't win.
#   vertex       — Google Vertex AI: CLAUDE_CODE_USE_VERTEX=1 + project/region
#                  envs + GCP credentials json bound RO into the container.
#                  Cloud-provider flags have the HIGHEST precedence in CC's
#                  auth chain, so subscription leftovers are inert here too.
CLAUDE_AUTH_MODES = ("subscription", "api_key", "vertex")

# GCP Application Default Credentials — the path `gcloud auth
# application-default login` writes to.
_GCP_ADC_DEFAULT = _HOME / ".config" / "gcloud" / "application_default_credentials.json"
# Container-side path the resolved GCP credentials json is bound to. Kept
# OUTSIDE /root — that whole tree is the RW .proposer_home bind, and nesting
# a RO file bind inside another bind is needlessly fragile.
_VERTEX_CREDS_IN_CONTAINER = "/gcp/credentials.json"


class ProposerLaunchError(RuntimeError):
    """Raised when the proposer subprocess cannot be launched or exits abnormally."""


def _resolve_anthropic_api_key() -> str:
    """ANTHROPIC_API_KEY for claude_auth=api_key: host env first, then the
    project .env (same convention the evaluator uses for container keys).
    Returns "" when neither source has it — caller decides how to fail."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    return dotenv_values(str(PROJECT_ROOT / ".env")).get("ANTHROPIC_API_KEY", "") or ""


def _resolve_gcp_credentials(vertex_cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Locate the GCP credentials json for claude_auth=vertex.

    Precedence: explicit `vertex.credentials` config path →
    $GOOGLE_APPLICATION_CREDENTIALS → gcloud ADC default location.
    Raises ProposerLaunchError with setup instructions if none exists.
    """
    explicit = (vertex_cfg or {}).get("credentials")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(_GCP_ADC_DEFAULT)

    for cand in candidates:
        if cand.is_file():
            return cand
    raise ProposerLaunchError(
        "claude_auth=vertex: no GCP credentials json found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". On the machine running the orchestrator, either run "
        "`gcloud auth application-default login` (writes the ADC file), or "
        "point GOOGLE_APPLICATION_CREDENTIALS / cfg.proposer.claude_code."
        "vertex.credentials at a service-account or ADC json."
    )


def _check_environment(
    agent: str = "claude_code",
    *,
    claude_auth: str = "subscription",
    vertex_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Pre-flight checks. Raise with actionable messages so the user knows
    exactly what to install or configure for the chosen agent + auth mode."""
    if not PROPOSER_BASE_SIF.exists():
        raise ProposerLaunchError(
            f"proposer-base.sif missing at {PROPOSER_BASE_SIF}. "
            f"Build it once with:\n"
            f"  PATH=$HOME/.local/bin:$PATH singularity build "
            f"{PROPOSER_BASE_SIF} containers/proposer-base.def"
        )
    if agent == "claude_code":
        if claude_auth not in CLAUDE_AUTH_MODES:
            raise ProposerLaunchError(
                f"unknown claude_auth: {claude_auth!r}; valid: {CLAUDE_AUTH_MODES}"
            )
        if not _HOST_CLAUDE_BIN.exists():
            raise ProposerLaunchError(
                f"host claude CLI not found at {_HOST_CLAUDE_BIN}. "
                f"Install Claude Code and ensure `which claude` resolves to it."
            )
        if claude_auth == "subscription":
            if not _HOST_CLAUDE_CREDS.exists() and not _load_oauth_token():
                raise ProposerLaunchError(
                    f"claude_auth=subscription: no credentials. Run `claude login` "
                    f"(writes {_HOST_CLAUDE_CREDS}) or `claude setup-token` "
                    f"(write the token to {_HOST_OAUTH_TOKEN_FILE})."
                )
        elif claude_auth == "api_key":
            if not _resolve_anthropic_api_key():
                raise ProposerLaunchError(
                    "claude_auth=api_key: ANTHROPIC_API_KEY not found. Export it "
                    "in the host environment or add it to the project .env."
                )
        elif claude_auth == "vertex":
            vc = vertex_cfg or {}
            if not vc.get("project_id") or not vc.get("region"):
                raise ProposerLaunchError(
                    "claude_auth=vertex requires cfg.proposer.claude_code.vertex."
                    "{project_id, region} to be set (e.g. project_id: "
                    "itpc-gcp-ai-eng-claude, region: us-east5)."
                )
            _resolve_gcp_credentials(vc)  # raises with instructions if absent
    elif agent == "codex":
        if not _HOST_CODEX_BIN.exists():
            raise ProposerLaunchError(
                f"host codex CLI not found at {_HOST_CODEX_BIN}. "
                f"Install OpenAI Codex CLI and ensure `which codex` resolves to it."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise ProposerLaunchError(
                "OPENAI_API_KEY env var not set on host. Required for codex "
                "agent. Export it before launching (the orchestrator does NOT "
                "load .env into the host environment — only the evaluator "
                "reads .env, for container injection)."
            )
    else:
        raise ProposerLaunchError(f"unknown agent: {agent!r}")


def _prepare_claude_home(claude_auth: str = "subscription") -> Path:
    """Stage a writable per-run HOME for the in-container coding agents.

    Both `claude` and `codex` are Bun/Rust-compiled binaries that appendFileSync
    to log files and cache state under $HOME/.<agent>/. If we only bind
    individual credential files RO, the agent crashes silently with ENOENT on
    its first log write.

    Workaround: create `<workspace>/.proposer_home/`, populate per-agent
    auth dirs into it, and bind the whole `.proposer_home` as /root inside
    the container. The agent then writes its logs/sessions there (per-run,
    ephemeral — not synced back to the host's ~/.<agent>/, which is the
    right isolation behavior).

    For claude_code with claude_auth=subscription:
      - copies `~/.claude/.credentials.json` into `.proposer_home/.claude/`
        as a fallback (when CLAUDE_CODE_OAUTH_TOKEN env var injection is
        not used — see `_load_oauth_token`).
      For api_key / vertex the copy is SKIPPED (and any stale copy from an
      earlier subscription run is removed) so the scratch HOME carries no
      second credential source — auth comes solely from the env vars
      injected by `_agent_auth_extra_env`.

    For codex:
      - the codex-specific apikey auth.json is staged by
        `_stage_codex_apikey_auth(proposer_home)` from the propose() entry
        point — NOT done here so claude-only runs don't need OPENAI_API_KEY.
    """
    proposer_home = paths.workspace / ".proposer_home"
    claude_dir = proposer_home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    creds_dst = claude_dir / ".credentials.json"
    if claude_auth == "subscription" and _HOST_CLAUDE_CREDS.exists():
        # Always refresh from host so an expired/refreshed host token reaches
        # the container next launch. Host token mtime check would be brittle.
        # Skip the copy if the host creds aren't present (e.g. user only uses
        # codex and hasn't run `claude login`) — _check_environment already
        # validated whichever agent's requirements are met.
        shutil.copy2(_HOST_CLAUDE_CREDS, creds_dst)
        creds_dst.chmod(0o600)
    elif claude_auth in ("api_key", "vertex") and creds_dst.exists():
        # A prior subscription run in this workspace may have staged a copy;
        # drop it so the only live credential is the injected env var.
        try:
            creds_dst.unlink()
        except OSError as exc:
            log.warning(f"could not remove stale credentials copy: {exc}")
    return proposer_home


def _stage_codex_apikey_auth(proposer_home: Path) -> None:
    """Write an apikey-mode auth.json into the container's /root/.codex/.

    Codex CLI doesn't read OPENAI_API_KEY from env directly — it requires a
    config file at $HOME/.codex/auth.json with shape:

        {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-..."}

    Without this, `codex exec` reaches `wss://api.openai.com/v1/responses`
    with no auth header and gets 401. The auth.json is per-run (lives in
    the run's workspace) and rewritten each propose so a manually-rotated
    OPENAI_API_KEY is picked up immediately.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        # _check_environment(agent="codex") already validates this; defensive.
        raise ProposerLaunchError(
            "_stage_codex_apikey_auth: OPENAI_API_KEY env var not set"
        )
    codex_dir = proposer_home / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    auth_json = codex_dir / "auth.json"
    auth_json.write_text(
        json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": api_key}),
        encoding="utf-8",
    )
    auth_json.chmod(0o600)


_HOST_OAUTH_TOKEN_FILE = Path.home() / ".claude" / "oauth_token"


def _load_oauth_token() -> Optional[str]:
    """Read the long-lived OAuth token from ~/.claude/oauth_token, if present.

    Generated by `claude setup-token` (interactive flow, subscription-scoped),
    the token starts with `sk-ant-oat01-` and is designed for headless/
    automation use cases — it bypasses the ~8h access_token + rotating
    refresh_token cycle that breaks long forge runs. When this returns a
    value, `propose()` injects it into the proposer container via
    SINGULARITYENV_CLAUDE_CODE_OAUTH_TOKEN; CC SDK then uses it directly
    (inference-only) instead of touching credentials.json's OAuth refresh
    path.

    Returns None if the file is missing or empty — caller falls back to
    the credentials.json copy mechanism in `_prepare_claude_home`.
    """
    if not _HOST_OAUTH_TOKEN_FILE.exists():
        return None
    try:
        token = _HOST_OAUTH_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        log.warning(f"could not read {_HOST_OAUTH_TOKEN_FILE}: {exc}")
        return None
    return token or None


_OAUTH_TOKEN_LOG_ONCE = False
_OPENAI_KEY_LOG_ONCE = False


def _agent_auth_extra_env(
    agent: str,
    *,
    claude_auth: str = "subscription",
    vertex_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Build the per-propose `extra_env` for `_stream_subprocess`.

    All secrets travel as SINGULARITYENV_* process env — Singularity strips
    the prefix and forwards them inside the container; nothing lands in argv
    (so `ps` can't leak it), and values are re-read on each propose so manual
    rotation is picked up immediately.

    For claude_code, dispatch on `claude_auth`:
      subscription — if a long-lived OAuth token exists at
        ~/.claude/oauth_token, inject CLAUDE_CODE_OAUTH_TOKEN; otherwise no
        env override (container falls back to the credentials.json copy,
        which has the 8h refresh-cycle limitation that motivated the token
        mechanism).
      api_key — inject ANTHROPIC_API_KEY (host env / project .env).
      vertex — inject CLAUDE_CODE_USE_VERTEX=1 + ANTHROPIC_VERTEX_PROJECT_ID
        + CLOUD_ML_REGION + GOOGLE_APPLICATION_CREDENTIALS pointing at the
        RO-bound credentials json (see _build_singularity_cmd's extra bind).

    For codex:
      Inject OPENAI_API_KEY (long-lived, no refresh) via
      SINGULARITYENV_OPENAI_API_KEY. Codex's ChatGPT-OAuth login path
      (codex login) is intentionally bypassed — API key is more stable
      for long unattended runs.
    """
    global _OAUTH_TOKEN_LOG_ONCE, _OPENAI_KEY_LOG_ONCE

    if agent == "claude_code" and claude_auth == "api_key":
        api_key = _resolve_anthropic_api_key()
        if not api_key:
            # _check_environment already validated this; defensive.
            raise ProposerLaunchError(
                "_agent_auth_extra_env: claude_auth=api_key but ANTHROPIC_API_KEY "
                "is not set (host env or project .env)"
            )
        return {"SINGULARITYENV_ANTHROPIC_API_KEY": api_key}

    if agent == "claude_code" and claude_auth == "vertex":
        vc = vertex_cfg or {}
        # Credentials json path is resolved host-side and bound by
        # _build_singularity_cmd; here we only point CC at the container path.
        return {
            "SINGULARITYENV_CLAUDE_CODE_USE_VERTEX": "1",
            "SINGULARITYENV_ANTHROPIC_VERTEX_PROJECT_ID": str(vc.get("project_id", "")),
            "SINGULARITYENV_CLOUD_ML_REGION": str(vc.get("region", "")),
            "SINGULARITYENV_GOOGLE_APPLICATION_CREDENTIALS": _VERTEX_CREDS_IN_CONTAINER,
        }

    if agent == "claude_code":
        token = _load_oauth_token()
        if not token:
            if not _OAUTH_TOKEN_LOG_ONCE:
                log.info(
                    f"oauth: {_HOST_OAUTH_TOKEN_FILE} missing or empty — "
                    f"proposer will use credentials.json (8h refresh cycle)"
                )
                _OAUTH_TOKEN_LOG_ONCE = True
            return {}
        if not _OAUTH_TOKEN_LOG_ONCE:
            log.info(
                f"oauth: using long-lived token from {_HOST_OAUTH_TOKEN_FILE} "
                f"(SINGULARITYENV_CLAUDE_CODE_OAUTH_TOKEN injection)"
            )
            _OAUTH_TOKEN_LOG_ONCE = True
        return {"SINGULARITYENV_CLAUDE_CODE_OAUTH_TOKEN": token}

    if agent == "codex":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # _check_environment already validated this, but defensive:
            return {}
        if not _OPENAI_KEY_LOG_ONCE:
            log.info(
                "codex: using OPENAI_API_KEY for auth "
                "(SINGULARITYENV_OPENAI_API_KEY injection)"
            )
            _OPENAI_KEY_LOG_ONCE = True
        return {"SINGULARITYENV_OPENAI_API_KEY": api_key}

    return {}


# Backwards-compat alias for the old, claude_code-only name. Kept in case
# any external script imports it; new code should call _agent_auth_extra_env.
def _oauth_extra_env() -> Dict[str, str]:
    return _agent_auth_extra_env("claude_code")


def _prepare_agent_auth(
    agent: str,
    *,
    claude_auth: str = "subscription",
    vertex_cfg: Optional[Dict[str, Any]] = None,
) -> "tuple[Path, List[str], Dict[str, str]]":
    """Auth staging shared by propose() / propose_with_fix().

    Returns (proposer_home, extra_binds, extra_env):
      proposer_home — per-run scratch HOME (credentials.json copy only for
                      claude_auth=subscription; codex auth.json staged here);
      extra_binds   — additional RO binds (the GCP credentials json for
                      claude_auth=vertex, resolved host-side);
      extra_env     — secret-carrying SINGULARITYENV_* overrides.
    """
    effective_auth = claude_auth if agent == "claude_code" else "subscription"
    proposer_home = _prepare_claude_home(effective_auth)
    if agent == "codex":
        _stage_codex_apikey_auth(proposer_home)
    extra_binds: List[str] = []
    if agent == "claude_code" and claude_auth == "vertex":
        creds = _resolve_gcp_credentials(vertex_cfg)
        extra_binds.append(f"{creds}:{_VERTEX_CREDS_IN_CONTAINER}:ro")
    extra_env = _agent_auth_extra_env(
        agent, claude_auth=claude_auth, vertex_cfg=vertex_cfg,
    )
    return proposer_home, extra_binds, extra_env


def _build_singularity_cmd(
    *,
    propose_args: List[str],
    proposer_home: Path,
    agent: str = "claude_code",
    extra_binds: Optional[List[str]] = None,
) -> List[str]:
    """Build the `singularity exec ...` argv with the selective bind list.

    Bind list rationale:
      - LAYER-1 wrapper script (propose_in_container.py + its forge.prompts
        import) needs to be reachable via /app/forge/. Agents never read these.
      - Agent reference materials live under /app/{common,datasets,forge}/. Only
        the files that contribute to writing a correct harness are bound:
          forge/harness_base.py    the MemoStructure base CC must inherit from
          common/harness_base.py   the underlying ABC (forge's subclasses it)
          common/recorder.py       Basic_Recorder data envelope
                                   (harness_base imports it)
          common/llm.py            Agent + Embedding helpers (token-tracked)
          common/logger.py         transitive dep of common.llm
          common/__init__.py       package marker
          datasets/                full directory: env.py, workflow.py,
                                   prompts.py per benchmark + raw data files.
                                   No selective filtering — see PROGRESS for
                                   the cheat-via-trace-traces note.
      - workspace/ is the agent's cwd (RW). Per-run isolation.
      - Agent binary + scratch HOME bound conditionally per `agent`:
          claude_code → claude binary + ~/.local/share/claude + /root/.claude
                        (proposer_home, contains credentials.json fallback)
          codex       → codex binary only; auth via OPENAI_API_KEY env var
                        injected by _agent_auth_extra_env (no state binds)
      - NOT bound (vs the previous /app:ro whole-root bind):
          /app/forge/{orchestrator,proposer,...}.py    host-side outer-loop
          /app/common/{workflow,judge,tokens}.py        internal infrastructure
          /app/baselines, /app/docs, /app/configs, /app/seeds, /app/venv
          /app/workspace                                cross-run isolation
          /app/.env, /app/.git                          not needed
    """
    binds = [
        # LAYER-1 wrapper script imports. Note: forge/prompts/ is intentionally
        # NOT bound — prompts are rendered HOST-SIDE by `_render_and_stage_prompts`
        # and staged to the per-harness workspace as `.prompt_system.txt` +
        # `.prompt_task.txt`; the container only reads those files via /workspace.
        f"{PROJECT_ROOT}/forge/__init__.py:/app/forge/__init__.py:ro",
        f"{PROJECT_ROOT}/forge/propose_in_container.py:/app/forge/propose_in_container.py:ro",

        # Agent reference materials
        f"{PROJECT_ROOT}/forge/harness_base.py:/app/forge/harness_base.py:ro",
        f"{PROJECT_ROOT}/common/__init__.py:/app/common/__init__.py:ro",
        f"{PROJECT_ROOT}/common/harness_base.py:/app/common/harness_base.py:ro",
        f"{PROJECT_ROOT}/common/recorder.py:/app/common/recorder.py:ro",
        f"{PROJECT_ROOT}/common/llm.py:/app/common/llm.py:ro",
        f"{PROJECT_ROOT}/common/logger.py:/app/common/logger.py:ro",
        f"{PROJECT_ROOT}/datasets:/app/datasets:ro",

        # Workspace (cwd) + scratch HOME for whichever agent
        f"{paths.workspace}:/workspace:rw",
        f"{proposer_home}:/root:rw",
    ]
    # Per-agent binary binds.
    if agent == "claude_code":
        binds += [
            f"{_HOST_CLAUDE_BIN}:/usr/local/bin/claude:ro",
            f"{_HOST_CLAUDE_SHARE}:/usr/local/share/claude:ro",
        ]
    elif agent == "codex":
        binds += [
            # Bind the resolved binary (symlink target). codex needs no
            # additional share/state dirs at runtime when using API-key auth.
            f"{_HOST_CODEX_BIN_RESOLVED}:/usr/local/bin/codex:ro",
        ]
    if extra_binds:
        # e.g. the RO GCP credentials json for claude_auth=vertex.
        binds += list(extra_binds)

    cmd = [
        "singularity", "exec",
        "--containall",
    ]
    for b in binds:
        cmd += ["--bind", b]
    cmd += [
        # SINGULARITYENV_HOME=/root is implied by --containall; no need to set.
        # CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING is a no-op for codex (and
        # we no longer use the SDK anyway), but harmless — leave for CC.
        "--env", "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING=true",
        # PYTHONPATH so any `python -c "from common.harness_base import ..."`
        # the agent runs for self-validation finds the bound /app/common package.
        "--env", "PYTHONPATH=/app",
        str(PROPOSER_BASE_SIF),
        "python", "/app/forge/propose_in_container.py",
    ] + propose_args
    return cmd


async def _stream_subprocess(
    cmd: List[str],
    timeout_s: int,
    label: str,
    *,
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    """Run `cmd` and forward stdout line-by-line to the host log.

    Returns the subprocess exit code. Raises TimeoutError on wall-clock
    timeout (after escalating SIGTERM → SIGKILL on the singularity exec
    process group).

    `extra_env` is merged onto `os.environ` for the subprocess. Used to
    pass SINGULARITYENV_CLAUDE_CODE_OAUTH_TOKEN through to the container
    without exposing the token in argv (which `ps` would see).
    """
    log.info(f"{label}: launching ({len(cmd)} argv tokens, image={PROPOSER_BASE_SIF.name})")
    env: Optional[Dict[str, str]] = None
    if extra_env:
        env = {**os.environ, **extra_env}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        # Put singularity in its own process group so we can kill it cleanly.
        preexec_fn=os.setsid,
        # Match the in-container readline limit. The container only emits
        # truncated `proposer·*: ...` lines so the host pump shouldn't ever
        # see > 64 KB in practice, but if the container crashes with a stack
        # trace (or a stray library prints a giant message) the default limit
        # would silently kill the outer pump too.
        limit=16 * 1024 * 1024,
    )

    async def _pump_stdout() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            # Lines from propose_in_container.py are pre-tagged
            # ("proposer·tool: ...", "proposer·text: ...", etc).
            # Map error tag → log.error, info / tool / text → log.info.
            # Anything untagged is unstructured output from the agent CLI's
            # own stderr (CC's bun-runtime exceptions, codex panics, etc.),
            # which we MUST surface — silent rc=1 was traced to these being
            # swallowed by log.debug. Use WARNING so they're visible at the
            # default level without disrupting the agent's structured event
            # stream.
            if line.startswith("proposer·error"):
                log.error(line)
            elif line.startswith("proposer·info"):
                log.info(line)
            elif line.startswith("proposer·tool"):
                log.info(line)
            elif line.startswith("proposer·text"):
                log.info(line)
            else:
                log.warning(f"proposer·raw: {line}")

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
# Host-side prompt rendering / staging
# ---------------------------------------------------------------------------

def _render_and_stage_prompts(
    *,
    new_id: str,
    mode: str,  # "propose" | "fix"
    prompts_version: Optional[str],
    sanity_enabled: bool,
    active_datasets: Optional[List[str]],
    error_trace: Optional[str] = None,
) -> Dict[str, str]:
    """Render system + task prompts HOST-SIDE for the chosen version and
    write them to the harness dir as forensic-friendly text files.

    The container then reads the two files via the /workspace RW bind — it
    never imports any version of `forge.prompts.*`. Effect: each harness dir
    permanently captures the exact prompt that produced it, available for
    diff between runs without spelunking git.

    Returns the *container-side* absolute paths suitable to pass as argv to
    `propose_in_container.py`.
    """
    new_dir = paths.harnesses_dir / new_id
    new_dir.mkdir(parents=True, exist_ok=True)

    system_text = build_proposer_system(
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets or None,
        version=prompts_version,
    )
    new_dir_rel = f"harnesses/{new_id}"
    if mode == "propose":
        task_text = proposer_task_prompt(new_dir_rel, version=prompts_version)
    elif mode == "fix":
        if error_trace is None:
            raise ValueError("fix mode requires error_trace")
        task_text = proposer_fix_prompt(
            new_dir_rel, error_trace, version=prompts_version
        )
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected propose|fix)")

    system_host = new_dir / ".prompt_system.txt"
    task_host = new_dir / ".prompt_task.txt"
    system_host.write_text(system_text, encoding="utf-8")
    task_host.write_text(task_text, encoding="utf-8")

    return {
        "system_in_container": f"/workspace/harnesses/{new_id}/.prompt_system.txt",
        "task_in_container": f"/workspace/harnesses/{new_id}/.prompt_task.txt",
    }


# ---------------------------------------------------------------------------
# Public API — same signatures as v5 host-side proposer; orchestrator unchanged.
# ---------------------------------------------------------------------------

async def propose(
    new_id: str,
    *,
    model: str = "claude-opus-4-7",
    max_turns: int = 80,
    timeout_s: int = 25 * 60,
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
    agent: str = "claude_code",
    agent_opts: Optional[Dict[str, Any]] = None,
    prompts_version: Optional[str] = None,
    claude_auth: str = "subscription",
    vertex_cfg: Optional[Dict[str, Any]] = None,
    extra_binds: Optional[List[str]] = None,
) -> Path:
    """Run a sandboxed proposer; return the new harness directory on success.

    `agent` selects the coding-agent backend: 'claude_code' (default) or
    'codex'. The in-container script (`propose_in_container.py`) dispatches
    on this and shells out to the matching CLI.

    `claude_auth` selects the claude_code credential route (see
    CLAUDE_AUTH_MODES): subscription | api_key | vertex. `vertex_cfg` is the
    cfg.proposer.claude_code.vertex dict ({project_id, region, credentials});
    both are ignored when agent='codex'.

    `prompts_version` selects the prompt template version (a stem name under
    `forge/prompts/templates/`); None / "latest" → templates/_default.
    Prompts are rendered HOST-SIDE and staged into the harness dir; the
    container does not import any forge.prompts.* code.

    No parent_id passed — the agent explores `harnesses/` itself and decides
    which prior(s) to draw from. Choices are recorded in `meta.json::parent_ids`.

    Raises:
      ProposerLaunchError  pre-flight (SIF/binary/creds missing).
      TimeoutError         the subprocess exceeds timeout_s.
      RuntimeError         the proposer exits without PROPOSAL_READY, or
                           the subprocess returns a non-zero exit code.
    """
    _check_environment(agent=agent, claude_auth=claude_auth, vertex_cfg=vertex_cfg)

    new_dir = paths.harnesses_dir / new_id
    # mkdir on host so the bind sees it; container has it RW via /workspace.
    new_dir.mkdir(parents=True, exist_ok=True)

    staged = _render_and_stage_prompts(
        new_id=new_id,
        mode="propose",
        prompts_version=prompts_version,
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets,
    )

    proposer_home, auth_binds, extra_env = _prepare_agent_auth(
        agent, claude_auth=claude_auth, vertex_cfg=vertex_cfg,
    )
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "propose",
        "--agent", agent,
        "--model", model,
        "--max-turns", str(max_turns),
        "--timeout-s", str(timeout_s),
        "--system-prompt-file", staged["system_in_container"],
        "--task-file", staged["task_in_container"],
        "--agent-opts", json.dumps(agent_opts or {}),
    ]
    cmd = _build_singularity_cmd(
        propose_args=propose_args, proposer_home=proposer_home, agent=agent,
        extra_binds=auth_binds + list(extra_binds or []),
    )

    rc = await _stream_subprocess(
        cmd, timeout_s=timeout_s + 60, label=f"proposer[{new_id}]",
        extra_env=extra_env,
    )
    if rc != 0:
        raise RuntimeError(
            f"proposer subprocess exited with rc={rc} for {new_id}; "
            f"see orchestrator log for the streamed agent output"
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
    sanity_enabled: bool = True,
    active_datasets: Optional[List[str]] = None,
    agent: str = "claude_code",
    agent_opts: Optional[Dict[str, Any]] = None,
    prompts_version: Optional[str] = None,
    claude_auth: str = "subscription",
    vertex_cfg: Optional[Dict[str, Any]] = None,
    extra_binds: Optional[List[str]] = None,
) -> Path:
    """Ask the agent to Read + Edit the existing harness to fix a sanity-check
    failure.

    The error trace is folded HOST-SIDE into the rendered task prompt by
    `proposer_fix_prompt`; no separate file is written into the harness dir.
    The staged `.prompt_task.txt` for a fix call therefore CONTAINS the trace
    inline (truncated if >90 lines). `claude_auth`/`vertex_cfg` behave as in
    `propose()`.
    """
    _check_environment(agent=agent, claude_auth=claude_auth, vertex_cfg=vertex_cfg)

    new_dir = paths.harnesses_dir / new_id
    if not (new_dir / "harness.py").exists():
        raise RuntimeError(
            f"propose_with_fix called but {new_dir}/harness.py does not exist"
        )

    staged = _render_and_stage_prompts(
        new_id=new_id,
        mode="fix",
        prompts_version=prompts_version,
        sanity_enabled=sanity_enabled,
        active_datasets=active_datasets,
        error_trace=error_trace,
    )

    proposer_home, auth_binds, extra_env = _prepare_agent_auth(
        agent, claude_auth=claude_auth, vertex_cfg=vertex_cfg,
    )
    propose_args = [
        "--new-id", new_id,
        "--workspace", "/workspace",
        "--mode", "fix",
        "--agent", agent,
        "--model", model,
        "--max-turns", str(max_turns),
        "--timeout-s", str(timeout_s),
        "--system-prompt-file", staged["system_in_container"],
        "--task-file", staged["task_in_container"],
        "--agent-opts", json.dumps(agent_opts or {}),
    ]
    cmd = _build_singularity_cmd(
        propose_args=propose_args, proposer_home=proposer_home, agent=agent,
        extra_binds=auth_binds + list(extra_binds or []),
    )
    rc = await _stream_subprocess(
        cmd, timeout_s=timeout_s + 60, label=f"propose_fix[{new_id}]",
        extra_env=extra_env,
    )

    if rc != 0:
        raise RuntimeError(
            f"propose_with_fix subprocess exited with rc={rc} for {new_id}"
        )
    return new_dir
