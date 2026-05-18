"""In-container proposer entrypoint.

Launched by `forge/proposer.py` via:

    singularity exec --containall --bind ... proposer-base.sif \\
        python /app/forge/propose_in_container.py \\
            --new-id <int> --workspace /workspace [--mode propose|fix] \\
            [--error-trace-file /workspace/.fix_error_trace.txt] \\
            --agent <claude_code|codex> --model <model> --timeout-s <s>

Inside the Singularity sandbox we have:
  /app                project root, RO  (forge.* + datasets.* importable)
  /workspace          this run's workspace, RW  (cwd for the agent)
  /usr/local/bin/claude         host's claude binary, RO bind (CC)
  /usr/local/share/claude/      host's claude install, RO bind (CC)
  /usr/local/bin/codex          host's codex binary, RO bind (codex)
  /root/.claude/.credentials.json     host's OAuth creds, RW (CC fallback)

The agent's filesystem-readable surface is exactly the binds above; nothing
else from the host is visible (`--containall` strips $HOME, /tmp, etc).

# Multi-agent design

This file used to depend on claude_code_sdk (with a private-API monkey-patch
to handle unknown event types). It now shells out to the agent CLI directly
in stream-json mode and parses NDJSON events from stdout. Each agent has:

  - `_build_cmd_*`       argv builder (returns list[str])
  - `_build_stdin_*`     stdin bytes (NDJSON for CC, plain text for codex)
  - `_handle_event_*`    NDJSON event → _emit() lines

The main loop is identical across agents — only dispatch differs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Callable, List

# Project root is bound at /app. Add to sys.path so `from forge.prompts` works.
_APP = Path("/app")
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from forge.prompts import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)


# Tool restrictions intentionally NOT set: filesystem isolation is enforced
# by the Singularity sandbox in forge/proposer.py — the agent's reachable
# surface is exactly the bind list (workspace + project root RO + agent
# CLI binary RO + per-run scratch HOME). Bash, WebFetch, WebSearch, mcp__*
# all permitted inside the container; the proposer image (extends
# eval-base.sif) has the same Python env as the evaluator, so the agent can
# `python -c "from harness import ..."` for sub-second import validation.
# See PROPOSER_SYSTEM for the workflow cheat sheet.


def _emit(prefix: str, msg: str) -> None:
    """Print a tagged line to stdout. Host-side launcher line-by-line
    forwards these into the orchestrator log.

    Tag conventions:
      proposer·text    agent assistant text — emitted line-by-line, full
                       content (one log line per source line; rendered at
                       INFO so the reasoning is visible without --verbose)
      proposer·tool    agent tool use (truncated to 140 chars)
      proposer·info    high-level status (model, finish, usage)
      proposer·error   actionable failure
    """
    # Use sys.stdout.write to avoid print's locking subtleties under asyncio
    sys.stdout.write(f"{prefix}: {msg}\n")
    sys.stdout.flush()


# ============================================================================
# Per-agent: build argv + build stdin + handle event
# ============================================================================

# ---- claude_code ----------------------------------------------------------

def _build_cmd_cc(
    *,
    system_prompt: str,
    model: str,
    agent_opts: dict,
) -> List[str]:
    cmd = [
        "claude", "-p",
        # --verbose is REQUIRED when --output-format=stream-json in -p mode.
        "--verbose",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--model", model,
        "--system-prompt", system_prompt,
        "--permission-mode", "bypassPermissions",
    ]
    disallowed = agent_opts.get("disallowed_tools") or []
    if disallowed:
        cmd += ["--disallowed-tools", ",".join(disallowed)]
    effort = agent_opts.get("effort")
    if effort:
        # CC --effort: low / medium / high / xhigh / max
        cmd += ["--effort", str(effort)]
    return cmd


def _build_stdin_cc(task: str) -> bytes:
    """One NDJSON user message line. CC -p with --input-format stream-json
    reads events from stdin until EOF."""
    return (
        json.dumps({"type": "user", "message": {"role": "user", "content": task}})
        + "\n"
    ).encode("utf-8")


# CC tool counter (state for _handle_event_cc; module-level since handler
# is called per-line on the event loop's single thread).
_cc_tool_count = 0


def _handle_event_cc(event: dict) -> None:
    """Parse one CC stream-json event → _emit() lines.

    Event types we recognize:
      assistant   — message with content blocks (text / thinking / tool_use)
      result      — final summary (duration, cost, turns, usage)
      system      — startup info; logged briefly
      rate_limit_event / others — silently ignored
    """
    global _cc_tool_count
    etype = event.get("type", "")
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []) or []:
            btype = block.get("type")
            if btype == "text":
                text = (block.get("text") or "").strip()
                if text:
                    # Emit each line separately so orchestrator.log preserves
                    # paragraph structure.
                    for line in text.splitlines():
                        _emit("proposer·text", line)
            elif btype == "tool_use":
                _cc_tool_count += 1
                _emit(
                    "proposer·tool",
                    f"#{_cc_tool_count} {block.get('name','')} "
                    f"{str(block.get('input',''))[:140]}",
                )
            # "thinking" blocks: skip (not previously logged either).
    elif etype == "result":
        cost = event.get("total_cost_usd") or 0
        _emit(
            "proposer·info",
            f"finished turns={event.get('num_turns','?')} "
            f"tools={_cc_tool_count} "
            f"duration={event.get('duration_ms','?')}ms "
            f"cost=${cost:.4f} "
            f"usage={event.get('usage','')}",
        )
        if event.get("is_error"):
            _emit("proposer·error", f"agent reported is_error=true: {event.get('result','')}")
    # else: system / rate_limit_event / unknown — skip silently.


# ---- codex ----------------------------------------------------------------

def _build_cmd_codex(
    *,
    model: str,
    workspace: str,
    agent_opts: dict,
) -> List[str]:
    # codex has no --system-prompt; system + task are combined into stdin
    # (see _build_stdin_codex). codex also has no --disallowed-tools;
    # sandboxing relies entirely on the Singularity bind list.
    cmd = [
        "codex", "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--cd", workspace,
        "--model", model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--sandbox", "danger-full-access",
    ]
    reasoning = agent_opts.get("reasoning_effort")
    if reasoning:
        # codex -c: TOML-typed override. Values that fail TOML parse become
        # literal strings, so unquoted `high` works equally well; we quote
        # for clarity.
        cmd += ["-c", f'model_reasoning_effort="{reasoning}"']
    cmd += ["-"]  # read prompt from stdin (must come after -c overrides)
    return cmd


def _build_stdin_codex(system_prompt: str, task: str) -> bytes:
    """Codex sees one user prompt. Concatenate the system prompt + the
    task prompt with a separator. Codex doesn't distinguish "system" from
    "user" at the CLI level, so the agent reads it as instructions in one
    big context.
    """
    payload = f"{system_prompt}\n\n---\n\n{task}\n"
    return payload.encode("utf-8")


_codex_tool_count = 0


def _handle_event_codex(event: dict) -> None:
    """Parse one codex --json event → _emit() lines.

    Event types observed:
      thread.started   — skip
      turn.started     — skip
      item.completed   — item.type ∈ {agent_message, command_call, ...}
      turn.completed   — final usage summary
      others           — skip silently
    """
    global _codex_tool_count
    etype = event.get("type", "")
    if etype == "item.completed":
        item = event.get("item") or {}
        itype = item.get("type", "")
        if itype == "agent_message":
            text = (item.get("text") or "").strip()
            if text:
                for line in text.splitlines():
                    _emit("proposer·text", line)
        else:
            # All non-text items are tool-like (command_call, file_write,
            # web_fetch, ...). Log compactly.
            _codex_tool_count += 1
            payload = json.dumps(item, ensure_ascii=False)[:140]
            _emit("proposer·tool", f"#{_codex_tool_count} {itype} {payload}")
    elif etype == "turn.completed":
        usage = event.get("usage", {}) or {}
        _emit(
            "proposer·info",
            f"finished tools={_codex_tool_count} usage={usage}",
        )


# Agent dispatch table
_AGENT_DISPATCH = {
    "claude_code": {
        "build_cmd": _build_cmd_cc,
        "build_stdin": _build_stdin_cc,
        "handle_event": _handle_event_cc,
        "uses_system_prompt_flag": True,
    },
    "codex": {
        "build_cmd": _build_cmd_codex,
        "build_stdin": _build_stdin_codex,
        "handle_event": _handle_event_codex,
        "uses_system_prompt_flag": False,
    },
}


# ============================================================================
# Main loop
# ============================================================================


async def _run(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    new_dir_rel = f"harnesses/{args.new_id}"
    new_dir = workspace / new_dir_rel

    if args.mode == "propose":
        new_dir.mkdir(parents=True, exist_ok=True)
        task = proposer_task_prompt(new_dir_rel=new_dir_rel)
    else:  # fix
        if not (new_dir / "harness.py").exists():
            _emit("proposer·error", f"fix mode but {new_dir}/harness.py missing")
            return 2
        if not args.error_trace_file:
            _emit("proposer·error", "fix mode requires --error-trace-file")
            return 2
        error_trace = Path(args.error_trace_file).read_text(encoding="utf-8")
        task = proposer_fix_prompt(new_dir_rel=new_dir_rel, error_trace=error_trace)

    active_datasets = [
        d.strip() for d in (args.active_datasets or "").split(",") if d.strip()
    ]

    system_prompt = build_proposer_system(
        sanity_enabled=args.sanity_enabled,
        active_datasets=active_datasets or None,
        update_type=args.update_type,
    )

    if args.agent not in _AGENT_DISPATCH:
        _emit("proposer·error", f"unknown agent: {args.agent}")
        return 5

    # Parse agent-specific opts (JSON dict). Empty / missing → {}. This dict
    # carries everything per-agent: disallowed_tools (CC) / effort (CC) /
    # reasoning_effort (codex). Unknown keys are silently ignored by each
    # agent's argv builder so adding new keys is forward-compatible.
    try:
        agent_opts: dict = json.loads(args.agent_opts or "{}")
    except json.JSONDecodeError as exc:
        _emit("proposer·error", f"--agent-opts is not valid JSON: {exc}")
        return 6
    if not isinstance(agent_opts, dict):
        _emit("proposer·error", f"--agent-opts must decode to a dict, got {type(agent_opts).__name__}")
        return 6

    # Build agent-specific argv + stdin payload.
    if args.agent == "claude_code":
        cmd = _build_cmd_cc(
            system_prompt=system_prompt,
            model=args.model,
            agent_opts=agent_opts,
        )
        stdin_bytes = _build_stdin_cc(task)
    elif args.agent == "codex":
        cmd = _build_cmd_codex(
            model=args.model,
            workspace=str(workspace),
            agent_opts=agent_opts,
        )
        stdin_bytes = _build_stdin_codex(system_prompt, task)

    if agent_opts:
        _emit("proposer·info", f"agent_opts={agent_opts}")
    _emit(
        "proposer·info",
        f"mode={args.mode} new_id={args.new_id} agent={args.agent} model={args.model}",
    )

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workspace),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdin is not None
    try:
        proc.stdin.write(stdin_bytes)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        _emit("proposer·error", f"failed writing stdin: {exc}")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    handler: Callable[[dict], None] = _AGENT_DISPATCH[args.agent]["handle_event"]  # type: ignore[assignment]

    async def _pump() -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON output — stderr leaking through, or pre-init
                # banner from the binary. Log a truncated form.
                _emit("proposer·text", line[:500])
                continue
            try:
                handler(event)
            except Exception as exc:
                _emit("proposer·error", f"event handler error: {exc} on {line[:200]}")

    try:
        await asyncio.wait_for(_pump(), timeout=args.timeout_s)
    except asyncio.TimeoutError:
        _emit("proposer·error", f"timed out after {args.timeout_s}s for {args.new_id}")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return 3

    rc = await proc.wait()

    if args.mode == "propose":
        sentinel = new_dir / "PROPOSAL_READY"
        if not sentinel.exists():
            _emit(
                "proposer·error",
                f"finished without writing PROPOSAL_READY at {new_dir}",
            )
            return 4

    return rc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-id", required=True)
    p.add_argument("--workspace", required=True,
                   help="Container path of the per-run workspace (typically /workspace).")
    p.add_argument("--mode", choices=["propose", "fix"], default="propose")
    p.add_argument("--error-trace-file", default=None,
                   help="Required when --mode=fix; path inside container to the trace text file.")
    p.add_argument("--agent", default="claude_code",
                   choices=list(_AGENT_DISPATCH.keys()),
                   help="Coding agent backend to drive (default claude_code).")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--max-turns", type=int, default=80,
                   help="Soft turn budget. NOT enforced — wall-clock --timeout-s "
                        "is the hard limit. Kept for backward compat with old "
                        "callers; no longer plumbed to the agent CLI.")
    p.add_argument("--timeout-s", type=int, default=25 * 60)
    p.add_argument("--sanity-enabled", default="true",
                   choices=["true", "false"],
                   help="Whether the orchestrator will run a sanity check after this propose. "
                        "Controls whether sanity-related sections appear in PROPOSER_SYSTEM.")
    p.add_argument("--active-datasets", default="",
                   help="Comma-separated dataset names this run is configured for "
                        "(e.g. 'dynamicmem,locomo'). Empty = full 3-benchmark prompt. "
                        "Controls which dataset shapes / dispatch examples / trace "
                        "field listings appear in PROPOSER_SYSTEM.")
    p.add_argument("--update-type", default="all_at_once",
                   choices=["all_at_once", "chunked", "sequential"],
                   help="Phase 1 dispatch mode for this run. Surfaced to the proposer "
                        "so it knows whether general_update will be called once with "
                        "the full payload or many times with smaller chunks.")
    p.add_argument("--agent-opts", default="{}",
                   help="JSON dict of agent-specific options. Recognized keys: "
                        "{claude_code: effort} {codex: reasoning_effort}. Unknown "
                        "keys are silently ignored by each agent's argv builder.")
    args = p.parse_args()
    # Convert string flag → bool (argparse choices keep it stringly-typed for
    # CLI symmetry with the host launcher).
    args.sanity_enabled = (args.sanity_enabled == "true")
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
