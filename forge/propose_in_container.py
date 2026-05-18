"""In-container proposer entrypoint.

Launched by `forge/proposer.py` via:

    singularity exec --containall --bind ... proposer-base.sif \\
        python /app/forge/propose_in_container.py \\
            --new-id <int> --workspace /workspace [--mode propose|fix] \\
            [--error-trace-file /workspace/.fix_error_trace.txt] \\
            --model <claude-model> --max-turns <N> --timeout-s <s>

Inside the Singularity sandbox we have:
  /app                project root, RO  (forge.* + datasets.* importable)
  /workspace          this run's workspace, RW  (cwd for CC)
  /seeds              seed library, RO
  /usr/local/bin/claude         host's claude binary, RO bind
  /usr/local/share/claude/      host's claude install, RO bind
  /root/.claude/.credentials.json     host's OAuth creds, RW (CLI may refresh)

CC's filesystem-readable surface is exactly the four binds above; nothing
else from the host is visible (`--containall` strips $HOME, /tmp, etc).

This file used to depend on claude_code_sdk (with a private-API monkey-patch
to handle unknown event types). It now shells out to the `claude` CLI
directly in stream-json mode and parses NDJSON events from stdout, removing
the SDK dependency + the `_internal.message_parser` monkey-patch. External
behavior (the lines emitted to stdout for the host orchestrator) is
unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List

# Project root is bound at /app. Add to sys.path so `from forge.prompts` works.
_APP = Path("/app")
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from forge.prompts import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)


# Tool restrictions intentionally NOT set here: filesystem isolation is
# enforced by the Singularity sandbox in forge/proposer.py — CC's reachable
# surface is exactly the bind list (workspace + project root RO + CLI binary
# RO + per-run scratch HOME). Bash, WebFetch, WebSearch, mcp__* all permitted
# inside the container; the proposer image (extends eval-base.sif) has the
# same Python env as the evaluator, so CC can `python -c "from harness import
# ..."` for sub-second import validation. See PROPOSER_SYSTEM for the workflow
# cheat sheet.


def _emit(prefix: str, msg: str) -> None:
    """Print a tagged line to stdout. Host-side launcher line-by-line
    forwards these into the orchestrator log.

    Tag conventions:
      proposer·text    CC assistant text — emitted line-by-line, full content
                       (one log line per source line; rendered at INFO so the
                       reasoning is visible without --verbose)
      proposer·tool    CC tool use (truncated to 140 chars)
      proposer·info    high-level status (model, finish, usage)
      proposer·error   actionable failure
    """
    # Use sys.stdout.write to avoid print's locking subtleties under asyncio
    sys.stdout.write(f"{prefix}: {msg}\n")
    sys.stdout.flush()


def _build_cc_cmd(
    *,
    system_prompt: str,
    model: str,
    disallowed_tools: List[str],
) -> List[str]:
    """Build the `claude` CLI argv for non-interactive stream-json mode."""
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
    if disallowed_tools:
        cmd += ["--disallowed-tools", ",".join(disallowed_tools)]
    return cmd


def _build_cc_stdin(task: str) -> bytes:
    """One NDJSON user message line. CC -p with --input-format stream-json
    reads events from stdin until EOF."""
    return (
        json.dumps({"type": "user", "message": {"role": "user", "content": task}})
        + "\n"
    ).encode("utf-8")


# CC tool counter (state for _handle_cc_event; module-level since the handler
# is called per-line on the event loop's single thread).
_cc_tool_count = 0


def _handle_cc_event(event: dict) -> None:
    """Parse one CC stream-json event → _emit() lines.

    Event types we recognize:
      assistant   — message with content blocks (text / thinking / tool_use)
      result      — final summary (duration, cost, turns, usage)
      system / rate_limit_event / others — skipped silently. The SDK we used
                    to depend on aborted on unknown event types; the CLI just
                    streams them and lets us choose what to ignore.
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
                    # paragraph structure (no truncation; CC's reasoning
                    # between tool calls is the richest signal for "why this
                    # decision").
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
            _emit("proposer·error",
                  f"agent reported is_error=true: {event.get('result','')}")


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

    # Parse disallowed_tools from comma-separated CLI arg.
    # Empty string / unset = no restrictions = allow all tools.
    disallowed_tools = [
        t.strip() for t in (args.disallowed_tools or "").split(",") if t.strip()
    ]

    # Active datasets (comma-separated; empty = render full 3-benchmark prompt
    # for back-compat with older orchestrators that don't yet wire this).
    active_datasets = [
        d.strip() for d in (args.active_datasets or "").split(",") if d.strip()
    ]

    system_prompt = build_proposer_system(
        sanity_enabled=args.sanity_enabled,
        active_datasets=active_datasets or None,
    )

    cmd = _build_cc_cmd(
        system_prompt=system_prompt,
        model=args.model,
        disallowed_tools=disallowed_tools,
    )
    stdin_bytes = _build_cc_stdin(task)

    if disallowed_tools:
        _emit("proposer·info", f"disallowed_tools={disallowed_tools}")
    _emit("proposer·info", f"mode={args.mode} new_id={args.new_id} model={args.model}")

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
                _handle_cc_event(event)
            except Exception as exc:
                _emit("proposer·error",
                      f"event handler error: {exc} on {line[:200]}")

    try:
        await asyncio.wait_for(_pump(), timeout=args.timeout_s)
    except asyncio.TimeoutError:
        _emit("proposer·error",
              f"timed out after {args.timeout_s}s for {args.new_id}")
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
            _emit("proposer·error",
                  f"finished without writing PROPOSAL_READY at {new_dir}")
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
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--max-turns", type=int, default=80,
                   help="Soft turn budget. NOT enforced — wall-clock --timeout-s "
                        "is the hard limit. Kept for backward compat with old "
                        "callers; no longer plumbed to the claude CLI (which has "
                        "no --max-turns flag).")
    p.add_argument("--timeout-s", type=int, default=25 * 60)
    p.add_argument("--disallowed-tools", default="",
                   help="Comma-separated tool names blocked inside CC. Empty = allow all.")
    p.add_argument("--sanity-enabled", default="true",
                   choices=["true", "false"],
                   help="Whether the orchestrator will run a sanity check after this propose. "
                        "Controls whether sanity-related sections appear in PROPOSER_SYSTEM.")
    p.add_argument("--active-datasets", default="",
                   help="Comma-separated dataset names this run is configured for "
                        "(e.g. 'dynamicmem,locomo'). Empty = full 3-benchmark prompt. "
                        "Controls which dataset shapes / dispatch examples / trace "
                        "field listings appear in PROPOSER_SYSTEM.")
    args = p.parse_args()
    # Convert string flag → bool (argparse choices keep it stringly-typed for
    # CLI symmetry with the host launcher).
    args.sanity_enabled = (args.sanity_enabled == "true")
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
