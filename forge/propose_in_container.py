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
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Project root is bound at /app. Add to sys.path so `from forge.prompts` works.
_APP = Path("/app")
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    SystemMessage,
)
from claude_code_sdk._errors import MessageParseError
import claude_code_sdk._internal.message_parser as _mp
import claude_code_sdk._internal.client as _cl

from forge.prompts import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)


# SDK 0.0.25 aborts the message stream on first unknown event type
# (e.g. rate_limit_event). Patch to return a harmless SystemMessage. Same
# fix as the original host-side proposer — preserved here for parity.
_original_parse = _mp.parse_message


def _lenient_parse(data):
    try:
        return _original_parse(data)
    except MessageParseError:
        return SystemMessage(subtype="unknown", data=data)


_cl.parse_message = _lenient_parse


# Tool restrictions intentionally NOT set: filesystem isolation is enforced
# by the Singularity sandbox in forge/proposer.py — CC's reachable surface is
# exactly the bind list (workspace + project root RO + CLI binary RO + per-run
# scratch HOME). Bash, WebFetch, WebSearch, mcp__* all permitted inside the
# container; the proposer image (extends eval-base.sif) has the same Python
# env as the evaluator, so CC can `python -c "from harness import ..."` for
# sub-second import validation. See PROPOSER_SYSTEM for the workflow cheat sheet.


def _emit(prefix: str, msg: str) -> None:
    """Print a tagged line to stdout. Host-side launcher line-by-line
    forwards these into the orchestrator log.

    Tag conventions:
      proposer·text    CC assistant text (truncated to 160 chars)
      proposer·tool    CC tool use (truncated to 140 chars)
      proposer·info    high-level status (model, finish, usage)
      proposer·error   actionable failure
    """
    # Use sys.stdout.write to avoid print's locking subtleties under asyncio
    sys.stdout.write(f"{prefix}: {msg}\n")
    sys.stdout.flush()


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
    disallowed_tools = [t.strip() for t in (args.disallowed_tools or "").split(",") if t.strip()]

    options = ClaudeCodeOptions(
        cwd=str(workspace),
        system_prompt=build_proposer_system(sanity_enabled=args.sanity_enabled),
        permission_mode="bypassPermissions",
        model=args.model,
        max_turns=args.max_turns,
        disallowed_tools=disallowed_tools,
        # No allowed_tools — sandbox is the bind list, not the tool list.
    )

    if disallowed_tools:
        _emit("proposer·info", f"disallowed_tools={disallowed_tools}")

    _emit("proposer·info", f"mode={args.mode} new_id={args.new_id} model={args.model}")

    tool_count = 0

    async def _stream() -> None:
        nonlocal tool_count
        async for msg in query(prompt=task, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        snippet = block.text.strip()[:160].replace("\n", " ")
                        if snippet:
                            _emit("proposer·text", snippet)
                    elif isinstance(block, ToolUseBlock):
                        tool_count += 1
                        _emit("proposer·tool",
                              f"#{tool_count} {block.name} {str(block.input)[:140]}")
            elif isinstance(msg, ResultMessage):
                _emit("proposer·info",
                      f"finished turns={msg.num_turns} tools={tool_count} "
                      f"duration={msg.duration_ms}ms cost=${msg.total_cost_usd or 0:.4f} "
                      f"usage={msg.usage}")

    try:
        await asyncio.wait_for(_stream(), timeout=args.timeout_s)
    except asyncio.TimeoutError:
        _emit("proposer·error",
              f"timed out after {args.timeout_s}s for {args.new_id}")
        return 3

    if args.mode == "propose":
        sentinel = new_dir / "PROPOSAL_READY"
        if not sentinel.exists():
            _emit("proposer·error",
                  f"finished without writing PROPOSAL_READY at {new_dir}")
            return 4

    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--new-id", required=True)
    p.add_argument("--workspace", required=True,
                   help="Container path of the per-run workspace (typically /workspace).")
    p.add_argument("--mode", choices=["propose", "fix"], default="propose")
    p.add_argument("--error-trace-file", default=None,
                   help="Required when --mode=fix; path inside container to the trace text file.")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--timeout-s", type=int, default=25 * 60)
    p.add_argument("--disallowed-tools", default="",
                   help="Comma-separated tool names blocked inside CC. Empty = allow all.")
    p.add_argument("--sanity-enabled", default="true",
                   choices=["true", "false"],
                   help="Whether the orchestrator will run a sanity check after this propose. "
                        "Controls whether sanity-related sections appear in PROPOSER_SYSTEM.")
    args = p.parse_args()
    # Convert string flag → bool (argparse choices keep it stringly-typed for
    # CLI symmetry with the host launcher).
    args.sanity_enabled = (args.sanity_enabled == "true")
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
