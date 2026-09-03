"""Coding-agent proposer — drives `claude` or `codex` and logs the session.

Meta-Harness's proposer is a coding agent with filesystem access to every
prior candidate's code, score, and execution traces; upstream ships a Claude
Code wrapper and notes that any agent works given "a wrapper that cleanly logs
proposer interactions". Both backends are supported here.

Each agent contributes three functions — argv, stdin payload, NDJSON event
handler — and the drive loop below is identical for both. This mirrors
`forge/propose_in_container.py`, deliberately copied rather than imported
(baselines never import a method's internals).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

AGENTS = ("claude_code", "codex")

# Reasoning effort when the config leaves it null. The paper runs its proposer
# at MAX reasoning (§4.1, "Opus-4.6 with max reasoning"), which is what upstream's
# wrapper passes; codex's scale tops out at "high", so the default is per-agent
# rather than one value that is wrong for one of them.
DEFAULT_EFFORT = {"claude_code": "max", "codex": "high"}


@dataclass
class ProposeResult:
    exit_code: int
    text: str = ""
    tools: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_s: float = 0.0
    log_dir: Optional[str] = None
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


# --------------------------------------------------------------------------
# claude_code
# --------------------------------------------------------------------------

def _cmd_claude(*, model: str, system_prompt_file: Path, effort: Optional[str]) -> List[str]:
    cmd = [
        "claude", "-p",
        # --verbose is REQUIRED alongside --output-format stream-json in -p mode.
        "--verbose",
        "--output-format", "stream-json",
        "--input-format", "stream-json",
        "--model", model,
        # File-based system prompt: keeps a ~20 KB blob out of argv (where `ps`
        # would expose it) and leaves the exact prior next to the session log.
        "--system-prompt-file", str(system_prompt_file),
        "--permission-mode", "bypassPermissions",
        # The proposer prior is injected as the system prompt, so the operator's
        # own settings, skills, plugins and MCP servers must not leak in.
        "--setting-sources", "",
        "--strict-mcp-config",
    ]
    if effort:
        cmd += ["--effort", str(effort)]
    return cmd


def _stdin_claude(system_prompt: str, task: str) -> bytes:
    """One NDJSON user message; the system prompt rides --system-prompt-file."""
    line = {"type": "user", "message": {"role": "user", "content": task}}
    return (json.dumps(line) + "\n").encode("utf-8")


def _event_claude(event: dict, result: ProposeResult) -> None:
    etype = event.get("type", "")
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []) or []:
            if block.get("type") == "text":
                result.text += block.get("text") or ""
            elif block.get("type") == "tool_use":
                tool = f"{block.get('name', '')}({str(block.get('input', ''))[:120]})"
                result.tools.append(tool)
    elif etype == "result":
        result.usage = event.get("usage", {}) or {}
        result.cost_usd = float(event.get("total_cost_usd") or 0.0)


# --------------------------------------------------------------------------
# codex
# --------------------------------------------------------------------------

def _cmd_codex(*, model: str, cwd: Path, effort: Optional[str]) -> List[str]:
    cmd = [
        "codex", "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--cd", str(cwd),
        "--model", model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--sandbox", "danger-full-access",
    ]
    if effort:
        cmd += ["-c", 'model_reasoning_effort="%s"' % effort]
    cmd += ["-"]  # prompt on stdin; must come after any -c override
    return cmd


def _stdin_codex(system_prompt: str, task: str) -> bytes:
    """Codex has no system-prompt channel — prior and task go in as one prompt."""
    return ("%s\n\n---\n\n%s\n" % (system_prompt, task)).encode("utf-8")


def _event_codex(event: dict, result: ProposeResult) -> None:
    etype = event.get("type", "")
    if etype == "item.completed":
        item = event.get("item") or {}
        if item.get("type") == "agent_message":
            result.text += item.get("text") or ""
        else:
            result.tools.append(f"{item.get('type', '')}({json.dumps(item)[:120]})")
    elif etype == "turn.completed":
        result.usage = event.get("usage", {}) or {}


_DISPATCH: Dict[str, Dict[str, Callable]] = {
    "claude_code": {"cmd": _cmd_claude, "stdin": _stdin_claude, "event": _event_claude},
    "codex": {"cmd": _cmd_codex, "stdin": _stdin_codex, "event": _event_codex},
}


# --------------------------------------------------------------------------
# drive loop
# --------------------------------------------------------------------------

def _child_env(agent: str, auth: str) -> Dict[str, str]:
    """Auth for the agent CLI. `subscription` drops ANTHROPIC_API_KEY so the
    claude CLI falls back to `claude login` OAuth (upstream does the same, to
    avoid API rate limits); `api_key` leaves the environment untouched. Codex
    authenticates through its own login and ignores this."""
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    if agent == "claude_code" and auth == "subscription":
        env.pop("ANTHROPIC_API_KEY", None)
    return env


async def _drive(
    *, agent: str, cmd: List[str], stdin_bytes: bytes, cwd: Path, env: Dict[str, str],
    timeout_s: int, result: ProposeResult, events_path: Path,
) -> None:
    handle = _DISPATCH[agent]["event"]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd), env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # A single NDJSON event can carry a multi-MB tool_result; asyncio's
        # 64 KB default would raise LimitOverrunError and kill the propose.
        limit=16 * 1024 * 1024,
    )

    assert proc.stdin is not None
    try:
        proc.stdin.write(stdin_bytes)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError) as exc:
        result.stderr += f"failed writing stdin: {exc}\n"
    finally:
        proc.stdin.close()

    async def pump() -> None:
        assert proc.stdout is not None
        with events_path.open("w", encoding="utf-8") as events:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                events.write(line + "\n")
                try:
                    handle(json.loads(line), result)
                except (json.JSONDecodeError, ValueError):
                    continue
                except Exception as exc:  # a bad event must not abort the run
                    result.stderr += f"event handler error: {exc}\n"
        result.exit_code = await proc.wait()

    try:
        await asyncio.wait_for(pump(), timeout=timeout_s)
    except asyncio.TimeoutError:
        result.exit_code = 124
        result.stderr += f"proposer timed out after {timeout_s}s\n"
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()

    if proc.stderr is not None:
        try:
            result.stderr += (await proc.stderr.read()).decode(errors="replace")
        except Exception:
            pass


async def propose(
    *,
    agent: str,
    model: str,
    system_prompt: str,
    task: str,
    cwd: Path,
    log_dir: Path,
    name: str,
    timeout_s: int,
    effort: Optional[str] = None,
    auth: str = "subscription",
) -> ProposeResult:
    """Run one proposer session. Everything it emitted is written under
    `log_dir/<name>/` — the run's audit trail of what the agent saw and did."""
    if agent not in _DISPATCH:
        raise ValueError(f"unknown agent {agent!r}; valid: {AGENTS}")
    effort = effort or DEFAULT_EFFORT[agent]

    session_dir = log_dir / name
    session_dir.mkdir(parents=True, exist_ok=True)
    system_file = session_dir / "system_prompt.txt"
    system_file.write_text(system_prompt, encoding="utf-8")
    (session_dir / "task_prompt.txt").write_text(task, encoding="utf-8")

    if agent == "claude_code":
        cmd = _cmd_claude(model=model, system_prompt_file=system_file, effort=effort)
    else:
        cmd = _cmd_codex(model=model, cwd=cwd, effort=effort)
    stdin_bytes = _DISPATCH[agent]["stdin"](system_prompt, task)

    result = ProposeResult(exit_code=0, log_dir=str(session_dir))
    started = datetime.now(timezone.utc)
    try:
        await _drive(
            agent=agent, cmd=cmd, stdin_bytes=stdin_bytes, cwd=cwd,
            env=_child_env(agent, auth), timeout_s=timeout_s,
            result=result, events_path=session_dir / "events.jsonl",
        )
    except FileNotFoundError as exc:
        result.exit_code = 127
        result.stderr += f"{exc} — is the `{cmd[0]}` CLI installed and on PATH?\n"
    result.duration_s = (datetime.now(timezone.utc) - started).total_seconds()

    (session_dir / "response.md").write_text(result.text, encoding="utf-8")
    (session_dir / "meta.json").write_text(json.dumps({
        "timestamp": started.isoformat(),
        "agent": agent, "model": model, "effort": effort, "auth": auth,
        "command": cmd, "cwd": str(cwd),
        "exit_code": result.exit_code,
        "duration_seconds": round(result.duration_s, 2),
        "cost_usd": result.cost_usd,
        "usage": result.usage,
        "tools": result.tools,
        "stderr": result.stderr[-4000:],
    }, indent=2, default=str), encoding="utf-8")
    return result
