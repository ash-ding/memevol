"""cc as a native-answer MemoStructure: Phase 1 (`general_update`) stashes the
currently-visible data into a per-user temp dir. Phase 2 (`general_retrieve`)
re-stashes the CURRENT visible data (the workflow may have grown the visible
prefix, e.g. DynamicMem checkpoints) and returns `{}` — cc injects NO memory
into the QA prompt; it answers by reading the temp-dir file itself via tools.

The actual answering happens in `CCMemo.general_answer`, which the workflow's
shared answer step calls first (falling back to the standard QA agent only
when a memo returns None) — cc runs Claude Code (with Read/Grep/Glob tool
access to the memo's temp dir) on the EXACT formatted prompt the main method
would pose to its own QA agent, which is what makes cc emit the benchmark's
required output format (e.g. DynamicMem TCE's "Return JSON only" + skeleton)
instead of free prose that the judge can't parse.

`ask_cc` / `DISALLOWED_TOOLS` / the SDK-parser monkey-patch are transplanted
verbatim from the old `baselines/cc/eval_cc.py` (DynamicMem-only script,
removed in favor of this multi-dataset module). The one adaptation: the old
script hardcoded a single-dataset `SYSTEM_PROMPT` constant; `ask_cc` now takes
an explicit `system_prompt` param (falling back to the per-dataset
file-schema prompt derived from `tmp_dir`'s contents — see
`_system_prompt_for_dir`) so callers can supply their own.
"""
from __future__ import annotations

import os

from claude_code_sdk import (
    query, ClaudeCodeOptions, AssistantMessage, ResultMessage, UserMessage,
    TextBlock, ToolUseBlock, ToolResultBlock, SystemMessage,
)
from claude_code_sdk._errors import MessageParseError

from common.harness_base import MemoStructure

# Monkey-patch SDK's parse_message to skip unknown message types
# (e.g. rate_limit_event not yet handled in SDK 0.0.25).
# Without this, the entire query stream aborts on first unknown event.
import claude_code_sdk._internal.message_parser as _mp
import claude_code_sdk._internal.client as _cl

_original_parse = _mp.parse_message


def _lenient_parse(data):
    try:
        return _original_parse(data)
    except MessageParseError:
        # Return a harmless SystemMessage so the stream continues
        return SystemMessage(subtype="unknown", data=data)


_cl.parse_message = _lenient_parse

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
}

# Disallow all MCP tools to ensure pure file-retrieval evaluation
DISALLOWED_TOOLS = [
    "mcp__*",  # wildcard may not work; list explicit prefixes below as fallback
    "mcp__claude_ai_Gmail__gmail_create_draft",
    "mcp__claude_ai_Gmail__gmail_get_profile",
    "mcp__claude_ai_Gmail__gmail_list_drafts",
    "mcp__claude_ai_Gmail__gmail_list_labels",
    "mcp__claude_ai_Gmail__gmail_read_message",
    "mcp__claude_ai_Gmail__gmail_read_thread",
    "mcp__claude_ai_Gmail__gmail_search_messages",
    "mcp__claude_ai_Google_Calendar__gcal_create_event",
    "mcp__claude_ai_Google_Calendar__gcal_delete_event",
    "mcp__claude_ai_Google_Calendar__gcal_find_meeting_times",
    "mcp__claude_ai_Google_Calendar__gcal_find_my_free_time",
    "mcp__claude_ai_Google_Calendar__gcal_get_event",
    "mcp__claude_ai_Google_Calendar__gcal_list_calendars",
    "mcp__claude_ai_Google_Calendar__gcal_list_events",
    "mcp__claude_ai_Google_Calendar__gcal_respond_to_event",
    "mcp__claude_ai_Google_Calendar__gcal_update_event",
    "mcp__leann-server__leann_list",
    "mcp__leann-server__leann_search",
]

# per-dataset context spec: init-key → (filename, schema description)
_CONTEXT = {
    "app_logs": ("app_log_large.json",
        "Each entry has: app_log_id, timestamp, app_name, api_name, request, response, and metadata."),
    "conversation": ("conversation.json",
        "A multi-session two-speaker conversation: keys speaker_a, speaker_b, session_1..N "
        "(each a list of {speaker, dia_id, text}), and session_N_date_time."),
    "sessions": ("sessions.json",
        "A list of chat sessions, each {session_id, date, messages: [{role, content}]}."),
}
_FILENAME_TO_KEY = {fname: key for key, (fname, _schema) in _CONTEXT.items()}


def _context_key(init: dict) -> str:
    for k in _CONTEXT:
        if k in init:
            return k
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


def _system_prompt(key: str) -> str:
    fname, schema = _CONTEXT[key]
    return (f"You are answering a task using a user's data.\n\n### Data\n"
            f"The file `{fname}` in your working directory contains the data. {schema}\n"
            f"This is the ONLY file you should read.\n\n### Instructions\n"
            f"Use your tools to read the data, then answer the task in the user message. "
            f"Follow the task's output-format instructions EXACTLY — if it asks for JSON, "
            f"output only the JSON object, no prose or preamble.")


def _system_prompt_for_dir(tmp_dir: str) -> str:
    """Derive the per-dataset system prompt from which known context file
    `_write_context` dropped into tmp_dir — used as ask_cc's fallback when no
    explicit `system_prompt` is passed."""
    for fname in os.listdir(tmp_dir):
        if fname in _FILENAME_TO_KEY:
            return _system_prompt(_FILENAME_TO_KEY[fname])
    raise FileNotFoundError(f"no recognized context file in {tmp_dir}: {os.listdir(tmp_dir)}")


async def ask_cc(question: str, tmp_dir: str, model: str, max_turns: int = 30,
                  system_prompt: str | None = None) -> tuple[str, dict, list]:
    """Ask Claude Code a question with tool access to tmp_dir.

    `system_prompt`: explicit system prompt to use (e.g. the per-dataset
    file-schema prompt built by CCMemo._run_cc). Falls back to deriving one
    from tmp_dir's contents (`_system_prompt_for_dir`) when not given.

    Returns (answer_text, usage_info, trace).
    trace is a list of dicts recording each step:
      {"role": "assistant", "type": "text", "content": "..."}
      {"role": "assistant", "type": "tool_use", "tool": "Grep", "input": {...}}
      {"role": "tool", "tool_use_id": "...", "content": "...(truncated)"}
    """
    options = ClaudeCodeOptions(
        cwd=tmp_dir,
        system_prompt=system_prompt or _system_prompt_for_dir(tmp_dir),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model=model,
        disallowed_tools=DISALLOWED_TOOLS,
    )

    final_text = ""
    usage_info = {}
    trace = []

    async for msg in query(prompt=f"Question: {question}", options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    final_text = block.text
                    trace.append({"role": "assistant", "type": "text", "content": block.text})
                elif isinstance(block, ToolUseBlock):
                    trace.append({
                        "role": "assistant",
                        "type": "tool_use",
                        "tool": block.name,
                        "tool_use_id": block.id,
                        "input": block.input,
                    })
                elif isinstance(block, ToolResultBlock):
                    # Tool results inside assistant message
                    content = block.content or ""
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + f"...(truncated, {len(block.content)} chars)"
                    trace.append({
                        "role": "tool",
                        "tool_use_id": block.tool_use_id,
                        "content": content,
                        "is_error": block.is_error,
                    })
        elif isinstance(msg, UserMessage):
            # UserMessage contains tool results
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content or ""
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + f"...(truncated, {len(block.content)} chars)"
                    trace.append({
                        "role": "tool",
                        "tool_use_id": block.tool_use_id,
                        "content": content,
                        "is_error": block.is_error,
                    })
        elif isinstance(msg, ResultMessage):
            final_text = msg.result or final_text
            usage_info = msg.usage or {}

    return final_text, usage_info, trace


class CCMemo(MemoStructure):
    def __init__(self):
        super().__init__()
        self._tmp_dir = None
        self._key = None

    def _write_context(self, init: dict, user_id: str):
        import tempfile, json as _json
        self._key = _context_key(init)
        fname, _ = _CONTEXT[self._key]
        if self._tmp_dir is None:
            self._tmp_dir = tempfile.mkdtemp(prefix=f"cc_{user_id}_")
        payload = init[self._key]
        with open(os.path.join(self._tmp_dir, fname), "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False)

    async def general_update(self, recorder) -> None:
        # stash the visible data (accumulating prefix for DynamicMem checkpoints)
        self._write_context(recorder.init, getattr(recorder, "user_id", "u"))

    async def general_retrieve(self, recorder) -> dict:
        # ensure the CURRENT visible data is on disk (Phase-2 init carries the
        # prefix); inject NO memory into the QA prompt — cc reads via file
        # tools (see general_answer, which calls _run_cc with the workflow's
        # own formatted prompt).
        self._write_context(recorder.init, getattr(recorder, "user_id", "u"))
        return {}

    async def general_answer(self, recorder, retrieved, prompt) -> str:
        """cc answers the workflow's formatted prompt via Claude Code (native
        agentic answer). self._tmp_dir was written by general_update/retrieve."""
        answer, _usage, _trace = await self._run_cc(prompt)
        return answer

    async def _run_cc(self, question: str) -> tuple:
        """Run Claude Code on `question` (the workflow's exact formatted
        prompt) with tool access to this user's temp dir. Requires
        general_update/general_retrieve to have already run at least once
        (sets self._tmp_dir + self._key)."""
        ask = self._cfg.get("_ask_cc", ask_cc)
        return await ask(question, self._tmp_dir, self._cfg["model"],
                         self._cfg.get("max_turns", 30),
                         system_prompt=_system_prompt(self._key))

    def __del__(self):
        # Best-effort cleanup of the per-instance scratch dir. One MemoStructure
        # instance == one user (no cross-user state, see CLAUDE.md), so this
        # never removes another user's data; ignore_errors guards against
        # already-gone dirs / interpreter-shutdown teardown ordering.
        tmp_dir = getattr(self, "_tmp_dir", None)
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
