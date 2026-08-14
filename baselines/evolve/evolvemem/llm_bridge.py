"""Sync `llm_call` for the vendored engine, routed through `common.llm`.

EvolveMem never imports an LLM SDK: every component takes an injected
`llm_call(messages, max_tokens, temperature) -> str`. That is a genuinely useful
property here — it means this baseline's INTERNAL LLM calls (extraction,
diagnosis, meta-analysis, answer generation) go through `common.llm.Agent` and
are therefore counted by `common.tokens`, unlike hipporag2 / mem0 / zep /
simplemem, whose internal calls hit the OpenAI SDK directly and never appear in
`token_usage.json`.

The engine is synchronous and is run in a worker thread (`asyncio.to_thread`),
so each call is bounced back onto the event loop that owns the shared client via
`run_coroutine_threadsafe`. Both `memo.py` (scoring) and `evolve.py` (search)
use this, so the loop and the scored artifact talk to the same LLM layer.
"""
from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional


def make_llm_call(
    model: str,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    on_error: str = "",
) -> Callable[..., str]:
    """Build the sync callable the vendored engine expects.

    Parameters
    ----------
    model : str
        Model name for `common.llm.Agent` (accepts the "/effort" suffix).
    loop : event loop, optional
        Loop to schedule the async call on. Defaults to the running loop, so
        this must be called from async context (both callers are).
    on_error : str
        What to return when a call fails. Upstream treats "" as a failed call
        and applies its OWN retry / chunk-splitting / coverage-verification
        fallbacks, so returning "" keeps those paths faithful; raising would
        abort a whole build or evolution round instead.
    """
    from common.llm import Agent

    target_loop = loop or asyncio.get_running_loop()

    def llm_call(messages: List[Dict], max_tokens: int = 4096,
                 temperature: float = 0.1) -> str:
        msgs = list(messages or [])
        system = ""
        if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            system = str(msgs[0].get("content", ""))
            msgs = msgs[1:]
        agent = Agent(system_prompt=system, model=model,
                      max_completion_tokens=max_tokens)
        future = asyncio.run_coroutine_threadsafe(
            agent.ask(msgs, with_full_msg=True, temperature=temperature),
            target_loop,
        )
        try:
            return str(future.result() or "").strip()
        except Exception:
            return on_error

    return llm_call
