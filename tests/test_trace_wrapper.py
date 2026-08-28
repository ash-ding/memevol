"""Wrapper behaviour: transparency, read-only retrieve, zero-cost when disabled.

Standalone — no eval path is exercised. Uses ``asyncio.run`` for the async
MemoClass hooks (matching the repo's no-pytest-asyncio convention).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_fakes import NoteListMemo, Recorder

import tracing
from tracing.wrapper import TracedMemo

_FACTS = ["Alice lives in Paris.", "Bob plays chess.", "Carol codes in Rust."]


def _drive(memo):
    """Run a scripted BUILD -> RETRIEVE -> ANSWER and return the results."""
    async def go():
        build = Recorder({"facts": _FACTS})
        b = await memo.build_memory_from_data(build)
        q = Recorder({"query": "who?"})
        r = await memo.retrieve_memory_for_query(q)
        a = await memo.use_memory_to_answer(q, r, "who?")
        return b, r, a
    return asyncio.run(go())


def _state(memo):
    inner = getattr(memo, "_inner", memo)
    return {nid: (n.content, tuple(n.keywords))
            for nid, n in inner._system.memories.items()}


def test_wrapper_transparency(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    monkeypatch.setenv("MEMEVOL_TRACE_ROOT", str(tmp_path))

    raw = NoteListMemo()
    wrapped = tracing.wrap_memo(NoteListMemo(), run_id="r", user_dir="u",
                                harness_name="NoteListMemo")
    assert isinstance(wrapped, TracedMemo)

    raw_b, raw_r, raw_a = _drive(raw)
    w_b, w_r, w_a = _drive(wrapped)

    # Identical hook return values...
    assert raw_b == w_b
    assert raw_r == w_r
    assert raw_a == w_a
    # ...and an identical inner memo end-state.
    assert _state(raw) == _state(wrapped)


def test_wrapper_delegates_unknown_attrs(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    monkeypatch.setenv("MEMEVOL_TRACE_ROOT", str(tmp_path))
    inner = NoteListMemo()
    wrapped = tracing.wrap_memo(inner, run_id="r", user_dir="u",
                                harness_name="NoteListMemo")
    # __getattr__ transparently delegates non-hook attributes to the inner memo.
    assert wrapped._system is inner._system


def test_trace_retrieve_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    monkeypatch.setenv("MEMEVOL_TRACE_ROOT", str(tmp_path))
    inner = NoteListMemo()
    wrapped = tracing.wrap_memo(inner, run_id="r", user_dir="u",
                                harness_name="NoteListMemo")

    async def go():
        await wrapped.build_memory_from_data(Recorder({"facts": _FACTS}))
        rec = Recorder({"query": "who?"})
        init_ref = rec.init
        before = _state(inner)
        result = await wrapped.retrieve_memory_for_query(rec)
        # recorder.init is the SAME object and unmodified.
        assert rec.init is init_ref
        assert rec.init == {"query": "who?"}
        # inner memo state is byte-identical before vs. after the traced read.
        assert _state(inner) == before
        return result

    assert asyncio.run(go())["hits"]  # retrieve still returns real content


def test_trace_disabled_is_free(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMEVOL_TRACE", raising=False)
    monkeypatch.setenv("MEMEVOL_TRACE_ROOT", str(tmp_path))

    inner = NoteListMemo()
    result = tracing.wrap_memo(inner, run_id="r", user_dir="u",
                               harness_name="NoteListMemo")
    # Passthrough: the ORIGINAL instance is returned, no TracedMemo constructed.
    assert result is inner
    assert not isinstance(result, TracedMemo)

    # mark() and the shared-kernel callback are single-branch no-ops.
    assert tracing.mark("checkpoint") is None
    assert tracing.on_kernel_call("llm") is None

    # Zero tracing work: nothing was written under the trace root.
    assert not any(tmp_path.rglob("*.md"))
    assert not any(tmp_path.rglob("*.git"))
