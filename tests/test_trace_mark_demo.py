"""mark() extension point, the shared-kernel observer callback, no cross-user
bleed, and the standalone demo smoke test.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_fakes import Note, NoteListMemo

import tracing
from tracing import demo, triggers
from tracing.adapters import NoteListAdapter
from tracing.git_store import GitStore
from tracing.triggers import (
    TraceSession,
    _current_phase,
    on_kernel_call,
    set_current_session,
)


def _session(tmp_path, memo, name="repo"):
    return TraceSession(
        memo=memo, run_id="r", user_dir="u", harness_name="test",
        adapter=NoteListAdapter(), git_store=GitStore(tmp_path / name))


def _set_notes(memo, mapping):
    memo._system.memories = {nid: Note(nid, content, ["k"])
                             for nid, content in mapping.items()}


def _reset_ctx():
    set_current_session(None)
    _current_phase.set(None)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


# --- mark() ----------------------------------------------------------------


def test_trace_mark_extension_point(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    repo = tmp_path / "repo"
    try:
        set_current_session(session)
        # (a) enabled + state changed -> exactly one commit tagged trigger=mark.
        _set_notes(memo, {"n0": "hello"})
        sha = tracing.mark("checkpoint")
        assert sha is not None
        assert len(session.store.log_lines()) == 1
        body = _git(repo, "log", "-1", "--format=%B").stdout
        assert "trigger=mark" in body and "label=checkpoint" in body

        # unchanged state -> mark() creates no empty commit.
        assert tracing.mark("again") is None
        assert len(session.store.log_lines()) == 1

        # (c) vector-shaped / secret-looking structural_meta never leaks.
        _set_notes(memo, {"n0": "hello", "n1": "world"})
        tracing.mark("cp", secret_vec=[0.987654] * 32, token="SHOULD_STAY_LABEL_ONLY")
        body2 = _git(repo, "log", "-1", "--format=%B").stdout
        assert "0.987654" not in body2
    finally:
        _reset_ctx()


def test_trace_mark_disabled_does_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMEVOL_TRACE", raising=False)
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    try:
        set_current_session(session)
        _set_notes(memo, {"n0": "hello"})
        # (b) disabled -> zero tracing work, no repo/commit created.
        assert tracing.mark("checkpoint") is None
        assert not (tmp_path / "repo" / ".git").exists()
    finally:
        _reset_ctx()


# --- shared-kernel observer callback (API unit-tested by direct invocation) --


def test_trace_kernel_call_trigger(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    repo = tmp_path / "repo"
    try:
        set_current_session(session)
        _current_phase.set("BUILD")
        _set_notes(memo, {"n0": "a"})
        sha = on_kernel_call("llm")  # simulate a completed LLM kernel call
        assert sha is not None
        assert "trigger=llm-call" in _git(repo, "log", "-1", "--format=%B").stdout

        _set_notes(memo, {"n0": "a", "n1": "b"})
        on_kernel_call("embed")     # simulate an embedding call
        assert "trigger=embed-call" in _git(repo, "log", "-1", "--format=%B").stdout
    finally:
        _reset_ctx()


def test_kernel_call_noop_without_session(monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    set_current_session(None)
    try:
        assert on_kernel_call("llm") is None  # no active session -> no-op
    finally:
        _reset_ctx()


# --- no cross-user bleed ---------------------------------------------------


def test_trace_concurrent_instances_no_bleed(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMEVOL_TRACE", "1")
    monkeypatch.setenv("MEMEVOL_TRACE_ROOT", str(tmp_path))

    w_alice = tracing.wrap_memo(NoteListMemo(), run_id="run", user_dir="alice",
                                harness_name="NoteListMemo")
    w_bob = tracing.wrap_memo(NoteListMemo(), run_id="run", user_dir="bob",
                              harness_name="NoteListMemo")

    from _trace_fakes import Recorder

    async def drive(wrapped, facts):
        await wrapped.build_memory_from_data(Recorder({"facts": facts}))
        await wrapped.retrieve_memory_for_query(Recorder({"query": "?"}))

    asyncio.run(_gather(
        drive(w_alice, ["ALICE_FACT_ONE", "ALICE_FACT_TWO"]),
        drive(w_bob, ["BOB_FACT_ONE"]),
    ))

    base = tmp_path / "run" / "traces" / "git"
    alice_repo = base / triggers._user_key("alice")
    bob_repo = base / triggers._user_key("bob")
    assert alice_repo.exists() and bob_repo.exists()
    assert alice_repo != bob_repo

    alice_text = _all_md_text(alice_repo)
    bob_text = _all_md_text(bob_repo)
    assert "ALICE_FACT_ONE" in alice_text and "ALICE_FACT_TWO" in alice_text
    assert "BOB_FACT_ONE" in bob_text
    # No content bleed across the per-user repos.
    assert "BOB_FACT_ONE" not in alice_text
    assert "ALICE_FACT_ONE" not in bob_text


async def _gather(*coros):
    await asyncio.gather(*coros)


def _all_md_text(repo: Path) -> str:
    return "\n".join(p.read_text() for p in repo.rglob("*.md"))


# --- demo smoke ------------------------------------------------------------


def test_trace_demo_smoke(tmp_path):
    result = demo.run_demo(tmp_path)
    repo_dir = result["repo_dir"]
    log = result["log"]

    assert repo_dir.exists() and (repo_dir / ".git").exists()
    assert len(log) >= 1
    # A BUILD/ingest commit is present...
    assert any("[BUILD]" in line and "pattern=ingest" in line for line in log)
    # ...and the read-only RETRIEVE calls produced NO state-change commit.
    assert not any("[RETRIEVE]" in line for line in log)
    # The repo really lives under the provided trace root.
    assert str(tmp_path) in str(repo_dir)
    # Sanity: the on-disk repo has the same number of commits as the log.
    count = subprocess.run(["git", "-C", str(repo_dir), "rev-list", "--count",
                            "HEAD"], capture_output=True, text=True).stdout.strip()
    assert int(count) == len(log)
