"""Two-section snapshot (Phase B): ``memory/`` (CORE) + ``state/`` (most-complete
REST), convention-based resolution, redaction, completeness signal, and
independent commit-on-change.

Exercises ``ConventionSession`` directly against its own temp per-user git repo
(the env gate lives in ``wrap_memo``; wiring the session live is Phase C).
Standalone — convention-following FAKES only, never the real ``baselines/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_convention_fakes import (
    DumpMemo,
    FakeChromaCollection,
    FakeQdrantClient,
    InHeapMemo,
    NoMemoryMemo,
    QPoint,
    SomeForgeInventedMemo,
)

from tracing.adapters import MemoryItem
from tracing.git_store import GitStore
from tracing.resolver import ConventionSession, resolve_and_snapshot


def _session(tmp_path, memo, name="repo"):
    return ConventionSession(memo=memo, run_id="r", user_dir="u",
                             harness_name="fake", git_store=GitStore(tmp_path / name))


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _read(repo: Path, rel: str) -> str:
    p = repo / rel
    return p.read_text() if p.exists() else ""


def _all_text(repo: Path, sub: str) -> str:
    base = repo / sub
    if not base.exists():
        return ""
    return "\n".join(p.read_text() for p in base.rglob("*.md"))


# --- (i) pure in-heap memory captured structurally ---------------------------


def test_inheap_memory_captured_structurally(tmp_path):
    memo = InHeapMemo({"notes": [{"id": "n0", "content": "Alice likes tea"},
                                 {"id": "n1", "content": "Bob plays chess"}]})
    session = _session(tmp_path, memo)
    result = session.snapshot("BUILD", "hook")
    repo = tmp_path / "repo"

    assert result["memory"] is not None
    mem = _all_text(repo, "memory/inheap")
    assert "Alice likes tea" in mem and "Bob plays chess" in mem


# --- resolution order: L1 dump_memory_text wins ------------------------------


def test_resolution_L1_dump_memory_text_wins():
    memo = DumpMemo([MemoryItem(item_id="d0", body="explicit dump body",
                                embedding_backed=True)])
    snap = resolve_and_snapshot(memo)
    assert snap.completeness.memory_source == "L1:dump_memory_text"
    bodies = "\n".join(ri.item.body for ri in snap.memory)
    assert "explicit dump body" in bodies
    assert "should not appear" not in bodies  # self.memory ignored when L1 present


# --- resolution order: L2 convention resolves an UNKNOWN class name ----------


def test_resolution_L2_convention_ignores_class_name():
    # A brand-new class name (as forge would invent) resolves by convention.
    memo = SomeForgeInventedMemo({"notes": [{"id": "n0", "content": "conv"}]})
    snap = resolve_and_snapshot(memo)
    assert snap.completeness.memory_source == "L2:self.memory"
    assert any("conv" in ri.item.body for ri in snap.memory)


# --- (vii) resolution order: L3 whole-memo fallback when no self.memory ------


def test_resolution_L3_whole_memo_fallback(tmp_path):
    memo = NoMemoryMemo()
    snap = resolve_and_snapshot(memo)
    assert snap.completeness.memory_source == "L3:whole-memo-fallback"
    # Fallback still yields a readable, non-crashing trace.
    session = _session(tmp_path, memo)
    assert session.snapshot("BUILD", "hook")["memory"] is not None
    assert "hello world" in _all_text(tmp_path / "repo", "memory")


# --- (viii) redaction: secret / vector / config never leak -------------------


def test_redaction_no_secret_or_vector_leaks(tmp_path):
    client = FakeQdrantClient([QPoint("p0", {"text": "safe", "embedding": [0.9] * 32})],
                              page=8)
    memo = SomeForgeInventedMemo(
        {"notes": [{"id": "n0", "content": "note", "secret_token": "sk-INHEAP"}],
         "vectors": client},
        _collection_name="mem",
        config={"api_key": "sk-STATELEAK", "top_k": 5})
    session = _session(tmp_path, memo)
    session.snapshot("BUILD", "hook")
    repo = tmp_path / "repo"

    everything = _all_text(repo, "memory") + _read(repo, "state/state.md")
    assert "sk-STATELEAK" not in everything     # config credential
    assert "sk-INHEAP" not in everything        # in-heap secret key dropped
    assert "0.9" not in everything              # vectors


# --- (xi) TWO-SECTION: state has config/counters, excludes memory subtree -----


def test_two_section_state_captures_config_excludes_memory(tmp_path):
    memo = SomeForgeInventedMemo(
        {"notes": [{"id": "n0", "content": "UNIQUE_MEM_CONTENT"}]},
        config={"api_key": "sk-STATELEAK", "top_k": 5},
        counter=3)
    session = _session(tmp_path, memo)
    session.snapshot("BUILD", "hook")
    repo = tmp_path / "repo"

    state = _read(repo, "state/state.md")
    memory = _all_text(repo, "memory")

    # state/ holds the hyperparameter + counter...
    assert "top_k" in state and "counter" in state
    # ...but NEVER the credential (round-1 Issue #1 protection holds)...
    assert "sk-STATELEAK" not in state
    # ...and EXCLUDES the self.memory subtree (no duplication)...
    assert "UNIQUE_MEM_CONTENT" not in state
    # ...while memory/ holds the item.
    assert "UNIQUE_MEM_CONTENT" in memory


# --- (xiii) client OUTSIDE self.memory -> reference marker, not dumped --------


def test_client_outside_memory_is_reference_marker_only(tmp_path):
    aux = FakeChromaCollection(["c0"], ["aux-doc-content"], [{"plumb": "SECRETPLUMB"}])
    memo = SomeForgeInventedMemo({"notes": [{"id": "n0", "content": "mem"}]})
    memo.aux_client = aux                      # a sibling of self.memory
    session = _session(tmp_path, memo)
    snap = session.snapshot("BUILD", "hook")["snapshot"]
    state = _read(tmp_path / "repo", "state/state.md")

    assert any("aux_client" in m and "chroma" in m
               for m in snap.completeness.state_reference_markers)
    assert "client-ref" in state and "chroma" in state
    # NOT structurally dumped: its plumbing/contents don't appear in state/.
    assert "SECRETPLUMB" not in state and "aux-doc-content" not in state


# --- (ix) memory/ commit-on-change ------------------------------------------


def test_memory_commit_on_change(tmp_path):
    memo = InHeapMemo({"notes": [{"id": "n0", "content": "v1"}]})
    session = _session(tmp_path, memo)

    first = session.snapshot("BUILD", "hook")
    assert first["memory"] is not None
    # No change -> no new memory commit.
    assert session.snapshot("BUILD", "hook")["memory"] is None
    # Change the note -> a new memory commit.
    memo.memory["notes"][0]["content"] = "v2"
    assert session.snapshot("BUILD", "hook")["memory"] is not None


# --- (xii) state/ commit-on-change (independent of memory/) -------------------


def test_state_commit_on_change_independent(tmp_path):
    memo = SomeForgeInventedMemo({"notes": [{"id": "n0", "content": "m"}]},
                                 counter=1)
    session = _session(tmp_path, memo)

    first = session.snapshot("BUILD", "hook")
    assert first["state"] is not None                 # initial state commit
    # Nothing changed -> no state commit.
    assert session.snapshot("BUILD", "hook")["state"] is None
    # Change ONLY a state-side counter -> a state commit, but NO memory commit.
    memo.counter = 2
    r = session.snapshot("BUILD", "hook")
    assert r["state"] is not None and r["memory"] is None


# --- (x) completeness signal flags an unreadable client ----------------------


def test_completeness_flags_unreadable_client(tmp_path):
    # A Qdrant client with NO collection scope cannot be read -> flagged.
    client = FakeQdrantClient([QPoint("p0", {"text": "x"})])
    memo = SomeForgeInventedMemo({"vectors": client})  # no collection attr
    session = _session(tmp_path, memo)
    snap = session.snapshot("BUILD", "hook")["snapshot"]

    assert snap.completeness.has_gaps
    assert any("qdrant" in u for u in snap.completeness.unreadable)
    # The completeness signal is persisted alongside the state section.
    assert "qdrant" in _read(tmp_path / "repo", "completeness.md")


# --- two independent sections both land in one repo --------------------------


def test_both_sections_present_and_clean_tree(tmp_path):
    memo = SomeForgeInventedMemo({"notes": [{"id": "n0", "content": "x"}]},
                                 counter=1)
    session = _session(tmp_path, memo)
    session.snapshot("BUILD", "hook")
    repo = tmp_path / "repo"

    assert (repo / "memory").exists() and (repo / "state" / "state.md").exists()
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""  # clean tree
