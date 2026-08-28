"""Git store + descriptor: commit-on-change dedup, deterministic commit
messages with machine-parseable trailers, and stable/followable per-item paths.

These exercise ``TraceSession`` directly (no env gate needed — the gate lives in
``wrap_memo``/``mark``), each against its own temp per-user git repo.
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

from _trace_fakes import Note, NoteListMemo

from tracing.adapters import NoteListAdapter
from tracing.git_store import GitStore
from tracing.triggers import (
    TraceSession,
    _Diff,
    build_commit_message,
    diff_shape_label,
)


def _session(tmp_path, memo):
    return TraceSession(
        memo=memo, run_id="r", user_dir="u", harness_name="test",
        adapter=NoteListAdapter(),
        git_store=GitStore(tmp_path / "repo"))


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)


def _set_notes(memo, mapping):
    memo._system.memories = {nid: Note(nid, content, ["k"])
                             for nid, content in mapping.items()}


# --- commit-on-change ------------------------------------------------------


def test_trace_commit_on_change(tmp_path):
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    repo = tmp_path / "repo"

    # Empty -> empty: no commit.
    assert session.snapshot("BUILD", "hook") is None
    assert session.store.log_lines() == []

    # Add an item: exactly one commit.
    _set_notes(memo, {"n0": "first"})
    sha1 = session.snapshot("BUILD", "hook")
    assert sha1 is not None
    assert len(session.store.log_lines()) == 1

    # No change: no new commit (no --allow-empty).
    assert session.snapshot("BUILD", "hook") is None
    assert len(session.store.log_lines()) == 1

    # Another change: one more commit.
    _set_notes(memo, {"n0": "first", "n1": "second"})
    sha2 = session.snapshot("BUILD", "hook")
    assert sha2 is not None and sha2 != sha1
    assert len(session.store.log_lines()) == 2
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""  # clean tree


# --- descriptor + trailers -------------------------------------------------


def test_trace_commit_message_descriptor(tmp_path):
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    repo = tmp_path / "repo"

    # Ingest two items (content carries a secret-looking token that must never
    # reach the message — the message holds only ids/counts/pattern).
    _set_notes(memo, {"n0": "SECRET_TOKEN_XYZ likes tea", "n1": "lives in NYC"})
    session.snapshot("BUILD", "hook")

    body = _git(repo, "log", "-1", "--format=%B").stdout
    assert body.startswith("[BUILD]")
    assert "trigger=hook" in body
    assert "+2 ~0 -0 items" in body
    assert "pattern=ingest" in body
    assert "op_seq=1" in body
    # No secret / vector / config content leaks into the message.
    assert "SECRET_TOKEN_XYZ" not in body

    # Machine-parseable git trailers are present and correct.
    def trailer(key):
        return _git(repo, "log", "-1",
                    f"--format=%(trailers:key={key},valueonly)").stdout.strip()
    assert trailer("Phase") == "BUILD"
    assert trailer("Trigger") == "hook"
    assert trailer("Op-Seq") == "1"
    assert trailer("Added") == "2"
    assert trailer("Updated") == "0"
    assert trailer("Deleted") == "0"
    assert trailer("Pattern") == "ingest"

    # Pure deletion (0 added, 0 updated, N deleted) -> prune/delete label.
    _set_notes(memo, {})
    session.snapshot("BUILD", "hook")
    body2 = _git(repo, "log", "-1", "--format=%B").stdout
    assert "+0 ~0 -2 items" in body2
    assert "pattern=prune/delete" in body2
    assert trailer("Pattern") == "prune/delete"


def test_descriptor_is_deterministic_and_label_map():
    d_ingest = _Diff(added=["a", "b"])
    d_prune = _Diff(deleted=["a", "b"])
    d_update = _Diff(updated=["a"])
    d_merge = _Diff(added=["a"], deleted=["b"])
    assert diff_shape_label(d_ingest) == "ingest"
    assert diff_shape_label(d_prune) == "prune/delete"
    assert diff_shape_label(d_update) == "link/update"
    assert diff_shape_label(d_merge) == "consolidate/merge"

    # Byte-identical output for identical inputs (no LLM, no timestamp).
    kw = dict(phase="BUILD", trigger="hook", op_seq=3, diff=_Diff(added=["x"]),
              label=None, anomaly=False)
    assert build_commit_message(**kw) == build_commit_message(**kw)

    # A retrieve-phase mutation is flagged as an anomaly.
    msg = build_commit_message(phase="RETRIEVE", trigger="hook", op_seq=1,
                               diff=_Diff(updated=["x"]), label=None, anomaly=True)
    assert "ANOMALY=retrieve-mutation" in msg
    assert "Anomaly: retrieve-mutation" in msg


# --- stable, followable per-item paths -------------------------------------


def test_trace_item_path_stable_and_followable(tmp_path):
    memo = NoteListMemo()
    session = _session(tmp_path, memo)
    repo = tmp_path / "repo"
    rel = session.store.item_rel_path("note_list", "n1")
    assert "n1" in rel and "/" in rel and rel.endswith(".md")

    # Create, then update the SAME item twice -> same file path each time.
    _set_notes(memo, {"n1": "version one"})
    session.snapshot("BUILD", "hook")
    _set_notes(memo, {"n1": "version two"})
    session.snapshot("BUILD", "hook")
    _set_notes(memo, {"n1": "version three"})
    session.snapshot("BUILD", "hook")

    # The item's multi-commit history is followable on the single stable file.
    history = session.store.file_history(rel)
    assert len(history) == 3
    follow = _git(repo, "log", "--follow", "--format=%H", "--", rel).stdout.split()
    assert len(follow) == 3
    assert (repo / rel).read_text().endswith("version three\"\n}\n") or \
        "version three" in (repo / rel).read_text()

    # Deleting the item removes that same file at HEAD.
    _set_notes(memo, {})
    session.snapshot("BUILD", "hook")
    assert not (repo / rel).exists()
    assert _git(repo, "cat-file", "-e", f"HEAD:{rel}").returncode != 0
    # The removal shows up as a deletion in the diff of the last commit.
    diff = _git(repo, "show", "--name-status", "--format=", "HEAD").stdout
    assert diff.strip().startswith("D") and rel in diff
