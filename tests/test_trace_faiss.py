"""FAISS handling (Phase B): a ``faiss.Index`` stores vectors only.

Recovery order: (a) discoverable parallel docstore -> (b) capture-at-write
ledger paired by insertion order (WITH the ntotal-vs-len mismatch guard) ->
(c) unreadable + vectors redacted. Plus the round-1 fix: a ``faiss.Index``
OBJECT reached during a walk renders ``<vector redacted>``, not ``repr()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_convention_fakes import (
    FakeDocstore,
    FakeFaissIndex,
    SomeForgeInventedMemo,
    _Document,
)

from tracing.adapters import _is_faiss_index, looks_like_vector, redact_value, render_text
from tracing.capture import clear_ledger, feed_embedding_text
from tracing.resolver import resolve_and_snapshot


def _faiss_items(snap):
    return [ri for ri in snap.memory if ri.kind == "faiss"]


# --- (a) docstore recovery: id is the DOCSTORE id, not the FAISS position -----


def test_faiss_recovers_text_from_parallel_docstore():
    index = FakeFaissIndex(ntotal=2)
    docstore = FakeDocstore({"d0": _Document("first doc"),
                             "d1": _Document("second doc")})
    memo = SomeForgeInventedMemo({
        "index": index,
        "index_to_docstore_id": {0: "d0", 1: "d1"},
        "docstore": docstore,
    })

    snap = resolve_and_snapshot(memo)
    items = _faiss_items(snap)
    assert {ri.item.item_id for ri in items} == {"d0", "d1"}  # docstore ids
    text = "\n".join(ri.item.body for ri in items)
    assert "first doc" in text and "second doc" in text
    assert any("docstore" in c for c in snap.completeness.clients)


# --- (b) capture-at-write ledger pairing by insertion order ------------------


def test_faiss_recovers_text_from_write_ledger():
    memo = SomeForgeInventedMemo({"index": FakeFaissIndex(ntotal=3)})
    clear_ledger(memo)
    for t in ["alpha", "beta", "gamma"]:
        feed_embedding_text(memo, t)   # fed directly (the tap is Phase C)

    snap = resolve_and_snapshot(memo)
    items = _faiss_items(snap)
    assert [ri.item.item_id for ri in items] == ["0", "1", "2"]
    text = "\n".join(ri.item.body for ri in items)
    assert "alpha" in text and "gamma" in text
    assert any("ledger" in c for c in snap.completeness.clients)


# --- (b') MISMATCH GUARD: refuse to pair, flag the gap -----------------------


def test_faiss_mismatch_guard_refuses_to_pair():
    memo = SomeForgeInventedMemo({"index": FakeFaissIndex(ntotal=3)})
    clear_ledger(memo)
    feed_embedding_text(memo, "only one")   # len(ledger)=1 != ntotal=3

    snap = resolve_and_snapshot(memo)
    # It refuses to mis-pair: no recovered text, mismatch recorded, vectors gone.
    assert any("refusing to pair" in m for m in snap.completeness.mismatches)
    text = "\n".join(ri.item.body for ri in _faiss_items(snap))
    assert "only one" not in text
    assert "<vector redacted" in text


# --- (c) unreadable: no docstore, no ledger -> flagged + redacted ------------


def test_faiss_unreadable_without_docstore_or_ledger():
    memo = SomeForgeInventedMemo({"index": FakeFaissIndex(ntotal=2)})
    clear_ledger(memo)

    snap = resolve_and_snapshot(memo)
    assert any("faiss" in u for u in snap.completeness.unreadable)
    assert snap.completeness.has_gaps
    text = "\n".join(ri.item.body for ri in _faiss_items(snap))
    assert "<vector redacted" in text


# --- round-1 fix: faiss.Index OBJECT redaction -------------------------------


def test_faiss_index_object_redacts_to_placeholder():
    index = FakeFaissIndex(ntotal=7)
    assert _is_faiss_index(index) is True
    assert looks_like_vector(index) is True
    # Reached during a walk, it renders a placeholder, NOT a SWIG-pointer repr.
    rendered = render_text({"index": index})
    assert "<vector redacted: faiss ntotal=7>" in rendered
    assert "FakeFaissIndex object at" not in rendered
    # And the value branch of redact_value coerces it too.
    assert redact_value(index) == "<vector redacted: faiss ntotal=7>"
