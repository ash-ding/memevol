"""Capture-at-write — the SECONDARY text-recovery path for a bare ``faiss.Index``
that has no discoverable parallel docstore.

A ``faiss.Index`` stores float arrays only; there is no text/payload to read
back. When no docstore sibling is found (see ``tracing.resolver``), the last
resort is to have recorded the text at *write* time. This module is the
MECHANISM ONLY:

  * a per-memo :class:`WriteLedger` (append-only list of the texts embedded, in
    insertion order), keyed off the memo by identity via a ``WeakKeyDictionary``
    so a finished memo's ledger is garbage-collected automatically;
  * a public :func:`feed_embedding_text` that the Phase-C shared-embedder tap
    (``baselines/harness/model_config.py::install_embedder_factory``) will call
    once wired — DELIBERATELY NOT wired here; Phase B feeds the ledger directly
    in tests;
  * :func:`pair_faiss_with_ledger`, which pairs ``ledger[i] <-> FAISS row i`` by
    INSERTION ORDER, guarded by a strict ``ntotal``-vs-``len(ledger)`` MISMATCH
    CHECK: on any mismatch (a delete / rebuild / ``IndexIDMap`` broke alignment)
    it REFUSES to pair and returns the reason for the completeness signal rather
    than silently mis-pairing text to the wrong vector.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple
from weakref import WeakKeyDictionary

from tracing.adapters import MemoryItem, render_text

log = logging.getLogger("main")

# Per-memo ledgers, keyed by identity; entries vanish when the memo is collected.
_LEDGERS: "WeakKeyDictionary[Any, WriteLedger]" = WeakKeyDictionary()


class WriteLedger:
    """Append-only, insertion-ordered record of embedded texts for one memo."""

    def __init__(self) -> None:
        self.texts: List[str] = []

    def add(self, text: Any) -> None:
        self.texts.append("" if text is None else str(text))

    def __len__(self) -> int:
        return len(self.texts)


def feed_embedding_text(memo: Any, text: Any) -> None:
    """Public feed API — record one text as it is embedded, in insertion order.

    Never raises (a memo that cannot be weak-referenced simply gets no ledger).
    The actual shared-embedder tap that calls this in a live run is Phase C.
    """
    try:
        ledger = _LEDGERS.get(memo)
        if ledger is None:
            ledger = WriteLedger()
            _LEDGERS[memo] = ledger
        ledger.add(text)
    except TypeError:
        # Non-weak-referenceable memo: capture-at-write is unavailable for it.
        log.debug("[tracing] capture-at-write unavailable for %r", type(memo))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("[tracing] feed_embedding_text failed: %r", exc)


def get_ledger(memo: Any) -> Optional[WriteLedger]:
    """Return the memo's write ledger, or None if nothing was fed."""
    try:
        return _LEDGERS.get(memo)
    except TypeError:
        return None


def clear_ledger(memo: Any) -> None:
    """Drop a memo's ledger (used by tests for isolation)."""
    try:
        _LEDGERS.pop(memo, None)
    except TypeError:
        pass


def pair_faiss_with_ledger(index: Any, ledger: Optional[WriteLedger]
                           ) -> Tuple[Optional[List[MemoryItem]], Optional[str]]:
    """Pair FAISS rows to ledger texts by insertion order — WITH the mismatch
    guard. Returns ``(items, None)`` on a clean pairing, or ``(None, reason)``
    when it refuses (no ledger, or ``ntotal != len(ledger)``). Vectors are never
    read; only the recorded text is emitted."""
    if ledger is None:
        return None, "FAISS: no capture-at-write ledger available (vectors redacted)"
    ntotal = getattr(index, "ntotal", None)
    try:
        ntotal_int = int(ntotal)
    except (TypeError, ValueError):
        return None, "FAISS: index has no readable ntotal (cannot pair; vectors redacted)"
    if ntotal_int != len(ledger):
        return None, (f"FAISS: ntotal={ntotal_int} != ledger len={len(ledger)} — "
                      "refusing to pair (insertion-order alignment broken; "
                      "delete/rebuild/IndexIDMap suspected)")
    items = [MemoryItem(item_id=str(i), body=render_text(text),
                        embedding_backed=True)
             for i, text in enumerate(ledger.texts)]
    return items, None


__all__ = ["WriteLedger", "clear_ledger", "feed_embedding_text", "get_ledger",
           "pair_faiss_with_ledger"]
