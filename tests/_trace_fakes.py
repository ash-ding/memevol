"""Shared, dependency-free fakes for the ``tracing/`` test-suite.

Not a ``test_*`` module, so pytest does not collect it. Each fake mimics one
harness's *public state shape* (the axis the adapter registry keys on) plus, for
the behavioural fake, the three MemoClass hooks — all in-process, no network, no
heavy deps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- note_list (amem) -------------------------------------------------------


class Note:
    def __init__(self, note_id: str, content: str,
                 keywords: Optional[List[str]] = None):
        self.id = note_id
        self.content = content
        self.keywords = keywords or []


class _System:
    def __init__(self) -> None:
        self.memories: Dict[str, Note] = {}


class NoteListMemo:
    """amem-style note_list shape + the three MemoClass hooks.

    ``build`` ingests ``recorder.init['facts']`` as notes; ``retrieve`` is
    strictly read-only. State is also directly mutable via ``self._system``.
    """

    def __init__(self) -> None:
        self._system = _System()

    async def build_memory_from_data(self, recorder: Any) -> None:
        facts = (getattr(recorder, "init", {}) or {}).get("facts", [])
        for i, text in enumerate(facts):
            nid = f"n{i}"
            self._system.memories[nid] = Note(nid, text, keywords=["k"])

    async def retrieve_memory_for_query(self, recorder: Any) -> Dict:
        return {"hits": [n.content for n in self._system.memories.values()]}

    async def use_memory_to_answer(self, recorder: Any, retrieved: Dict,
                                   prompt: str) -> Optional[str]:
        return f"answer:{len(retrieved.get('hits', []))}"


# --- plain_text_list (hipporag2) -------------------------------------------


class PassageMemo:
    def __init__(self, passages: Optional[List[str]] = None):
        self._passages: List[str] = list(passages or [])


# --- vector_store (mem0 / lightmem / simplemem) ----------------------------


class VectorStoreMemo:
    """Exposes a vector-suppressing ``get_all`` public accessor."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def get_all(self, with_vectors: bool = True, with_payload: bool = False):
        out = []
        for row in self._rows:
            r = dict(row)
            if not with_vectors:
                r.pop("embedding", None)
            out.append(r)
        return {"results": out}


# --- dict_with_vector_fields (memoryos) ------------------------------------


class MemoryOSMemo:
    def __init__(self, stm: List[Dict[str, Any]]):
        self.short_term_memory = stm


# --- graph (zep / graphiti) ------------------------------------------------


class GraphNode:
    def __init__(self, uuid: str, name: str, summary: str = "",
                 embedding: Optional[List[float]] = None):
        self.uuid = uuid
        self.name = name
        self.summary = summary
        self.name_embedding = embedding or [0.1] * 32


class GraphMemo:
    def __init__(self, nodes: List[GraphNode], edges: Optional[List[Any]] = None):
        self.nodes = nodes
        self.edges = edges or []


# --- unknown shape (fallback) ----------------------------------------------


class UnknownMemo:
    def __init__(self) -> None:
        self.some_text = "hello world"
        self.count = 3
        # A vector value under a NON-vector key name: exercises the length
        # backstop (redacted to a placeholder rather than key-dropped).
        self.stray_numbers = [0.5] * 64


# --- recorder stand-in ------------------------------------------------------


class Recorder:
    """Minimal recorder: only ``.init`` is read by the fakes."""

    def __init__(self, init: Optional[Dict[str, Any]] = None):
        self.init: Dict[str, Any] = init or {}


# Opt the behavioural note_list fake into the note_list shape adapter via the
# public extension point, so ``wrap_memo`` resolves a real adapter for it.
def _register_fake_shapes() -> None:
    from tracing.adapters import (
        DictWithVectorFieldsAdapter,
        GraphAdapter,
        NoteListAdapter,
        PlainTextListAdapter,
        VectorStoreAdapter,
        register_adapter,
    )
    register_adapter("NoteListMemo", NoteListAdapter())
    register_adapter("PassageMemo", PlainTextListAdapter())
    register_adapter("VectorStoreMemo", VectorStoreAdapter())
    register_adapter("MemoryOSMemo", DictWithVectorFieldsAdapter())
    register_adapter("GraphMemo", GraphAdapter())


_register_fake_shapes()
