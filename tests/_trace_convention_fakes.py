"""Convention-following FAKE memo/backend classes for the Phase-B tracer tests.

NOT a ``test_*`` module (pytest does not collect it). Every fake mimics a
backend client's PUBLIC API SHAPE only — no network, no heavy deps, and it
NEVER imports the real ``baselines/``. Real-DB / real-baseline validation is
heavier infra deferred to Phase C.

The uniform convention: a memo's whole memory lives under ``self.memory`` (a
container that may hold in-heap data AND/OR backend clients AND/OR a mix).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Qdrant-like (mem0 / lightmem)
# ---------------------------------------------------------------------------


class QPoint:
    def __init__(self, pid: str, payload: Dict[str, Any],
                 vector: Optional[List[float]] = None):
        self.id = pid
        self.payload = payload
        self.vector = vector or [0.111] * 32


class FakeQdrantClient:
    """Low-level ``scroll`` + ``get_collections``. ``scroll`` PAGINATES via
    ``next_offset`` and honours ``with_vectors``. ``get_all`` is a convenience
    wrapper that DEFAULTS ``with_vectors=True`` (the wrong default) and records
    if it was ever called — the adapter must never touch it."""

    def __init__(self, points: List[QPoint], page: int = 2):
        self._points = points
        self._page = page
        self.get_all_called = False

    def get_collections(self):
        return ["mem"]

    def scroll(self, collection_name, limit=256, offset=None, with_payload=True,
               with_vectors=False):
        start = int(offset or 0)
        chunk = self._points[start:start + self._page]
        out = []
        for p in chunk:
            payload = dict(p.payload)
            if with_vectors:  # the adapter PINS this False
                payload["embedding"] = p.vector
            out.append(QPoint(p.id, payload, p.vector if with_vectors else None))
        nxt = start + self._page
        next_offset = nxt if nxt < len(self._points) else None
        return out, next_offset

    def get_all(self, with_vectors=True):  # WRONG default; must NOT be called
        self.get_all_called = True
        return [{"id": p.id, **p.payload, "embedding": p.vector}
                for p in self._points]


# ---------------------------------------------------------------------------
# LanceDB-like (simplemem)
# ---------------------------------------------------------------------------


class _Arrow:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def to_pylist(self):
        return [dict(r) for r in self._rows]  # FULL rows, incl. the vector column


class FakeLanceTable:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows

    def to_arrow(self):
        return _Arrow(self._rows)

    def search(self, *a, **k):  # part of the duck-type
        return self


# ---------------------------------------------------------------------------
# graphiti-like (zep) — GraphDriver duck-type + group-scoped read
# ---------------------------------------------------------------------------


class GNode:
    def __init__(self, uuid: str, name: str, summary: str = "", fact: str = ""):
        self.uuid = uuid
        self.name = name
        self.summary = summary
        self.fact = fact
        self.name_embedding = [0.5] * 32  # must never be read (with_embeddings=False)


class FakeGraphDriver:
    """Duck-typed ``GraphDriver``: async ``execute_query`` + a group-scoped
    read. Records the ``with_embeddings`` flag it was called with."""

    def __init__(self, nodes: List[GNode]):
        self._nodes = nodes
        self.with_embeddings_seen: Optional[bool] = None
        self.graph_operations_interface = object()

    async def execute_query(self, *a, **k):  # pragma: no cover - shape only
        return []

    def get_by_group_ids(self, group_ids, with_embeddings=True):
        self.with_embeddings_seen = with_embeddings
        return list(self._nodes)


# ---------------------------------------------------------------------------
# Milvus-like / Chroma-like (DOC-LEVEL-ONLY, UNVERIFIED)
# ---------------------------------------------------------------------------


class _Field:
    def __init__(self, name, dtype="VARCHAR", is_primary=False):
        self.name = name
        self.dtype = dtype
        self.is_primary = is_primary


class _Schema:
    def __init__(self, fields):
        self.fields = fields


class FakeMilvusClient:
    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.schema = _Schema([
            _Field("pk", "INT64", is_primary=True),
            _Field("text", "VARCHAR"),
            _Field("embedding", "FLOAT_VECTOR"),
        ])
        self.output_fields_seen: Optional[List[str]] = None

    def query(self, collection_name, filter, output_fields):
        self.output_fields_seen = list(output_fields)
        out = []
        for r in self._rows:
            out.append({k: r[k] for k in output_fields if k in r})
        return out


class FakeChromaCollection:
    def __init__(self, ids, docs, metas):
        self._ids, self._docs, self._metas = ids, docs, metas
        self.name = "coll"
        self.include_seen: Optional[List[str]] = None

    def add(self, *a, **k):  # part of the duck-type
        pass

    def count(self):
        return len(self._ids)

    def get(self, include=None):
        self.include_seen = list(include or [])
        return {"ids": self._ids, "documents": self._docs, "metadatas": self._metas}


# ---------------------------------------------------------------------------
# FAISS-like stubs (a pure vector store) + a docstore
# ---------------------------------------------------------------------------


class FakeFaissIndex:
    """Duck-types ``faiss.Index`` (recognised by ``_is_faiss_index``): integer
    ``ntotal``/``d`` and callable ``search``/``reconstruct``. Stores nothing
    readable as text."""

    def __init__(self, ntotal: int, d: int = 32):
        self.ntotal = ntotal
        self.d = d

    def search(self, *a, **k):  # pragma: no cover - shape only
        return [], []

    def reconstruct(self, *a, **k):  # pragma: no cover - shape only
        return [0.0] * self.d


class _Document:
    def __init__(self, page_content: str):
        self.page_content = page_content


class FakeDocstore:
    def __init__(self, mapping: Dict[str, _Document]):
        self._dict = mapping

    def search(self, docid):
        return self._dict.get(docid)


# ---------------------------------------------------------------------------
# Memo fakes (unfamiliar class names — resolution must be by convention)
# ---------------------------------------------------------------------------


class InHeapMemo:
    """Pure in-heap memory under ``self.memory`` (no backend client)."""

    def __init__(self, memory: Any):
        self.memory = memory


class SomeForgeInventedMemo:
    """A brand-new class name a forge run might invent; conforms to the
    convention, so resolution must succeed WITHOUT any name lookup."""

    def __init__(self, memory: Any, **attrs: Any):
        self.memory = memory
        for k, v in attrs.items():
            setattr(self, k, v)


class NoMemoryMemo:
    """Non-conforming: no ``self.memory`` -> whole-memo fallback (L3)."""

    def __init__(self):
        self.some_text = "hello world"
        self.count = 3


class DumpMemo:
    """Opt-in override: ``dump_memory_text`` wins outright (L1)."""

    def __init__(self, items):
        self.memory = {"ignored": ["should not appear"]}
        self._items = items

    def dump_memory_text(self):
        return list(self._items)
