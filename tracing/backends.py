"""Backend capability adapters — read text-only, vector-free memory items back
out of a *discovered* backend client.

Phase B removes the class-name registry: these adapters are no longer dispatched
by a memo's class name. Instead :func:`probe_capability` is called on every
object reached under ``self.memory`` during discovery (see ``tracing.discovery``)
and returns the first adapter whose ``matches`` recognises the client by its
*capability* (``isinstance`` on the real class when importable, else a strict
duck-type). Each adapter then reads the client back through its LOWEST-LEVEL
method with the vector-suppressing flag PINNED — never a harness convenience
wrapper (lightmem's ``get_all`` defaults ``with_vectors=True``; LanceDB
``.select()`` does not reliably drop the vector column, GitHub #953).

Repo-grounded gotchas honoured here (each verified against this repo's vendored
code / pinned ``uv.lock`` during the spike):
  * Qdrant  — low-level ``client.scroll(with_payload=True, with_vectors=False)``,
    paginated on ``next_offset`` until ``None``.
  * LanceDB — read the FULL row (``to_arrow().to_pylist()``) then drop the vector
    column BY KEY; id from an app column (``entry_id``), tolerate absence.
  * graphiti — detect the ``GraphDriver`` ABC (survives a Neo4j swap), read
    ``get_by_group_ids(..., with_embeddings=False)`` PINNED; id = ``uuid``.
  * Milvus / Chroma — DOC-LEVEL-ONLY, UNVERIFIED against any pinned version in
    this repo (no baseline depends on either); lower confidence.

Every read call is wrapped defensively; a failure is surfaced to the caller
(recorded in the snapshot's completeness signal), never swallowed silently.
FAISS is special (a pure vector store, no text) — see ``FAISSAdapter`` and the
recovery logic in ``tracing.resolver``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tracing.adapters import (
    MemoryItem,
    _is_faiss_index,
    is_vector_key,
    looks_like_vector,
    redact_value,
    render_text,
)

log = logging.getLogger("main")

#: Page size for paginated scans. Reasoned default, unmeasured against a live
#: backend (profiling deferred to Phase C).
SCROLL_PAGE = 256


class BackendReadError(Exception):
    """Raised inside an adapter when a client cannot be read (missing scope,
    remote failure, unexpected shape). The resolver records it in the
    completeness signal rather than letting it crash the eval."""


class BackendAdapter:
    """Capability adapter protocol. ``kind`` names the git subfolder
    (``memory/<kind>/<item>.md``); ``matches`` recognises a client by
    capability; ``extract`` reads it back text-only, vectors suppressed."""

    kind: str = "backend"
    #: Doc-level-only / unverified against any pinned version in this repo.
    unverified: bool = False

    def matches(self, client: Any) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:  # pragma: no cover
        raise NotImplementedError


def _row_body_and_id(row: Any, index: int, id_keys: tuple) -> tuple:
    """Redact a row to a text body and pick a stable id (drops vectors)."""
    body = render_text(row)  # redact_value drops *_embedding/*_vec/long-vectors
    if isinstance(row, dict):
        for k in id_keys:
            v = row.get(k)
            if v is not None:
                return body, str(v)
    else:
        for k in id_keys:
            v = getattr(row, k, None)
            if v is not None:
                return body, str(v)
    return body, str(index)


# ---------------------------------------------------------------------------
# Qdrant  (mem0 / lightmem)
# ---------------------------------------------------------------------------


class QdrantAdapter(BackendAdapter):
    kind = "qdrant"

    def matches(self, client: Any) -> bool:
        try:
            from qdrant_client import AsyncQdrantClient, QdrantClient  # type: ignore
            if isinstance(client, (QdrantClient, AsyncQdrantClient)):
                return True
        except Exception:
            pass
        # Duck-type: the low-level scroll + collection-listing surface.
        return callable(getattr(client, "scroll", None)) and callable(
            getattr(client, "get_collections", None))

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        collection = scope.get("collection")
        if not collection:
            raise BackendReadError("qdrant: no collection scope on the memo")
        items: List[MemoryItem] = []
        offset: Any = None
        # PIN with_vectors=False at the LOWEST level (never a get_all wrapper,
        # which lightmem defaults to with_vectors=True). Paginate on next_offset.
        while True:
            result = client.scroll(collection_name=collection, limit=SCROLL_PAGE,
                                   offset=offset, with_payload=True,
                                   with_vectors=False)
            points, next_offset = _unpack_scroll(result)
            for i, point in enumerate(points):
                payload = _get(point, "payload", {}) or {}
                pid = _get(point, "id", i)
                items.append(MemoryItem(
                    item_id=str(pid),
                    body=render_text(payload),  # vectors already suppressed + redacted
                    embedding_backed=True,
                ))
            if next_offset is None:
                break
            offset = next_offset
        return items


def _unpack_scroll(result: Any) -> tuple:
    """qdrant ``scroll`` returns ``(records, next_page_offset)``; tolerate a bare
    list too."""
    if isinstance(result, tuple) and len(result) == 2:
        return list(result[0] or []), result[1]
    if isinstance(result, list):
        return result, None
    return list(result or []), None


# ---------------------------------------------------------------------------
# LanceDB  (simplemem)
# ---------------------------------------------------------------------------


class LanceDBAdapter(BackendAdapter):
    kind = "lancedb"

    def matches(self, client: Any) -> bool:
        mod = (getattr(type(client), "__module__", "") or "")
        if mod.startswith("lancedb"):
            return True
        # Duck-type a LanceDB Table: to_arrow/to_pandas readback + search.
        has_read = callable(getattr(client, "to_arrow", None)) or callable(
            getattr(client, "to_pandas", None))
        return has_read and callable(getattr(client, "search", None))

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        rows = self._read_full_rows(client)
        items: List[MemoryItem] = []
        for i, row in enumerate(rows):
            # Drop the vector column BY KEY (do NOT rely on .select() — #953);
            # redact_value drops both *_embedding/*_vec keys and long vectors.
            body, item_id = _row_body_and_id(row, i, ("entry_id", "id", "_id"))
            items.append(MemoryItem(item_id=item_id, body=body,
                                    embedding_backed=True))
        return items

    @staticmethod
    def _read_full_rows(client: Any) -> List[Any]:
        to_arrow = getattr(client, "to_arrow", None)
        if callable(to_arrow):
            table = to_arrow()
            to_pylist = getattr(table, "to_pylist", None)
            if callable(to_pylist):
                return list(to_pylist())
            return list(table or [])
        to_pandas = getattr(client, "to_pandas", None)
        if callable(to_pandas):
            df = to_pandas()
            records = getattr(df, "to_dict", None)
            if callable(records):
                return list(df.to_dict(orient="records"))
        raise BackendReadError("lancedb: no readable full-row accessor")


# ---------------------------------------------------------------------------
# graphiti  (zep)
# ---------------------------------------------------------------------------

_GRAPH_TEXT_FIELDS = ("name", "summary", "fact", "content", "episode_body")


class GraphitiAdapter(BackendAdapter):
    kind = "graphiti"

    def matches(self, client: Any) -> bool:
        try:
            from graphiti_core.driver.driver import GraphDriver  # type: ignore
            if isinstance(client, GraphDriver):
                return True
        except Exception:
            pass
        # Duck-type the GraphDriver ABC (survives a Neo4j/Falkor swap): an async
        # query surface plus a group-scoped read.
        return callable(getattr(client, "execute_query", None)) and (
            callable(getattr(client, "get_by_group_ids", None))
            or getattr(client, "graph_operations_interface", None) is not None)

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        gid = scope.get("group_id") or scope.get("user_id")
        if not gid:
            raise BackendReadError("graphiti: no group_id scope on the memo")
        elements = self._read_group(client, gid)
        items: List[MemoryItem] = []
        for i, el in enumerate(elements):
            fields: Dict[str, Any] = {}
            for f in _GRAPH_TEXT_FIELDS + ("labels", "source_node_uuid",
                                           "target_node_uuid", "created_at"):
                v = _get(el, f)
                if v is not None:
                    fields[f] = v
            node_id = _get(el, "uuid") or _get(el, "id") or i
            items.append(MemoryItem(item_id=str(node_id),
                                    body=render_text(fields),
                                    embedding_backed=True))
        return items

    def _read_group(self, client: Any, gid: str) -> List[Any]:
        # Real path: EntityNode/EntityEdge/EpisodicNode.get_by_group_ids(
        #   driver, group_ids=[gid], with_embeddings=False)  <- PINNED.
        # Wiring the real classes is Phase C; here we prefer them if importable
        # and otherwise use a duck-typed driver-level shim (the fakes mimic it).
        node_classes = _graphiti_node_classes()
        out: List[Any] = []
        if node_classes:
            for cls in node_classes:
                fn = getattr(cls, "get_by_group_ids", None)
                if not callable(fn):
                    continue
                res = _maybe_await(fn(client, group_ids=[gid],
                                      with_embeddings=False))
                out.extend(list(res or []))
            return out
        shim = getattr(client, "get_by_group_ids", None)
        if callable(shim):
            res = _maybe_await(shim(group_ids=[gid], with_embeddings=False))
            return list(res or [])
        raise BackendReadError("graphiti: no readable group accessor")


def _graphiti_node_classes() -> List[Any]:
    try:  # pragma: no cover - graphiti absent in host env
        from graphiti_core.edges import EntityEdge  # type: ignore
        from graphiti_core.nodes import EntityNode, EpisodicNode  # type: ignore
        return [EntityNode, EntityEdge, EpisodicNode]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Milvus / Chroma  (DOC-LEVEL-ONLY, UNVERIFIED — no baseline depends on either)
# ---------------------------------------------------------------------------


class MilvusAdapter(BackendAdapter):
    kind = "milvus"
    unverified = True  # signatures NOT verified against any pinned version here

    def matches(self, client: Any) -> bool:
        mod = (getattr(type(client), "__module__", "") or "")
        if mod.startswith("pymilvus"):
            return True
        return callable(getattr(client, "query", None)) and (
            getattr(client, "schema", None) is not None
            or callable(getattr(client, "describe_collection", None)))

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        collection = scope.get("collection")
        if not collection:
            raise BackendReadError("milvus: no collection scope on the memo")
        # output_fields is inclusion-only: introspect the schema and EXCLUDE
        # vector fields (one extra call). UNVERIFIED signature.
        fields, pk = self._non_vector_fields(client, collection)
        rows = client.query(collection_name=collection,
                            filter=f"{pk} >= 0", output_fields=fields)
        items: List[MemoryItem] = []
        for i, row in enumerate(list(rows or [])):
            body, item_id = _row_body_and_id(row, i, (pk, "id", "pk"))
            items.append(MemoryItem(item_id=item_id, body=body,
                                    embedding_backed=True))
        return items

    @staticmethod
    def _non_vector_fields(client: Any, collection: str) -> tuple:
        schema = getattr(client, "schema", None)
        if schema is None and callable(getattr(client, "describe_collection", None)):
            schema = client.describe_collection(collection)
        field_defs = _get(schema, "fields", []) or []
        names: List[str] = []
        pk = "id"
        for fd in field_defs:
            name = _get(fd, "name")
            if name is None:
                continue
            if _get(fd, "is_primary"):
                pk = name
            if is_vector_key(name) or _is_vector_field(fd):
                continue
            names.append(name)
        return (names or ["*"]), pk


def _is_vector_field(field_def: Any) -> bool:
    dtype = str(_get(field_def, "dtype", "") or _get(field_def, "type", ""))
    return "VECTOR" in dtype.upper() or "FLOAT_VECTOR" in dtype.upper()


class ChromaAdapter(BackendAdapter):
    kind = "chroma"
    unverified = True

    def matches(self, client: Any) -> bool:
        mod = (getattr(type(client), "__module__", "") or "")
        if mod.startswith("chromadb"):
            return True
        return (callable(getattr(client, "get", None))
                and callable(getattr(client, "add", None))
                and callable(getattr(client, "count", None))
                and getattr(client, "name", None) is not None)

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        # Embeddings are excluded simply by NOT requesting them. UNVERIFIED.
        result = client.get(include=["documents", "metadatas"])
        ids = list(_get(result, "ids", []) or [])
        docs = list(_get(result, "documents", []) or [])
        metas = list(_get(result, "metadatas", []) or [])
        items: List[MemoryItem] = []
        for i, doc_id in enumerate(ids):
            body = render_text({
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else None,
            })
            items.append(MemoryItem(item_id=str(doc_id), body=body,
                                    embedding_backed=True))
        return items


# ---------------------------------------------------------------------------
# FAISS  (a pure vector store — recovery handled in tracing.resolver)
# ---------------------------------------------------------------------------


class FAISSAdapter(BackendAdapter):
    """A ``faiss.Index`` stores VECTORS ONLY — no text/payload concept. This
    adapter only *recognises* the index; text recovery (parallel docstore ->
    capture-at-write ledger -> unreadable-and-redacted) lives in
    ``tracing.resolver`` because it needs the index's discovered siblings and
    the per-memo write ledger."""

    kind = "faiss"

    def matches(self, client: Any) -> bool:
        return _is_faiss_index(client)

    def extract(self, client: Any, *, scope: Dict[str, Any]) -> List[MemoryItem]:
        # No text on the index itself; recovery is the resolver's job.
        return []


# ---------------------------------------------------------------------------
# Capability probe — the ONLY dispatch (no class-name registry)
# ---------------------------------------------------------------------------

#: Order matters only for disjoint duck-types; the concrete backends are
#: mutually exclusive in practice. FAISS first so an index is never mistaken.
_ADAPTERS: List[BackendAdapter] = [
    FAISSAdapter(),
    QdrantAdapter(),
    LanceDBAdapter(),
    GraphitiAdapter(),
    MilvusAdapter(),
    ChromaAdapter(),
]


def probe_capability(obj: Any) -> Optional[BackendAdapter]:
    """Return the first capability adapter that recognises ``obj`` as a backend
    client, else ``None``. Never raises — a matcher that blows up is skipped."""
    # Never treat plain in-heap containers / scalars / raw vectors as clients.
    if obj is None or isinstance(obj, (str, bytes, bytearray, bool, int, float,
                                       list, tuple, set, frozenset)):
        # ... except a faiss.Index, which is a client even though it is
        # vector-shaped; handled by FAISSAdapter below via the explicit check.
        if not _is_faiss_index(obj):
            return None
    if isinstance(obj, dict):
        return None
    for adapter in _ADAPTERS:
        try:
            if adapter.matches(obj):
                return adapter
        except Exception:  # a matcher must never break discovery
            continue
    return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _maybe_await(value: Any) -> Any:
    """Resolve a coroutine synchronously if a read accessor is async. Read-only
    by contract; never opens a mutating path."""
    import inspect

    if inspect.isawaitable(value):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Already inside a loop — run on a fresh loop in a worker thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(lambda: asyncio.run(value)).result()
        return asyncio.run(value)
    return value


# Re-export the vector guards so a caller importing from backends has them.
__all__ = [
    "SCROLL_PAGE",
    "BackendAdapter",
    "BackendReadError",
    "ChromaAdapter",
    "FAISSAdapter",
    "GraphitiAdapter",
    "LanceDBAdapter",
    "MilvusAdapter",
    "QdrantAdapter",
    "is_vector_key",
    "looks_like_vector",
    "probe_capability",
    "redact_value",
]
