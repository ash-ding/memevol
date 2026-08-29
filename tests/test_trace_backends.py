"""Capability backend adapters (Phase B): text-only, vector-free read-back from
clients discovered under ``self.memory``.

Standalone — against convention-following FAKE clients that mimic each backend's
PUBLIC API SHAPE only (no live DB, no heavy deps, no real ``baselines/``).
Real-DB validation is heavier infra deferred to Phase C.
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
    FakeChromaCollection,
    FakeGraphDriver,
    FakeLanceTable,
    FakeMilvusClient,
    FakeQdrantClient,
    GNode,
    QPoint,
    SomeForgeInventedMemo,
)

from tracing.resolver import resolve_and_snapshot


def _by_kind(snap, kind):
    return [ri for ri in snap.memory if ri.kind == kind]


def _all_memory_text(snap):
    return "\n".join(ri.item.body for ri in snap.memory)


# --- (ii) Qdrant: low-level scroll, vectors dropped, pagination, no wrapper ---


def test_qdrant_readback_drops_vectors_and_paginates():
    points = [QPoint(f"p{i}", {"text": f"fact {i}", "embedding": [0.9] * 32})
              for i in range(5)]  # > page size (2) -> forces pagination
    client = FakeQdrantClient(points, page=2)
    memo = SomeForgeInventedMemo({"vectors": client}, _collection_name="mem")

    snap = resolve_and_snapshot(memo)
    qitems = _by_kind(snap, "qdrant")

    # Pagination via next_offset returned ALL points, not just the first page.
    assert {ri.item.item_id for ri in qitems} == {f"p{i}" for i in range(5)}
    # Text is present, vectors are gone, and the wrong-default wrapper was NEVER
    # called (the adapter pins the low-level scroll with_vectors=False path).
    text = _all_memory_text(snap)
    assert "fact 0" in text and "fact 4" in text
    assert "0.9" not in text
    assert client.get_all_called is False
    assert all(ri.item.embedding_backed for ri in qitems)


def test_qdrant_no_collection_scope_is_flagged_unreadable():
    client = FakeQdrantClient([QPoint("p0", {"text": "x"})])
    memo = SomeForgeInventedMemo({"vectors": client})  # no collection attr

    snap = resolve_and_snapshot(memo)
    assert not _by_kind(snap, "qdrant")
    assert any("qdrant" in u for u in snap.completeness.unreadable)


# --- (iii) LanceDB: full row read, vector dropped by key, id fallback ---------


def test_lancedb_drops_vector_by_key_with_entry_id():
    tbl = FakeLanceTable([{"entry_id": "e0", "content": "lance text",
                           "vector": [0.7] * 32}])
    memo = SomeForgeInventedMemo({"db": tbl})

    snap = resolve_and_snapshot(memo)
    items = _by_kind(snap, "lancedb")
    assert len(items) == 1
    assert items[0].item.item_id == "e0"
    body = items[0].item.body
    assert "lance text" in body
    assert "0.7" not in body and "vector" not in body


def test_lancedb_missing_id_column_falls_back_to_row_index():
    tbl = FakeLanceTable([{"content": "row zero", "vector": [0.1] * 32},
                          {"content": "row one", "vector": [0.2] * 32}])
    memo = SomeForgeInventedMemo({"db": tbl})

    snap = resolve_and_snapshot(memo)
    items = _by_kind(snap, "lancedb")
    assert {ri.item.item_id for ri in items} == {"0", "1"}
    assert "0.1" not in _all_memory_text(snap)


# --- (iv) graphiti: GraphDriver duck-type, with_embeddings=False, uuid id -----


def test_graphiti_pins_with_embeddings_false_and_uses_uuid():
    driver = FakeGraphDriver([GNode("uuid-1", "Alice", summary="likes tea"),
                              GNode("uuid-2", "Bob", fact="knows Alice")])
    memo = SomeForgeInventedMemo({"graph": driver}, _gid="grp-1")

    snap = resolve_and_snapshot(memo)
    items = _by_kind(snap, "graphiti")

    assert driver.with_embeddings_seen is False  # PINNED off
    assert {ri.item.item_id for ri in items} == {"uuid-1", "uuid-2"}
    text = _all_memory_text(snap)
    assert "Alice" in text and "likes tea" in text and "knows Alice" in text
    assert "0.5" not in text  # name_embedding never read


def test_graphiti_no_group_scope_is_flagged_unreadable():
    driver = FakeGraphDriver([GNode("uuid-1", "Alice")])
    memo = SomeForgeInventedMemo({"graph": driver})  # no group id

    snap = resolve_and_snapshot(memo)
    assert not _by_kind(snap, "graphiti")
    assert any("graphiti" in u for u in snap.completeness.unreadable)


# --- Milvus / Chroma (doc-level-only, UNVERIFIED) ----------------------------


def test_milvus_excludes_vector_fields_from_output():
    client = FakeMilvusClient([{"pk": 1, "text": "mtext", "embedding": [0.1] * 32}])
    memo = SomeForgeInventedMemo({"v": client}, _collection_name="c")

    snap = resolve_and_snapshot(memo)
    items = _by_kind(snap, "milvus")
    assert client.output_fields_seen == ["pk", "text"]  # embedding EXCLUDED
    assert items[0].item.item_id == "1"
    assert "mtext" in items[0].item.body and "0.1" not in items[0].item.body
    # The unverified backend is flagged as such in the completeness signal.
    assert any("unverified" in c for c in snap.completeness.clients)


def test_chroma_requests_documents_and_metadatas_only():
    client = FakeChromaCollection(["c0"], ["chroma doc"], [{"k": "v"}])
    memo = SomeForgeInventedMemo({"c": client})

    snap = resolve_and_snapshot(memo)
    items = _by_kind(snap, "chroma")
    assert client.include_seen == ["documents", "metadatas"]  # embeddings excluded
    assert items[0].item.item_id == "c0"
    assert "chroma doc" in items[0].item.body


# --- (v) MULTI-COMPONENT self.memory: in-heap + client merged ----------------


def test_multi_component_memory_merges_inheap_and_client():
    client = FakeQdrantClient([QPoint("p0", {"text": "vector-backed fact"})],
                              page=8)
    memo = SomeForgeInventedMemo(
        {"notes": [{"id": "n0", "content": "in-heap fact"}], "vectors": client},
        _collection_name="mem")

    snap = resolve_and_snapshot(memo)
    kinds = {ri.kind for ri in snap.memory}
    assert "inheap" in kinds and "qdrant" in kinds  # merged into one snapshot
    text = _all_memory_text(snap)
    assert "in-heap fact" in text and "vector-backed fact" in text
