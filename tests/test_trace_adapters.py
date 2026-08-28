"""Per-adapter unit tests (5 shape adapters + never-crash fallback) and the
exotic-payload redaction guard. All text-only; no vector ever escapes.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _trace_fakes import (
    GraphMemo,
    GraphNode,
    MemoryOSMemo,
    NoteListMemo,
    PassageMemo,
    UnknownMemo,
    VectorStoreMemo,
)

from tracing import adapters
from tracing.adapters import (
    DictWithVectorFieldsAdapter,
    FallbackAdapter,
    GraphAdapter,
    NoteListAdapter,
    PlainTextListAdapter,
    VectorStoreAdapter,
    looks_like_vector,
    redact_value,
    render_text,
    resolve_adapter,
)


def _bodies(items):
    return "\n".join(i.body for i in items)


# --- note_list -------------------------------------------------------------


def test_adapter_note_list_extracts_text_only():
    from _trace_fakes import Note
    memo = NoteListMemo()
    memo._system.memories = {
        "n0": Note("n0", "hello", ["a"]),
        "n1": Note("n1", "world", ["b"]),
    }
    items = NoteListAdapter().extract(memo)
    ids = sorted(i.item_id for i in items)
    assert ids == ["n0", "n1"]
    assert "hello" in _bodies(items) and "world" in _bodies(items)
    assert all(i.embedding_backed for i in items)  # amem notes are embedding-backed


# --- plain_text_list -------------------------------------------------------


def test_adapter_plain_text_list_one_item_per_passage():
    memo = PassageMemo(["fact one", "fact two", "fact three"])
    items = PlainTextListAdapter().extract(memo)
    assert [i.item_id for i in items] == ["0", "1", "2"]
    assert "fact two" in _bodies(items)
    assert not any(i.embedding_backed for i in items)  # plain passages, no vectors


# --- vector_store ----------------------------------------------------------


def test_adapter_vector_store_drops_vectors_keeps_text():
    memo = VectorStoreMemo([
        {"id": "m1", "memory": "likes tea", "embedding": [0.1] * 128},
        {"id": "m2", "memory": "lives in NYC", "embedding": [0.2] * 128},
    ])
    items = VectorStoreAdapter().extract(memo)
    assert sorted(i.item_id for i in items) == ["m1", "m2"]
    body = _bodies(items)
    assert "likes tea" in body and "lives in NYC" in body
    assert "0.1" not in body and "0.2" not in body  # no raw coefficients
    assert all(i.embedding_backed for i in items)  # recorded as embedding-backed


# --- dict_with_vector_fields -----------------------------------------------


def test_adapter_dict_with_vector_fields_recursively_drops_embeddings():
    memo = MemoryOSMemo([
        {"id": "s1", "text": "morning run", "_embedding": [0.3] * 64},
        {"id": "s2", "text": "coffee", "meta": {"topic_vec": [0.4] * 64}},
    ])
    items = DictWithVectorFieldsAdapter().extract(memo)
    assert sorted(i.item_id for i in items) == ["s1", "s2"]
    body = _bodies(items)
    assert "morning run" in body and "coffee" in body
    assert "0.3" not in body and "0.4" not in body
    assert all(i.embedding_backed for i in items)  # had *_embedding / *_vec keys


# --- graph -----------------------------------------------------------------


def test_adapter_graph_extracts_node_text_without_embeddings():
    memo = GraphMemo([
        GraphNode("u1", "Alice", "the protagonist"),
        GraphNode("u2", "Paris", "a city in France"),
    ])
    items = GraphAdapter().extract(memo)
    body = _bodies(items)
    assert "Alice" in body and "Paris" in body and "a city in France" in body
    assert "0.1" not in body  # name_embedding coefficients never leak
    assert all(i.embedding_backed for i in items)
    assert all(i.item_id.startswith("nodes:") for i in items)


# --- fallback --------------------------------------------------------------


def test_adapter_fallback_handles_unlisted_shape_without_crashing():
    items = FallbackAdapter().extract(UnknownMemo())
    assert len(items) == 1
    body = items[0].body
    assert "unsupported" in body
    assert "hello world" in body  # textual public state is still captured
    assert "0.5" not in body      # stray_vector must be redacted
    assert "<vector redacted" in body


def test_adapter_fallback_never_leaks_config_or_secret_keys():
    # An UNREGISTERED memo class carrying a secret-bearing ``config`` (as every
    # MemoClass does) plus benign textual state and a ``keywords`` field. The
    # fallback dumps ``__dict__``; it must keep the benign text and ``keywords``
    # but NEVER render ``config`` / ``api_key`` / the secret value.
    class SecretBearingMemo:
        def __init__(self):
            self.config = {"api_key": "sk-SECRETVALUE", "base_url": "http://x"}
            self.api_key = "sk-SECRETVALUE"
            self.auth_token = "sk-SECRETVALUE"
            self.notes = "benign public state"
            self.keywords = ["alpha", "beta"]

    memo = SecretBearingMemo()
    # An unregistered class resolves to the fallback (not a targeted adapter).
    assert isinstance(resolve_adapter(memo, "mystery"), FallbackAdapter)

    items = FallbackAdapter().extract(memo)
    assert len(items) == 1
    body = items[0].body

    # Benign textual state and the keywords field survive.
    assert "benign public state" in body
    assert "keywords" in body
    assert "alpha" in body and "beta" in body

    # The secret value and the sensitive key names never appear.
    assert "sk-SECRETVALUE" not in body
    assert "api_key" not in body
    assert "auth_token" not in body
    assert "config" not in body

    # redact_value's dict branch is belt-and-suspenders: even a nested dict
    # carrying a sensitive key drops it while keeping siblings and ``keywords``.
    rendered = render_text(redact_value(
        {"api_key": "sk-SECRETVALUE", "keywords": ["k1"], "note": "keep me"}))
    assert "sk-SECRETVALUE" not in rendered
    assert "api_key" not in rendered
    assert "keywords" in rendered and "k1" in rendered
    assert "keep me" in rendered


def test_resolve_adapter_falls_back_for_unknown_class():
    # An unregistered class name resolves to the never-crash fallback, proving
    # the registry keys on class name rather than sniffing shape.
    assert isinstance(resolve_adapter(UnknownMemo(), "mystery"), FallbackAdapter)
    # A registered baseline class name resolves to its shape adapter.
    assert isinstance(adapters.CLASS_ADAPTER_MAP["AMemMemo"], NoteListAdapter)


def test_register_adapter_opts_a_class_into_a_known_shape():
    class TempShapeMemo(UnknownMemo):
        pass

    assert isinstance(resolve_adapter(TempShapeMemo(), "x"), FallbackAdapter)
    adapters.register_adapter("TempShapeMemo", NoteListAdapter())
    try:
        assert isinstance(resolve_adapter(TempShapeMemo(), "x"), NoteListAdapter)
    finally:
        adapters.CLASS_ADAPTER_MAP.pop("TempShapeMemo", None)


# --- exotic payload never leaks -------------------------------------------


def test_trace_exotic_payload_never_leaks():
    payloads = [
        [0.123456] * 32,                       # long list[float] -> vector
        (1.5,) * 20,                           # long tuple of floats -> vector
        {"embedding": [9.9] * 8, "text": "keep me"},  # vector-named key dropped
        {"nested_vector": [1.0] * 40},         # renamed vector, len-backstop drops
        {frozenset({"a", "b"}): "x", (1, 2): "y"},    # exotic keys
        collections.Counter({"apple": 2, "pear": 1}),
        {1, 2, 3},
        object(),                              # bare unknown object
    ]
    for p in payloads:
        rendered = render_text(redact_value(p))
        # No raw coefficients from any vector-shaped input survive.
        for coeff in ("0.123456", "9.9", "1.5"):
            assert coeff not in rendered

    # numpy / torch arrays, if available, are redacted structurally (never repr'd).
    try:
        import numpy as np
        arr = np.arange(100, dtype="float32")
        assert looks_like_vector(arr)
        # A non-vector key name so the value hits the redaction path (not the
        # key-drop path); the coefficients must be replaced by a placeholder.
        out = render_text(redact_value({"payload": arr, "label": "keep"}))
        assert "<vector redacted" in out
        assert "keep" in out
        assert "99.0" not in out
    except ImportError:  # pragma: no cover - numpy is present in the host env
        pass

    # A dict carrying both text and an embedding: text kept, vector dropped.
    out = render_text(redact_value({"content": "hello", "embedding": [0.1] * 50}))
    assert "hello" in out
    assert "0.1" not in out
