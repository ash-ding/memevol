"""State-shape adapters — extract text-only, vector-free memory items.

The registry is keyed by the memo's *state shape* (resolved from its class
name), NOT by harness name, so a forge-evolved harness that reuses a known
shape keeps working and anything unrecognised falls back safely rather than
crashing. Every adapter touches ONLY the harness's own public read
accessors and NEVER ``memo.config`` (the structural-secrets boundary — there
is no post-hoc redaction pass to forget). All rendered text routes through
:func:`redact_value`, so no raw vector, embedding, or binary blob can ever
reach the git store.

Round-1 note: the heavy-store adapters (vector store / dict-with-vectors /
graph) are validated in round 1 against lightweight fixtures that mimic each
backend's *public API shape*. Wiring them against the live mem0/Qdrant,
memoryos, and zep/graphiti backends is the deferred integration round.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("main")

# ``faiss`` is a heavy, Singularity-only dependency — it is NOT importable in
# the host env. Guard the import so the module always loads; a real
# ``faiss.Index`` is recognised via ``isinstance`` when available, and by a
# strict duck-type otherwise (round-1 Phase-B fix: a ``faiss.Index`` object
# reached during a walk must redact to a placeholder, not fall to ``repr()``).
try:  # pragma: no cover - depends on the runtime env
    import faiss as _faiss  # type: ignore
    _FAISS_INDEX = getattr(_faiss, "Index", None)
except Exception:  # pragma: no cover - host env has no faiss
    _faiss = None
    _FAISS_INDEX = None


def _is_faiss_index(value: Any) -> bool:
    """True for a ``faiss.Index`` object (a pure vector store, no text).

    Uses ``isinstance`` when faiss is importable, else a strict duck-type
    (callable ``search`` + ``reconstruct`` and integer ``ntotal`` + ``d``) so a
    stub mimicking the index type is still recognised in the host env.
    """
    if _FAISS_INDEX is not None:
        try:
            if isinstance(value, _FAISS_INDEX):
                return True
        except Exception:
            pass
    mod = (getattr(type(value), "__module__", "") or "").split(".")[0]
    if mod == "faiss":
        return True
    return (
        callable(getattr(value, "search", None))
        and callable(getattr(value, "reconstruct", None))
        and isinstance(getattr(value, "ntotal", None), int)
        and isinstance(getattr(value, "d", None), int)
    )


# ---------------------------------------------------------------------------
# Redaction — the safety core. Nothing vector-shaped or binary ever escapes.
# ---------------------------------------------------------------------------

#: A list/tuple of numbers longer than this is treated as a raw vector and
#: redacted regardless of the key it lives under (defense-in-depth backstop
#: for renamed embedding fields).
VECTOR_LEN_THRESHOLD = 16

#: Keys whose *suffix* marks them as embeddings/vectors — dropped wholesale.
_VECTOR_KEY_SUFFIXES = ("_embedding", "_embeddings", "_vec", "_vector", "_vectors")
#: Keys whose exact name marks them as embeddings/vectors.
_VECTOR_KEY_EXACT = {"embedding", "embeddings", "vector", "vectors", "vec"}

#: Fields never dumped from any adapter (siblings that hold model handles or
#: raw vector stores rather than text).
_SKIP_FIELDS = {"retriever", "embeddings", "embedding_model_obj", "client",
                "vector_store", "_client", "model"}

#: Keys whose exact (lowercased) name marks them as holding secrets/config or a
#: backend connection — NEVER rendered into a trace. ``config`` is the critical
#: one: every ``MemoClass`` carries ``self.config`` (which can hold API keys /
#: base-URLs / credentials), so any ``__dict__``-walking path must drop it.
_SENSITIVE_KEY_EXACT = {"config", "api_key", "apikey", "token", "secret",
                        "password", "passwd", "credential", "credentials",
                        "auth", "database"}
#: Suffixes that mark a key as secret-bearing. Deliberately NOT the bare
#: substring ``key`` — that would drop legitimate fields such as ``keywords``.
_SENSITIVE_KEY_SUFFIXES = ("_key", "_token", "_secret", "_password", "_credential")


def is_vector_key(key: Any) -> bool:
    """True if a dict key names an embedding/vector field."""
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in _VECTOR_KEY_EXACT:
        return True
    return any(k.endswith(suf) for suf in _VECTOR_KEY_SUFFIXES)


def is_sensitive_key(key: Any) -> bool:
    """True if a dict key names a secret / credential / config / connection
    field that must never be rendered into a trace.

    Exact-name matches plus ``*_key`` / ``*_token`` / ``*_secret`` /
    ``*_password`` / ``*_credential`` suffixes only — never the bare substring
    ``key``, so legitimate fields like ``keywords`` are preserved.
    """
    if not isinstance(key, str):
        return False
    k = key.lower()
    if k in _SENSITIVE_KEY_EXACT:
        return True
    return any(k.endswith(suf) for suf in _SENSITIVE_KEY_SUFFIXES)


#: Prefixes that mark a string VALUE (not just a key) as a credential. A
#: defense-in-depth backstop for the ``state/`` walk so a secret stored under an
#: innocuous key (not caught by :func:`is_sensitive_key`) is still dropped.
_SECRET_VALUE_PREFIXES = ("sk-", "sk_", "ghp_", "gho_", "github_pat_", "xoxb-",
                          "xoxp-", "AKIA", "ASIA", "AIza", "Bearer ",
                          "-----BEGIN")


def looks_like_secret_value(value: Any) -> bool:
    """True if a *string value* looks like a credential/token/private key.

    Conservative prefix match only (no entropy heuristic) so ordinary text is
    never dropped. Complements :func:`is_sensitive_key` (key-based) for the
    ``state/`` completeness walk, which descends into config dicts.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    return any(v.startswith(p) for p in _SECRET_VALUE_PREFIXES)


def _is_numpy_array(value: Any) -> bool:
    mod = type(value).__module__
    return mod.split(".")[0] == "numpy" and type(value).__name__ == "ndarray"


def _is_torch_tensor(value: Any) -> bool:
    mod = type(value).__module__
    return mod.split(".")[0] == "torch" and type(value).__name__ == "Tensor"


def looks_like_vector(value: Any) -> bool:
    """True for numpy arrays, torch tensors, faiss indexes, or long lists/tuples
    of numbers — anything that is raw coefficients rather than text."""
    if _is_numpy_array(value) or _is_torch_tensor(value) or _is_faiss_index(value):
        return True
    if isinstance(value, (list, tuple)) and len(value) > VECTOR_LEN_THRESHOLD:
        # bool is a Number subclass; require genuine numeric coefficients.
        return all(isinstance(x, Number) and not isinstance(x, bool) for x in value)
    return False


def _vector_placeholder(value: Any) -> str:
    if _is_faiss_index(value):
        return f"<vector redacted: faiss ntotal={getattr(value, 'ntotal', '?')}>"
    if _is_numpy_array(value) or _is_torch_tensor(value):
        shape = getattr(value, "shape", None)
        return f"<vector redacted: shape={tuple(shape) if shape is not None else '?'}>"
    return f"<vector redacted: len={len(value)}>"


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Recursively strip vectors/embeddings, coerce exotic types to text-safe
    forms, and never crash. Returns a JSON-serialisable structure.

    - numpy ndarray / torch tensor / long numeric list -> ``"<vector ...>"``
    - dict -> drop vector-named keys; recurse values; redact vector values
    - list/tuple -> recurse each element
    - set / frozenset -> sorted (if all-str) else listed, recursed
    - str / int / float / bool / None -> unchanged
    - anything else -> truncated ``repr`` (never a raw vector)
    """
    if _depth > 12:
        return "<max-depth>"
    if looks_like_vector(value):
        return _vector_placeholder(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if is_vector_key(k) or is_sensitive_key(k):
                continue
            key = k if isinstance(k, str) else str(k)
            out[key] = redact_value(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_value(x, _depth + 1) for x in value]
    if isinstance(value, (set, frozenset)):
        items = list(value)
        if all(isinstance(x, str) for x in items):
            items = sorted(items)
        return [redact_value(x, _depth + 1) for x in items]
    # numpy / torch scalars carry an ``.item()``; keep them as plain numbers.
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, (int, float, bool, str)):
                return scalar
        except Exception:
            pass
    # Unknown object: a bounded, coefficient-free repr.
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "...<truncated>"


def render_text(structure: Any) -> str:
    """Deterministic, human-readable text for a redacted item body."""
    if isinstance(structure, str):
        return structure
    return json.dumps(redact_value(structure), sort_keys=True, ensure_ascii=False,
                      indent=2, default=str)


# ---------------------------------------------------------------------------
# Item model + adapter protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryItem:
    """One extracted memory item — text only, vectors already redacted."""

    item_id: str
    body: str
    embedding_backed: bool = False
    embedding_model: Optional[str] = None


class Adapter:
    """A state-shape adapter. ``kind`` names the store shape (used in the git
    path ``memory/<kind>/<item>.md``); ``extract`` reads the memo's own public
    surface and returns text-only items."""

    kind: str = "unknown"

    def extract(self, memo: Any) -> List[MemoryItem]:  # pragma: no cover
        raise NotImplementedError


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key read that never raises."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# note_list  (amem: memo._system.memories)
# ---------------------------------------------------------------------------

_NOTE_FIELDS = ("content", "context", "keywords", "tags", "timestamp", "id",
                "category", "links")


class NoteListAdapter(Adapter):
    kind = "note_list"

    def extract(self, memo: Any) -> List[MemoryItem]:
        system = _get(memo, "_system")
        memories = _get(system, "memories", {}) if system is not None else {}
        items: List[MemoryItem] = []
        if isinstance(memories, dict):
            entries = list(memories.items())
        else:  # list/iterable of notes
            entries = [(None, n) for n in list(memories or [])]
        for key, note in entries:
            note_id = _get(note, "id", key)
            fields: Dict[str, Any] = {}
            for f in _NOTE_FIELDS:
                if f in _SKIP_FIELDS:
                    continue
                v = _get(note, f)
                if v is not None:
                    fields[f] = v
            item_id = str(note_id if note_id is not None else key if key is not None
                          else len(items))
            items.append(MemoryItem(
                item_id=item_id,
                body=render_text(fields),
                embedding_backed=True,  # amem notes are backed by an embedding retriever
            ))
        return items


# ---------------------------------------------------------------------------
# plain_text_list  (hipporag2: memo._passages -> List[str])
# ---------------------------------------------------------------------------


class PlainTextListAdapter(Adapter):
    kind = "plain_text_list"

    def extract(self, memo: Any) -> List[MemoryItem]:
        passages = _get(memo, "_passages", []) or []
        items: List[MemoryItem] = []
        for i, passage in enumerate(list(passages)):
            items.append(MemoryItem(
                item_id=str(i),
                body=render_text(str(passage)),
                embedding_backed=False,
            ))
        return items


# ---------------------------------------------------------------------------
# vector_store_get_all  (mem0 / lightmem / simplemem)
# ---------------------------------------------------------------------------

_ID_KEYS = ("id", "memory_id", "hash", "uuid", "_id", "key")


def _row_id(row: Any, index: int) -> str:
    for k in _ID_KEYS:
        v = _get(row, k)
        if v is not None:
            return str(v)
    return str(index)


class VectorStoreAdapter(Adapter):
    kind = "vector_store"

    def extract(self, memo: Any) -> List[MemoryItem]:
        rows = self._read_rows(memo)
        items: List[MemoryItem] = []
        for i, row in enumerate(rows):
            items.append(MemoryItem(
                item_id=_row_id(row, i),
                body=render_text(row),  # redact_value drops any vector payload
                embedding_backed=True,
            ))
        return items

    def _read_rows(self, memo: Any) -> List[Any]:
        # Only ever call vector-suppressing signatures of the public accessor.
        attempts: List[Callable[[], Any]] = [
            lambda: memo.get_all(with_vectors=False),
            lambda: memo.get_all(with_vectors=False, with_payload=True),
            lambda: memo.get_all_entries(),
        ]
        for call in attempts:
            try:
                result = call()
            except (AttributeError, TypeError):
                continue
            except Exception as exc:
                log.warning("[tracing] vector-store accessor failed: %r", exc)
                continue
            return self._normalise(result)
        return []

    @staticmethod
    def _normalise(result: Any) -> List[Any]:
        if isinstance(result, dict):
            # mem0-style {"results": [...]} or a plain id->row mapping.
            if "results" in result and isinstance(result["results"], list):
                return list(result["results"])
            return list(result.values())
        if isinstance(result, list):
            return result
        return list(result or [])


# ---------------------------------------------------------------------------
# dict_with_vector_fields  (memoryos: STM / MTM / LPM dict structures)
# ---------------------------------------------------------------------------

_MEMORYOS_TIERS = ("short_term_memory", "mid_term_memory", "long_term_memory",
                   "_stm", "_mtm", "_lpm", "stm", "mtm", "lpm")


class DictWithVectorFieldsAdapter(Adapter):
    kind = "dict_with_vector_fields"

    def extract(self, memo: Any) -> List[MemoryItem]:
        raw = self._collect(memo)
        items: List[MemoryItem] = []
        for i, entry in enumerate(raw):
            had_vector = _dict_has_vector(entry)
            items.append(MemoryItem(
                item_id=_row_id(entry, i),
                body=render_text(entry),  # recursively drops *_embedding/*_vec/...
                embedding_backed=had_vector,
            ))
        return items

    def _collect(self, memo: Any) -> List[Any]:
        # Prefer a public snapshot accessor if the backend exposes one.
        for name in ("get_all", "dump_all", "get_all_entries"):
            fn = getattr(memo, name, None)
            if callable(fn):
                try:
                    return _as_item_list(fn())
                except (AttributeError, TypeError):
                    pass
                except Exception as exc:
                    log.warning("[tracing] memoryos accessor %s failed: %r", name, exc)
        items: List[Any] = []
        for tier in _MEMORYOS_TIERS:
            container = getattr(memo, tier, None)
            if container is not None:
                items.extend(_as_item_list(container))
        return items


def _as_item_list(container: Any) -> List[Any]:
    if container is None:
        return []
    if isinstance(container, dict):
        # a mapping of id->entry, or a single entry dict
        values = list(container.values())
        if values and all(isinstance(v, (dict, str)) for v in values):
            return values
        return [container]
    if isinstance(container, (list, tuple)):
        return list(container)
    return [container]


def _dict_has_vector(obj: Any, _depth: int = 0) -> bool:
    if _depth > 12:
        return False
    if looks_like_vector(obj):
        return True
    if isinstance(obj, dict):
        for k, v in obj.items():
            if is_vector_key(k):
                return True
            if _dict_has_vector(v, _depth + 1):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_dict_has_vector(x, _depth + 1) for x in obj)
    return False


# ---------------------------------------------------------------------------
# graph  (zep / graphiti: entity + episodic nodes and edges)
# ---------------------------------------------------------------------------


class GraphAdapter(Adapter):
    """Graph store (zep/graphiti).

    Reads nodes/edges/episodes via a text-only, embeddings-suppressed surface.
    In round 1 this consumes a duck-typed surface (``nodes`` / ``edges`` /
    ``episodes`` collections, or a ``graph_snapshot(with_embeddings=False)``
    accessor) that the fixtures mimic. The integration round wires the real
    ``EntityNode/EntityEdge/EpisodicNode.get_by_group_ids(..., with_embeddings
    =False)`` call (pinned at the call site) into this same surface.
    """

    kind = "graph"

    def extract(self, memo: Any) -> List[MemoryItem]:
        collections = self._collections(memo)
        items: List[MemoryItem] = []
        for prefix, elements in collections:
            for i, el in enumerate(elements):
                node_id = _get(el, "uuid") or _get(el, "id") or _get(el, "name") or i
                fields: Dict[str, Any] = {}
                for f in ("name", "summary", "fact", "content", "labels",
                          "source_node_uuid", "target_node_uuid", "created_at"):
                    v = _get(el, f)
                    if v is not None:
                        fields[f] = v
                items.append(MemoryItem(
                    item_id=f"{prefix}:{node_id}",
                    body=render_text(fields),
                    embedding_backed=True,  # graphiti nodes/edges carry embeddings
                ))
        return items

    def _collections(self, memo: Any):
        snap = getattr(memo, "graph_snapshot", None)
        if callable(snap):
            try:
                data = snap(with_embeddings=False)
                return [(k, _as_item_list(v)) for k, v in dict(data).items()]
            except (AttributeError, TypeError):
                pass
            except Exception as exc:
                log.warning("[tracing] graph_snapshot failed: %r", exc)
        out = []
        for name in ("nodes", "edges", "episodes"):
            coll = getattr(memo, name, None)
            if coll is not None:
                out.append((name, _as_item_list(coll)))
        return out


# ---------------------------------------------------------------------------
# fallback  (never crashes; redacts anything vector-shaped)
# ---------------------------------------------------------------------------


class FallbackAdapter(Adapter):
    kind = "unsupported"

    def extract(self, memo: Any) -> List[MemoryItem]:
        payload = {
            "type": type(memo).__name__,
            "extraction": "unsupported",
        }
        # Best-effort: capture any obviously-textual public state, fully
        # redacted, so an unlisted shape still yields *some* readable trace.
        state = getattr(memo, "__dict__", None)
        if isinstance(state, dict):
            safe: Dict[str, Any] = {}
            for k, v in state.items():
                if not isinstance(k, str) or k.startswith("_trace"):
                    continue
                if k in _SKIP_FIELDS or is_vector_key(k) or is_sensitive_key(k):
                    continue
                safe[k] = redact_value(v)
            if safe:
                payload["state"] = safe
        return [MemoryItem(
            item_id="state",
            body=render_text(payload),
            embedding_backed=False,
        )]


# ---------------------------------------------------------------------------
# Registry — class name -> shape adapter, else fallback
# ---------------------------------------------------------------------------

_NOTE_LIST = NoteListAdapter()
_PLAIN_TEXT = PlainTextListAdapter()
_VECTOR_STORE = VectorStoreAdapter()
_DICT_VECTOR = DictWithVectorFieldsAdapter()
_GRAPH = GraphAdapter()
_FALLBACK = FallbackAdapter()

#: Explicit ``memo class name -> adapter`` map. No ``if harness == "..."``
#: chains; anything unlisted resolves to the never-crash fallback.
CLASS_ADAPTER_MAP: Dict[str, Adapter] = {
    "AMemMemo": _NOTE_LIST,
    "HippoRAGMemo": _PLAIN_TEXT,
    "Mem0Memo": _VECTOR_STORE,
    "LightMemMemo": _VECTOR_STORE,
    "SimpleMemMemo": _VECTOR_STORE,
    "MemoryOSMemo": _DICT_VECTOR,
    "ZepMemo": _GRAPH,
}

_warned_fallback: set = set()


def register_adapter(class_name: str, adapter: Adapter) -> None:
    """Register a shape adapter for a memo class name. The extension point a
    forge-evolved harness (or the demo) uses to opt a new class into a known
    shape without editing the registry literal."""
    CLASS_ADAPTER_MAP[class_name] = adapter


def resolve_adapter(memo: Any, harness_name: Optional[str] = None) -> Adapter:
    """Resolve the shape adapter for a memo by class name, else fallback.

    Warns once per ``(harness, type)`` pair when falling back so an unlisted
    (e.g. forge-evolved) shape is visible without log spam.
    """
    cls = type(memo).__name__
    adapter = CLASS_ADAPTER_MAP.get(cls)
    if adapter is not None:
        return adapter
    warn_key = (harness_name or "?", cls)
    if warn_key not in _warned_fallback:
        _warned_fallback.add(warn_key)
        log.warning("[tracing] no shape adapter for %s (harness=%s) — using "
                    "never-crash fallback", cls, harness_name)
    return _FALLBACK
