"""Bounded, never-crash object-graph discovery for the convention-based tracer.

Two walks, both ``__dict__``-only, cycle-guarded, and bounded:

* :func:`discover` — ROOTED AT ``memo.memory`` (the USER-DECISION convention).
  Returns (a) the in-heap structural content as a list of ``MemoryItem`` (one
  per discovered record) and (b) a list of MULTIPLE backend clients found under
  the root (``self.memory`` may hold several components at once). Capability
  probing is inline: a matched client is recorded and NOT descended into, but
  the walk continues over its siblings so every component is found.

* :func:`discover_state` — the ALWAYS-ON companion walk over the WHOLE memo
  EXCLUDING the ``self.memory`` subtree (its ids seed the cycle guard). It
  captures the rest of the memo structurally (config redacted, hyperparameters,
  counters, buffers, misc state), records any backend client only as a REFERENCE
  MARKER (never dumped), and skips machinery.

Hard rules: read ``__dict__`` directly (``vars``) — NEVER trigger ``@property``
getters (tracing must not open a socket as a side effect); every attribute read
is wrapped so a raising property can never crash the walk; ``gc.get_referents``
is the fallback for ``__dict__``-less C-ext/SWIG objects.
"""

from __future__ import annotations

import gc
import logging
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from tracing.adapters import (
    MemoryItem,
    _dict_has_vector,
    is_sensitive_key,
    is_vector_key,
    looks_like_secret_value,
    looks_like_vector,
    redact_value,
    render_text,
)
from tracing.backends import BackendAdapter, probe_capability

log = logging.getLogger("main")

# Reasoned, UNMEASURED defaults — no real memo was profiled in the spike;
# profiling to replace these with measured constants is deferred to Phase C.
# Exceeding a bound degrades to "some content/clients not found" (recorded in the
# completeness signal), NEVER a hang.
MAX_DEPTH = 6
MAX_VISITED = 2000

_PRIMITIVES = (str, bytes, bytearray, int, float, bool)
_ID_KEYS = ("id", "uuid", "_id", "memory_id", "hash", "entry_id", "key")

#: Config-like container keys that are normally dropped WHOLESALE by
#: ``is_sensitive_key`` (``config``) but which the state walk DESCENDS into so
#: hyperparameters survive while credential sub-keys/values are still dropped.
_STATE_DESCEND_KEYS = {"config"}

_SKIP_LIKE_NAMES = {
    "lock", "rlock", "condition", "event", "semaphore", "thread", "connection",
    "socket", "session", "engine", "pool", "executor", "loop", "logger",
}


@dataclass
class ClientRef:
    """A backend client found under ``self.memory``."""

    path: str
    obj: Any
    cap: BackendAdapter
    parent: Any


@dataclass
class DiscoveryResult:
    items: List[MemoryItem] = field(default_factory=list)   # in-heap records
    clients: List[ClientRef] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)          # skips / bounds hit
    truncated: bool = False


# ---------------------------------------------------------------------------
# Shared predicates
# ---------------------------------------------------------------------------


def _obj_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """The instance ``__dict__`` (never triggers a ``@property``), or None."""
    try:
        raw = object.__getattribute__(obj, "__dict__")
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _is_client(obj: Any) -> Optional[BackendAdapter]:
    try:
        return probe_capability(obj)
    except Exception:
        return None


def _should_skip(obj: Any) -> bool:
    """Machinery pre-filter (NOT the client/str/record cases). Skips bare
    non-str scalars, raw vectors, modules/classes, torch weights, callables, and
    obvious locks/connections."""
    if obj is None or isinstance(obj, (bool, int, float, bytes, bytearray)):
        return True
    if looks_like_vector(obj):  # numpy / torch / faiss / long numeric list
        return True
    if isinstance(obj, (types.ModuleType, type)):
        return True
    mod = (getattr(type(obj), "__module__", "") or "").split(".")[0]
    if mod == "torch":  # model weights / nn.Module
        return True
    if isinstance(obj, (types.FunctionType, types.MethodType,
                        types.BuiltinFunctionType, types.BuiltinMethodType,
                        types.LambdaType)):
        return True
    if type(obj).__name__.lower() in _SKIP_LIKE_NAMES:
        return True
    return False


def _referents(obj: Any) -> List[Any]:
    """Non-scalar gc referents of a ``__dict__``-less object (e.g. faiss/SWIG)."""
    out: List[Any] = []
    try:
        for ref in gc.get_referents(obj):
            if isinstance(ref, _PRIMITIVES) or ref is None:
                continue
            if isinstance(ref, (types.ModuleType, type)):
                continue
            out.append(ref)
    except Exception:
        pass
    return out


def _record_id(mapping: Dict[str, Any], path: str) -> str:
    for k in _ID_KEYS:
        v = mapping.get(k)
        if v is not None and isinstance(v, _PRIMITIVES):
            return str(v)
    return path


def _is_record_like(v: Any) -> bool:
    """A dict or object-with-__dict__ that is not a client / machinery / vector
    — i.e. a bag of fields that stands on its own as a memory record."""
    if _is_client(v) is not None or looks_like_vector(v) or _should_skip(v):
        return False
    if isinstance(v, dict):
        return True
    return _obj_dict(v) is not None


def _is_collection_of_records(v: Any) -> bool:
    if isinstance(v, (list, tuple, set, frozenset)):
        return any(_is_record_like(e) or _is_client(e) is not None for e in v)
    if isinstance(v, dict):
        return any(_is_record_like(x) or _is_client(x) is not None
                   for x in v.values())
    return False


def _is_namespace(v: Any) -> bool:
    """A dict/object that itself holds a client or a collection-of-records — so
    it must be descended into (not emitted whole as a record)."""
    if _is_client(v) is not None:
        return False
    d = v if isinstance(v, dict) else _obj_dict(v)
    if d is None:
        return False
    for x in d.values():
        if _is_client(x) is not None or _is_collection_of_records(x):
            return True
    return False


def _is_opaque_container(v: Any) -> bool:
    """A ``__dict__``-less, non-scalar C-ext/SWIG object (e.g. a wrapper holding
    a client in a slot) — descend via ``gc.get_referents`` to find hidden
    clients."""
    if isinstance(v, _PRIMITIVES) or v is None:
        return False
    if looks_like_vector(v) or _should_skip(v):
        return False
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        return False
    if _obj_dict(v) is not None:
        return False
    return bool(_referents(v))


def _is_structural(v: Any) -> bool:
    """A value that must be DESCENDED into (holds records/clients), as opposed to
    a field that belongs inline in its parent record."""
    return (_is_client(v) is not None or _is_collection_of_records(v)
            or _is_namespace(v) or _is_opaque_container(v))


def _is_pure_record_mapping(d: Dict[str, Any]) -> bool:
    """True when EVERY value is itself a record / client / collection-of-records
    (no bare scalar/str/vector field) — i.e. a mapping OF records, not a single
    record that merely has a nested dict field."""
    vals = list(d.values())
    if not vals:
        return False
    for v in vals:
        if (_is_client(v) is not None or _is_collection_of_records(v)
                or _is_record_like(v)):
            continue
        return False
    return True


# ---------------------------------------------------------------------------
# discover() — in-heap itemisation + client discovery, rooted at self.memory
# ---------------------------------------------------------------------------


class _MemoryWalker:
    def __init__(self) -> None:
        self.seen: Set[int] = set()
        self.visited = 0
        self.result = DiscoveryResult()

    def _budget_ok(self, depth: int, path: str) -> bool:
        if self.visited >= MAX_VISITED:
            if not self.result.truncated:
                self.result.truncated = True
                self.result.notes.append(
                    f"MAX_VISITED={MAX_VISITED} reached; some content not captured")
            return False
        if depth > MAX_DEPTH:
            self.result.notes.append(
                f"MAX_DEPTH={MAX_DEPTH} reached at {path}; not descended")
            return False
        return True

    def walk(self, path: str, obj: Any, parent: Any, depth: int) -> None:
        if not self._budget_ok(depth, path):
            return
        cap = _is_client(obj)
        if cap is not None:
            oid = id(obj)
            if oid in self.seen:
                return
            self.seen.add(oid)
            self.result.clients.append(
                ClientRef(path=path, obj=obj, cap=cap, parent=parent))
            return
        if isinstance(obj, str):
            if obj:
                self._emit(path, obj, embedding=False)
            return
        if _should_skip(obj):
            if looks_like_vector(obj):
                self.result.notes.append(f"raw vector skipped at {path}")
            return
        oid = id(obj)
        if oid in self.seen:
            return
        self.seen.add(oid)
        self.visited += 1

        if isinstance(obj, (list, tuple, set, frozenset)):
            for i, el in enumerate(obj):
                self.walk(f"{path}[{i}]", el, obj, depth + 1)
            return
        if isinstance(obj, dict):
            self._walk_mapping_or_record(path, obj, obj, depth)
            return
        d = _obj_dict(obj)
        if d is not None:
            self._walk_mapping_or_record(path, obj, d, depth)
            return
        # __dict__-less non-scalar (SWIG/C-ext): descend via gc referents.
        for i, ref in enumerate(_referents(obj)):
            self.walk(f"{path}.<ref#{i}>", ref, obj, depth + 1)

    def _walk_mapping_or_record(self, path: str, obj: Any,
                                d: Dict[str, Any], depth: int) -> None:
        # A pure mapping OF records -> each value is its own item.
        if _is_pure_record_mapping(d):
            for k, v in d.items():
                self.walk(f"{path}.{_key(k)}", v, obj, depth + 1)
            return
        # Otherwise a record node: descend into any embedded clients /
        # collections-of-records, and emit the remaining scalar-ish fields as ONE
        # record so config/counters at this level are still captured.
        structural = {}
        fields = {}
        for k, v in d.items():
            if isinstance(k, str) and k.startswith("_trace"):
                continue
            if _is_structural(v):
                structural[k] = v
            else:
                fields[k] = v
        if structural:
            for k, v in structural.items():
                self.walk(f"{path}.{_key(k)}", v, obj, depth + 1)
            if fields:
                self._emit_record(path, fields)
        else:
            # Whole node is one record (dict or object).
            self._emit_record(path, d if isinstance(obj, dict) else _public(d))

    def _emit(self, path: str, body_obj: Any, embedding: bool) -> None:
        self.result.items.append(MemoryItem(
            item_id=path, body=render_text(body_obj), embedding_backed=embedding))

    def _emit_record(self, path: str, mapping: Dict[str, Any]) -> None:
        safe = _public(mapping)
        item_id = _record_id(safe, path)
        self.result.items.append(MemoryItem(
            item_id=item_id,
            body=render_text(safe),  # redact_value drops vectors + secret keys
            embedding_backed=_dict_has_vector(safe)))


def _public(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Drop private ``_trace*`` bookkeeping keys; keep everything else (redaction
    happens in ``render_text``/``redact_value``)."""
    out: Dict[str, Any] = {}
    for k, v in mapping.items():
        if isinstance(k, str) and k.startswith("_trace"):
            continue
        out[k if isinstance(k, str) else str(k)] = v
    return out


def _key(k: Any) -> str:
    return k if isinstance(k, str) else str(k)


def discover(root: Any) -> DiscoveryResult:
    """Walk the object graph starting at ``self.memory``; return in-heap records
    + every discovered backend client + completeness notes. Never raises."""
    walker = _MemoryWalker()
    try:
        walker.walk("memory", root, None, 0)
    except Exception as exc:  # pragma: no cover - defensive
        walker.result.notes.append(f"discovery aborted: {exc!r}")
    return walker.result


def collect_subtree_ids(root: Any, *, limit: int = MAX_VISITED) -> Set[int]:
    """Ids of every object reachable under ``root`` (bounded) — used to EXCLUDE
    the ``self.memory`` subtree from the state walk so it is not duplicated."""
    ids: Set[int] = set()
    if root is None:
        return ids
    stack = [(root, 0)]
    while stack and len(ids) < limit:
        obj, depth = stack.pop()
        oid = id(obj)
        if oid in ids or depth > MAX_DEPTH:
            continue
        if isinstance(obj, _PRIMITIVES) or obj is None:
            continue
        ids.add(oid)
        try:
            if isinstance(obj, dict):
                children = list(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                children = list(obj)
            else:
                d = _obj_dict(obj)
                children = list(d.values()) if d is not None else _referents(obj)
        except Exception:
            children = []
        for c in children:
            if not (isinstance(c, _PRIMITIVES) or c is None):
                stack.append((c, depth + 1))
    return ids


# ---------------------------------------------------------------------------
# discover_state() — always-on whole-memo walk, excluding self.memory
# ---------------------------------------------------------------------------


@dataclass
class StateResult:
    structure: Dict[str, Any] = field(default_factory=dict)
    reference_markers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


_STATE_SENTINEL = object()  # "drop this value" marker


class _StateWalker:
    def __init__(self, exclude: Set[int]) -> None:
        self.exclude = set(exclude)
        self.seen: Set[int] = set()
        self.result = StateResult()
        self.visited = 0

    def walk_root(self, memo: Any) -> None:
        d = _obj_dict(memo)
        if d is None:
            self.result.notes.append("memo has no __dict__; state capture empty")
            return
        self.seen.add(id(memo))
        self.result.structure = self._walk_mapping(d, depth=0, path="state",
                                                   descend_config=False)

    def _value(self, k: Any, v: Any, depth: int, path: str,
               descend_config: bool) -> Any:
        key = _key(k)
        if isinstance(key, str) and key.startswith("_trace"):
            return _STATE_SENTINEL
        if key == "memory":  # the self.memory subtree lives in memory/, not here
            return _STATE_SENTINEL
        if id(v) in self.exclude:
            return _STATE_SENTINEL
        if is_vector_key(key):
            return _STATE_SENTINEL
        if is_sensitive_key(key) and key not in _STATE_DESCEND_KEYS:
            return _STATE_SENTINEL  # credential/config-connection key dropped
        descend = descend_config or key in _STATE_DESCEND_KEYS
        return self._walk_value(v, depth + 1, f"{path}.{key}", descend)

    def _walk_value(self, v: Any, depth: int, path: str,
                    descend_config: bool) -> Any:
        if depth > MAX_DEPTH or self.visited >= MAX_VISITED:
            if self.visited >= MAX_VISITED:
                self.result.notes.append("state walk hit MAX_VISITED")
            return "<bounded>"
        cap = _is_client(v)
        if cap is not None:
            marker = f"<client-ref: {cap.kind} at {path}>"
            self.result.reference_markers.append(marker)
            return marker
        if looks_like_vector(v):
            return redact_value(v)  # -> vector placeholder
        if isinstance(v, str):
            return "<redacted secret>" if looks_like_secret_value(v) else v
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        if _should_skip(v):
            return _STATE_SENTINEL
        oid = id(v)
        if oid in self.seen or oid in self.exclude:
            return "<cycle-or-excluded>"
        self.seen.add(oid)
        self.visited += 1
        if isinstance(v, dict):
            return self._walk_mapping(v, depth, path, descend_config)
        if isinstance(v, (list, tuple, set, frozenset)):
            out = []
            for i, e in enumerate(v):
                rv = self._walk_value(e, depth + 1, f"{path}[{i}]", descend_config)
                if rv is not _STATE_SENTINEL:
                    out.append(rv)
            return out
        d = _obj_dict(v)
        if d is not None:
            body = self._walk_mapping(d, depth, path, descend_config)
            body["<type>"] = type(v).__name__
            return body
        return redact_value(v)

    def _walk_mapping(self, d: Dict[str, Any], depth: int, path: str,
                      descend_config: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            rv = self._value(k, v, depth, path, descend_config)
            if rv is not _STATE_SENTINEL:
                out[_key(k)] = rv
        return out


def discover_state(memo: Any, *, exclude: Set[int]) -> StateResult:
    """Whole-memo structural walk EXCLUDING the ``self.memory`` subtree. Config
    is captured but every value passes ``is_sensitive_key`` + secret-value +
    vector redaction; clients are reference markers only; machinery is skipped.
    Never raises."""
    walker = _StateWalker(exclude)
    try:
        walker.walk_root(memo)
    except Exception as exc:  # pragma: no cover - defensive
        walker.result.notes.append(f"state walk aborted: {exc!r}")
    return walker.result
