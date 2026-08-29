"""Convention-first resolution + the two-section snapshot.

There is NO class-name registry here and NO dispatch on ``type(memo).__name__``.
Generality comes from the ``self.memory`` CONVENTION plus a never-crash
whole-memo fallback.

Every snapshot has TWO complementary sections, committed independently:

* ``memory/`` — the high-signal CORE (a clean, diffable per-item timeline).
  Resolution order:
    L1  optional ``memo.dump_memory_text()`` (opt-in override, checked FIRST);
    L2  root at ``self.memory``: in-heap records captured structurally + a
        capability read-back from EVERY discovered backend client, MERGED
        (L2b: capture-at-write folds in for a bare FAISS with no docstore);
    L3  never-crash whole-memo generic walk, ONLY when ``self.memory`` is absent.

* ``state/`` — the most-complete REST of the memo: an ALWAYS-ON whole-memo walk
  EXCLUDING the ``self.memory`` subtree (config redacted, hyperparameters,
  counters, buffers, misc state; clients as reference markers; machinery
  skipped).

A COMPLETENESS SIGNAL spanning both sections makes coverage gaps visible.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tracing.adapters import (
    FallbackAdapter,
    MemoryItem,
    redact_value,
    render_text,
)
from tracing.backends import BackendReadError
from tracing.capture import get_ledger, pair_faiss_with_ledger
from tracing.discovery import (
    ClientRef,
    collect_subtree_ids,
    discover,
    discover_state,
)
from tracing.git_store import GitStore
from tracing.triggers import _Diff, _trace_root, _user_key, build_commit_message

log = logging.getLogger("main")

_STATE_FILE = "state/state.md"
_COMPLETENESS_FILE = "completeness.md"

# Docstore sibling names for FAISS text recovery (LangChain/LlamaIndex shapes).
_DOCSTORE_KEYS = ("docstore", "_docstore", "index_to_docstore_id",
                  "id_to_text", "doc_ids", "texts", "_dict")


# ---------------------------------------------------------------------------
# Snapshot model
# ---------------------------------------------------------------------------


@dataclass
class ResolvedItem:
    kind: str            # git subfolder under memory/
    item: MemoryItem


@dataclass
class Completeness:
    memory_source: str = ""                       # L1 / L2 / L3
    inheap_items: int = 0
    clients: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    unknown_clients: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    state_reference_markers: List[str] = field(default_factory=list)
    state_notes: List[str] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.unreadable or self.unknown_clients or self.mismatches)

    def render(self) -> str:
        return render_text({
            "memory_source": self.memory_source,
            "inheap_items": self.inheap_items,
            "clients": self.clients,
            "unreadable": self.unreadable,
            "unknown_clients": self.unknown_clients,
            "mismatches": self.mismatches,
            "skipped": self.skipped,
            "state_reference_markers": self.state_reference_markers,
            "state_notes": self.state_notes,
            "has_gaps": self.has_gaps,
        })


@dataclass
class Snapshot:
    memory: List[ResolvedItem] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    completeness: Completeness = field(default_factory=Completeness)


# ---------------------------------------------------------------------------
# Scope — the ONE thin per-harness hint, read from the memo's own attributes
# ---------------------------------------------------------------------------


def _first_attr(memo: Any, *names: str) -> Optional[Any]:
    for n in names:
        v = getattr(memo, n, None)
        if v is not None:
            return v
    return None


def _scope(memo: Any) -> Dict[str, Any]:
    return {
        "collection": _first_attr(memo, "collection_name", "_collection_name",
                                  "_collection", "collection"),
        "table": _first_attr(memo, "table_name", "_table_name", "_table", "table"),
        "group_id": _first_attr(memo, "_gid", "group_id", "_group_id", "gid"),
        "user_id": _first_attr(memo, "_user_id", "user_id", "_instance_id",
                               "instance_id"),
    }


# ---------------------------------------------------------------------------
# FAISS text recovery: docstore -> capture-at-write ledger -> unreadable
# ---------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _find_docstore(parent: Any) -> Dict[str, Any]:
    if parent is None:
        return {}
    found: Dict[str, Any] = {}
    for key in _DOCSTORE_KEYS:
        v = _get(parent, key)
        if v is not None:
            found[key] = v
    return found


def _doc_text(entry: Any) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    for attr in ("page_content", "text", "content"):
        v = _get(entry, attr)
        if isinstance(v, str):
            return v
    if entry is not None:
        return render_text(entry)
    return None


def _faiss_from_docstore(index: Any, siblings: Dict[str, Any]) -> Optional[List[MemoryItem]]:
    """LangChain shape: docstore.search(index_to_docstore_id[i]).page_content —
    id is the DOCSTORE id (stable), not the FAISS integer position."""
    ntotal = int(_get(index, "ntotal", 0) or 0)
    id_map = siblings.get("index_to_docstore_id")
    docstore = siblings.get("docstore") or siblings.get("_docstore")
    if id_map is not None and docstore is not None:
        items: List[MemoryItem] = []
        for i in range(ntotal):
            docid = id_map.get(i) if isinstance(id_map, dict) else (
                id_map[i] if i < len(id_map) else None)
            if docid is None:
                continue
            entry = None
            search = getattr(docstore, "search", None) or getattr(docstore, "get", None)
            if callable(search):
                entry = search(docid)
            elif isinstance(docstore, dict):
                entry = docstore.get(docid)
            text = _doc_text(entry)
            if text is not None:
                items.append(MemoryItem(item_id=str(docid), body=render_text(text),
                                        embedding_backed=True))
        if items:
            return items
    # Plain parallel text list/dict of length ntotal.
    for key in ("id_to_text", "texts"):
        seq = siblings.get(key)
        if isinstance(seq, dict) and seq:
            return [MemoryItem(item_id=str(k), body=render_text(_doc_text(v) or ""),
                               embedding_backed=True) for k, v in seq.items()]
        if isinstance(seq, (list, tuple)) and len(seq) == ntotal and ntotal:
            return [MemoryItem(item_id=str(i), body=render_text(_doc_text(v) or ""),
                               embedding_backed=True) for i, v in enumerate(seq)]
    return None


def _recover_faiss(memo: Any, ref: ClientRef,
                   completeness: Completeness) -> List[ResolvedItem]:
    index = ref.obj
    siblings = _find_docstore(ref.parent)
    items = _faiss_from_docstore(index, siblings)
    if items:
        completeness.clients.append(f"faiss at {ref.path}: {len(items)} items (docstore)")
        return [ResolvedItem("faiss", it) for it in items]
    # Capture-at-write ledger (with the ntotal-vs-len mismatch guard).
    paired, reason = pair_faiss_with_ledger(index, get_ledger(memo))
    if paired:
        completeness.clients.append(f"faiss at {ref.path}: {len(paired)} items (ledger)")
        return [ResolvedItem("faiss", it) for it in paired]
    if reason:
        if "refusing to pair" in reason:
            completeness.mismatches.append(reason)
        else:
            completeness.unreadable.append(reason)
    # Unreadable: record in the completeness signal + redact the vectors.
    completeness.unreadable.append(f"faiss at {ref.path}: unreadable (vectors redacted)")
    return [ResolvedItem("faiss", MemoryItem(
        item_id="faiss_index",
        body=render_text(redact_value(index)),  # -> "<vector redacted: faiss ...>"
        embedding_backed=True))]


# ---------------------------------------------------------------------------
# resolve_and_snapshot
# ---------------------------------------------------------------------------

_MEMORY_SENTINEL = object()


def _call_dump(memo: Any) -> Optional[List[MemoryItem]]:
    fn = getattr(memo, "dump_memory_text", None)
    if not callable(fn):
        return None
    try:
        res = fn()
        if inspect.isawaitable(res):
            import asyncio
            res = asyncio.run(res)
        if isinstance(res, list) and all(isinstance(x, MemoryItem) for x in res):
            return res
        log.warning("[tracing] dump_memory_text returned non-MemoryItem list; ignoring")
    except Exception as exc:
        log.warning("[tracing] dump_memory_text failed: %r", exc)
    return None


def resolve_and_snapshot(memo: Any) -> Snapshot:
    """Produce the two-section snapshot for a memo. Never raises."""
    snap = Snapshot()
    comp = snap.completeness

    # -- memory/ section ----------------------------------------------------
    dump = _call_dump(memo)
    root = getattr(memo, "memory", _MEMORY_SENTINEL)
    if dump is not None:
        comp.memory_source = "L1:dump_memory_text"
        snap.memory = [ResolvedItem("dump", it) for it in dump]
    elif root is not _MEMORY_SENTINEL:
        comp.memory_source = "L2:self.memory"
        snap.memory = _resolve_from_memory(memo, root, comp)
    else:
        comp.memory_source = "L3:whole-memo-fallback"
        try:
            items = FallbackAdapter().extract(memo)
        except Exception as exc:  # pragma: no cover - fallback is defensive
            log.warning("[tracing] fallback extract failed: %r", exc)
            items = []
        snap.memory = [ResolvedItem("unsupported", it) for it in items]

    # -- state/ section (ALWAYS ON) ----------------------------------------
    exclude = set()
    if root is not _MEMORY_SENTINEL:
        exclude = collect_subtree_ids(root)
        exclude.add(id(root))
    try:
        state = discover_state(memo, exclude=exclude)
        snap.state = state.structure
        comp.state_reference_markers = state.reference_markers
        comp.state_notes = state.notes
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[tracing] state walk failed: %r", exc)
    return snap


def _resolve_from_memory(memo: Any, root: Any,
                         comp: Completeness) -> List[ResolvedItem]:
    dr = discover(root)
    comp.inheap_items = len(dr.items)
    comp.skipped.extend(dr.notes)
    resolved: List[ResolvedItem] = [ResolvedItem("inheap", it) for it in dr.items]

    scope = _scope(memo)
    for ref in dr.clients:
        if ref.cap.kind == "faiss":
            resolved.extend(_recover_faiss(memo, ref, comp))
            continue
        try:
            items = ref.cap.extract(ref.obj, scope=scope)
        except BackendReadError as exc:
            comp.unreadable.append(f"{ref.cap.kind} at {ref.path}: {exc}")
            continue
        except Exception as exc:  # remote/hosted backends can fail at runtime
            comp.unreadable.append(f"{ref.cap.kind} at {ref.path}: read failed ({exc!r})")
            continue
        tag = f"{ref.cap.kind} at {ref.path}: {len(items)} items"
        if ref.cap.unverified:
            tag += " (unverified backend)"
        comp.clients.append(tag)
        resolved.extend(ResolvedItem(ref.cap.kind, it) for it in items)
    return resolved


# ---------------------------------------------------------------------------
# ConventionSession — two-section commit-on-change over a per-user git store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rendered:
    kind: str
    item_id: str
    rel_path: str
    content: str
    signature: str


def _frontmatter(fields: Dict[str, Any]) -> str:
    import json

    lines = ["---"]
    for key in ("harness", "section", "kind", "item_id", "embedding_backed",
                "embedding_model", "op", "op_seq", "timestamp"):
        v = fields.get(key)
        if v is None:
            rendered = "null"
        elif isinstance(v, bool):
            rendered = "true" if v else "false"
        elif isinstance(v, (int, float)):
            rendered = str(v)
        else:
            rendered = json.dumps(str(v).replace("\r", " ").replace("\n", " "),
                                  ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.extend(["---", ""])
    return "\n".join(lines) + "\n"


class ConventionSession:
    """Owns one user's two-section git store with independent commit-on-change.

    Standalone in Phase B (exercised directly, like round 1's ``TraceSession``);
    wiring it into ``wrap_memo``/``common`` is Phase C. The round-1 single-section
    ``TraceSession`` is untouched so existing traces keep working.
    """

    def __init__(self, *, memo: Any, run_id: str, user_dir: str,
                 harness_name: str, git_store: Optional[GitStore] = None):
        self._memo = memo
        self.run_id = run_id
        self.user_dir = user_dir
        self.harness_name = harness_name
        self.user_key = _user_key(user_dir)
        if git_store is not None:
            self.store = git_store
        else:
            repo_dir = (_trace_root() / str(run_id) / "traces" / "git" / self.user_key)
            self.store = GitStore(repo_dir)
        self._last_memory: Dict[str, _Rendered] = {}
        self._last_state_sig = ""
        self._op_seq = 0

    # -- public ------------------------------------------------------------

    def snapshot(self, phase: str, trigger: str, *,
                 label: Optional[str] = None) -> Dict[str, Any]:
        """Render both sections; commit each ONLY IF CHANGED (independently).
        Returns ``{"memory": sha|None, "state": sha|None, "snapshot": Snapshot}``.
        Never raises."""
        try:
            snap = resolve_and_snapshot(self._memo)
            mem_sha = self._commit_memory(snap, phase, trigger, label)
            state_sha = self._commit_state(snap, phase, trigger, label)
            return {"memory": mem_sha, "state": state_sha, "snapshot": snap}
        except Exception as exc:  # a trace failure must never perturb the eval
            log.warning("[tracing] convention snapshot failed (phase=%s): %r",
                        phase, exc)
            return {"memory": None, "state": None, "snapshot": None}

    # -- memory/ section ---------------------------------------------------

    def _render_memory(self, snap: Snapshot) -> Dict[str, _Rendered]:
        rendered: Dict[str, _Rendered] = {}
        for ri in snap.memory:
            rel_path = self.store.item_rel_path(ri.kind, ri.item.item_id)
            sig_src = "\x00".join([ri.kind, ri.item.item_id,
                                   "1" if ri.item.embedding_backed else "0",
                                   ri.item.body])
            sig = hashlib.sha1(sig_src.encode("utf-8")).hexdigest()
            fm = _frontmatter({
                "harness": self.harness_name, "section": "memory", "kind": ri.kind,
                "item_id": ri.item.item_id,
                "embedding_backed": ri.item.embedding_backed,
                "embedding_model": ri.item.embedding_model, "op": "snapshot",
                "op_seq": self._op_seq + 1,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            # Last write wins on a rel_path collision (distinct ids never collide).
            rendered[rel_path] = _Rendered(ri.kind, ri.item.item_id, rel_path,
                                           fm + ri.item.body + "\n", sig)
        return rendered

    def _commit_memory(self, snap: Snapshot, phase: str, trigger: str,
                       label: Optional[str]) -> Optional[str]:
        new = self._render_memory(snap)
        diff = _Diff()
        for path, r in new.items():
            if path not in self._last_memory:
                diff.added.append(r.item_id)
            elif self._last_memory[path].signature != r.signature:
                diff.updated.append(r.item_id)
        for path in self._last_memory:
            if path not in new:
                diff.deleted.append(self._last_memory[path].item_id)
        if not diff.changed:
            return None

        self._op_seq += 1
        changed_paths: List[str] = []
        for path, r in new.items():
            if path not in self._last_memory or \
                    self._last_memory[path].signature != r.signature:
                self.store.write_item(path, r.content)
                changed_paths.append(path)
        for path in self._last_memory:
            if path not in new:
                self.store.delete_item(path)
                changed_paths.append(path)

        msg = build_commit_message(phase=phase, trigger=trigger,
                                   op_seq=self._op_seq, diff=diff,
                                   label=_sec_label("memory", label), anomaly=False)
        sha = self.store.commit_paths(changed_paths, msg)
        self._last_memory = new
        return sha

    # -- state/ section ----------------------------------------------------

    def _commit_state(self, snap: Snapshot, phase: str, trigger: str,
                      label: Optional[str]) -> Optional[str]:
        # ``snap.state`` is ALREADY fully redacted by ``discover_state`` (secrets
        # dropped, vectors placeheld, config DESCENDED, clients as markers).
        # Serialise it directly — running it back through ``render_text``/
        # ``redact_value`` would wrongly drop the (already-safe) ``config`` key.
        state_body = _dump_json(snap.state)
        comp_body = snap.completeness.render()
        sig = hashlib.sha1((state_body + "\x00" + comp_body).encode("utf-8")).hexdigest()
        if sig == self._last_state_sig:
            return None
        first = self._last_state_sig == ""
        self._op_seq += 1
        fm = _frontmatter({
            "harness": self.harness_name, "section": "state", "kind": "state",
            "item_id": "state", "embedding_backed": False, "embedding_model": None,
            "op": "snapshot", "op_seq": self._op_seq,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self.store.write_item(_STATE_FILE, fm + state_body + "\n")
        self.store.write_item(_COMPLETENESS_FILE, comp_body + "\n")
        diff = _Diff(added=["state"]) if first else _Diff(updated=["state"])
        msg = build_commit_message(phase=phase, trigger=trigger,
                                   op_seq=self._op_seq, diff=diff,
                                   label=_sec_label("state", label), anomaly=False)
        sha = self.store.commit_paths([_STATE_FILE, _COMPLETENESS_FILE], msg)
        self._last_state_sig = sig
        return sha


def _dump_json(obj: Any) -> str:
    """Deterministic JSON for an ALREADY-redacted structure (no second redaction
    pass — that would drop safe ``config``/``database`` keys)."""
    import json

    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2,
                      default=str)


def _sec_label(section: str, label: Optional[str]) -> str:
    return f"section={section} {label}" if label else f"section={section}"


def new_convention_session(*, memo: Any, run_id: str, user_dir: str,
                           harness_name: str,
                           git_store: Optional[GitStore] = None) -> ConventionSession:
    return ConventionSession(memo=memo, run_id=run_id, user_dir=user_dir,
                             harness_name=harness_name, git_store=git_store)


__all__ = [
    "Completeness", "ConventionSession", "ResolvedItem", "Snapshot",
    "new_convention_session", "resolve_and_snapshot",
]
