"""Trigger layer — render -> hash -> commit-on-change, plus the deterministic
commit-message descriptor.

Three trigger sources register through ONE uniform path so new sources compose
without special-casing:

  * **hook-boundary** — fired by ``tracing.wrapper.TracedMemo`` before/after
    each build/retrieve/answer hook (the guaranteed-quiescent backbone).
  * **shared-kernel** — :func:`on_kernel_call`, the observer callback a later
    integration round wires into ``common/llm.py`` and ``common/openai_usage
    .py``. In round 1 it is exercised by invoking it directly.
  * **mark** — :func:`mark`, an opt-in semantic checkpoint any code may call.

Every source funnels into :meth:`TraceSession.snapshot`, which renders the
current memo state via its shape adapter, hashes it against the last committed
signature, and commits ONLY on change (no empty commits). The same diff that
decides *whether* to commit also produces the commit message — a human summary
line plus machine-parseable git trailers — all code-computed, no LLM, and
carrying only structural/aggregate descriptors (never prompt text, user data,
secrets, ``memo.config``, or raw vectors).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tracing.adapters import Adapter, MemoryItem, redact_value, resolve_adapter
from tracing.git_store import GitStore

log = logging.getLogger("main")

# Correlation plumbing. The wrapper binds the active session + phase for the
# duration of each hook (and leaves the session bound afterwards so a following
# ``mark()`` targets the right memo); ContextVar copies isolate concurrent
# users/tasks by construction.
_current_session: ContextVar[Optional["TraceSession"]] = ContextVar(
    "memevol_trace_session", default=None)
_current_phase: ContextVar[Optional[str]] = ContextVar(
    "memevol_trace_phase", default=None)


def _trace_root() -> Path:
    return Path(os.environ.get("MEMEVOL_TRACE_ROOT", "workspace"))


def _user_key(user_dir: str) -> str:
    try:
        from common.memory_cache import user_key  # reuse the canonical sanitizer
        return user_key(user_dir)
    except Exception:
        import re
        name = Path(str(user_dir)).name or str(user_dir)
        return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# ---------------------------------------------------------------------------
# Rendered snapshot model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Rendered:
    item_id: str
    rel_path: str
    body: str
    embedding_backed: bool
    embedding_model: Optional[str]
    signature: str  # hash of STABLE content only (excludes op/op_seq/timestamp)


@dataclass
class _Diff:
    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


# ---------------------------------------------------------------------------
# Descriptor — deterministic, code-computed commit message + git trailers
# ---------------------------------------------------------------------------

_TRAILER_KEYS = ("Phase", "Trigger", "Op-Seq", "Added", "Updated", "Deleted",
                 "Pattern")


def _sanitize_line(text: str, limit: int = 80) -> str:
    """One-line, redacted, bounded string safe for a summary/trailer value."""
    flat = str(text).replace("\r", " ").replace("\n", " ").strip()
    return flat[:limit]


def diff_shape_label(diff: _Diff) -> str:
    """Heuristic operation-pattern label inferred from the diff SHAPE only."""
    a, u, d = len(diff.added), len(diff.updated), len(diff.deleted)
    if a and not u and not d:
        return "ingest"
    if d and not a and not u:
        return "prune/delete"
    if u and not a and not d:
        return "link/update"
    if a and d and not u:
        return "consolidate/merge"
    return "consolidate/merge"


def build_commit_message(*, phase: str, trigger: str, op_seq: int, diff: _Diff,
                         label: Optional[str], anomaly: bool) -> str:
    """Deterministic message: human summary line + machine-parseable trailers.

    Contains ONLY structural/aggregate descriptors — never prompt text, user
    data, secrets, config, or vectors. Byte-identical for identical inputs.
    """
    pattern = diff_shape_label(diff)
    ids = sorted(diff.added) + sorted(diff.updated) + sorted(diff.deleted)
    ids_str = ",".join(_sanitize_line(i, 40) for i in ids[:12])
    if len(ids) > 12:
        ids_str += ",..."

    trig = f"trigger={trigger}"
    safe_label = _sanitize_line(label) if label else None
    if safe_label:
        trig += f" label={safe_label}"

    summary = (f"[{phase}] {trig} op_seq={op_seq} | "
               f"+{len(diff.added)} ~{len(diff.updated)} -{len(diff.deleted)} items | "
               f"pattern={pattern}")
    if ids_str:
        summary += f" | ids={ids_str}"
    if anomaly:
        summary += " | ANOMALY=retrieve-mutation"

    trailers = {
        "Phase": phase,
        "Trigger": trigger,
        "Op-Seq": str(op_seq),
        "Added": str(len(diff.added)),
        "Updated": str(len(diff.updated)),
        "Deleted": str(len(diff.deleted)),
        "Pattern": pattern,
    }
    trailer_lines = [f"{k}: {trailers[k]}" for k in _TRAILER_KEYS]
    if safe_label:
        trailer_lines.append(f"Label: {safe_label}")
    if anomaly:
        trailer_lines.append("Anomaly: retrieve-mutation")

    return summary + "\n\n" + "\n".join(trailer_lines) + "\n"


# ---------------------------------------------------------------------------
# TraceSession — one per wrapped memo / user
# ---------------------------------------------------------------------------


class TraceSession:
    """Owns a user's git store, last-committed signature, and op-sequence."""

    def __init__(self, *, memo: Any, run_id: str, user_dir: str,
                 harness_name: str, adapter: Optional[Adapter] = None,
                 git_store: Optional[GitStore] = None):
        self._memo = memo
        self.run_id = run_id
        self.user_dir = user_dir
        self.harness_name = harness_name
        self.user_key = _user_key(user_dir)
        self.adapter = adapter if adapter is not None else resolve_adapter(memo, harness_name)
        if git_store is not None:
            self.store = git_store
        else:
            repo_dir = (_trace_root() / str(run_id) / "traces" / "git" / self.user_key)
            self.store = GitStore(repo_dir)
        # Baseline is an EMPTY snapshot so a first build's "before" (empty)
        # produces no commit and its "after" (populated) produces exactly one.
        self._last: Dict[str, _Rendered] = {}
        self._op_seq = 0

    # -- rendering ---------------------------------------------------------

    def _render(self) -> Dict[str, _Rendered]:
        try:
            items: List[MemoryItem] = self.adapter.extract(self._memo)
        except Exception as exc:  # never crash the eval
            log.warning("[tracing] adapter %s failed on %s: %r — empty snapshot",
                        type(self.adapter).__name__, type(self._memo).__name__, exc)
            return {}
        rendered: Dict[str, _Rendered] = {}
        for item in items:
            rel_path = self.store.item_rel_path(self.adapter.kind, item.item_id)
            model = _sanitize_model_name(item.embedding_model)
            sig_src = "\x00".join([
                item.item_id,
                "1" if item.embedding_backed else "0",
                model or "",
                item.body,
            ])
            signature = hashlib.sha1(sig_src.encode("utf-8")).hexdigest()
            rendered[item.item_id] = _Rendered(
                item_id=item.item_id, rel_path=rel_path, body=item.body,
                embedding_backed=item.embedding_backed, embedding_model=model,
                signature=signature)
        return rendered

    def _snapshot_signature(self, rendered: Dict[str, _Rendered]) -> str:
        parts = [f"{r.item_id}:{r.signature}" for r in
                 sorted(rendered.values(), key=lambda r: r.item_id)]
        return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()

    def _diff(self, new: Dict[str, _Rendered]) -> _Diff:
        diff = _Diff()
        for item_id, r in new.items():
            if item_id not in self._last:
                diff.added.append(item_id)
            elif self._last[item_id].signature != r.signature:
                diff.updated.append(item_id)
        for item_id in self._last:
            if item_id not in new:
                diff.deleted.append(item_id)
        return diff

    # -- the one uniform trigger path --------------------------------------

    def snapshot(self, phase: str, trigger: str, *, label: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Render current state; commit ONLY IF CHANGED. Returns commit SHA or
        None. Never raises — a trace failure must not perturb the eval."""
        try:
            return self._snapshot_inner(phase, trigger, label, meta)
        except Exception as exc:
            log.warning("[tracing] snapshot failed (phase=%s trigger=%s): %r",
                        phase, trigger, exc)
            return None

    def _snapshot_inner(self, phase: str, trigger: str, label: Optional[str],
                        meta: Optional[Dict[str, Any]]) -> Optional[str]:
        new = self._render()
        new_sig = self._snapshot_signature(new)
        # Cheap short-circuit: nothing changed since the last commit.
        if new_sig == self._snapshot_signature(self._last):
            return None

        diff = self._diff(new)
        if not diff.changed:
            return None

        # `meta` is descriptor-only: fold a redacted, bounded hint into the
        # label so it can never leak secrets/vectors. (Kept minimal on purpose.)
        eff_label = label
        if meta:
            safe_meta = redact_value(meta)
            hint = _sanitize_line(safe_meta, 40)
            eff_label = f"{label} {hint}" if label else hint

        anomaly = phase == "RETRIEVE" and diff.changed
        self._op_seq += 1
        op_seq = self._op_seq

        # Write added/updated files (with fresh frontmatter), delete removed.
        for item_id in diff.added:
            self._write(new[item_id], "create", op_seq)
        for item_id in diff.updated:
            self._write(new[item_id], "update", op_seq)
        for item_id in diff.deleted:
            self.store.delete_item(self._last[item_id].rel_path)

        message = build_commit_message(
            phase=phase, trigger=trigger, op_seq=op_seq, diff=diff,
            label=eff_label, anomaly=anomaly)
        sha = self.store.commit(message)
        # Advance the baseline regardless of commit success so we don't loop
        # trying to re-commit the same (uncommittable) state every trigger.
        self._last = new
        return sha

    def _write(self, r: _Rendered, op: str, op_seq: int) -> None:
        frontmatter = _frontmatter({
            "harness": self.harness_name,
            "adapter": self.adapter.kind,
            "item_id": r.item_id,
            "embedding_backed": r.embedding_backed,
            "embedding_model": r.embedding_model,
            "op": op,
            "op_seq": op_seq,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        self.store.write_item(r.rel_path, frontmatter + r.body + "\n")


def _sanitize_model_name(model: Any) -> Optional[str]:
    """Keep only a short model NAME (never a vector or object)."""
    if model is None:
        return None
    if isinstance(model, str):
        return _sanitize_line(model, 80)
    return _sanitize_line(type(model).__name__, 80)


def _frontmatter(fields: Dict[str, Any]) -> str:
    """Minimal, deterministic YAML frontmatter (hand-rolled, no dependency)."""
    lines = ["---"]
    for key in ("harness", "adapter", "item_id", "embedding_backed",
                "embedding_model", "op", "op_seq", "timestamp"):
        value = fields.get(key)
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\r", " ").replace("\n", " ")
    return json.dumps(text, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Session construction + shared-kernel / mark trigger sources
# ---------------------------------------------------------------------------


def new_session(*, memo: Any, run_id: str, user_dir: str, harness_name: str,
                git_store: Optional[GitStore] = None) -> TraceSession:
    return TraceSession(memo=memo, run_id=run_id, user_dir=user_dir,
                        harness_name=harness_name, git_store=git_store)


def bind_session(session: Optional["TraceSession"], phase: Optional[str]
                 ) -> Tuple[Any, Any]:
    """Bind the active session/phase; returns the reset tokens. Used by the
    wrapper (and by tests exercising the shared-kernel callback directly)."""
    return _current_session.set(session), _current_phase.set(phase)


def set_current_session(session: Optional["TraceSession"]) -> None:
    """Leave a session bound as the context's active one (so a later
    ``mark()`` targets it). Set by the wrapper at construction and per hook."""
    _current_session.set(session)


def on_kernel_call(kind: str = "llm") -> Optional[str]:
    """Shared-kernel observer callback — a peer trigger source. A later
    integration round fires this from ``common/llm.py`` and
    ``common/openai_usage.py`` when a kernel call completes; round 1 invokes it
    directly. No-op when disabled or when no session is bound."""
    from tracing import is_enabled
    if not is_enabled():
        return None
    session = _current_session.get(None)
    if session is None:
        return None
    phase = _current_phase.get(None) or "UNKNOWN"
    trigger = "embed-call" if kind == "embed" else "llm-call"
    return session.snapshot(phase, trigger=trigger)


def mark(label: str = "", **structural_meta: Any) -> Optional[str]:
    """Opt-in semantic-checkpoint trigger — a peer to the hook-boundary and
    shared-kernel sources. Single-branch no-op when tracing is disabled. When
    enabled, forces a labeled snapshot through the SAME commit-on-change path;
    unchanged state produces no (empty) commit. ``label``/``structural_meta``
    are descriptor-only and routed through the redaction fallback — they can
    never emit secrets, prompt text, user data, or vectors. Keep ``label`` a
    short string; never pass sensitive data."""
    from tracing import is_enabled
    if not is_enabled():
        return None
    session = _current_session.get(None)
    if session is None:
        log.debug("[tracing] mark(%r) called with no active session — ignoring",
                  label)
        return None
    phase = _current_phase.get(None) or "MARK"
    return session.snapshot(phase, trigger="mark", label=label,
                            meta=structural_meta or None)
