"""``TracedMemo`` — a behaviour-preserving composition wrapper.

Wraps a live ``common.memo_class.MemoClass`` instance by *composition* (never
subclassing or monkeypatching): ``self._inner`` holds the real memo and
``__getattr__`` delegates every attribute. Only the three eval hooks are
overridden — WITHOUT changing their signatures — to fire before/after snapshot
triggers around the (unchanged) inner call:

  * ``build_memory_from_data``   -> phase BUILD
  * ``retrieve_memory_for_query`` -> phase RETRIEVE (stays strictly read-only)
  * ``use_memory_to_answer``      -> phase ANSWER

Each override returns the inner result byte-for-byte and re-raises any inner
exception unchanged (no control-flow change). Snapshots are read-only (they
only call the adapter's pure read accessors) and are wrapped so a tracing
failure can never perturb the eval. When tracing is disabled, ``wrap_memo``
never constructs a ``TracedMemo`` at all — see ``tracing.wrap_memo``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from common.recorder import Basic_Recorder
from tracing.triggers import (
    TraceSession,
    _current_phase,
    _current_session,
    set_current_session,
)

log = logging.getLogger("main")


class TracedMemo:
    """Composition wrapper around a MemoClass instance (see module docstring)."""

    def __init__(self, inner: Any, *, session: TraceSession):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_trace_session", session)
        # Bind this session as the context's active one so a following
        # ``mark()`` (called between hooks) targets this memo. ContextVar
        # copies keep concurrent users/tasks isolated.
        set_current_session(session)

    # -- transparent delegation -------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found on the wrapper itself.
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"TracedMemo({self._inner!r})"

    # -- traced eval hooks (signatures preserved verbatim) ----------------

    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        return await self._traced("BUILD", self._inner.build_memory_from_data,
                                  recorder)

    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        return await self._traced("RETRIEVE",
                                  self._inner.retrieve_memory_for_query, recorder)

    async def use_memory_to_answer(self, recorder: Basic_Recorder,
                                   retrieved: Dict, prompt: str) -> Optional[str]:
        return await self._traced("ANSWER", self._inner.use_memory_to_answer,
                                  recorder, retrieved, prompt)

    # -- shared machinery --------------------------------------------------

    async def _traced(self, phase: str, fn, *args):
        session: TraceSession = self._trace_session
        # Bind session + phase for the duration of the hook so shared-kernel
        # callbacks and mark() correlate correctly. Leave the session bound
        # afterwards (reset only the phase) so a post-hook mark() still works.
        _current_session.set(session)
        phase_token = _current_phase.set(phase)
        try:
            self._safe_snapshot(session, phase)          # before
            result = await fn(*args)                      # inner call — propagates
            self._safe_snapshot(session, phase)          # after
            return result
        finally:
            _current_phase.reset(phase_token)

    @staticmethod
    def _safe_snapshot(session: TraceSession, phase: str) -> None:
        # session.snapshot already swallows its own errors, but guard here too
        # so nothing on the hook hot-path can ever escape into the eval.
        try:
            session.snapshot(phase, trigger="hook")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("[tracing] hook-boundary snapshot failed: %r", exc)
