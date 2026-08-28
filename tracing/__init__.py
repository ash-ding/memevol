"""``tracing`` — a gated, wrapper-based memory-state tracer.

Persists a git-diffable timeline of a memory system's evolving internal state:
one git repo per user, one stable file per memory item, deterministic
code-computed commit descriptors with machine-parseable trailers. A future
analysis agent can then use the GIT TOOLCHAIN ITSELF (``git diff`` / ``log`` /
``blame`` / ``bisect`` / ``branch``) as its toolset over a harness's runtime
memory behaviour.

Round 1 is a STANDALONE, purely-additive package: the wrapper, the shape-adapter
registry, the per-user git store, and the full pluggable trigger layer
(hook-boundary + shared-kernel callback API + ``mark()``) are all implemented
and tested here in complete isolation from the eval path. Wiring the three
gated ``common/`` activation hooks into a live eval run is a deferred
integration round.

Everything is gated on a single ``MEMEVOL_TRACE`` env check (default OFF):
when disabled, ``wrap_memo`` returns the original memo untouched, ``mark()`` is
a single-branch no-op, and no tracing work runs.
"""

from __future__ import annotations

import os
from typing import Any

# Re-exported trigger sources (defined in tracing.triggers).
from tracing.triggers import mark, on_kernel_call

__all__ = ["is_enabled", "mark", "on_kernel_call", "wrap_memo"]

_TRUTHY = {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Single cheap env-var check — the one gate for all tracing work."""
    return os.environ.get("MEMEVOL_TRACE", "").strip().lower() in _TRUTHY


def wrap_memo(memo: Any, *, run_id: str, user_dir: str, harness_name: str) -> Any:
    """Return a ``TracedMemo`` wrapping ``memo`` when tracing is enabled, else
    the original ``memo`` untouched (a single branch, zero tracing work).

    The wrapper is behaviour-preserving: identical hook return values, an
    identical inner memo end-state, a read-only retrieve path, and exceptions
    propagated unchanged. In round 1 this is called by the standalone
    driver/demo (and the tests); wiring it into ``common/workflow.py`` at
    ``self._new_memo()`` is the deferred integration round.
    """
    if not is_enabled():
        return memo
    # Imported lazily so the disabled path never touches the wrapper/session.
    from tracing.triggers import new_session
    from tracing.wrapper import TracedMemo

    session = new_session(memo=memo, run_id=run_id, user_dir=user_dir,
                          harness_name=harness_name)
    return TracedMemo(memo, session=session)
