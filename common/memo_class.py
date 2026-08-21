"""The standardized memory-system contract every benchmark evaluates through.

`MemoClass` exposes three OPTIONAL-override hooks — BUILD
(`build_memory_from_data`) / RETRIEVE (`retrieve_memory_for_query`) /
ANSWER (`use_memory_to_answer`) — all with safe defaults. A fresh instance
is created per user/sample BY THE FRAMEWORK (the per-dataset workflow), so no
cross-user state is possible.

Configuration: the framework passes an optional `config` dict to the
constructor (each instance gets its own copy at `self.config`). Parameterized
memory systems (the harness baselines) receive their run.py-resolved settings
this way — `run_baseline(memo_class=MyMemo, memo_config={...})`; a subclass
that overrides `__init__` must accept and forward it:

    class MyMemo(MemoClass):
        def __init__(self, config=None):
            super().__init__(config)
            ...
        # read settings via self.config.get("top_k", 10)

forge-evolved harnesses are never handed a config (their settings live in the
generated code itself), so a plain `def __init__(self)` override also keeps
working there — the workflow only passes `config=` when one was provided.

The data envelope the hooks receive is `Basic_Recorder`, defined in
`common/recorder.py` (re-imported here for convenience).

(Hard-renamed 2026-08-06 — no legacy shim or alias exists; every consumer
imports `common.memo_class.MemoClass`.)
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional

from common.recorder import Basic_Recorder  # noqa: F401  (re-export: legacy import path)


class MemoClass(ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Per-instance copy — instances must never share mutable config state
        # (the fresh-instance-per-user guarantee extends to configuration).
        self.config: Dict[str, Any] = dict(config) if config else {}
        self.database: Optional[Any] = None

    # -------- Standardized eval hooks (all OPTIONAL overrides) --------

    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        """BUILD (Phase 1). `recorder.init` holds the data newly visible for
        THIS call; accumulate state across calls and choose your own ingestion
        granularity. Default: no-op (build no memory)."""
        return None

    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        """RETRIEVE (Phase 2). `recorder.init` holds the query (+ context).
        Return retrieved context; `{"inline_memory_blocks": [str,...]}` controls
        inline rendering. MUST be read-only w.r.t. memory. Default: `{}`."""
        return {}

    async def use_memory_to_answer(self, recorder: Basic_Recorder, retrieved: Dict,
                                    prompt: str) -> Optional[str]:
        """ANSWER (optional). Return the answer string, or None to defer to the
        benchmark's standard QA agent (the default). `prompt` is the workflow's
        fully-formatted answer prompt. Default: None."""
        return None

    # -------- Optional memory-cache hooks (common/memory_cache.py) --------
    # Override BOTH when the default pickle can't capture your state — a live
    # DB connection, an HTTP pool, a thread pool or a subprocess makes an
    # instance unpicklable (`cannot pickle '_thread.RLock' object`), and the
    # evaluator then silently rebuilds Phase-1 memory from scratch at every
    # stage instead of reusing it. These are the same hooks `forge/memo_class.py`
    # declares for evolved harnesses; declared here so the harness baselines
    # can see them too.
    #
    # `path` is a per-snapshot filename PREFIX, not a file — write/read any
    # file(s) at `str(path) + <your suffix>`. Return True on success; returning
    # False (the default) tells the evaluator to fall back to its own pickle
    # (save) or to rebuild from scratch (load). `load_memory` is called on a
    # FRESH instance built exactly like any other, so `self.config` is
    # populated and can be used to reopen a backend.

    def save_memory(self, path) -> bool:
        """Persist this memo's state under the `path` prefix. Default: False."""
        return False

    def load_memory(self, path) -> bool:
        """Restore state written by `save_memory`. Default: False."""
        return False
