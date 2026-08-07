"""The standardized memory-system contract every benchmark evaluates through.

`MemoClass` exposes three OPTIONAL-override hooks — BUILD
(`build_memory_from_data`) / RETRIEVE (`retrieve_memory_for_query`) /
ANSWER (`use_memory_to_answer`) — all with safe defaults. A fresh instance
is created per user/sample, so no cross-user state is possible.

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

    def __init__(self):
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
