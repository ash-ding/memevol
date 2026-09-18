"""The standardized memory-system contract every benchmark evaluates through.

`MemoClass` exposes two hooks, BUILD (`build_memory_from_data`) and RETRIEVE
(`retrieve_memory_for_query`), both ABSTRACT — a subclass that misses either (a
typo'd name included) fails at instantiation instead of silently evaluating as
a no-memory system. A fresh instance is created per user/sample BY THE
FRAMEWORK (the per-dataset workflow), so no cross-user state is possible.

ANSWERING IS NOT A HOOK. Every benchmark answers through its own shared QA
agent, from what RETRIEVE returned — so a score compares memories, not
answerers, and the answer-side cost is the same for every system. (An optional
`use_memory_to_answer` hook existed until 2026-09; no baseline ever
implemented it, the forge proposer was never told about it, and the baseline
tests asserted against overriding it.)

Configuration: the framework may pass a `config` dict to the constructor;
each instance keeps its own copy at `self.config`. The class holds no defaults
and no resolution logic — whoever constructs the memo supplies the complete
config (the harness baselines' is resolved by baselines/harness/eval_harness.py):

    class MyMemo(MemoClass):
        async def retrieve_memory_for_query(self, recorder):
            k = self.config["top_k"]    # supplied by the caller, never defaulted here

forge-evolved harnesses are never handed a config (their settings live in the
generated code itself), so a plain `def __init__(self)` override also keeps
working there — the workflow only passes `config=` when one was provided.

The data envelope the hooks receive is `Basic_Recorder`, defined in
`common/recorder.py` (re-imported here for convenience).

(Hard-renamed 2026-08-06 — no legacy shim or alias exists; every consumer
imports `common.memo_class.MemoClass`.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from common.recorder import Basic_Recorder  # noqa: F401  (re-export: legacy import path)


class MemoClass(ABC):
    #: Declared here for readers and type checkers; assigned per instance in
    #: __init__ — a class-level dict would be shared across users.
    config: Dict[str, Any]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Per-instance copy — instances must never share mutable config state
        # (the fresh-instance-per-user guarantee extends to configuration).
        self.config = dict(config) if config else {}

    # -------- Standardized eval hooks --------

    @abstractmethod
    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        """BUILD (Phase 1, REQUIRED). `recorder.init` holds the data newly
        visible for THIS call; accumulate state across calls and choose your
        own ingestion granularity. A system that stores nothing implements
        this as `return None`."""

    @abstractmethod
    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        """RETRIEVE (Phase 2, REQUIRED). `recorder.init` holds the query (+
        context). Return retrieved context; `{"inline_memory_blocks": [str,...]}`
        controls inline rendering. MUST be read-only w.r.t. memory."""

