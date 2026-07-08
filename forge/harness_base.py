"""forge's harness contract — the base class every EVOLVED harness inherits.

This is forge's own evolution surface, split from `common.harness_base` so
the forge contract can gain documentation / optional hooks without touching
the frozen baseline contract that alma's meta-agent reads
(`baselines/alma/meta_agent_prompt.py::_read_harness_base`).

`MemoStructure` here SUBCLASSES `common.harness_base.MemoStructure`, so
everything downstream that type-checks against the common ABC
(`forge/launch.py::_load_harness_class`, `forge/contract.py`,
`common/workflow.py`) accepts forge harnesses unchanged — including
historical harnesses in old workspaces that still import
`common.harness_base` directly.

The contract (unchanged from common):

  Phase 1 — `general_update(recorder)` is called one or more times with
  `recorder.init` holding a chunk of raw benchmark data (see the per-dataset
  shapes in the proposer system prompt). Build/extend your memory.
  For DynamicMem the calls follow the official TCE checkpoint protocol:
  chronological app-log segments, interleaved with Phase-2 queries at each
  checkpoint — never assume you have seen the full stream.

  Phase 2 — `general_retrieve(recorder)` is called once per query with
  `recorder.init` holding the query (+ benchmark-specific context). Return a
  Dict of retrieved context for the QA agent. MUST be read-only with respect
  to memory state (queries must not pollute memory — checkpoint isolation
  depends on it). Tip: return `{"inline_memory_blocks": [str, ...]}` to
  control exactly how retrieved memory is rendered into the official
  DynamicMem answer prompt (blocks are joined verbatim); any other dict
  shape is serialized as one JSON block.

A fresh MemoStructure instance is created per user/sample — no cross-user
state is possible.
"""

from __future__ import annotations

from common.harness_base import Basic_Recorder  # noqa: F401  (re-export)
from common.harness_base import MemoStructure as _CommonMemoStructure


class MemoStructure(_CommonMemoStructure):
    """Evolution-target base class for forge harnesses.

    Inherit from this and implement both abstract methods:

        class MyHarness(MemoStructure):
            async def general_update(self, recorder) -> None: ...
            async def general_retrieve(self, recorder) -> dict: ...

    MEMORY CACHING (staged evaluation): the evaluator snapshots your built
    memory after Phase 1 (per checkpoint for DynamicMem) and reuses it at
    deeper stages instead of rebuilding — evals of your harness get much
    faster/cheaper when your state is serializable. The DEFAULT snapshot
    mechanism is `pickle` of the whole memo object, which works when your
    state on `self` is plain data (dicts, lists, numpy arrays, BM25Okapi,
    `common.llm.Agent`/`Embedding` config objects are all fine). Do NOT keep
    unpicklable handles on `self` (open clients, locks, loaded torch models,
    chromadb clients) — build such things lazily and cache them in attributes
    you can drop, or override the two hooks below. If pickling fails, the
    eval still runs; it just rebuilds memory at every stage.
    """

    # ---- Optional memory-cache hooks -------------------------------------
    # Override BOTH when the default pickle can't capture your state (e.g. a
    # chromadb PersistentClient). `path` is a per-snapshot filename PREFIX —
    # write/read any file(s) at `str(path) + <your suffix>`. Return True on
    # success; returning False (the default) tells the evaluator to use its
    # default pickle (save) or rebuild from scratch (load).

    def save_memory(self, path) -> bool:
        return False

    def load_memory(self, path) -> bool:
        return False
