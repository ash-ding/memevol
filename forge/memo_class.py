"""forge's memo contract — the base class every EVOLVED harness inherits.

This is forge's own evolution surface, split from `common.memo_class` so
the forge contract can gain documentation / optional hooks without touching
the frozen baseline contract that alma's meta-agent reads
(`baselines/evolve/alma/meta_agent_prompt.py::_read_contract_files`).

`MemoClass` here SUBCLASSES `common.memo_class.MemoClass`, so
everything downstream that type-checks against the common ABC
(`forge/launch.py::_load_harness_class`, `forge/contract.py`,
`common/workflow.py`) accepts forge harnesses unchanged — including
historical harnesses in old workspaces that still import
`common.memo_class` directly (pre-2026-08 workspaces need their harness imports updated to load).

The contract (unchanged from common):

  Phase 1 — `build_memory_from_data(recorder)` is called ONCE per build call with
  `recorder.init` holding the data newly visible for that call (see the
  per-dataset shapes in the proposer system prompt). Build/extend your memory;
  choose your own ingestion granularity (write your own loop if you want to
  ingest in pieces).
  For DynamicMem the calls follow the official TCE checkpoint protocol: one call
  per checkpoint with that checkpoint's new app-log segment, interleaved with
  Phase-2 queries — accumulate across calls; never assume you have the full stream.

  Phase 2 — `retrieve_memory_for_query(recorder)` is called once per query with
  `recorder.init` holding the query (+ benchmark-specific context). Return a
  Dict of retrieved context for the QA agent. MUST be read-only with respect
  to memory state (queries must not pollute memory — checkpoint isolation
  depends on it). Tip: return `{"inline_memory_blocks": [str, ...]}` to
  control exactly how retrieved memory is rendered into the official
  DynamicMem answer prompt (blocks are joined verbatim); any other dict
  shape is serialized as one JSON block.

A fresh MemoClass instance is created per user/sample — no cross-user
state is possible.
"""

from __future__ import annotations

from common.recorder import Basic_Recorder  # noqa: F401  (re-export)
from common.memo_class import MemoClass as _CommonMemoClass


class MemoClass(_CommonMemoClass):
    """Evolution-target base class for forge harnesses.

    Inherit from this and implement build (`build_memory_from_data`) and retrieve
    (`retrieve_memory_for_query`):

        class MyHarness(MemoClass):
            async def build_memory_from_data(self, recorder) -> None: ...
            async def retrieve_memory_for_query(self, recorder) -> dict: ...
    """
