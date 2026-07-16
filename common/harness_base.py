"""Abstract base classes for DynamicMem-style memory harnesses.

`Basic_Recorder` — per-user state container used by all benchmarks and
passed into `general_update` / `general_retrieve`. Benchmark-specific
recorders (e.g. `DynamicMemRecorder`, `LoCoMoRecorder`) inherit from it.

`MemoStructure` / `Sub_memo_layer` — the two-phase contract every harness
must implement.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Basic_Recorder:
    """Base class for task recorders. Domain-specific recorders inherit from this."""

    init: Dict[str, Any] = field(default_factory=dict)
    steps: list = field(default_factory=list)
    reward: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def log_init(self, *args, **kwargs):
        async with self._lock:
            self.init = kwargs

    async def log_step(self, **kwargs):
        async with self._lock:
            self.steps.append(kwargs)

    async def set_reward(self, reward: float):
        async with self._lock:
            self.reward = reward

# only for alma method, claude code should ignore this class
@dataclass
class Sub_memo_layer(ABC):
    """Abstract class for retrieve/update sub-function."""
    layer_intro: str = "Introduction of the structure of current defined Database(if any), corresponding Update and Retrieve method."
    database: Optional[Any] = None

    @abstractmethod
    async def retrieve(self, **kwargs):
        """The retrieve function of current layer."""
        pass

    @abstractmethod
    async def update(self, **kwargs):
        """The update function of current layer."""
        pass


class MemoStructure(ABC):

    # Ingestion granularity is the memory system's design choice. `chunked`
    # is a convenience the build hook may use; a memo may override _update_type
    # / _n_chunks (or ignore chunking entirely and ingest the whole data).
    _update_type: str = "all_at_once"
    _n_chunks: int = 5

    def __init__(self):
        self.database: Optional[Any] = None

    # -------- Standardized eval hooks (all OPTIONAL overrides) --------

    async def general_update(self, recorder: Basic_Recorder) -> None:
        """BUILD (Phase 1). `recorder.init` holds the data newly visible for
        THIS call; accumulate state across calls and choose your own ingestion
        granularity (e.g. `for chunk in self.chunked(recorder.init[...]): ...`).
        Default: no-op (build no memory)."""
        return None

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        """RETRIEVE (Phase 2). `recorder.init` holds the query (+ context).
        Return retrieved context; `{"inline_memory_blocks": [str,...]}` controls
        inline rendering. MUST be read-only w.r.t. memory. Default: `{}`."""
        return {}

    async def general_answer(self, recorder: Basic_Recorder, retrieved: Dict,
                             prompt: str) -> Optional[str]:
        """ANSWER (optional). Return the answer string, or None to defer to the
        benchmark's standard QA agent (the default). `prompt` is the workflow's
        fully-formatted answer prompt. Default: None."""
        return None

    def chunked(self, data):
        """Yield partitions of `data` per self._update_type / self._n_chunks:
        all_at_once → [data]; sequential → one item each; chunked → _n_chunks
        near-equal contiguous partitions. Order preserved, no loss."""
        data = list(data)
        if self._update_type == "sequential":
            for item in data:
                yield [item]
        elif self._update_type == "chunked":
            n = max(1, self._n_chunks)
            n = min(n, len(data)) or 1
            base, extra = divmod(len(data), n)
            start = 0
            for i in range(n):
                size = base + (1 if i < extra else 0)
                yield data[start:start + size]
                start += size
        else:  # all_at_once
            yield data
