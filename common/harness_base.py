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

    def __init__(self):
        self.database: Optional[Any] = None

    # -------- Pipeline Runner --------
    @abstractmethod
    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        """General retrieve method — orders and chains per-layer retrieves."""
        pass

    @abstractmethod
    async def general_update(self, recorder: Basic_Recorder) -> None:
        """General update method — orders and chains per-layer updates."""
        pass
