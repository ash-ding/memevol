"""Minimal base classes for memevol."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict


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
