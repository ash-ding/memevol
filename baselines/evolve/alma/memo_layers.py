"""`Sub_memo_layer` — alma's layered-memory design vocabulary.

alma's meta-agent decomposes a memory system into multiple specialised
layers, each owning a database plus a retrieve/update pair, orchestrated
by the final `MemoClass` subclass. This is alma's OWN design opinion,
NOT part of the shared eval contract (`common/memo_class.py`).
Generated memo code imports it from here:

    from baselines.evolve.alma.memo_layers import Sub_memo_layer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


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
