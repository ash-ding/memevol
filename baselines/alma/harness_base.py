"""
Abstract base classes for alma-style memo structures.

Kept inside baselines/alma/ because the two-phase interface
(general_update + general_retrieve) is specific to alma's design. Other
methods (e.g. Meta-Harness) may define their own harness contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from datasets.dynamicmem.env import Basic_Recorder


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
