"""
No-memory baseline for DynamicMem.

general_update: no-op (nothing is stored).
general_retrieve: returns empty dict (no memory context for the agent).

This establishes the zero-memory baseline reward used to calibrate normalized scores.
"""

from common.harness_base import MemoStructure


class NoMemMemoStructure(MemoStructure):
    async def general_update(self, recorder) -> None:
        pass

    async def general_retrieve(self, recorder) -> dict:
        return {}
