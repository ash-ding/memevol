"""
No-memory baseline for DynamicMem.

build_memory_from_data: no-op (nothing is stored).
retrieve_memory_for_query: returns empty dict (no memory context for the agent).

This establishes the zero-memory baseline reward used to calibrate normalized scores.
"""

from common.memo_class import MemoClass


class NoMemMemoClass(MemoClass):
    async def build_memory_from_data(self, recorder) -> None:
        pass

    async def retrieve_memory_for_query(self, recorder) -> dict:
        return {}
