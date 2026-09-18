"""Seed harness: no memory.

Serves as the baseline (calibration point). Phase 1 does nothing; Phase 2
returns an empty dict, so the QA agent answers purely from the question
with no retrieved context.

Subsequent candidates should beat this by organizing the user's app logs
into some form of memory.
"""
from typing import Dict

from forge.memo_class import MemoClass

class NoMemoryHarness(MemoClass):
    async def build_memory_from_data(self, recorder) -> None:
        return None

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return {}
