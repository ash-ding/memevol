"""Seed harness: no memory.

Serves as the baseline (calibration point). Phase 1 does nothing; Phase 2
returns an empty dict, so the QA agent answers purely from the question
with no retrieved context.

Subsequent candidates should beat this by organizing the user's app logs
into some form of memory.
"""
from typing import Dict

from forge.harness_base import MemoStructure

class NoMemoryHarness(MemoStructure):
    async def general_update(self, recorder) -> None:
        return None

    async def general_retrieve(self, recorder) -> Dict:
        return {}
