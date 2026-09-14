"""Baseline harness: no memory.

The calibration floor — Phase 1 stores nothing, Phase 2 returns nothing, so
the shared QA agent answers from the question alone. Every proposed candidate
is measured against this.
"""

from typing import Dict

from common.memo_class import MemoClass


class NoMemoryHarness(MemoClass):

    async def build_memory_from_data(self, recorder) -> None:
        return None

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return {}
