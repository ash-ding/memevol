"""Shared deterministic evaluation sampling — the ONE place the search loop's
per-step seed and the shuffle-then-prefix nesting primitive live. Used by
forge, alma, and the harness baselines (via datasets/*/env.py + common/workflow.py).

Backward compatibility: with seed=None, shuffle_prefix returns a raw prefix and
combine_seed returns user_dir verbatim — byte-identical to the pre-seed behavior.
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Optional, Sequence


def derive_sample_seed(sampling_seed: int, step: int, dataset: str) -> str:
    """Per-(base seed, search step, dataset) seed string. Pure + stable across
    processes (hashlib, NOT the salted builtin hash), so the same config
    reproduces the same per-step subset sequence on any machine."""
    h = hashlib.sha256(f"{int(sampling_seed)}|{int(step)}|{dataset}".encode("utf-8"))
    return h.hexdigest()[:16]


def combine_seed(sample_seed: Optional[str], user_dir: str) -> str:
    """Seed for a user's per-QA/item sampling. sample_seed=None → user_dir
    verbatim (the historical seed) so nothing changes at default flags."""
    if sample_seed is None:
        return user_dir
    return f"{sample_seed}::{user_dir}"


def shuffle_prefix(pool: Sequence[str], n: Optional[int], seed: Optional[str]) -> List[str]:
    """Select up to n items from pool.

    seed=None  → raw prefix `list(pool)[:n]` (historical behavior; n=None → all).
    seed set   → deterministic random.Random(seed)-shuffled copy, then prefix.

    Never mutates `pool`. With a fixed seed, a smaller n's result is always a
    prefix of a larger n's (the staged-nesting invariant)."""
    items = list(pool)
    if seed is not None:
        random.Random(seed).shuffle(items)
    if n is None:
        return items
    return items[: max(0, int(n))]
