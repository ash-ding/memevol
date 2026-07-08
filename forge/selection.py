"""Pareto frontier + selection helpers.

`Frontier` records every evaluated harness with its objectives and the list
of `parent_ids` (priors that the proposer drew from when writing it).

v4 (Meta-Harness paper alignment): the search loop NO LONGER calls
`sample_parent` — the proposer is responsible for browsing prior
candidates and picking what to build on. `sample_parent` is preserved
solely as a utility for held-out-test top-K selection.

`OBJECTIVES` is still ("accuracy",) — multi-axis Pareto frontier is on the
roadmap (PROGRESS.md issue #5) but not yet wired in.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Entry:
    id: str
    objectives: Dict[str, float]
    parent_ids: List[str] = field(default_factory=list)
    # v5 fields — None for entries written by older versions.
    content_hash: Optional[str] = None
    created_at: Optional[str] = None


# Highest gauntlet tier (stage3). `stage_<ds>` objectives record the tier a
# benchmark reached; see forge/orchestrator.py::evaluate_harness.
FINAL_STAGE = 3.0


def _fully_staged(entry: Entry) -> bool:
    """True iff every evaluated benchmark reached the final gauntlet stage.

    Scores from different stages are NOT comparable (a lucky stage1 score
    over 2 users vs a stage3 score over 6), so selection restricts to
    fully-staged entries when any exist. Entries without per-dataset axes
    (pre-staged-era or dev runs) count as not fully staged."""
    ds_axes = [k for k in entry.objectives if k.startswith("accuracy_")]
    if not ds_axes:
        return False
    for k in ds_axes:
        stage_key = "stage_" + k[len("accuracy_"):]
        if float(entry.objectives.get(stage_key, 0.0)) < FINAL_STAGE:
            return False
    return True


class Frontier:
    """Population + Pareto helpers."""

    OBJECTIVES: Tuple[str, ...] = ("accuracy",)

    def __init__(self, entries: Optional[List[Entry]] = None):
        self._entries: List[Entry] = list(entries or [])

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def add(self, entry: Entry) -> None:
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = entry
                return
        self._entries.append(entry)

    def get(self, id: str) -> Optional[Entry]:
        for e in self._entries:
            if e.id == id:
                return e
        return None

    def all_entries(self) -> List[Entry]:
        return list(self._entries)

    def remove_by_ids(self, ids) -> int:
        """Drop entries whose id is in `ids` (set or any container).

        Returns the count removed. Used by the orchestrator's polluting-entry
        cleanup at search_loop startup.
        """
        ids = set(ids)
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id not in ids]
        return before - len(self._entries)

    # ------------------------------------------------------------------
    # Frontier
    # ------------------------------------------------------------------

    def _selection_pool(self) -> List[Entry]:
        """Entries eligible for score comparison: the fully-staged subset
        when non-empty (same-stage scores are the only comparable ones),
        otherwise everyone (early runs where nothing finished stage3)."""
        staged = [e for e in self._entries if _fully_staged(e)]
        return staged or list(self._entries)

    def pareto_ids(self) -> List[str]:
        """Ids on the Pareto frontier (for multi-axis) or top-1 (single axis).

        Compares within `_selection_pool()` — an entry eliminated at stage1
        with a lucky small-sample score must not outrank a stage3 entry."""
        pool = self._selection_pool()
        if not pool:
            return []
        if len(self.OBJECTIVES) == 1:
            axis = self.OBJECTIVES[0]
            best = max(pool, key=lambda e: e.objectives.get(axis, 0.0))
            return [best.id]
        frontier_ids: List[str] = []
        for a in pool:
            dominated = False
            for b in pool:
                if a.id == b.id:
                    continue
                ge_all = all(
                    b.objectives.get(o, 0.0) >= a.objectives.get(o, 0.0)
                    for o in self.OBJECTIVES
                )
                gt_any = any(
                    b.objectives.get(o, 0.0) > a.objectives.get(o, 0.0)
                    for o in self.OBJECTIVES
                )
                if ge_all and gt_any:
                    dominated = True
                    break
            if not dominated:
                frontier_ids.append(a.id)
        return frontier_ids

    def sample_parent(
        self,
        tau: float = 0.5,
        seed: Optional[int] = None,
    ) -> Optional[str]:
        """Softmax-sample over all entries.

        ⚠ NOT used by the search loop anymore (v4 delegates parent
        selection to the proposer agent). Kept as a utility for held-out
        test top-K selection where you might want a stochastic but
        score-weighted pick.

        score(e)  = e.objectives[primary_axis]
        prob(e)   ∝ exp(score / tau)
        Primary axis is OBJECTIVES[0].

        Samples within `_selection_pool()` — restricted to fully-staged
        entries when any exist, since cross-stage scores aren't comparable.
        """
        pool = self._selection_pool()
        if not pool:
            return None
        axis = self.OBJECTIVES[0]
        logits = [float(e.objectives.get(axis, 0.0)) / max(1e-6, tau) for e in pool]
        m = max(logits)
        exps = [math.exp(l - m) for l in logits]
        total = sum(exps)
        probs = [x / total for x in exps] if total > 0 else [1.0 / len(exps)] * len(exps)
        rng = random.Random(seed)
        r = rng.random()
        acc = 0.0
        for e, p in zip(pool, probs):
            acc += p
            if r <= acc:
                return e.id
        return pool[-1].id

    # ------------------------------------------------------------------
    # Persistence (with backward-compat for v3 frontier.json)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        out_entries = []
        for e in self._entries:
            d: Dict[str, object] = {
                "id": e.id,
                "objectives": e.objectives,
                "parent_ids": list(e.parent_ids),
            }
            if e.content_hash is not None:
                d["content_hash"] = e.content_hash
            if e.created_at is not None:
                d["created_at"] = e.created_at
            out_entries.append(d)
        return {
            "objectives": list(self.OBJECTIVES),
            "entries": out_entries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Frontier":
        entries: List[Entry] = []
        for e in data.get("entries", []):
            # Backward compat: v3 used `parent_id: Optional[str]` + `visit_count: int`.
            # If we see those, convert: parent_id="0" → parent_ids=["0"]; visit_count ignored.
            if "parent_ids" in e:
                parent_ids = list(e["parent_ids"]) if e["parent_ids"] else []
            else:
                pid = e.get("parent_id")
                parent_ids = [pid] if pid else []
            entries.append(Entry(
                id=e["id"],
                objectives=dict(e["objectives"]),
                parent_ids=parent_ids,
                content_hash=e.get("content_hash"),     # v5; None for older entries
                created_at=e.get("created_at"),         # v5; None for older entries
            ))
        return cls(entries=entries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Frontier":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
