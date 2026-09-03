"""Run state — the filesystem the proposer reads and the loop writes.

Meta-Harness keeps no compressed history: the proposer inspects raw prior
artifacts directly. This module owns only the small index files that make that
filesystem navigable, plus the finalization lock that keeps the test split
untouched until a run is frozen.

Layout under `logs/<run_name>/`:

    evolution_summary.jsonl   one row per evaluated candidate
    frontier_val.json         Pareto front over (score up, context cost down)
    pending_eval.json         the proposer's handoff for this iteration
    finalized.json            test-evaluation lock
    proposer/<iterN>/         proposer session logs
    evals/<system>/           per-candidate eval artifacts (score, traces)
    reports/                  proposer-written post-eval notes
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunPaths:
    """Every path a run touches, derived from the baseline root + run name."""

    root: Path
    run_name: str

    @property
    def logs(self) -> Path:
        return self.root / "logs" / self.run_name

    @property
    def summary(self) -> Path:
        return self.logs / "evolution_summary.jsonl"

    @property
    def frontier(self) -> Path:
        return self.logs / "frontier_val.json"

    @property
    def pending(self) -> Path:
        return self.logs / "pending_eval.json"

    @property
    def finalized(self) -> Path:
        return self.logs / "finalized.json"

    @property
    def proposer_logs(self) -> Path:
        return self.logs / "proposer"

    @property
    def evals(self) -> Path:
        return self.logs / "evals"

    @property
    def reports(self) -> Path:
        return self.logs / "reports"

    @property
    def harnesses(self) -> Path:
        return self.root / "harnesses"

    def test_results(self, dataset: str) -> Path:
        return self.root / "results" / dataset / "test"

    def mkdirs(self) -> None:
        for path in (self.logs, self.proposer_logs, self.evals, self.reports, self.harnesses):
            path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# evolution_summary.jsonl
# --------------------------------------------------------------------------

def append_row(paths: RunPaths, row: Dict[str, Any]) -> None:
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_rows(paths: RunPaths) -> List[Dict[str, Any]]:
    if not paths.summary.exists():
        return []
    rows = []
    for line in paths.summary.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def last_iteration(paths: RunPaths) -> int:
    return max((int(r.get("iteration", 0)) for r in read_rows(paths)), default=0)


def summary_row(
    *, iteration: int, system: str, metrics: Dict[str, Any],
    candidate: Optional[Dict[str, Any]] = None, best_score: float = 0.0,
) -> Dict[str, Any]:
    """One evolution_summary.jsonl row. `score` is on the same 0-1 scale as
    every other baseline's `accuracy_<dataset>`."""
    from evaluator import normalized_score

    score = normalized_score(metrics)
    candidate = candidate or {}
    row: Dict[str, Any] = {
        "iteration": iteration,
        "system": system,
        "score": round(score, 4),
        "delta": round(score - best_score, 4),
        "context_cost": round(float(metrics.get("memory_tokens_per_query", 0.0)), 1),
        "stage": metrics.get("stage"),
        "eliminated": bool(metrics.get("eliminated", False)),
        "tokens": metrics.get("tokens"),
        "hypothesis": candidate.get("hypothesis", ""),
        "axis": candidate.get("axis", ""),
        "base_system": candidate.get("base_system", ""),
    }
    if metrics.get("error"):
        row["error"] = str(metrics["error"])[:500]
    return row


# --------------------------------------------------------------------------
# frontier_val.json
# --------------------------------------------------------------------------

def _pareto_front(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Entries not dominated on (score up, context_cost down) — the paper
    evaluates candidates under Pareto dominance when accuracy and context cost
    both matter. Ties on both axes keep the first entry seen."""
    front = []
    for e in entries:
        dominated = any(
            o is not e
            and o["score"] >= e["score"]
            and o["context_cost"] <= e["context_cost"]
            and (o["score"] > e["score"] or o["context_cost"] < e["context_cost"])
            for o in entries
        )
        if not dominated:
            front.append(e)
    return sorted(front, key=lambda e: (-e["score"], e["context_cost"]))


def rebuild_frontier(paths: RunPaths) -> Dict[str, Any]:
    """Recompute frontier_val.json from the summary. The newest row wins when a
    system was evaluated more than once."""
    latest: Dict[str, Dict[str, Any]] = {}
    for row in read_rows(paths):
        if not row.get("eliminated"):
            latest[row["system"]] = row

    entries = [
        {"system": r["system"], "score": r["score"], "context_cost": r["context_cost"],
         "iteration": r.get("iteration", 0)}
        for r in latest.values()
    ]
    ranked = sorted(entries, key=lambda e: -e["score"])
    frontier = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "best": ranked[0] if ranked else None,
        "_pareto": _pareto_front(entries),
        "all": ranked,
    }
    paths.frontier.parent.mkdir(parents=True, exist_ok=True)
    paths.frontier.write_text(json.dumps(frontier, indent=2), encoding="utf-8")
    return frontier


def read_frontier(paths: RunPaths) -> Dict[str, Any]:
    if not paths.frontier.exists():
        return {}
    return json.loads(paths.frontier.read_text(encoding="utf-8"))


def best_score(paths: RunPaths) -> float:
    best = read_frontier(paths).get("best")
    return float(best["score"]) if best else 0.0


# --------------------------------------------------------------------------
# finalized.json — the test-split lock
# --------------------------------------------------------------------------

def is_finalized(paths: RunPaths) -> bool:
    if not paths.finalized.exists():
        return False
    return json.loads(paths.finalized.read_text(encoding="utf-8")).get("status") == "complete"


def mark_finalizing(paths: RunPaths, systems: List[str]) -> None:
    paths.finalized.parent.mkdir(parents=True, exist_ok=True)
    paths.finalized.write_text(json.dumps({
        "status": "in_progress",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "systems": sorted(systems),
    }, indent=2), encoding="utf-8")


def mark_finalized(paths: RunPaths, scores: Dict[str, float]) -> None:
    state = json.loads(paths.finalized.read_text(encoding="utf-8"))
    state["status"] = "complete"
    state["completed_at"] = datetime.now().isoformat(timespec="seconds")
    state["test_scores"] = {k: round(v, 4) for k, v in scores.items()}
    paths.finalized.write_text(json.dumps(state, indent=2), encoding="utf-8")
