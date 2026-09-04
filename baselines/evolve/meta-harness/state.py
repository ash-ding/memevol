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
    proposer_usage.jsonl      what the SEARCH cost, one row per session
    evals/<system>/           per-candidate eval artifacts (score, traces)
    harnesses/                this run's harnesses (baselines + candidates)
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
        """This run's harnesses — baselines seeded in at start, candidates
        written here. PER RUN on purpose: a shared directory lets one run see
        (and `--fresh` delete) another run's candidates, and makes a name
        collision across runs possible while each run's uniqueness check only
        sees its own summary."""
        return self.logs / "harnesses"

    @property
    def seed_harnesses(self) -> Path:
        """The tracked baseline harnesses, copied into each run at start."""
        return self.root / "harnesses"

    @property
    def proposer_usage(self) -> Path:
        return self.logs / "proposer_usage.jsonl"

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
    # Ties on score are broken by the cheaper system: with two equal scores the
    # one that spends fewer tokens per query is strictly the better harness.
    ranked = sorted(entries, key=lambda e: (-e["score"], e["context_cost"]))
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


# --------------------------------------------------------------------------
# proposer_usage.jsonl — what the SEARCH itself cost
# --------------------------------------------------------------------------
# The proposer is a coding-agent CLI, so its tokens never pass through
# `common.llm` and cannot reach `common.tokens` (forge's containerized proposer
# has the same boundary). Both CLIs do report usage on their own event stream,
# though, so the numbers exist — this is where they are kept, per session, next
# to the evaluation costs they should be read beside. The paper puts one
# iteration at roughly 10 MTok of proposer context (Table 1), which makes this
# the larger half of a run's cost.

_TOKEN_FIELDS = (
    "input_tokens", "output_tokens",
    "cached_input_tokens", "reasoning_output_tokens",        # codex
    "cache_creation_input_tokens", "cache_read_input_tokens",  # claude
)


def record_proposer_usage(paths: RunPaths, iteration: int, agent: str,
                          model: Optional[str], result: Any) -> Dict[str, Any]:
    """Append one session's usage. Returns the row."""
    usage = result.usage or {}
    row: Dict[str, Any] = {
        "iteration": iteration,
        "agent": agent,
        "model": model or "(cli default)",
        "exit_code": result.exit_code,
        "duration_s": round(result.duration_s, 1),
        "tool_calls": len(result.tools),
        "cost_usd": round(result.cost_usd, 4),
    }
    row.update({f: int(usage.get(f, 0) or 0) for f in _TOKEN_FIELDS})
    paths.proposer_usage.parent.mkdir(parents=True, exist_ok=True)
    with paths.proposer_usage.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_proposer_usage(paths: RunPaths) -> List[Dict[str, Any]]:
    if not paths.proposer_usage.exists():
        return []
    rows = []
    for line in paths.proposer_usage.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def proposer_usage_total(paths: RunPaths) -> Dict[str, Any]:
    """Run totals. `input_tokens` already includes the cached reads both CLIs
    report separately, so cached counts are carried for context, never added."""
    rows = read_proposer_usage(paths)
    total: Dict[str, Any] = {"sessions": len(rows), "cost_usd": 0.0, "duration_s": 0.0}
    total.update({f: 0 for f in _TOKEN_FIELDS})
    for row in rows:
        total["cost_usd"] += float(row.get("cost_usd", 0.0))
        total["duration_s"] += float(row.get("duration_s", 0.0))
        for f in _TOKEN_FIELDS:
            total[f] += int(row.get(f, 0))
    total["cost_usd"] = round(total["cost_usd"], 4)
    total["duration_s"] = round(total["duration_s"], 1)
    return total
