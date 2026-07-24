"""MemEvolve's meta-evolution operator F (paper §4.2): architectural
selection (Pareto non-dominated sort over F = (perf, −cost, −delay), Perf
tiebreak, top-K parents) + diagnose-and-design evolution (defect profile →
S constrained variants per parent, each sanity-checked with repair retries).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.logger import get_logger

from baselines.evolve.memevolve.design_space import validate_operators
from baselines.evolve.memevolve.genotype_manager import GenotypeArchive
from baselines.evolve.memevolve.meta_prompts import (
    DEFECT_SCHEMA,
    DESIGN_SCHEMA,
    build_design_prompt,
    build_diagnose_prompt,
    build_repair_prompt,
)

log = get_logger("main")

# Evidence field name per dataset in the saved traces.
EVIDENCE_KEY = {
    "dynamicmem": "relevant_app_logs",
    "locomo": "relevant_turns",
    "longmemeval_s": "relevant_sessions",
    "longmemeval_m": "relevant_sessions",
}


# ---------------------------------------------------------------------------
# Architectural selection
# ---------------------------------------------------------------------------

def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """a dominates b on (perf↑, cost↓, delay↓)."""
    ge = (a["perf"] >= b["perf"] and a["cost"] <= b["cost"] and a["delay"] <= b["delay"])
    gt = (a["perf"] > b["perf"] or a["cost"] < b["cost"] or a["delay"] < b["delay"])
    return ge and gt


def pareto_rank(candidates: List[Dict[str, Any]]) -> List[int]:
    """Non-dominated sorting rank per candidate (0 = first front)."""
    n = len(candidates)
    ranks = [0] * n
    remaining = set(range(n))
    front = 0
    while remaining:
        current = {i for i in remaining
                   if not any(_dominates(candidates[j], candidates[i])
                              for j in remaining if j != i)}
        if not current:          # numeric ties can stall the loop — flush
            current = set(remaining)
        for i in current:
            ranks[i] = front
        remaining -= current
        front += 1
    return ranks


def select_parents(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Top-K by (Pareto rank asc, perf desc)."""
    ranks = pareto_rank(candidates)
    order = sorted(range(len(candidates)),
                   key=lambda i: (ranks[i], -candidates[i]["perf"]))
    chosen = [candidates[i] for i in order[:top_k]]
    log.info("selection: " + ", ".join(
        f"{c['sha']}(perf={c['perf']:.3f},cost={c['cost']:.0f},delay={c['delay']:.0f}s,rank={ranks[order[j]]})"
        for j, c in enumerate(chosen)))
    return chosen


# ---------------------------------------------------------------------------
# Diagnosis evidence
# ---------------------------------------------------------------------------

def build_failure_log(
    run_dir: Path,
    dataset: str,
    max_examples: int = 10,
    content_cap: int = 450,
) -> Dict[str, Any]:
    """Worst-first sampled QA steps + contrast from a candidate's own
    execution batch (the paper's replay interface, trace-file flavored)."""
    run_dir = Path(run_dir)
    evidence_key = EVIDENCE_KEY.get(dataset, "relevant_context")

    steps: List[Dict[str, Any]] = []
    traces_dir = run_dir / "traces"
    if traces_dir.is_dir():
        for p in sorted(traces_dir.glob("*.json")):
            try:
                with p.open(encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                log.warning(f"failure_log: unreadable trace {p}: {exc}")
                continue
            uid = payload.get("user_id", p.stem)
            for step in payload.get("steps", []) or []:
                step["user_id"] = uid
                steps.append(step)

    def _trim(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
        return text[:content_cap]

    ordered = sorted(steps, key=lambda s: s.get("score", 0.0))
    n_fail = max(1, max_examples - 2)
    picked = ordered[:n_fail] + ordered[-2:] if len(ordered) > n_fail else ordered
    return {
        "n_steps_total": len(steps),
        "examples": [{
            "user_id": s.get("user_id", ""),
            "query": _trim(s.get("query", "")),
            "retrieved_memory": _trim(s.get("retrieved_memory", {})),
            "predicted": _trim(s.get("predicted", "")),
            "reference": _trim(s.get("reference", "")),
            "score": s.get("score", 0.0),
            "judge_reason": _trim(s.get("judge_reason", "")),
            "gold_evidence": _trim(s.get(evidence_key, [])),
        } for s in picked],
    }


# ---------------------------------------------------------------------------
# Diagnose-and-design
# ---------------------------------------------------------------------------

async def diagnose(
    operators: Dict[str, str],
    feedback: Dict[str, Any],
    failure_log: Dict[str, Any],
    meta_model: str = "gpt-5",
) -> Dict[str, Any]:
    from common.llm import Agent
    system, user = build_diagnose_prompt(
        feedback={k: feedback[k] for k in ("perf", "cost", "delay", "invalid_users")},
        operators=operators,
        failure_log=failure_log,
    )
    agent = Agent(system_prompt=system, output_schema=DEFECT_SCHEMA,
                  model=meta_model, timeout=900)
    profile = await agent.ask(user, reasoning_effort="medium")
    log.info(f"diagnosis: {profile.get('summary', '')[:160]}")
    return profile


async def design_variant(
    parent_operators: Dict[str, str],
    defect_profile: Dict[str, Any],
    variant_index: int,
    n_variants: int,
    sibling_rationales: List[str],
    meta_model: str = "gpt-5",
) -> Tuple[Dict[str, str], str]:
    """One Design(Ω, D, s) call → (operators, rationale). Raises on invalid
    output after Agent's own retry ladder; caller decides how to proceed."""
    from common.llm import Agent
    system, user = build_design_prompt(
        parent_operators, defect_profile, variant_index, n_variants, sibling_rationales)
    agent = Agent(system_prompt=system, output_schema=DESIGN_SCHEMA,
                  model=meta_model, timeout=1200)
    result = await agent.ask(user, reasoning_effort="medium")
    operators = {name: _strip_fence(result[name])
                 for name in ("encode", "store", "retrieve", "manage")}
    validate_operators(operators)
    return operators, str(result.get("design_rationale", ""))


def _strip_fence(src: str) -> str:
    """LLMs sometimes wrap code in ``` fences despite the JSON contract."""
    import re
    m = re.search(r"```(?:python)?\n?(.*?)```", src, re.DOTALL)
    return (m.group(1) if m else src).strip()


async def sanity_check_with_repair(
    operators: Dict[str, str],
    archive: GenotypeArchive,
    dataset: str,
    meta: Dict[str, Any],
    run_check,                    # async (sha, module_path) -> feedback dict
    meta_model: str = "gpt-5",
    max_repairs: int = 2,
    max_check_wall_s: Optional[float] = None,
) -> Optional[str]:
    """Archive + sanity-gate a designed genotype; on failure, ask the LLM to
    repair and retry (alma's examine loop, design-space flavored). Returns
    the surviving sha, or None if the variant never passes.

    `max_check_wall_s` — COST GUARD (operational): a check pass whose
    wall-clock (feedback["delay"]) exceeds this budget fails the gate with
    an actionable message, so LLM-heavy Phase-1 designs are repaired into
    cheaper ones BEFORE the full inner-loop eval pays for them."""
    from common.llm import Agent

    current = dict(operators)
    for attempt in range(max_repairs + 1):
        try:
            validate_operators(current)
            sha = archive.save(current, {**meta, "repair_attempts": attempt})
        except ValueError as exc:
            err = f"static validation failed: {exc}"
            log.warning(f"sanity[{attempt}]: {err}")
            sha = None
        else:
            feedback = await run_check(sha, archive.assembled_path(sha))
            bad = feedback["invalid_users"] or any(
                (u.get("failure_info") or "") for u in feedback["per_user"].values())
            too_slow = (max_check_wall_s is not None
                        and feedback.get("delay", 0.0) > max_check_wall_s)
            if not bad and not too_slow:
                return sha
            if too_slow and not bad:
                err = (
                    f"COST GUARD TRIPPED: the sanity check took "
                    f"{feedback['delay']:.0f}s wall-clock (budget: "
                    f"{max_check_wall_s:.0f}s) for {len(feedback['per_user'])} "
                    f"users. encode/store are far too slow/expensive — usually "
                    f"per-item or per-session LLM calls during ingestion. "
                    f"Redesign so ingestion uses NO per-item LLM calls (batch "
                    f"or purely programmatic indexing); keep quality via "
                    f"indexing + query-time selection instead."
                )
            else:
                err = json.dumps(feedback["invalid_users"][:3]
                                 or [u.get("failure_info") for u in feedback["per_user"].values()
                                     if u.get("failure_info")][:3],
                                 ensure_ascii=False, default=str)
            log.warning(f"sanity[{attempt}] failed for {sha}: {err[:300]}")

        if attempt == max_repairs:
            break
        system, user = build_repair_prompt(current, err)
        agent = Agent(system_prompt=system, output_schema=DESIGN_SCHEMA,
                      model=meta_model, timeout=1200)
        try:
            result = await agent.ask(user)
            current = {name: _strip_fence(result[name])
                       for name in ("encode", "store", "retrieve", "manage")}
        except Exception as exc:
            log.warning(f"repair LLM failed on attempt {attempt}: {exc}")

    log.warning("variant dropped after exhausting repair attempts")
    return None
