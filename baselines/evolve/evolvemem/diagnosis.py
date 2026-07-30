"""EvolveMem failure diagnosis (§3.3) — reads a finished run's per-question
raw logs and asks an LLM to propose a structured configuration adjustment Δθ.

The rubric is phrased in failure PATTERNS, not benchmarks, so proposals stay
valid when the diagnosis LLM invents new dimensions (paper: "the rubric is
written in terms of failure patterns rather than specific benchmarks").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import get_logger

log = get_logger("main")

# Evidence field name per dataset in the saved traces (kept local — the
# baselines convention is copy, not import across evolve/ methods).
EVIDENCE_KEY = {
    "dynamicmem": "relevant_app_logs",
    "locomo": "relevant_turns",
    "longmemeval_s": "relevant_sessions",
    "longmemeval_m": "relevant_sessions",
}

DIAGNOSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "evidence": {"type": "string"},
                    "affected_fraction": {"type": "number"},
                },
                "required": ["category", "evidence"],
            },
        },
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "new_value": {},
                    "rationale": {"type": "string"},
                },
                "required": ["parameter", "new_value", "rationale"],
            },
        },
        "per_category": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "pattern": {"type": "string"},
                    "overrides": {"type": "object"},
                },
                "required": ["name", "pattern", "overrides"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["root_causes", "adjustments", "summary"],
}

_RUBRIC = """\
Common failure patterns to check for (diagnose from evidence, not intuition):
- WRONG-ENTITY RETRIEVAL: retrieved memories are about a different person/app/topic
  than the question. Levers: k_str / w_str up, entity_swap toggle, per-category override.
- INSUFFICIENT CONTEXT: gold evidence absent from retrieved memories entirely.
  Levers: k_sem / k_kw / b_ctx up, fusion_mode change, query_expand_keywords extra.
- TEMPORAL CONFUSION: answer cites the wrong time period, or "most recent X"
  resolves to a stale memory. Levers: lambda_recency, recency_halflife_days,
  per-category override matched on temporal phrasing.
- MULTI-HOP MISS: question needs composing several memories; retrieval surfaced
  only one hop. Levers: query_decomposition on, b_ctx up.
- NOISE DILUTION: gold evidence retrieved but buried under irrelevant memories.
  Levers: b_ctx down, k_* down, dedup_tau down, importance/recency weights.
- STORE-QUALITY GAP: memories are raw/uninformative so no retrieval setting can
  work. Levers: extraction_mode="llm", extraction_window, unit_granularity.
- ANSWER-STYLE MISMATCH: right evidence, wrong answer form (verbose, hedged,
  missing inference). Levers: answer_style.
"""


def build_failure_log(
    run_dir: Path,
    dataset: str,
    max_examples: int = 14,
    content_cap: int = 500,
) -> Dict[str, Any]:
    """Assemble the per-question raw log the diagnosis prompt consumes.

    Prioritizes the lowest-scoring steps across ALL users (the paper logs
    every question; we cap for prompt budget, worst-first, plus a couple of
    high-scoring contrast examples)."""
    run_dir = Path(run_dir)
    evidence_key = EVIDENCE_KEY.get(dataset, "relevant_context")

    with (run_dir / "score.json").open(encoding="utf-8") as f:
        score = json.load(f)

    steps: List[Dict[str, Any]] = []
    traces_dir = run_dir / "traces"
    if traces_dir.is_dir():
        for p in sorted(traces_dir.glob("*.json")):
            try:
                with p.open(encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                log.warning(f"diagnosis: unreadable trace {p}: {exc}")
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
    examples = [{
        "user_id": s.get("user_id", ""),
        "query": _trim(s.get("query", "")),
        "retrieved_memory": _trim(s.get("retrieved_memory", {})),
        "predicted": _trim(s.get("predicted", "")),
        "reference": _trim(s.get("reference", "")),
        "score": s.get("score", 0.0),
        "judge_reason": _trim(s.get("judge_reason", "")),
        "gold_evidence": _trim(s.get(evidence_key, [])),
    } for s in picked]

    return {
        "overall": score.get("benchmark_eval_score", {}),
        "invalid_users": score.get("invalid_users", []),
        "n_steps_total": len(steps),
        "examples": examples,
    }


async def diagnose(
    failure_log: Dict[str, Any],
    current_config: Dict[str, Any],
    round_history: List[Dict[str, Any]],
    dataset: str,
    meta_model: str = "gpt-5",
    space=None,
    max_adjustments: Optional[int] = None,
) -> Dict[str, Any]:
    """One diagnosis call → validated proposal dict (schema above).

    `space` — an action-space module exposing `space_description()` and
    `RUBRIC` (e.g. action_space_simplemem for the real-substrate θ).
    None keeps the native EvolveMemMemo space (backward compatible)."""
    from common.llm import Agent

    if space is not None:
        space_desc = space.space_description()
        rubric = space.RUBRIC
    else:
        from baselines.evolve.evolvemem.action_space import (
            BOUNDS, DEFAULT_CONFIG, ENUM_KEYS, EXTRA_HOOKS,
        )
        space_desc = {
            "dimensions": {k: v for k, v in DEFAULT_CONFIG.items() if k not in ("per_category", "extras")},
            "numeric_bounds": {k: list(v) for k, v in BOUNDS.items()},
            "enums": {k: list(v) for k, v in ENUM_KEYS.items()},
            "extras_hooks_implemented": EXTRA_HOOKS,
        }
        rubric = _RUBRIC
    history_view = [
        {"round": h["round"], "score": h["score"], "action": h["action"],
         "summary": h.get("diagnosis_summary", "")}
        for h in round_history[-6:]
    ]

    param_rules = (
        "- `parameter` must name a dimension of the action space, an "
        "`extras.<hook>` key from extras_hooks_implemented, or a genuinely "
        "new `extras.<name>` dimension (state its intended semantics in the "
        "rationale; unimplemented ones are recorded but inert).\n"
        "- Use `per_category` (regex on the question text) when a failure "
        "pattern is category-specific rather than global.\n"
    ) if space is None else (
        "- `parameter` must name a dimension of the action space exactly — "
        "this substrate has NO extras hooks and NO per_category overrides "
        "(leave `per_category` absent/empty).\n"
    )
    system = (
        "You are the diagnosis module of EvolveMem, a self-evolving memory "
        "system. You read per-question failure logs from the last evaluation "
        "round and propose a targeted retrieval-configuration adjustment Δθ.\n\n"
        + rubric +
        "\nRules:\n"
        "- Ground every root cause in specific examples (quote the query).\n"
        + (f"- Propose AT MOST {max_adjustments} adjustments, ORDERED by "
           f"expected impact (highest first) — only the top "
           f"{max_adjustments} will be applied this round.\n"
           if max_adjustments else
           "- Propose FEW, TARGETED adjustments (1-4), not a shotgun rewrite.\n")
        + param_rules +
        "- Avoid re-proposing an adjustment that the history shows was "
        "already tried and reverted."
    )
    user = (
        f"Dataset: {dataset}\n\n"
        f"Action space:\n{json.dumps(space_desc, indent=1, ensure_ascii=False)}\n\n"
        f"Current configuration θ_r:\n{json.dumps(current_config, indent=1, ensure_ascii=False)}\n\n"
        f"Round history:\n{json.dumps(history_view, indent=1, ensure_ascii=False)}\n\n"
        f"Failure log:\n{json.dumps(failure_log, indent=1, ensure_ascii=False)}"
    )

    agent = Agent(system_prompt=system, output_schema=DIAGNOSIS_SCHEMA,
                  model=meta_model, timeout=900)
    proposal = await agent.ask(user, reasoning_effort="medium")
    log.info(f"diagnosis: {proposal.get('summary', '')[:160]}")
    return proposal
