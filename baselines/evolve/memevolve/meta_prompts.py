"""Prompt builders for MemEvolve's meta-evolution operator F (paper §4.2):
DIAGNOSE (trajectory evidence → structured defect profile over the four
components) and DESIGN (defect profile → S constrained operator redesigns).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from baselines.evolve.memevolve.design_space import (
    OPERATOR_SIGNATURES,
    SKELETON_HEADER,
)

DEFECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "component": {"type": "string",
                                  "enum": ["encode", "store", "retrieve", "manage"]},
                    "defect": {"type": "string"},
                    "evidence": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["component", "defect", "evidence", "severity"],
            },
        },
        "resource_notes": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["defects", "summary"],
}

DESIGN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "design_rationale": {"type": "string"},
        "encode": {"type": "string"},
        "store": {"type": "string"},
        "retrieve": {"type": "string"},
        "manage": {"type": "string"},
    },
    "required": ["design_rationale", "encode", "store", "retrieve", "manage"],
}

_CONTRACT = f"""\
Design-space contract (immutable — you may ONLY rewrite the four operator
bodies):

  {OPERATOR_SIGNATURES['encode']}      E: items → memory units
      `items`: list of {{"content": str, "ts": float|None, "source_id": str}}
      (benchmark-normalized by the fixed adapter). Return a list of unit
      dicts of YOUR chosen shape — store/retrieve must agree with it.
  {OPERATOR_SIGNATURES['store']}       U: integrate units into `state`
  {OPERATOR_SIGNATURES['retrieve']}    R: query str → dict handed VERBATIM to
      the QA agent (keep it clean/structured; a "memories" list works well).
      MUST be read-only w.r.t. `state` (checkpoint isolation).
  {OPERATOR_SIGNATURES['manage']}      G: periodic maintenance after each
      build call (consolidation, forgetting, reorganization).

`state` is a plain dict your operators own entirely (no other code touches
it). All four are async; LLM calls are allowed via
`from common.llm import Agent` (model "gpt-5-mini", `await agent.ask(msg)`)
and embeddings via `from common.llm import Embedding`
(`await Embedding(model="text-embedding-3-small").get_batch_embeddings(texts)`).
Mind token cost and latency — they are selection objectives alongside score.

Toolkit available at module level inside the assembled file (do not
re-implement): `_tokenize(text)`, `_parse_ts(raw)`, `_BM25` (add/scores),
plus stdlib `math`, `re`, `json`, `datetime`, `timezone`, `Counter`,
`defaultdict`.

Each operator must be a complete top-level `async def` with EXACTLY the
signature shown. No code outside the four function definitions (module
imports inside function bodies are fine).
"""


def build_diagnose_prompt(
    feedback: Dict[str, Any],
    operators: Dict[str, str],
    failure_log: Dict[str, Any],
) -> Tuple[str, str]:
    system = (
        "You are the diagnosis phase of MemEvolve's meta-evolution operator. "
        "Given a memory architecture's four operator implementations and "
        "trajectory-level evidence from its own execution batch, produce a "
        "structured defect profile: which component (encode/store/retrieve/"
        "manage) is the bottleneck, with concrete evidence. Typical defects: "
        "retrieval failures (gold evidence never surfaced), ineffective "
        "abstractions (units too raw or too lossy), storage inefficiencies "
        "(unbounded growth, duplicates), missing maintenance (no forgetting/"
        "consolidation), and resource waste (token cost or latency out of "
        "proportion to score). Compare gold_evidence against retrieved_memory "
        "in the examples to localize the failure."
    )
    ops_view = "\n\n".join(f"### {name}\n```python\n{src.strip()}\n```"
                           for name, src in operators.items())
    user = (
        f"Inner-loop feedback F = (perf, cost, delay):\n"
        f"{json.dumps(feedback, indent=1, ensure_ascii=False)}\n\n"
        f"Operator implementations:\n{ops_view}\n\n"
        f"Execution evidence (worst-first sampled QA steps + contrast):\n"
        f"{json.dumps(failure_log, indent=1, ensure_ascii=False)}"
    )
    return system, user


def build_design_prompt(
    operators: Dict[str, str],
    defect_profile: Dict[str, Any],
    variant_index: int,
    n_variants: int,
    sibling_rationales: List[str],
) -> Tuple[str, str]:
    system = (
        "You are the design phase of MemEvolve's meta-evolution operator. "
        "Conditioned on a parent architecture and its defect profile, produce "
        "ONE redesigned architecture by rewriting the four operators within "
        "the immutable design-space contract below. Fix the diagnosed "
        "high-severity defects first; keep what demonstrably works. Variants "
        "must be genuinely distinct: you are variant "
        f"{variant_index + 1} of {n_variants} — take a different architectural "
        "route than the sibling rationales listed (if any).\n\n" + _CONTRACT +
        "\nReturn the full source of all four operators (rewritten or kept) "
        "plus a short design_rationale."
    )
    ops_view = "\n\n".join(f"### {name}\n```python\n{src.strip()}\n```"
                           for name, src in operators.items())
    user = (
        f"Parent operators:\n{ops_view}\n\n"
        f"Defect profile:\n{json.dumps(defect_profile, indent=1, ensure_ascii=False)}\n\n"
        f"Sibling variant rationales to diverge from:\n"
        f"{json.dumps(sibling_rationales, ensure_ascii=False)}"
    )
    return system, user


def build_repair_prompt(
    operators: Dict[str, str],
    error_msg: str,
) -> Tuple[str, str]:
    system = (
        "A designed MemEvolve architecture failed its sanity check. Repair "
        "the four operators so the error cannot recur, changing as little of "
        "the design as possible.\n\n" + _CONTRACT
    )
    ops_view = "\n\n".join(f"### {name}\n```python\n{src.strip()}\n```"
                           for name, src in operators.items())
    user = f"Operators:\n{ops_view}\n\nError:\n{error_msg[:4000]}"
    return system, user


# Reference note for the design prompt author: SKELETON_HEADER is what the
# operators are concatenated onto — kept importable here for tooling/tests.
_ = SKELETON_HEADER
