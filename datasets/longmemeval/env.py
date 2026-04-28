"""LongMemEval environment — dataset-layer utilities.

Supports two variants that share the same 500 questions but differ in
haystack density:

  - "s" (longmemeval_s_cleaned.json,  ~277 MB, ~48 sessions/sample)
  - "m" (longmemeval_m_cleaned.json,  ~2.7 GB, ~475 sessions/sample)

Each sample is a single (question, answer) pair with a "haystack" of chat
sessions to retrieve from. Forge's BaseWorkflow treats one sample as one
"user" and the single QA as one Phase-2 step.

Train/test split (deterministic, same question_ids across variants):
  * search  — 50 question_ids, stratified by question_type
                multi-session: 13, temporal-reasoning: 13,
                knowledge-update: 8, single-session-user: 7,
                single-session-assistant: 6, single-session-preference: 3
  * test    — remaining 450 question_ids

Judging is provided by `common.judge.Judge` (used by BaseWorkflow's default
`judge()` method); this module no longer re-exports any judge symbol.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.harness_base import Basic_Recorder


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_data_env = os.environ.get("LONGMEMEVAL_DATA", "")
_DATA_DIR: Path = Path(_data_env) if _data_env else Path(__file__).resolve().parent

DATA_PATHS: Dict[str, Path] = {
    "s": _DATA_DIR / "longmemeval_s_cleaned.json",
    "m": _DATA_DIR / "longmemeval_m_cleaned.json",
}

SEARCH_SIZE = 50
SEARCH_SPLIT_SEED = "longmemeval.search.v1"


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

@dataclass
class LongMemEvalRecorder(Basic_Recorder):
    """Records one sample's haystack ingestion + single QA answer.

    init:
      Phase 1 chunk:     {"sessions": List[session_dict]}
      Phase 2 retrieve:  {"sessions": List[session_dict], "query": str,
                          "question_date": str}

    Where each `session_dict` is:
      {"session_id": str, "date": str,
       "messages": List[{"role": "user"|"assistant", "content": str}]}

    steps: exactly one entry (LongMemEval has 1 QA per sample).
      {
        'query': str,
        'retrieved_memory': Dict,
        'predicted': str, 'reference': str,
        'score': int, 'judge_reason': str,
        'qa_metadata': {'question_type': str, 'question_date': str,
                        'answer_session_ids': List[str]},
        'relevant_sessions': List[session_dict],
      }
    """

    init: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    user_id: str = field(default="")
    reward: float = field(default=0.0)
    failure_info: Optional[str] = field(default=None)

    async def log_init(self, sessions: List[Dict]) -> None:
        """Phase 1 chunk ingest."""
        async with self._lock:
            self.init = {"sessions": sessions}

    async def log_step(
        self,
        query: str,
        predicted: str,
        reference: str,
        score: int,
        judge_reason: str = "",
        qa_metadata: Dict = None,
        retrieved_memory: Dict = None,
        relevant_sessions: List[Dict] = None,
    ) -> None:
        async with self._lock:
            self.steps.append(
                {
                    "query": query,
                    "retrieved_memory": retrieved_memory or {},
                    "predicted": predicted,
                    "reference": reference,
                    "score": score,
                    "judge_reason": judge_reason,
                    "qa_metadata": qa_metadata or {},
                    "relevant_sessions": relevant_sessions or [],
                }
            )

    async def set_reward(self, reward: float) -> None:
        async with self._lock:
            self.reward = reward


# ---------------------------------------------------------------------------
# Lazy variant loader
# ---------------------------------------------------------------------------

_CACHE: Dict[str, List[Dict]] = {}


def _load_variant(variant: str) -> List[Dict]:
    if variant not in DATA_PATHS:
        raise KeyError(f"Unknown LongMemEval variant '{variant}'; known: {sorted(DATA_PATHS)}")
    if variant not in _CACHE:
        with open(str(DATA_PATHS[variant]), encoding="utf-8") as f:
            _CACHE[variant] = json.load(f)
    return _CACHE[variant]


# ---------------------------------------------------------------------------
# Stratified search/test split — variant-agnostic (same 500 qids across s/m)
# ---------------------------------------------------------------------------

_SPLIT_CACHE: Optional[Tuple[List[str], List[str]]] = None


def _proportional_counts(
    by_type: Dict[str, List[str]], target: int
) -> Dict[str, int]:
    """Split `target` into per-type integer counts proportional to |ids|."""
    total = sum(len(ids) for ids in by_type.values())
    raw = {t: len(ids) * target / total for t, ids in by_type.items()}
    counts = {t: int(raw[t]) for t in by_type}  # floor
    remainder = target - sum(counts.values())
    if remainder > 0:
        # Distribute remainder to types with largest fractional parts
        fracs = sorted(
            ((raw[t] - counts[t], t) for t in by_type),
            key=lambda x: (-x[0], x[1]),  # deterministic tie-break
        )
        for _, t in fracs[:remainder]:
            counts[t] += 1
    return counts


def _compute_split() -> Tuple[List[str], List[str]]:
    """Returns (search_qids, test_qids). Deterministic, cached.

    Works off the "s" variant (cheapest to load) — question_ids are identical
    across variants.
    """
    global _SPLIT_CACHE
    if _SPLIT_CACHE is not None:
        return _SPLIT_CACHE

    samples = _load_variant("s")
    by_type: Dict[str, List[str]] = {}
    for s in samples:
        by_type.setdefault(s["question_type"], []).append(s["question_id"])

    # Deterministic per-type order
    for t in by_type:
        by_type[t] = sorted(by_type[t])

    counts = _proportional_counts(by_type, SEARCH_SIZE)

    rng = random.Random(SEARCH_SPLIT_SEED)
    search_qids: List[str] = []
    test_qids: List[str] = []
    for t in sorted(by_type):
        ids = by_type[t][:]
        rng.shuffle(ids)
        search_qids.extend(ids[:counts[t]])
        test_qids.extend(ids[counts[t]:])

    _SPLIT_CACHE = (sorted(search_qids), sorted(test_qids))
    return _SPLIT_CACHE


def get_task_list(status: str, eval_n_samples: int) -> List[str]:
    """Return question_id strings for the requested split, capped at eval_n_samples."""
    search_qids, test_qids = _compute_split()
    pool = search_qids if status == "search" else test_qids
    return pool[:int(eval_n_samples)]


# ---------------------------------------------------------------------------
# Per-sample loading
# ---------------------------------------------------------------------------

def _find_sample(samples: List[Dict], question_id: str) -> Dict:
    for s in samples:
        if s.get("question_id") == question_id:
            return s
    raise KeyError(f"LongMemEval question_id not found: {question_id}")


def _build_sessions(sample: Dict) -> List[Dict]:
    """Convert parallel lists into a list of session dicts."""
    out: List[Dict] = []
    for sid, date, msgs in zip(
        sample.get("haystack_session_ids", []),
        sample.get("haystack_dates", []),
        sample.get("haystack_sessions", []),
    ):
        out.append({
            "session_id": sid,
            "date": date,
            "messages": msgs,
        })
    return out


def load_user_data(
    user_dir: str,
    eval_n_qa: Optional[int] = None,
    *,
    variant: str = "s",
) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Load one LongMemEval sample for the given variant.

    Parameters
    ----------
    user_dir : str
        The question_id (e.g., "e47becba"). Named `user_dir` to match the
        DynamicMem/LoCoMo signature consumed by forge.BaseWorkflow.
    eval_n_qa : Optional[int]
        Ignored — LongMemEval has 1 QA per sample. Kept for API parity.
    variant : str
        "s" or "m" — which haystack density to use.

    Returns
    -------
    (sessions, user_profile, qa_pairs)
      - sessions:     List[session_dict]. Each session_dict has session_id,
                      date, messages.
      - user_profile: empty dict (API parity).
      - qa_pairs:     list with exactly one normalized QA dict.
    """
    samples = _load_variant(variant)
    sample = _find_sample(samples, user_dir)

    sessions = _build_sessions(sample)

    qa_pairs = [{
        "query": sample.get("question", ""),
        "reference": str(sample.get("answer", "")),
        "metadata": {
            "question_type": sample.get("question_type", ""),
            "question_date": sample.get("question_date", ""),
            "answer_session_ids": list(sample.get("answer_session_ids", []) or []),
        },
    }]
    # eval_n_qa is a no-op for LongMemEval (one QA per haystack) but we honour
    # "0 means empty" to stay predictable.
    if eval_n_qa is not None and eval_n_qa <= 0:
        qa_pairs = []

    return sessions, {}, qa_pairs
