"""LoCoMo environment — dataset-layer utilities.

Single-file dataset (`locomo10.json`, 10 multi-session conversation samples).
Mirrors the DynamicMem env interface so forge / BaseWorkflow can plug it in
without changes:

  - LoCoMoRecorder:      per-user recorder with `init = {"conversation": ..., "query": ...}`
  - get_task_list():     returns sample_id strings for the requested split
  - load_user_data():    parses locomo10.json → (conversation_dict, normalized_qa_pairs)

Only `sample["conversation"]` and `sample["qa"]` are used — `event_summary`,
`observation`, and `session_summary` are intentionally ignored.

Judging is provided by `common.metric.Judge` (used by BaseWorkflow's default
`judge()` method); this module no longer re-exports any judge symbol.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.recorder import Basic_Recorder


# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------

_data_env = os.environ.get("LOCOMO_DATA", "")
DATA_PATH: Path = (
    Path(_data_env) if _data_env else Path(__file__).resolve().parent / "locomo10.json"
)

TRAIN_SAMPLES = 6
EVAL_SAMPLES = 4

# LoCoMo category ids -> the names the papers report per category. cat-5
# (adversarial) is absent deliberately: it carries no gold answer and is
# excluded at load time below — the papers exclude it too.
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoRecorder(Basic_Recorder):
    """Records one sample's conversation ingestion + QA session.

    init:
      Phase 1 chunk:     {"conversation": <minimal conversation dict — only
                          speaker_*, session_N, session_N_date_time keys;
                          possibly a session-subset chunk>}
      Phase 2 retrieve:  {"conversation": <full conversation dict>, "query": str}

    steps: one entry per answered QA pair.
      {
        'query': str,
        'predicted': str,
        'reference': str,
        'score': int,           # 0/1 (LoCoMo paper-aligned binary judge)
        'judge_reason': str,
        'qa_metadata': {'category': int, 'evidence': List[str]},
        'retrieved_memory': Dict,
        'relevant_turns': List[Dict],   # turns the QA's evidence (dia_ids) points to
      }

    reward: mean(steps[i].score), in [0, 1].
    """

    init: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    user_id: str = field(default="")
    reward: float = field(default=0.0)
    failure_info: Optional[str] = field(default=None)

    async def log_init(self, conversation: Dict) -> None:
        """Phase 1: populate init with conversation subset (a chunk of sessions)."""
        async with self._lock:
            self.init = {"conversation": conversation}

    async def log_step(
        self,
        query: str,
        predicted: str,
        reference: str,
        score: int,
        judge_reason: str = "",
        qa_metadata: Dict = None,
        retrieved_memory: Dict = None,
        relevant_turns: List[Dict] = None,
        token_f1: Optional[float] = None,
        bleu1: Optional[float] = None,
    ) -> None:
        step: Dict[str, Any] = {
            "query": query,
            "retrieved_memory": retrieved_memory or {},
            "predicted": predicted,
            "reference": reference,
            "score": score,
            "judge_reason": judge_reason,
            "qa_metadata": qa_metadata or {},
            "relevant_turns": relevant_turns or [],
        }
        # Lexical metrics (reporting only, alongside the judge). Omitted rather
        # than stored as null when absent — aggregation treats a missing key as
        # "recompute from predicted/reference", which is also how it reads
        # traces written before these existed.
        if token_f1 is not None:
            step["token_f1"] = token_f1
        if bleu1 is not None:
            step["bleu1"] = bleu1
        async with self._lock:
            self.steps.append(step)

    async def set_reward(self, reward: float) -> None:
        async with self._lock:
            self.reward = reward


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_CACHE: Optional[List[Dict]] = None


def _load_all() -> List[Dict]:
    """Load locomo10.json once and cache in-process."""
    global _CACHE
    if _CACHE is None:
        with open(str(DATA_PATH), encoding="utf-8") as f:
            _CACHE = json.load(f)
    return _CACHE


def get_task_list(status: str, eval_n_samples: Optional[int], seed: Optional[str] = None) -> List[str]:
    """Return sample_id strings for the requested split.

    status='search' → first TRAIN_SAMPLES (6) samples, capped at eval_n_samples
    status='test'   → last EVAL_SAMPLES (4) samples (held-out), same cap

    Both splits honour the eval_n_samples cap (deterministic prefix ⇒
    staged nesting holds on the test split too — heldout single-stage sizing
    was silently void before 2026-07-08).
    """
    from common.sampling import shuffle_prefix
    all_samples = _load_all()
    sample_ids = [s["sample_id"] for s in all_samples]
    train_ids = sample_ids[:TRAIN_SAMPLES]
    eval_ids = sample_ids[TRAIN_SAMPLES:TRAIN_SAMPLES + EVAL_SAMPLES]
    pool = train_ids if status == "search" else eval_ids
    return shuffle_prefix(pool, eval_n_samples, seed)


def _find_sample(sample_id: str) -> Dict:
    for s in _load_all():
        if s["sample_id"] == sample_id:
            return s
    raise KeyError(f"LoCoMo sample not found: {sample_id}")


def load_user_data(
    user_dir: str, eval_n_qa: Optional[int] = None, sample_seed: Optional[str] = None
) -> Tuple[Dict, Dict, List[Dict]]:
    """Load one LoCoMo sample.

    Parameters
    ----------
    user_dir : str
        The sample_id (e.g., "conv-26"). Named `user_dir` to mirror the
        DynamicMem signature consumed by forge.BaseWorkflow.
    eval_n_qa : Optional[int]
        If not None, sample this many QAs deterministically.
    sample_seed : Optional[str]
        Per-step seed (stage_spec["sample_seed"]); combined with user_dir via
        common.sampling.combine_seed. None (default) → seeds on user_dir
        alone, exactly as before this parameter existed (back-compat).

    Returns
    -------
    (conversation, user_profile, qa_pairs) tuple, where:
      - conversation:  sample["conversation"] dict (raw). Only speaker_*,
                       session_N, session_N_date_time keys are present.
      - user_profile:  empty dict (returned for API parity with DynamicMem).
      - qa_pairs:      list of normalized {query, reference, metadata} dicts.

    We intentionally drop `event_summary`, `observation`, `session_summary`
    from the sample — they are not part of the pipeline inputs.
    """
    sample = _find_sample(user_dir)

    conversation = sample["conversation"]

    # Category-5 (adversarial) QAs are EXCLUDED (2026-07-08): 444/446 of them
    # carry no `answer` key at all (only `adversarial_answer`, the trap
    # option), so every prior run judged them against an empty gold — scores
    # were noise. Protocol decision: LoCoMo runs on categories 1-4 only.
    # NOTE this changes the QA pool the shuffle-then-prefix sampler sees, so
    # LoCoMo scores are NOT comparable with pre-2026-07-08 runs (which were
    # contaminated anyway).
    qa_src: List[Dict] = [
        q for q in sample.get("qa", []) if int(q.get("category", 0)) != 5
    ]
    qa_pairs: List[Dict] = []
    for q in qa_src:
        qa_pairs.append({
            "query": q.get("question", ""),
            "reference": str(q.get("answer", "")),
            "metadata": {
                "category": q.get("category", 0),
                "evidence": list(q.get("evidence", []) or []),
            },
        })

    if eval_n_qa is not None:
        # Shuffle-then-prefix (NOT rng.sample): guarantees the nesting
        # property staged evaluation depends on — the n=20 selection is
        # always the first 20 of the n=40 selection for the same sample.
        from common.sampling import combine_seed
        rng = random.Random(combine_seed(sample_seed, user_dir))
        shuffled = list(qa_pairs)
        rng.shuffle(shuffled)
        qa_pairs = shuffled[: min(eval_n_qa, len(shuffled))]

    return conversation, {}, qa_pairs


# ---------------------------------------------------------------------------
# Session extraction helper — shared by workflow for Phase 1 chunking.
# ---------------------------------------------------------------------------

def extract_sessions(conversation: Dict) -> List[Tuple[int, str, List[Dict]]]:
    """Return an ordered list of (session_idx, date_time, turns) tuples.

    Sorted by session_idx ascending.
    """
    sessions: List[Tuple[int, str, List[Dict]]] = []
    for key, value in conversation.items():
        if not key.startswith("session_"):
            continue
        if key.endswith("_date_time"):
            continue
        try:
            idx = int(key.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        date_time = conversation.get(f"{key}_date_time", "")
        if not isinstance(value, list):
            continue
        sessions.append((idx, date_time, value))
    sessions.sort(key=lambda t: t[0])
    return sessions


def _split_dia_ids(raw: str) -> List[str]:
    """LoCoMo sometimes packs multiple dia_ids into a single evidence string
    (e.g. "D8:6; D9:17"). Split on `;` / `,` and strip whitespace."""
    if not isinstance(raw, str):
        return []
    parts = raw.replace(",", ";").split(";")
    return [p.strip() for p in parts if p.strip()]


def lookup_turns_by_dia_ids(conversation: Dict, dia_ids: List[str]) -> List[Dict]:
    """Look up dialogue turns by their dia_id (format: 'D<sess>:<turn>').

    Handles both normalized single ids and LoCoMo's packed "D1:2; D3:4" form.
    """
    if not dia_ids:
        return []
    flat: List[str] = []
    for d in dia_ids:
        flat.extend(_split_dia_ids(d))

    index: Dict[str, Dict] = {}
    for _, _, turns in extract_sessions(conversation):
        for turn in turns:
            if isinstance(turn, dict) and "dia_id" in turn:
                index[turn["dia_id"]] = turn
    return [index[d] for d in flat if d in index]
