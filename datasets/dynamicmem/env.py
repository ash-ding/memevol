"""
DynamicMem environment — dataset-layer utilities shared by all methods
(alma, cc, hipporag2, future meta-harness).

Kept deliberately free of any method-layer dependencies: this module imports
only from the standard library, openai, and optionally jsonschema. Method
packages may inject a token tracker at runtime via `set_token_tracker`.

Provides:
  - Basic_Recorder:       minimal async recorder base class
  - DynamicMemRecorder:   records one user's app-log ingestion + QA session
  - get_task_list():      user directory paths for a split
  - load_user_data():     app_logs, user_profile, sampled QA pairs for a user
  - judge_answer():       LLM judge that scores a predicted answer (raw openai, no Agent dep)
  - set_token_tracker():  inject a tracker (from a method package) to record judge usage

Design note: the interaction protocol is two-phase (per user):
  Phase 1  — general_update is called N times with app_log chunks to build a user profile
  Phase 2  — general_retrieve is called once per QA question; agent answers using the profile
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("main")

try:
    import httpx
    import openai
    from openai import AsyncOpenAI
except ImportError:  # openai is a hard dependency of the project, but keep import guarded
    openai = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


# OpenAI rejects `reasoning_effort` for non-reasoning models. Keep a
# narrow allowlist by name prefix; add new model families as they appear.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)

# ---------------------------------------------------------------------------
# Optional token-tracker injection (set by the calling method package)
# ---------------------------------------------------------------------------

_token_tracker: Optional[Any] = None


def set_token_tracker(tracker: Any) -> None:
    """Inject a TokenTracker-like object. The tracker must expose an
    async update(model_name: str, usage) coroutine. Pass None to disable.
    """
    global _token_tracker
    _token_tracker = tracker


# ---------------------------------------------------------------------------
# Data-path resolution
#
# Data (user_data/, user_qa/) sits directly under this package directory:
#     datasets/dynamicmem/user_data/
#     datasets/dynamicmem/user_qa/
# Override with the DYNAMICMEM_DATA env var if you want to point elsewhere.
# ---------------------------------------------------------------------------

_data_env = os.environ.get("DYNAMICMEM_DATA", "")
DATA_DIR: Path = Path(_data_env) if _data_env else Path(__file__).resolve().parent

TRAIN_USERS = 6
EVAL_USERS = 4


# ---------------------------------------------------------------------------
# Recorder base + domain-specific recorder
# ---------------------------------------------------------------------------

@dataclass
class Basic_Recorder:
    """Base class for task recorders. Domain-specific recorders inherit from this."""

    init: Dict[str, Any] = field(default_factory=dict)
    steps: list = field(default_factory=list)
    reward: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def log_init(self, *args, **kwargs):
        async with self._lock:
            self.init = kwargs

    async def log_step(self, **kwargs):
        async with self._lock:
            self.steps.append(kwargs)

    async def set_reward(self, reward: float):
        async with self._lock:
            self.reward = reward


@dataclass
class DynamicMemRecorder(Basic_Recorder):
    """Records one user's app-log ingestion and QA session.

    init:   loaded once in load_user_data / set by workflow.
        {
            'app_logs':     List[dict],   # entries from app_log_large.json
            'query':        str,          # injected before each general_retrieve call
        }
        Each app_log entry: app_log_id, timestamp, app_name, api_name, request, response.

    steps:  one entry per answered QA pair.
        {
            'query':       str,
            'predicted':   str,
            'reference':   str,
            'score':       int,   # 0–10 (LLM judge)
            'judge_reason': str,  # brief explanation from the judge
            'qa_metadata': {'domain': str, 'belonged': str, 'app_log_ids': List[str]}
        }

    reward: mean(steps[i].score), in [0, 10].

    Each step stores its own retrieved_memory (per-QA, not per-recorder).
    """

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "App logs loaded at session start. "
                "'app_logs' is a list of app-log dicts (app_log_large.json). "
                "'query' is injected before each general_retrieve call."
            ),
            "type": "Dict[str, Any]",
            "example": {
                "app_logs": [
                    {
                        "app_log_id": "log_00001",
                        "timestamp": "2023-10-01 06:30:00",
                        "app_name": "Fitbit",
                        "api_name": "RecordActivity",
                        "request": {"activity_type": "walking", "duration_minutes": 60},
                        "response": {"calories_burned": 312},
                    }
                ],
                "query": "When does the user usually exercise?",
            },
        },
    )

    steps: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "description": "One entry per answered QA pair.",
            "type": "List[Dict[str, Any]]",
            "example": [
                {
                    "query": "When does the user usually attend weekly briefings?",
                    "retrieved_memory": {"habits": "User attends Tuesday 10:00 AM briefings."},
                    "predicted": "Every Tuesday at 10:00 AM.",
                    "reference": "Every Tuesday morning at 10:00 the user attended weekly briefings.",
                    "score": 9,
                    "judge_reason": "Correctly identifies Tuesday 10:00 AM schedule. Minor omission of location.",
                    "qa_metadata": {
                        "domain": "habits_state",
                        "belonged": "Work & Education",
                        "app_log_ids": ["log_00010", "log_00022"],
                    },
                    "relevant_app_logs": [
                        {
                            "app_log_id": "log_00010",
                            "timestamp": "2023-10-03 10:00:00",
                            "app_name": "Google Calendar",
                            "api_name": "CreateEvent",
                            "request": {"title": "Weekly Briefing", "start": "10:00"},
                            "response": {"event_id": "evt_001"},
                        }
                    ],
                }
            ],
        },
    )

    user_id: str = field(
        default="",
        metadata={
            "description": "User identifier (e.g. 'user_001') for per-user tracking.",
            "type": "str",
        },
    )

    reward: float = field(
        default=0.0,
        metadata={
            "description": "Mean QA accuracy (0.0–1.0) across all sampled QA pairs.",
            "type": "float",
        },
    )

    failure_info: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Populated when Phase 2 terminated early due to retrieve failure "
                "(break out of the QA loop, partial recorder preserved). "
                "None for fully-completed users."
            ),
            "type": "Optional[str]",
        },
    )

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, metadata={"internal": True})

    async def log_init(self, app_logs: List[Dict]) -> None:
        async with self._lock:
            self.init = {"app_logs": app_logs}

    async def log_step(
        self,
        query: str,
        predicted: str,
        reference: str,
        score: int,
        judge_reason: str = "",
        qa_metadata: Dict = None,
        retrieved_memory: Dict = None,
        relevant_app_logs: List[Dict] = None,
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
                    "relevant_app_logs": relevant_app_logs or [],
                }
            )

    async def set_reward(self, reward: float) -> None:
        async with self._lock:
            self.reward = reward


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_all_user_dirs() -> List[str]:
    """Return absolute paths to all user directories, sorted numerically."""
    user_data_dir = DATA_DIR / "user_data"
    try:
        dirs = [d for d in os.listdir(str(user_data_dir)) if os.path.isdir(str(user_data_dir / d))]
    except FileNotFoundError:
        log.error(f"user_data directory not found: {user_data_dir}")
        return []
    dirs.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return [str(user_data_dir / d) for d in dirs]


def _get_qa_path_for_user(user_dir: str) -> Path:
    """Return the QA json path for a given user directory path."""
    qa_dir = DATA_DIR / "user_qa"
    dir_name = Path(user_dir).name
    user_idx = int(re.search(r"(\d+)", dir_name).group(1))
    qa_files = [f for f in os.listdir(str(qa_dir)) if f.endswith(".json")]
    qa_files.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))
    return qa_dir / qa_files[user_idx - 1]


def get_task_list(status: str, eval_n_users: int) -> List[str]:
    """Return user-directory paths for the requested split.

    status:
      'search' → first eval_n_users users from the search split (users 001–006)
      'test'   → held-out test users (users 007–010)
    """
    all_dirs = _get_all_user_dirs()
    train_dirs = all_dirs[:TRAIN_USERS]
    eval_dirs = all_dirs[TRAIN_USERS:]

    if status == "search":
        return train_dirs[:int(eval_n_users)]
    else:  # test
        return eval_dirs


def load_user_data(user_dir: str, eval_n_qa: Optional[int] = None) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Load app logs, user profile, and QA pairs.

    eval_n_qa=None (default): use all available QA pairs (for evaluation).
    eval_n_qa=N: deterministically sample N pairs (for training, cost control).

    Returns (app_logs, user_profile, qa_pairs).
    """
    user_path = Path(user_dir)

    with open(str(user_path / "app_log_large.json"), encoding="utf-8") as fh:
        app_logs: List[Dict] = json.load(fh)

    with open(str(user_path / "user_basic_profile.json"), encoding="utf-8") as fh:
        user_profile: Dict = json.load(fh)

    with open(str(_get_qa_path_for_user(user_dir)), encoding="utf-8") as fh:
        all_qa: List[Dict] = json.load(fh)

    if eval_n_qa is None:
        qa_pairs = all_qa
    else:
        rng = random.Random(user_dir)
        qa_pairs = rng.sample(all_qa, min(eval_n_qa, len(all_qa)))

    return app_logs, user_profile, qa_pairs


# ---------------------------------------------------------------------------
# Judge (raw openai, no Agent dependency)
# ---------------------------------------------------------------------------

BASIC_JUDGE_PROMPT = """You are an expert evaluator. Your task is to rate the quality of an AI-generated answer based on a standard reference.

[Question]: {query}
[Standard Reference]: {reference}
[AI Prediction]: {prediction}

Rate the prediction on a scale of 0 to 10 based on whether it correctly covers the KEY POINTS in the reference.

**Important Evaluation Principle:**
- Focus on whether the prediction captures the ESSENTIAL information from the reference.
- Additional details, elaborations, or supplementary information in the prediction should NOT be penalized, as long as they don't contradict the reference.
- Only deduct points for: missing key points, factual errors, or contradictions with the reference.

Scoring criteria:
- 0-2: Mostly or completely incorrect. Contains major factual errors, contradicts the reference, or misses almost all key points.
- 3-5: Partially correct but missing significant key points from the reference, or contains notable factual errors.
- 6-8: Covers most key points from the reference correctly. Minor omissions or inaccuracies may exist, but core information is accurate.
- 9-10: Fully covers all key points from the reference accurately. No factual errors or contradictions. (Extra details beyond the reference are acceptable.)

Output ONLY a JSON object with two keys:
- "reason": a brief explanation for the score (focus on which key points were hit or missed)
- "score": a number from 0 to 10 (integer)

Output format(JSON):
{{"reason": "...", "score": ...}}
"""


async def judge_answer(
    query: str,
    predicted: str,
    reference: str,
    judge_model: str = "gpt-5-mini",
) -> Tuple[int, str]:
    """Score a predicted answer against the reference using an LLM judge.

    Uses the openai SDK directly (no Agent wrapper) so this module stays
    free of method-layer dependencies. If a token tracker has been injected
    via `set_token_tracker`, per-call usage is reported to it.

    Returns (score, reason) where score is 0–10 and reason is a brief explanation.
    """
    if AsyncOpenAI is None:
        return 0, "openai SDK unavailable"

    user_msg = BASIC_JUDGE_PROMPT.format(query=query, reference=reference, prediction=predicted)

    api_key = os.getenv("OPENAI_API_KEY", "")
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(150.0, connect=10.0),
        max_retries=0,
    )

    chat_kwargs = {
        "model": judge_model,
        "messages": [{"role": "user", "content": user_msg}],
        "response_format": {"type": "json_object"},
    }
    if _supports_reasoning(judge_model):
        chat_kwargs["reasoning_effort"] = "low"

    # 6 attempts with exponential backoff capped at 60s covers ~15 min of
    # sustained OpenAI-side flakiness — chosen to match the robustness of the
    # pre-reorganization judge (which stacked judge_answer's retries on top of
    # Agent.ask's internal retries for an effective ~18 attempts).
    MAX_JUDGE_ATTEMPTS = 6
    for attempt in range(MAX_JUDGE_ATTEMPTS):
        try:
            resp = await client.chat.completions.create(**chat_kwargs)
            if _token_tracker is not None and hasattr(resp, "usage") and resp.usage is not None:
                try:
                    await _token_tracker.update(model_name=judge_model, usage=resp.usage)
                except Exception as tracker_exc:
                    log.debug(f"token tracker update failed: {tracker_exc}")

            raw = resp.choices[0].message.content or ""
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                return 0, f"Judge returned non-JSON: {raw[:200]}"

            score = int(result.get("score", 0))
            score = max(0, min(10, score))
            reason = str(result.get("reason", ""))
            return score, reason

        except (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            if attempt < MAX_JUDGE_ATTEMPTS - 1:
                delay = min(60, 2 ** (attempt + 1))  # 2, 4, 8, 16, 32, (cap) 60
                log.warning(f"Judge retry {attempt+1}/{MAX_JUDGE_ATTEMPTS}: {repr(exc)}")
                await asyncio.sleep(delay)
                continue
            log.warning(f"Judge failed after {MAX_JUDGE_ATTEMPTS} attempts: {repr(exc)}")
            return 0, f"Judge error: {exc}"
        except Exception as exc:
            log.warning(f"Judge failed: {repr(exc)}")
            return 0, f"Judge error: {exc}"

    return 0, "Judge exhausted retries"
