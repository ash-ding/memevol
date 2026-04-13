"""
DynamicMem environment for memevol.

Provides:
  - DynamicMemRecorder: records one user's app-log ingestion + QA session
  - get_task_list(): returns user directory paths for a given split
  - load_user_data(): loads app_logs, user_profile, sampled QA pairs for a user
  - judge_answer(): LLM judge that scores a predicted answer against reference

Design note: the interaction protocol is two-phase (per user):
  Phase 1  — general_update is called N times with app_log chunks to build a user profile
  Phase 2  — general_retrieve is called once per QA question; agent answers using the profile

recorder.init['query'] is injected by DynamicMem_Workflow before each general_retrieve call
so that memory code can do query-aware retrieval.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_envs.base_envs import Basic_Recorder

try:
    from logger import get_logger
    log = get_logger("main")
except Exception:
    import logging
    log = logging.getLogger("main")

try:
    from utils.hire_agent import Agent
except Exception:
    Agent = None  # type: ignore

# ---------------------------------------------------------------------------
# Data-path resolution
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "run_main.py").exists():
            return p
    return Path(__file__).resolve().parent.parent


_data_env = os.environ.get("DYNAMICMEM_DATA", "")
DATA_DIR: Path = Path(_data_env) if _data_env else (_find_project_root() / "dynamicmem")

TRAIN_USERS = 6
EVAL_USERS = 4

# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

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
# Data helpers (free functions, no class needed)
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
    else:  # eval
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


async def judge_answer(query: str, predicted: str, reference: str, judge_model: str = "gpt-5-mini") -> Tuple[int, str]:
    """Score a predicted answer against the reference using an LLM judge.

    Returns (score, reason) where score is 0–10 and reason is a brief explanation.
    """
    if Agent is None:
        return 0, "Agent unavailable"

    judge_schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Brief explanation for the score."},
            "score": {"type": "integer", "minimum": 0, "maximum": 10, "description": "Score from 0 to 10."},
        },
        "required": ["reason", "score"],
    }
    judge_user = BASIC_JUDGE_PROMPT.format(query=query, reference=reference, prediction=predicted)
    import openai
    for attempt in range(3):
        try:
            agent = Agent(system_prompt="", output_schema=judge_schema, model=judge_model, timeout=150)
            result = await agent.ask(judge_user, reasoning_effort="low")
            if isinstance(result, dict) and "error" not in result:
                score = int(result.get("score", 1))
                score = max(0, min(10, score))
                reason = str(result.get("reason", ""))
                return score, reason
            return 0, "Judge returned invalid result"
        except (asyncio.TimeoutError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            if attempt < 2:
                delay = 2 ** (attempt + 1)
                log.warning(f"Judge retry {attempt+1}/3: {repr(exc)}")
                await asyncio.sleep(delay)
                continue
            log.warning(f"Judge failed after 3 attempts: {repr(exc)}")
            return 0, f"Judge error: {exc}"
        except Exception as exc:
            log.warning(f"Judge failed: {repr(exc)}")
            return 0, f"Judge error: {exc}"
