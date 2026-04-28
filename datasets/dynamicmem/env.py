"""
DynamicMem environment — dataset-layer utilities shared by all methods
(alma, cc, hipporag2, future meta-harness).

Pure data-layer module: stdlib only, no LLM client, no Agent/Judge import.
Judging is provided by `common.judge.Judge` (host-agnostic), which is wired
in via `BaseWorkflow._make_judge()`. Token usage is tracked by
`common.tokens.GLOBAL_TOKEN_TRACKER` (read lazily by Judge/Agent).

Provides:
  - Basic_Recorder:       minimal async recorder base class
  - DynamicMemRecorder:   records one user's app-log ingestion + QA session
  - get_task_list():      user directory paths for a split
  - load_user_data():     app_logs, user_profile, sampled QA pairs for a user

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
# Recorder — Basic_Recorder lives in common/; DynamicMemRecorder is the
# domain-specific subclass used by DynamicMem.
# ---------------------------------------------------------------------------

from common.harness_base import Basic_Recorder  # noqa: E402  (re-export)


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


def get_task_list(status: str, eval_n_samples: int) -> List[str]:
    """Return user-directory paths for the requested split.

    status:
      'search' → first eval_n_samples users from the search split (users 001–006)
      'test'   → held-out test users (users 007–010)
    """
    all_dirs = _get_all_user_dirs()
    train_dirs = all_dirs[:TRAIN_USERS]
    eval_dirs = all_dirs[TRAIN_USERS:]

    if status == "search":
        return train_dirs[:int(eval_n_samples)]
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


