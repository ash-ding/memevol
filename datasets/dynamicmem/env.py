"""
DynamicMem environment — dataset-layer utilities shared by all methods
(alma, cc, hipporag2, forge meta-harness).

Data format: official DynamicMem TCE v2 (taskabc_v2 / rq_v2, pack-first).
Each user directory holds exactly two files:
    user_data/<NNN_user_NNN>/app_log_large.json   # list[app_log], chronological
    user_data/<NNN_user_NNN>/task_packs.json      # dict with 5 checkpoints

Pure data-layer module: stdlib + datasets.dynamicmem.tce_prompts (pure) only.
No LLM client here; judging lives in workflow.py + tce_prompts.py.

Provides:
  - DynamicMemRecorder:        records one user's ingestion + task session
  - get_task_list():           user directory paths for a split
  - load_user_checkpoints():   app_logs + normalized checkpoint task items
  - sample_items():            deterministic stratified task-item sampling
  - load_user_data():          BACKWARD-COMPAT shim (last checkpoint only,
                               flat {query, reference, metadata} QA) for the
                               cc / hipporag2 standalone baselines.

Checkpoint protocol (official): each checkpoint cp_i has an `as_of` cutoff;
memory visible at cp_i is EXACTLY app_logs[:as_of.log_index+1]. Ingestion and
task answering interleave per checkpoint — see workflow.py.
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

from datasets.dynamicmem.tce_prompts import (
    flatten_observability,
    flatten_snapshot,
    normalize_task_a_current_value,
)

log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Data-path resolution
#
# Data (user_data/<user>/{app_log_large.json,task_packs.json}) sits directly
# under this package directory:
#     datasets/dynamicmem/user_data/
# Override with the DYNAMICMEM_DATA env var if you want to point elsewhere.
# ---------------------------------------------------------------------------

_data_env = os.environ.get("DYNAMICMEM_DATA", "")
DATA_DIR: Path = Path(_data_env) if _data_env else Path(__file__).resolve().parent

TRAIN_USERS = 6
EVAL_USERS = 4

TASK_FAMILY_STATE_COMPLETION = "state_completion"
TASK_FAMILY_APPLY_SERVICE = "apply_service"


# ---------------------------------------------------------------------------
# Recorder — Basic_Recorder lives in common/; DynamicMemRecorder is the
# domain-specific subclass used by DynamicMem.
# ---------------------------------------------------------------------------

from common.harness_base import Basic_Recorder  # noqa: E402  (re-export)


@dataclass
class DynamicMemRecorder(Basic_Recorder):
    """Records one user's app-log ingestion and TCE task session.

    init:   set by the workflow.
        {
            'app_logs':     List[dict],   # entries from app_log_large.json
            'query':        str,          # injected before each general_retrieve call
        }
        Each app_log entry: app_log_id, timestamp, app_name, api_name, request, response.

    steps:  one entry per answered task item (Task A state-completion key or
        Task C personalized-service item).
        {
            'query':       str,           # the item's retrieval_query
            'predicted':   str,           # model answer (JSON string for structured)
            'reference':   str,           # golden, JSON string for structured
            'score':       float,         # 0.0-1.0 official holistic Core+Detail
            'judge_reason': str,
            'qa_metadata': {'task_family', 'checkpoint_id', 'state_key', 'qa_id',
                            'service_family', 'domain', 'app_log_ids',
                            'field_judgments', 'evidence_prf', ...}
        }

    reward: mean(steps[i].score), in [0.0, 1.0].
    """

    init: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": (
                "App logs visible to the memory system. "
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
                "query": "[Scenario]\\n...\\n\\n[Task Instruction]\\nDraft a reminder message...",
            },
        },
    )

    steps: List[Dict[str, Any]] = field(
        default_factory=list,
        metadata={
            "description": "One entry per answered task item.",
            "type": "List[Dict[str, Any]]",
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
            "description": "Mean task score (0.0-1.0) across all sampled items.",
            "type": "float",
        },
    )

    failure_info: Optional[str] = field(
        default=None,
        metadata={
            "description": (
                "Populated when the task loop terminated early due to failure "
                "(partial recorder preserved). None for fully-completed users."
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
        score: float,
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


def get_task_list(status: str, eval_n_samples: int) -> List[str]:
    """Return user-directory paths for the requested split.

    status:
      'search' → first eval_n_samples users from the search split (users 001–006)
      'test'   → first eval_n_samples held-out test users (users 007–010)

    Both splits honour the eval_n_samples cap (a deterministic prefix, so
    staged nesting holds on the test split too — mode=test stage sizing
    was silently void before 2026-07-08).
    """
    all_dirs = _get_all_user_dirs()
    train_dirs = all_dirs[:TRAIN_USERS]
    eval_dirs = all_dirs[TRAIN_USERS:]

    if eval_n_samples is None:  # coverage=full: whole split, no cap
        return train_dirs if status == "search" else eval_dirs
    if status == "search":
        return train_dirs[:int(eval_n_samples)]
    else:  # test
        return eval_dirs[:int(eval_n_samples)]


# ---------------------------------------------------------------------------
# Checkpoint loader (official TCE v2 pack-first format)
# ---------------------------------------------------------------------------

def _extract_evidence_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _task_a_items(checkpoint: Dict, task_contract_version: str, domain: str, checkpoint_id: str) -> List[Dict]:
    """Task A (State Completion) items — one per pack key with a non-None
    golden after official normalization.

    Mirrors task_packs.py::build_state_completion_targets_from_pack (template
    from `answer_template`) + tce_core/evaluation.py:835-845 (expected set =
    keys whose normalize_task_a_current_value(validated) is not None; keys
    outside that set are not scored by the official eval).
    """
    pack = checkpoint.get("state_completion_pack")
    keys_obj = pack.get("keys") if isinstance(pack, dict) else None
    if not isinstance(keys_obj, dict):
        return []

    validated_flat = flatten_snapshot(checkpoint.get("validated_snapshot_state") or {})
    obs_flat = flatten_observability(checkpoint.get("state_observability") or {})

    items: List[Dict] = []
    for state_key in sorted(keys_obj.keys()):
        pack_item = keys_obj.get(state_key)
        if not isinstance(pack_item, dict):
            continue
        answer_template = pack_item.get("answer_template")
        if answer_template is None:
            continue
        golden = normalize_task_a_current_value(
            validated_flat.get(state_key), task_contract_version=task_contract_version
        )
        if golden is None:
            continue
        retrieval_query = str(pack_item.get("retrieval_query") or "").strip()
        question_text = str(pack_item.get("question_text") or "").strip() or retrieval_query
        if not retrieval_query:
            continue
        obs = obs_flat.get(state_key) or {}
        evidence_ids = _extract_evidence_ids(obs.get("evidence_app_log_ids") if isinstance(obs, dict) else None)
        if not evidence_ids and isinstance(obs, dict):
            last_id = obs.get("last_app_log_id")
            evidence_ids = [str(last_id)] if last_id is not None and str(last_id).strip() else []
        items.append(
            {
                "task_family": TASK_FAMILY_STATE_COMPLETION,
                "checkpoint_id": checkpoint_id,
                "state_key": state_key,
                "qa_id": "",
                "query": retrieval_query,
                "answer_query_text": question_text,
                "reference": golden,
                "answer_template": answer_template,
                "output_template": None,
                "service_family": "",
                "scenario": "",
                "task_instruction": "",
                "scoring_points": [],
                "evidence_ids": evidence_ids,
                "domain": domain,
            }
        )
    return items


def _task_c_items(checkpoint: Dict, domain: str, checkpoint_id: str) -> List[Dict]:
    """Task C (Personalized Service) items — official v2 extraction
    (pipeline.py::_extract_rq3_apply_pack_for_checkpoint :499-517 requires
    scenario + task_instruction; items[:1] per key)."""
    payload = checkpoint.get("rq3_apply_service_qa")
    keys_obj = payload.get("keys") if isinstance(payload, dict) else None
    if not isinstance(keys_obj, dict):
        return []

    items: List[Dict] = []
    for state_key in sorted(keys_obj.keys()):
        key_obj = keys_obj.get(state_key)
        raw_items = key_obj.get("items") if isinstance(key_obj, dict) else None
        if not isinstance(raw_items, list):
            continue
        for idx, raw in enumerate(raw_items[:1]):  # official: one item per key
            if not isinstance(raw, dict):
                continue
            scenario = str(raw.get("scenario") or "").strip()
            task_instruction = str(raw.get("task_instruction") or "").strip()
            if not scenario or not task_instruction:
                continue
            service_family = str(raw.get("service_family") or "").strip()
            reference_answer = raw.get("reference_answer")
            reference_output = raw.get("reference_output")
            items.append(
                {
                    "task_family": TASK_FAMILY_APPLY_SERVICE,
                    "checkpoint_id": checkpoint_id,
                    "state_key": state_key,
                    "qa_id": str(raw.get("qa_id") or f"q{idx + 1}"),
                    "query": str(raw.get("retrieval_query") or "").strip(),
                    "answer_query_text": str(raw.get("retrieval_query") or "").strip(),
                    "reference": reference_answer if reference_answer is not None else reference_output,
                    "answer_template": None,
                    "output_template": raw.get("output_template"),
                    "service_family": service_family,
                    "scenario": scenario,
                    "task_instruction": task_instruction,
                    "reference_answer": str(reference_answer or ""),
                    "reference_output": reference_output,
                    "scoring_points": list(raw.get("answer_scoring_points") or []),
                    "evidence_ids": _extract_evidence_ids(raw.get("gold_memory_evidence_app_log_ids")),
                    "domain": domain,
                }
            )
    return items


def load_user_checkpoints(user_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load one user's app logs + normalized checkpoint task items.

    Returns (app_logs, checkpoints) where each checkpoint is
        {checkpoint_id, as_of: {log_index, timestamp}, items: List[item]}
    ordered chronologically. Item shape: see _task_a_items / _task_c_items.
    """
    user_path = Path(user_dir)

    with open(str(user_path / "app_log_large.json"), encoding="utf-8") as fh:
        app_logs_raw = json.load(fh)
    # Official loaders tolerate both a bare list and {"app_logs": [...]}.
    app_logs: List[Dict] = (
        app_logs_raw.get("app_logs", []) if isinstance(app_logs_raw, dict) else app_logs_raw
    )

    with open(str(user_path / "task_packs.json"), encoding="utf-8") as fh:
        task_pack: Dict = json.load(fh)

    task_contract_version = str(task_pack.get("task_contract_version") or "taskabc_v2")

    checkpoints: List[Dict] = []
    for cp in task_pack.get("checkpoints", []):
        if not isinstance(cp, dict):
            continue
        checkpoint_id = str(cp.get("checkpoint_id") or "")
        as_of = cp.get("as_of") or {}
        domain = str(as_of.get("domain") or "")
        items = _task_a_items(cp, task_contract_version, domain, checkpoint_id) + _task_c_items(
            cp, domain, checkpoint_id
        )
        checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "as_of": {
                    "log_index": as_of.get("log_index"),
                    "timestamp": str(as_of.get("timestamp") or ""),
                },
                "items": items,
            }
        )
    return app_logs, checkpoints


def observed_logs_for_checkpoint(checkpoint: Dict, app_logs: List[Dict]) -> List[Dict]:
    """Prefix of app_logs visible at this checkpoint — official semantics
    (tce_core/pipeline.py::observed_logs_for_checkpoint :918-934):
    app_logs[:log_index+1]."""
    log_index = (checkpoint.get("as_of") or {}).get("log_index")
    if isinstance(log_index, int) and 0 <= log_index < len(app_logs):
        return app_logs[: log_index + 1]
    return list(app_logs)


def sample_items(checkpoints: List[Dict], n: Optional[int], seed: Any) -> List[Dict]:
    """Deterministic stratified sampling of task items.

    n=None → all items (checkpoint order, Task A before Task C within each).
    n=N    → round-robin over (checkpoint × task_family) buckets so every
             checkpoint and both families are covered as evenly as N allows.
    """
    if n is None:
        return [item for cp in checkpoints for item in cp.get("items", [])]

    rng = random.Random(str(seed))
    buckets: Dict[Tuple[str, str], List[Dict]] = {}
    cids: List[str] = []
    for cp in checkpoints:
        cid = str(cp.get("checkpoint_id") or "")
        cids.append(cid)
        for fam in (TASK_FAMILY_STATE_COMPLETION, TASK_FAMILY_APPLY_SERVICE):
            items = [i for i in cp.get("items", []) if i.get("task_family") == fam]
            rng.shuffle(items)
            buckets[(cid, fam)] = items

    # Bucket order: checkpoint coverage first, families alternating by
    # checkpoint index, then the complementary pass — so even a tiny n
    # spreads across ALL checkpoints and BOTH families (e.g. n=5 over
    # 5 checkpoints → one item per checkpoint, families A,C,A,C,A).
    fams = (TASK_FAMILY_STATE_COMPLETION, TASK_FAMILY_APPLY_SERVICE)
    primary = [(cid, fams[i % 2]) for i, cid in enumerate(cids)]
    secondary = [(cid, fams[(i + 1) % 2]) for i, cid in enumerate(cids)]
    order: List[Tuple[str, str]] = primary + secondary

    out: List[Dict] = []
    round_idx = 0
    while len(out) < n:
        progressed = False
        for key in order:
            if len(out) >= n:
                break
            bucket = buckets[key]
            if round_idx < len(bucket):
                out.append(bucket[round_idx])
                progressed = True
        if not progressed:
            break
        round_idx += 1
    return out


def sample_items_staged(
    checkpoints: List[Dict],
    *,
    n_checkpoints: Optional[int],
    n_task_a: Optional[int],
    n_task_c: Optional[int],
    seed: Any,
) -> List[Dict]:
    """Per-checkpoint task sampling for the staged-evaluation pipeline.

    Takes the FIRST `n_checkpoints` checkpoints (chronological prefix) and,
    from each, the first `n_task_a` / `n_task_c` items of that checkpoint's
    deterministically shuffled Task-A / Task-C buckets. Counts beyond a
    bucket's size return what exists.

    NESTING GUARANTEE (stage gating depends on it): with the same seed, a
    smaller (n_checkpoints, n_task_a, n_task_c) selection is always a subset
    of a larger one — checkpoint prefix × shuffled-bucket prefix.
    """
    rng = random.Random(str(seed))
    counts = {TASK_FAMILY_STATE_COMPLETION: n_task_a, TASK_FAMILY_APPLY_SERVICE: n_task_c}
    out: List[Dict] = []
    # None = no cap (coverage=full): all checkpoints / whole buckets. The
    # shuffles still run so capped selections stay prefixes of the full set.
    cps = checkpoints if n_checkpoints is None else checkpoints[: max(0, int(n_checkpoints))]
    for cp in cps:
        # Shuffles run in fixed checkpoint order, so a shorter checkpoint
        # prefix consumes the identical rng sequence for the checkpoints it
        # does include — selections within each checkpoint are independent
        # of n_checkpoints.
        for fam in (TASK_FAMILY_STATE_COMPLETION, TASK_FAMILY_APPLY_SERVICE):
            items = [i for i in cp.get("items", []) if i.get("task_family") == fam]
            rng.shuffle(items)
            n = counts[fam]
            out.extend(items if n is None else items[: max(0, int(n))])
    return out


# ---------------------------------------------------------------------------
# Backward-compat shim — cc / hipporag2 standalone baselines
# ---------------------------------------------------------------------------

def _reference_to_str(reference: Any) -> str:
    if isinstance(reference, str):
        return reference
    return json.dumps(reference, ensure_ascii=False)


def load_user_data(user_dir: str, eval_n_qa: Optional[int] = None) -> Tuple[List[Dict], Dict, List[Dict]]:
    """BACKWARD-COMPAT shim for the cc / hipporag2 standalone baselines
    (two-phase, no checkpoint interleaving).

    Uses ONLY the LAST checkpoint's task items — its as_of covers the whole
    log stream, so full-history ingestion + these items is temporally
    self-consistent. Items are flattened to the legacy
    {query, reference, metadata} shape with a string reference.

    The forge evaluation path does NOT use this — see workflow.py
    (checkpoint-interleaved, official TCE protocol).
    """
    app_logs, checkpoints = load_user_checkpoints(user_dir)

    qa_pairs: List[Dict] = []
    if checkpoints:
        last_cp = checkpoints[-1]
        for item in last_cp.get("items", []):
            qa_pairs.append(
                {
                    "query": item.get("query", ""),
                    "reference": _reference_to_str(item.get("reference")),
                    "metadata": {
                        "domain": item.get("domain", ""),
                        "belonged": item.get("service_family") or item.get("task_family", ""),
                        "app_log_ids": list(item.get("evidence_ids") or []),
                        "checkpoint_id": item.get("checkpoint_id", ""),
                        "task_family": item.get("task_family", ""),
                        "state_key": item.get("state_key", ""),
                    },
                }
            )

    if eval_n_qa is not None:
        rng = random.Random(user_dir)
        qa_pairs = rng.sample(qa_pairs, min(eval_n_qa, len(qa_pairs)))

    return app_logs, {}, qa_pairs
