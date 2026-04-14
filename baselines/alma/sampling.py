"""
Alma-specific post-processing: read the full traces + score.json produced by
the eval subprocess, perform single-user score-bin sampling, and build the
compressed feedback artifact that the meta-agent's analysis prompt expects.

This is the only place sampling lives — other methods consume the full traces
directly and do not need this file.

Kept a "data artifact" shape compatible with alma's old launch.py JSON so
the analysis / generation prompts can be reused without changes:

    {
      "benchmark_eval_score": { overall, stddev },
      "examples": [
        {"user_id", "query", "retrieved_memory", "predicted", "reference",
         "score", "judge_reason", "relevant_app_logs"},
        ...,
        {"error_info": "...", "score": 0.0},  # for each invalid user / partial failure
      ],
      "token_usage": { per_model: {...} },
    }
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from baselines.alma.logger import get_logger

log = get_logger("main")


def _sample_steps_from_bins(steps: List[Dict[str, Any]], n_score_bins: int, samples_per_bin: int) -> List[Dict[str, Any]]:
    """Bin steps by score and sample up to samples_per_bin from each bin."""
    bin_width = 10.0 / n_score_bins
    binned: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_score_bins)}
    for step in steps:
        s = step["score"]
        idx = min(int(s / bin_width), n_score_bins - 1)
        idx = max(0, idx)
        binned[idx].append(step)

    sampled: List[Dict[str, Any]] = []
    for i in range(n_score_bins):
        recs = binned[i]
        k = min(samples_per_bin, len(recs))
        if k > 0:
            sampled.extend(random.sample(recs, k))
    return sampled


def _read_score(output_run_dir: Path) -> Dict[str, Any]:
    path = output_run_dir / "score.json"
    if not path.exists():
        raise FileNotFoundError(f"score.json missing under {output_run_dir}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _read_token_usage(output_run_dir: Path) -> Dict[str, Any]:
    path = output_run_dir / "token_usage.json"
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"Failed to read token_usage.json: {exc}")
        return {}


def _read_traces(output_run_dir: Path) -> List[Dict[str, Any]]:
    """Read every traces/<user_id>.json file; return list of payload dicts."""
    traces_dir = output_run_dir / "traces"
    if not traces_dir.is_dir():
        return []
    payloads = []
    for p in sorted(traces_dir.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                payloads.append(json.load(f))
        except Exception as exc:
            log.warning(f"Failed to read trace {p}: {exc}")
    return payloads


def build_analysis_artifact(
    output_run_dir: Path,
    n_score_bins: int = 3,
    samples_per_bin: int = 3,
) -> Dict[str, Any]:
    """Assemble the compressed analysis artifact from a finished run directory.

    Behaviour:
      - Pick exactly one valid (non-empty) user at random; bin its QA steps by
        score and sample up to samples_per_bin from each bin. This matches
        alma's historical single-user sampling policy.
      - Surface partial-user failures (failure_info on the recorder) as
        synthetic error_info entries so the meta-agent's reflection prompt
        sees them.
      - Overall score / stddev are copied verbatim from score.json.
    """
    score = _read_score(output_run_dir)
    token_usage = _read_token_usage(output_run_dir)
    trace_payloads = _read_traces(output_run_dir)

    # Collect valid user trace lists (skip empty / all-failure users).
    valid_users: List[tuple] = []  # (user_id, steps)
    invalid_records_info: List[str] = []

    for item in score.get("invalid_users", []) or []:
        uid = item.get("user_id", "unknown")
        err = item.get("error", "<unknown error>")
        invalid_records_info.append(f"User {uid} failed: {err}")

    per_user_meta = score.get("per_user", {}) or {}

    for payload in trace_payloads:
        uid = payload.get("user_id", "") or "unknown"
        steps = payload.get("steps", []) or []
        fi = payload.get("failure_info")
        if fi:
            invalid_records_info.append(f"User {uid} partially failed: {fi}")
        # Tag each step with user_id so the downstream prompt can cite per-user patterns.
        for step in steps:
            step["user_id"] = uid
        if steps:
            valid_users.append((uid, steps))

    # Single-user selection + bin sampling.
    sampled_steps: List[Dict[str, Any]] = []
    chosen_uid = None
    if valid_users:
        chosen_uid, chosen_steps = random.choice(valid_users)
        sampled_steps = _sample_steps_from_bins(chosen_steps, n_score_bins, samples_per_bin)

    for uid, steps in valid_users:
        marker = "[selected]" if uid == chosen_uid else ""
        n = len(sampled_steps) if uid == chosen_uid else 0
        log.info(f"[User {uid}] {len(steps)} QAs, sampled {n} {marker}".rstrip())

    # Include ALL execution errors — there is at most one per user (Phase 1
    # total-fail and Phase 2 partial-fail are mutually exclusive for a given
    # user), so with DynamicMem's 6 users this caps at ~6 entries of ~200
    # bytes each, which is cheap relative to the QA sample block and lets the
    # meta-agent see the full failure pattern instead of a random subset.
    # Order: Phase 1 fails first (more severe signal), then Phase 2 partial
    # fails in trace-file order.
    invalid_sample = list(invalid_records_info)

    examples: List[Dict[str, Any]] = []
    for step in sampled_steps:
        examples.append({
            "user_id": step.get("user_id", ""),
            "query": step["query"],
            "retrieved_memory": step.get("retrieved_memory", {}),
            "predicted": step["predicted"],
            "reference": step["reference"],
            "score": step["score"],
            "judge_reason": step.get("judge_reason", ""),
            "relevant_app_logs": step.get("relevant_app_logs", []),
        })
    for err in invalid_sample:
        examples.append({"error_info": err, "score": 0.0})

    avg = score.get("benchmark_eval_score", {})

    log.info(
        f"--- AVG Reward: {avg.get('benchmark_overall_eval_score', 0.0):.3f} | "
        f"SE: {avg.get('benchmark_overall_eval_standard_deviation', 0.0):.3f} | "
        f"Valid users: {len(per_user_meta)} | "
        f"Sampled {len(sampled_steps)} QAs from user={chosen_uid} "
        f"({n_score_bins} bins × {samples_per_bin} samples/bin) ---"
    )

    artifact = {
        "benchmark_eval_score": avg,
        "examples": examples,
        "token_usage": token_usage,
    }

    # Persist this sampling snapshot alongside the full traces, so the exact
    # input the meta-agent saw at analysis time is always recoverable later.
    # build_analysis_artifact may run several times per output_run_dir (once
    # after eval, more on sanity-check retries, again on analyze) — each call
    # gets its own UTC-timestamped file.
    try:
        samples_dir = output_run_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%f")
        snapshot = {
            "timestamp_utc": ts,
            "chosen_user_id": chosen_uid,
            "n_score_bins": n_score_bins,
            "samples_per_bin": samples_per_bin,
            **artifact,
        }
        with (samples_dir / f"{ts}.json").open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning(f"Failed to persist sampling snapshot: {exc}")

    return artifact
