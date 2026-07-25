"""Shared staged-evaluation config + gauntlet driver, used by forge AND the
baselines (which must not import forge). The stage sizes/plan/nesting live here;
run_gauntlet (Task 5) drives the promotion loop with an injected stage runner.
Moved from forge/orchestrator.py 2026-07-25."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Staged evaluation — per-benchmark stage schema
#
# Each benchmark's `datasets.<ds>` block carries a `stages` mapping with four
# entries (sanity_check + stage1..3). Sizes use the benchmark's NATIVE
# hierarchy fields; `threshold` on stage1/stage2 is the promotion gate (the
# stage's normalized score must be >= threshold to advance; per-benchmark
# independent). stage3 is terminal (no threshold). Sizes are PER-UNIT
# (DynamicMem task counts are per checkpoint; LoCoMo QA counts are per
# conversation). Nested sampling guarantees a smaller stage's task set is a
# subset of a larger one.
# ---------------------------------------------------------------------------

STAGE_ORDER = ("stage1", "stage2", "stage3")

# Benchmark-family field vocabulary: <family> -> (sample-count field, extras)
_FAMILY_FIELDS = {
    "dynamicmem": ("n_users", ("n_checkpoints", "n_task_a", "n_task_c")),
    "locomo": ("n_conversations", ("n_qa",)),
    "longmemeval": ("n_questions", ()),
}

# Initial thresholds are DELIBERATELY conservative (only clearly-broken
# candidates get eliminated early) — calibrate against stage-score
# distributions after the first real search run. LoCoMo/LongMemEval have a
# non-zero no-memory floor (some questions are answerable from the question
# text alone), hence higher thresholds than DynamicMem. (LoCoMo runs on
# categories 1-4 only since 2026-07-08 — cat-5 adversarial QAs are excluded.)
DEFAULT_STAGES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "dynamicmem": {
        "sanity_check": {"n_users": 1, "n_checkpoints": 1, "n_task_a": 1, "n_task_c": 1},
        "stage1": {"n_users": 2, "n_checkpoints": 1, "n_task_a": 5, "n_task_c": 5, "threshold": 0.05},
        "stage2": {"n_users": 4, "n_checkpoints": 3, "n_task_a": 5, "n_task_c": 5, "threshold": 0.10},
        "stage3": {"n_users": 6, "n_checkpoints": 5, "n_task_a": 5, "n_task_c": 5},
    },
    "locomo": {
        "sanity_check": {"n_conversations": 1, "n_qa": 3},
        "stage1": {"n_conversations": 2, "n_qa": 20, "threshold": 0.30},
        "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
        "stage3": {"n_conversations": 6, "n_qa": 60},
    },
    "longmemeval": {
        "sanity_check": {"n_questions": 2},
        "stage1": {"n_questions": 20, "threshold": 0.25},
        "stage2": {"n_questions": 50, "threshold": 0.30},
        "stage3": {"n_questions": 100},
    },
}

# stage_<ds> objective value for a coverage=full evaluation. Numerically above
# the last gauntlet tier (3) so full-coverage entries are distinguishable in
# frontier telemetry; scores at stage 3 vs 4 are still NOT mutually comparable —
# coverage is uniform within a run.
FULL_STAGE = 4.0

_OLD_SIZE_FIELDS = ("eval_n_samples", "eval_n_qa", "check_n_samples", "check_n_qa")


def _benchmark_family(ds: str) -> str:
    """Map a dataset name to its stage-schema family (longmemeval_s/m share one)."""
    if ds.startswith("longmemeval"):
        return "longmemeval"
    if ds not in _FAMILY_FIELDS:
        raise ValueError(
            f"Unknown benchmark {ds!r} — no stage schema. Known families: "
            f"{sorted(_FAMILY_FIELDS)} (longmemeval_s / longmemeval_m share 'longmemeval')."
        )
    return ds


def _wire_size(v: Any) -> Optional[int]:
    """Stage size field → wire value. None (a `null` / `full` / `all` field,
    normalized by _resolve_dataset_stages) passes through as None = no cap
    (whole pool / whole split), reusing the same downstream machinery as
    coverage=full; anything else is coerced to int."""
    return None if v is None else int(v)


def stage_wire_spec(ds: str, stage_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stage's benchmark-native size fields to the wire spec the
    container consumes: {"n_samples": int|None, ...family extras}. A None
    field means full coverage of that dimension. `threshold` is an
    orchestrator-side concern and is stripped."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": _wire_size(stage_params[sample_field])}
    for f in extras:
        spec[f] = _wire_size(stage_params[f])
    return spec


def full_wire_spec(ds: str) -> Dict[str, Any]:
    """Wire spec for coverage=full: every size field None = no cap (the
    container's env/get_task_list and the workflows treat None as "whole
    split / all checkpoints / whole buckets")."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": None}
    for f in extras:
        spec[f] = None
    return spec


def stage_plan(ds: str, ds_params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any], Optional[float]]]:
    """Ordered promotion plan for one benchmark:
    [(stage_name, wire_spec, threshold_or_None), ...] for stage1..3."""
    stages = ds_params["stages"]
    plan = []
    for name in STAGE_ORDER:
        params = stages[name]
        threshold = params.get("threshold")
        plan.append((name, stage_wire_spec(ds, params), float(threshold) if threshold is not None else None))
    return plan


def _resolve_dataset_stages(ds: str, params: Dict[str, Any]) -> None:
    """Fill defaults + validate the `stages` block of one dataset, in place."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    allowed = {sample_field, *extras, "threshold"}

    # Old flat schema is gone — fail loudly with a migration hint.
    stale = [f for f in _OLD_SIZE_FIELDS if f in params]
    if stale:
        raise ValueError(
            f"datasets.{ds}: fields {stale} were removed — evaluation sizes now "
            f"live in a `stages` block. Example:\n"
            f"  {ds}:\n    stages:\n"
            f"      sanity_check: {DEFAULT_STAGES[family]['sanity_check']}\n"
            f"      stage1: {DEFAULT_STAGES[family]['stage1']}\n"
            f"      stage2: {DEFAULT_STAGES[family]['stage2']}\n"
            f"      stage3: {DEFAULT_STAGES[family]['stage3']}"
        )

    stages = params.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError(f"datasets.{ds}.stages must be a mapping")
    for name in ("sanity_check", *STAGE_ORDER):
        block = stages.setdefault(name, {})
        if not isinstance(block, dict):
            raise ValueError(f"datasets.{ds}.stages.{name} must be a mapping")
        unknown = set(block) - allowed
        if unknown:
            raise ValueError(
                f"datasets.{ds}.stages.{name}: unknown field(s) {sorted(unknown)} "
                f"for family {family!r}; allowed: {sorted(allowed)}"
            )
        for field, default in DEFAULT_STAGES[family][name].items():
            block.setdefault(field, default)
        # Full-coverage sentinel: a size field of null / "full" / "all"
        # (case-insensitive) → None = that dimension's whole pool / split,
        # reusing coverage=full's downstream None path. Intended for stage3
        # (progressive gauntlet with a full final stage); the monotonicity
        # check below treats None as +inf, so a stray null at an earlier
        # stage forces later stages full or errors out.
        for field in (sample_field, *extras):
            v = block.get(field)
            if isinstance(v, str) and v.strip().lower() in ("full", "all"):
                block[field] = None
        # sanity_check and stage3 never gate
        if name in ("sanity_check", "stage3"):
            block.pop("threshold", None)
        thr = block.get("threshold")
        if thr is not None and not (0.0 <= float(thr) <= 1.0):
            raise ValueError(f"datasets.{ds}.stages.{name}.threshold must be in [0,1], got {thr}")

    # Monotonic non-decreasing sizes across stage1..3 (nesting depends on it).
    # None (full coverage) counts as the largest value, so it must not precede
    # a concrete size — a null/full field forces every later stage full too.
    for field in (sample_field, *extras):
        seq = [stages[name][field] for name in STAGE_ORDER]
        cmp = [float("inf") if v is None else int(v) for v in seq]
        if any(b < a for a, b in zip(cmp, cmp[1:])):
            raise ValueError(
                f"datasets.{ds}.stages: {field} must be non-decreasing across "
                f"stage1..stage3 (got {seq}) — staged nesting depends on it; "
                f"a null/full field (= full coverage) may only be followed by "
                f"another null/full field"
            )
