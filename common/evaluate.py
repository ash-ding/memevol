"""Shared evaluation layer, used by forge AND the baselines (which must not
import forge). The stage sizes/plan/nesting live here, and `evaluate_memo` — the
single, execution-independent evaluator — runs the whole staged gauntlet /
single pass / smoke run in the current process (isolation is the caller's
wrapper: forge wraps it in a Singularity container, alma in a plain subprocess,
harness baselines call it directly)."""
from __future__ import annotations

import copy
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

# stage_<ds> objective value for a progressive=false single-stage evaluation. Numerically above
# the last gauntlet tier (3) so whole-split (progressive=false) entries are distinguishable in
# frontier telemetry; scores at stage 3 vs 4 are still NOT mutually comparable —
# the tier is uniform within a run.
FULL_STAGE = 4.0

_OLD_SIZE_FIELDS = ("eval_n_samples", "eval_n_qa", "check_n_samples", "check_n_qa")


def _benchmark_family(ds: str) -> str:
    """Map a dataset name to its stage-schema family."""
    if ds.startswith("longmemeval"):
        return "longmemeval"
    if ds not in _FAMILY_FIELDS:
        raise ValueError(
            f"Unknown benchmark {ds!r} — no stage schema. Known families: "
            f"{sorted(_FAMILY_FIELDS)} (longmemeval_s maps to the 'longmemeval' family)."
        )
    return ds


def _wire_size(v: Any) -> Optional[int]:
    """Stage size field → wire value. None (a `null` / `full` / `all` field,
    normalized by _resolve_dataset_stages) passes through as None = no cap
    (whole pool / whole split), reusing the same downstream machinery as
    the whole split; anything else is coerced to int."""
    return None if v is None else int(v)


def stage_wire_spec(ds: str, stage_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stage's benchmark-native size fields to the wire spec the
    container consumes: {"n_samples": int|None, ...family extras}. A None
    field means the whole split for that dimension. `threshold` is an
    orchestrator-side concern and is stripped."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": _wire_size(stage_params[sample_field])}
    for f in extras:
        spec[f] = _wire_size(stage_params[f])
    return spec


def full_wire_spec(ds: str) -> Dict[str, Any]:
    """Wire spec: every size field None = no cap (the
    container's env/get_task_list and the workflows treat None as "whole
    split / all checkpoints / whole buckets")."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": None}
    for f in extras:
        spec[f] = None
    return spec


def single_stage_wire_spec(ds: str, single_stage_params: Dict[str, Any]) -> Dict[str, Any]:
    """Wire spec for the progressive=false single pass — the `single_stage`
    block, same size fields as a stage (NO threshold). null field → None (whole
    split for that dimension)."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": _wire_size(single_stage_params.get(sample_field))}
    for f in extras:
        spec[f] = _wire_size(single_stage_params.get(f))
    return spec


def _resolve_single_stage_block(ds: str, params: Dict[str, Any]) -> None:
    """Validate + normalize the `single_stage` block (progressive=false sizing)
    in place, when present. Same size fields as a stage, NO threshold; a
    null/"full"/"all" field → None (whole split for that dim); an omitted field
    → None. A missing/None block is left as-is — required-ness is enforced by
    resolve_single_stage_spec (which knows `progressive`)."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    single = params.get("single_stage")
    if single is None:
        return
    if not isinstance(single, dict):
        raise ValueError(f"datasets.{ds}.single_stage must be a mapping")
    ss_allowed = {sample_field, *extras}   # NO threshold
    unknown = set(single) - ss_allowed
    if unknown:
        raise ValueError(
            f"datasets.{ds}.single_stage: unknown field(s) {sorted(unknown)} "
            f"for family {family!r}; allowed: {sorted(ss_allowed)} (no threshold)"
        )
    for field in (sample_field, *extras):
        v = single.get(field)
        if isinstance(v, str) and v.strip().lower() in ("full", "all"):
            single[field] = None
        single.setdefault(field, None)   # omitted field → null = whole for that dim


def resolve_single_stage_spec(ds: str, single_stage: Optional[Dict[str, Any]]
                              ) -> Dict[str, Any]:
    """Presence-check + validate + normalize the `single_stage` block for the
    progressive=false single pass, returning its wire spec. Self-validating:
    rejects unknown fields / a stray threshold even when called directly (does
    NOT depend on a prior _resolve_dataset_stages call). An ABSENT block (None)
    raises ValueError — but an empty `{}` counts as PRESENT (all-null = whole
    split), NOT absent. The single source of the progressive=false sizing
    semantics, shared by forge (resolve_sampling_plan) AND the baseline
    single-pass paths (harness eval_utility, alma launch)."""
    if single_stage is None:
        sample_field = _FAMILY_FIELDS[_benchmark_family(ds)][0]
        raise ValueError(
            f"datasets.{ds}: progressive=false requires a `single_stage` block "
            f"(same size fields as a stage; use all-null for the whole split), "
            f"e.g. single_stage: {{{sample_field}: null}}"
        )
    # Validate + normalize a throwaway copy so the caller's dict is untouched.
    _ss = {"single_stage": copy.deepcopy(single_stage)}
    _resolve_single_stage_block(ds, _ss)
    return single_stage_wire_spec(ds, _ss["single_stage"])


def resolve_sampling_plan(ds: str, params: Dict[str, Any], progressive: bool
                          ) -> List[Tuple[str, Dict[str, Any], Optional[float]]]:
    """Unified sampling plan for one dataset, shared by forge + baselines.
    progressive=True  → the staged gauntlet (stage1..3 + thresholds).
    progressive=False → ONE pass sized by the REQUIRED `single_stage` block
    (via resolve_single_stage_spec: absent → ValueError; empty {} = whole)."""
    if progressive:
        return stage_plan(ds, params)
    return [("single", resolve_single_stage_spec(ds, params.get("single_stage")), None)]


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


def missing_sizing_config(dataset: str, params: Dict[str, Any], progressive: bool,
                           path_prefix: str) -> List[str]:
    """Missing sizing leaf paths (strict mode) for ONE dataset's RAW yaml
    `params`. progressive → `stages` with all four entries, each listing its
    full native size fields (+ `threshold` on stage1/stage2); else →
    `single_stage` with its full native size fields. A null/absent block or any
    missing leaf is a returned path (dotted, prefixed by path_prefix). [] when
    complete. VALIDATION-ONLY — does not mutate params."""
    sample_field, extras = _FAMILY_FIELDS[_benchmark_family(dataset)]
    size_fields = [sample_field, *extras]
    pre = f"{path_prefix}." if path_prefix else "."
    missing = []
    if progressive:
        stages = (params or {}).get("stages")
        if not isinstance(stages, dict):
            return [f"{pre}stages"]
        for entry in ("sanity_check", "stage1", "stage2", "stage3"):
            block = stages.get(entry)
            if not isinstance(block, dict):
                missing.append(f"{pre}stages.{entry}")
                continue
            need = size_fields + (["threshold"] if entry in ("stage1", "stage2") else [])
            missing += [f"{pre}stages.{entry}.{f}" for f in need if f not in block]
    else:
        single = (params or {}).get("single_stage")
        if not isinstance(single, dict):
            return [f"{pre}single_stage"]
        missing += [f"{pre}single_stage.{f}" for f in size_fields if f not in single]
    return missing


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
        # Whole-split sentinel: a size field of null / "full" / "all"
        # (case-insensitive) → None = that dimension's whole pool / split,
        # reusing the whole-split downstream None path. Intended for stage3
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
    # None (whole split) counts as the largest value, so it must not precede
    # a concrete size — a null/full field forces every later stage full too.
    for field in (sample_field, *extras):
        seq = [stages[name][field] for name in STAGE_ORDER]
        cmp = [float("inf") if v is None else int(v) for v in seq]
        if any(b < a for a, b in zip(cmp, cmp[1:])):
            raise ValueError(
                f"datasets.{ds}.stages: {field} must be non-decreasing across "
                f"stage1..stage3 (got {seq}) — staged nesting depends on it; "
                f"a null/full field (= whole split) may only be followed by "
                f"another null/full field"
            )

    # `single_stage` block (progressive=false sizing): validated + normalized in
    # place when present. Required-ness is enforced by resolve_single_stage_spec
    # (which knows `progressive`), not here.
    _resolve_single_stage_block(ds, params)


# ---------------------------------------------------------------------------
# Gauntlet driver — the promotion/elimination loop, shared by forge AND the
# baselines. The stage EXECUTION (container exec for forge, in-process for the
# baselines) is injected as `run_stage_fn`; the output dir + real split are the
# CALLER's concern (captured in its closure), so they are NOT parameters here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# evaluate_memo — the single, execution-independent evaluation entry point.
# ---------------------------------------------------------------------------

def _build_score_json(recorder_list: List[Any]) -> Dict[str, Any]:
    """Summarize a recorder list into the canonical score.json dict (mean per-user
    reward + SE, per_user details, invalid_users). Promoted from alma's launch.py
    (eval_utility used to borrow alma's copy) — the ONE score-summary builder."""
    rewards: List[float] = []
    per_user: Dict[str, Any] = {}
    invalid_users: List[Dict[str, Any]] = []

    for rec in recorder_list:
        if isinstance(rec, Exception):
            uid = getattr(rec, "user_id", "unknown")
            invalid_users.append({"user_id": uid, "error": repr(rec)})
            continue
        try:
            uid = getattr(rec, "user_id", "") or "unknown"
            reward = float(getattr(rec, "reward", 0.0))
            steps = getattr(rec, "steps", [])
            fi = getattr(rec, "failure_info", None)
            rewards.append(reward)
            per_user[uid] = {"reward": reward, "n_qa": len(steps), "failure_info": fi}
        except Exception as exc:
            invalid_users.append({"user_id": "unknown", "error": f"record access error: {exc!r}"})

    if rewards:
        mean = sum(rewards) / len(rewards)
        if len(rewards) > 1:
            var = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
            se = (var ** 0.5) / (len(rewards) ** 0.5)
        else:
            se = 0.0
    else:
        mean, se = 0.0, 0.0

    return {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": mean,
            "benchmark_overall_eval_standard_deviation": se,
        },
        "per_user": per_user,
        "invalid_users": invalid_users,
    }


def _per_user_stddev(score: Dict[str, Any]) -> Optional[float]:
    """Population stddev over per_user rewards (None with <2 users) — the
    robustness telemetry forge's host used to compute off score.json."""
    rewards = [float(u.get("reward", 0.0)) for u in score.get("per_user", {}).values()]
    if len(rewards) < 2:
        return None
    mean = sum(rewards) / len(rewards)
    return (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5


def _stage_cost_metrics(stage_usage: Dict[str, Any],
                        workflow: Any) -> Dict[str, Any]:
    """Cost block for one stage's metrics dict, from that stage's usage delta.

    `tokens` stays the ALL-PHASE total (unchanged semantics — forge's
    `tokens_total` frontier objective and every frontier entry recorded before
    the phase dimension existed mean exactly this). The phase split arrives as
    ADDITIONAL keys rather than by redefining the old one:

      tokens_build     the cost of HAVING the memory — the issue's headline
      tokens_retrieve  memory-system work at query time
      tokens_memory    build + retrieve = the full cost of operating the memory
      tokens_answer    the QA agent (cost of answering, not of the memory)
      tokens_judge     evaluation overhead — never part of a cost claim
      llm_calls        request count, all phases

    `build_cache_hits` rides along because a cached build spends no tokens:
    a `tokens_build` of 0 next to a non-zero hit count means "not measured
    here", not "free".
    """
    from common.tokens import (ANSWER, BUILD, JUDGE, MEMORY_PHASES, RETRIEVE,
                               phase_tokens, total_tokens)
    return {
        "tokens": total_tokens(stage_usage),
        "tokens_build": phase_tokens(stage_usage, BUILD),
        "tokens_retrieve": phase_tokens(stage_usage, RETRIEVE),
        "tokens_memory": phase_tokens(stage_usage, *MEMORY_PHASES),
        "tokens_answer": phase_tokens(stage_usage, ANSWER),
        "tokens_judge": phase_tokens(stage_usage, JUDGE),
        "llm_calls": int(stage_usage.get("totals", {}).get("calls", 0)),
        "phase_seconds": dict(stage_usage.get("phase_seconds", {})),
        "build_cache_hits": int(getattr(workflow, "build_cache_hits", 0)),
        "build_cache_misses": int(getattr(workflow, "build_cache_misses", 0)),
    }


def _memory_token_metrics(recorder_list: List[Any]) -> Dict[str, Any]:
    """Memory tokens for one stage, summed over its users.

    `memory_tokens_per_query` divides by the queries actually PROMPTED —
    steps where retrieval returned and a QA prompt was built. A query whose
    QA call then failed still had the memory stitched into a prompt, so its
    cost was genuinely incurred. `memory_tokens_n_answered` carries the
    stricter denominator so the other reading stays recoverable; neither is
    the sampled count, which differs whenever a query fails.
    """
    total = 0
    n_prompted = 0
    n_answered = 0
    for rec in recorder_list:
        if isinstance(rec, Exception):
            continue
        total += int(getattr(rec, "memory_tokens_total", 0))
        n_prompted += int(getattr(rec, "memory_tokens_n", 0))
        n_answered += int(getattr(rec, "memory_tokens_answered_n", 0))
    return {
        "memory_tokens_total": total,
        "memory_tokens_per_query": (total / n_prompted) if n_prompted else 0.0,
        "memory_tokens_n_queries": n_prompted,
        "memory_tokens_n_answered": n_answered,
    }


#: Cost fields accumulated ACROSS executed stages (cost accounting spans the
#: whole gauntlet, not just the last stage). `phase_seconds` and the cache
#: tallies are merged separately — they are not flat ints.
_CUMULATIVE_COST_FIELDS = (
    "tokens", "tokens_build", "tokens_retrieve", "tokens_memory",
    "tokens_answer", "tokens_judge", "llm_calls",
    "build_cache_hits", "build_cache_misses",
    "memory_tokens_total", "memory_tokens_n_queries", "memory_tokens_n_answered",
    # NOTE: memory_tokens_per_query is a RATIO — recomputed from the
    # accumulated total/denominator, never summed.
)


def _memo_source_dir(memo_class: type) -> Optional["Path"]:
    """Directory holding the memo module — fingerprinted (common.memory_cache.
    harness_fingerprint) so a change to the memo's source invalidates the
    cross-stage cache. `memo_class` may be a configured wrapper subclass whose
    source file is elsewhere; walk the MRO to the first class defined in its own
    real file (skipping this module)."""
    import inspect
    from pathlib import Path
    this_file = Path(__file__).resolve()
    for cls in memo_class.__mro__:
        try:
            src = inspect.getsourcefile(cls)
        except TypeError:
            src = None
        if not src:
            continue
        srcp = Path(src).resolve()
        if srcp == this_file:
            continue
        if cls.__module__ in ("common.memo_class", "forge.memo_class", "abc", "builtins"):
            break  # reached the framework base — no useful source dir above it
        return srcp.parent
    return None


async def evaluate_memo(
    *,
    memo_class: type,
    dataset: str,
    split: str,
    progressive: bool,
    out_dir: "Path",
    qa_model: str,
    judge_model: str,
    stages: Optional[Dict[str, Any]] = None,
    single_stage: Optional[Dict[str, Any]] = None,
    max_sample_concurrent: int = 3,
    sample_seed: Optional[str] = None,
    memory_cache: bool = True,
    memcache_fingerprint: Optional[str] = None,
    memcache_dir: Optional["Path"] = None,
    smoke: bool = False,
    max_logs: Optional[int] = None,
    memo_sha: str = "",
    memo_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one memo on one dataset/split — the single, EXECUTION-INDEPENDENT
    entry point. It runs the whole evaluation in the CURRENT process; isolation
    is the CALLER's wrapper, not a parameter: forge's container runs this inside
    `singularity exec` (launch.py), alma inside its plain subprocess, the harness
    baselines call it directly in-process. No injected executor.

    Modes:
    - ``smoke=True``          → ONE sanity_check-sized pass (sizes from the
      `stages` block / family DEFAULT_STAGES), artifacts at the out_dir ROOT,
      telemetry stage=0.0, no stages.json. Used by forge's sanity gate +
      smoke_test and alma's mode=check.
    - ``progressive=True``    → the staged stage1→2→3 gauntlet with promotion
      thresholds, sized by `stages` (None → family DEFAULT_STAGES).
    - ``progressive=False``   → ONE ('single', spec, None) pass sized by the
      REQUIRED `single_stage` block (raises ValueError if absent).

    Per executed stage it writes ``out_dir/<stage>/{score.json, token_usage.json,
    traces/}``; afterwards the reached stage's score.json / token_usage.json /
    traces/ are copied to the out_dir root and a ``stages.json`` summary is
    written (non-smoke). A root score.json is guaranteed even when the first
    stage crashes (error score), so callers can always read one.

    Returns the metrics dict, accumulated across every executed stage:

      raw_score, score_max, per_user_stddev, stage, eliminated
      tokens                     ALL-PHASE token total (unchanged meaning —
                                 forge's `tokens_total` objective)
      tokens_build               cost of BUILDING the memory
      tokens_retrieve            memory-system work at query time
      tokens_memory              build + retrieve
      tokens_answer/_judge       QA agent / evaluation overhead
      llm_calls                  request count
      phase_seconds              wall-clock per phase — the only cost signal
                                 that also covers local models
      build_cache_hits/_misses   a cached build spends no tokens; a
                                 tokens_build of 0 beside a non-zero hit
                                 count means "not measured", not "free"
      memory_tokens_total        tokens the memory added to the QA prompts
      memory_tokens_per_query    the recurring cost of USING the memory
      memory_tokens_n_queries    denominator: queries actually prompted
      memory_tokens_n_answered   stricter denominator (answer succeeded)

    (stage: 1/2/3 gauntlet tier, FULL_STAGE for the single pass, 0.0 for smoke.)

    ``sample_seed`` rides every stage spec (constant across stages so the
    stage1 ⊂ stage2 ⊂ stage3 nesting holds). ``memory_cache`` mounts the
    cross-stage Phase-1 memory cache at ``memcache_dir`` (default
    out_dir/memory_cache; forge's container passes its persistent host-bound
    /memcache so snapshots survive across evaluator invocations), skipped for
    smoke; ``memcache_fingerprint`` overrides the source-dir fingerprint
    (alma fingerprints its single staged memo file; forge fingerprints the
    whole harness dir). ``max_logs`` / ``memo_sha`` are forwarded to the
    workflow (alma's knobs — forge passes memo_sha=harness id)."""
    import json as _json
    import shutil as _shutil
    from pathlib import Path
    from benchmarks.registry import resolve as _resolve_dataset
    from common.tokens import diff_summary as _diff_summary, init_global_tracker
    from common.memory_cache import harness_fingerprint as _dir_fingerprint

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    family = _benchmark_family(dataset)

    # ---- plan ----
    if smoke or progressive:
        params: Dict[str, Any] = {"stages": copy.deepcopy(stages) if stages else copy.deepcopy(DEFAULT_STAGES[family])}
        _resolve_dataset_stages(dataset, params)
    if smoke:
        plan: List[Tuple[str, Dict[str, Any], Optional[float]]] = [
            ("sanity", stage_wire_spec(dataset, params["stages"]["sanity_check"]), None)
        ]
    elif progressive:
        plan = stage_plan(dataset, params)
    else:
        plan = [("single", resolve_single_stage_spec(dataset, single_stage), None)]

    workflow_cls, env_module, _rec = _resolve_dataset(dataset)
    tracker = init_global_tracker()

    # ---- cross-stage memory cache (gauntlet + single; never smoke/sanity) ----
    fingerprint = ""
    if memory_cache and not smoke:
        fingerprint = memcache_fingerprint if memcache_fingerprint is not None else ""
        if not fingerprint:
            src_dir = _memo_source_dir(memo_class)
            fingerprint = _dir_fingerprint(src_dir) if src_dir is not None else ""
        memcache_dir = Path(memcache_dir) if memcache_dir is not None else out_dir / "memory_cache"
        memcache_dir.mkdir(parents=True, exist_ok=True)
    else:
        memcache_dir = None

    stage_summary: Dict[str, Any] = {}
    final_metrics: Dict[str, Any] = {}
    reached = ""
    eliminated = False

    for stage_name, wire_spec, threshold in plan:
        stage_dir = out_dir if smoke else out_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec = dict(wire_spec)
        if sample_seed is not None:
            spec["sample_seed"] = sample_seed

        n = spec.get("n_samples")
        task_list = env_module.get_task_list(
            status=split,
            eval_n_samples=None if n is None else int(n),
            seed=spec.get("sample_seed"),
        )

        workflow = workflow_cls(
            memo_class=memo_class, model=qa_model, max_logs=max_logs,
            judge_model=judge_model, memo_config=memo_config,
        )
        workflow.memo_sha = memo_sha
        workflow.status = split
        workflow.output_run_dir = stage_dir
        if memcache_dir is not None:
            workflow.memory_cache_dir = memcache_dir
            workflow.harness_fingerprint = fingerprint

        # The tracker is process-global and CUMULATIVE, but each stage's
        # token_usage.json must describe that stage alone — snapshot before,
        # subtract after. (Before this, stage2's file silently contained
        # stage1's tokens while the `tokens` metric beside it was a correct
        # delta, and the two disagreed.)
        usage_before = tracker.summary()
        raw_score = 0.0
        stddev: Optional[float] = None
        crashed: Optional[Exception] = None
        stage_usage: Dict[str, Any] = {}
        stage_records: List[Any] = []
        try:
            records, rlen = await workflow.run_all_users(
                task_list, stage=stage_name, stage_spec=spec,
                max_sample_concurrent=max_sample_concurrent,
            )
            stage_records = records[:rlen]
            workflow.save_full_traces(stage_records)
            score = _build_score_json(stage_records)
            raw_score = float(score["benchmark_eval_score"]["benchmark_overall_eval_score"])
            stddev = _per_user_stddev(score)
            stage_usage = _diff_summary(tracker.summary(), usage_before)
            with (stage_dir / "score.json").open("w", encoding="utf-8") as f:
                _json.dump(score, f, indent=2, ensure_ascii=False)
            with (stage_dir / "token_usage.json").open("w", encoding="utf-8") as f:
                _json.dump(stage_usage, f, indent=2, ensure_ascii=False)
        except Exception as exc:  # a crashed stage eliminates + stops
            crashed = exc
        # A crashed stage still burned tokens before it died — count them.
        if not stage_usage:
            stage_usage = _diff_summary(tracker.summary(), usage_before)

        cost = {**_stage_cost_metrics(stage_usage, workflow),
                **_memory_token_metrics(stage_records)}
        m: Dict[str, Any] = {
            "raw_score": raw_score,
            "score_max": workflow.judge_score_max,
            "per_user_stddev": stddev,
            **cost,
        }
        score_max = int(m["score_max"]) or 1
        normalized = raw_score / score_max
        # stages.json keeps THIS stage's cost; `m` below accumulates across
        # stages. `cost` is deep-copied into both so the accumulation cannot
        # mutate the per-stage record through the shared phase_seconds dict.
        stage_summary[stage_name] = {
            "raw_score": raw_score,
            "score_max": m["score_max"],
            "normalized": normalized,
            "threshold": threshold,
            **copy.deepcopy(cost),
            "spec": spec,
        }
        m["phase_seconds"] = dict(cost["phase_seconds"])
        # Cost accounting spans ALL executed stages, not just the last.
        for k, s in stage_summary.items():
            if k == stage_name:
                continue
            for field in _CUMULATIVE_COST_FIELDS:
                m[field] += int(s.get(field, 0))
            for ph, secs in s.get("phase_seconds", {}).items():
                m["phase_seconds"][ph] = round(m["phase_seconds"].get(ph, 0.0) + secs, 3)
        # A ratio, so recomputed from the accumulated pair rather than summed.
        m["memory_tokens_per_query"] = (
            m["memory_tokens_total"] / m["memory_tokens_n_queries"]
            if m["memory_tokens_n_queries"] else 0.0
        )
        final_metrics = m
        reached = stage_name
        if crashed is not None:
            eliminated = True
            stage_summary[stage_name]["crashed"] = repr(crashed)
            break
        if threshold is not None and normalized < threshold:
            eliminated = True
            break

    if smoke:
        final_metrics["stage"] = 0.0  # sanity-size smoke run, not a gauntlet tier
        final_metrics["eliminated"] = False
    else:
        final_metrics["stage"] = {
            "stage1": 1.0, "stage2": 2.0, "stage3": 3.0, "single": FULL_STAGE,
        }.get(reached, 0.0)
        final_metrics["eliminated"] = eliminated
        # Reached-stage artifacts up to the root (score/tokens/traces) + stages.json.
        reached_dir = out_dir / reached
        for fname in ("score.json", "token_usage.json"):
            src = reached_dir / fname
            if src.exists():
                _shutil.copy2(src, out_dir / fname)
        src_traces = reached_dir / "traces"
        if src_traces.is_dir():
            dst_traces = out_dir / "traces"
            if dst_traces.exists():
                _shutil.rmtree(dst_traces)
            _shutil.copytree(src_traces, dst_traces)
        with (out_dir / "stages.json").open("w", encoding="utf-8") as f:
            _json.dump(
                {"reached": reached, "eliminated": eliminated, "stages": stage_summary},
                f, indent=2, ensure_ascii=False,
            )

    # Guarantee a root score.json even if the first stage crashed before writing
    # one (callers — alma's memo_manager, forge's sanity gate — always read one).
    if not (out_dir / "score.json").exists():
        err = stage_summary.get(reached, {}).get("crashed", "no stage produced a score")
        with (out_dir / "score.json").open("w", encoding="utf-8") as f:
            _json.dump({
                "benchmark_eval_score": {
                    "benchmark_overall_eval_score": 0.0,
                    "benchmark_overall_eval_standard_deviation": 0.0,
                },
                "per_user": {},
                "invalid_users": [{"user_id": "eval_crashed", "error": str(err)}],
            }, f, indent=2, ensure_ascii=False)

    return final_metrics
