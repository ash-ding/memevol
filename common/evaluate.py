"""Shared staged-evaluation config + gauntlet driver, used by forge AND the
baselines (which must not import forge). The stage sizes/plan/nesting live here;
run_gauntlet (Task 5) drives the promotion loop with an injected stage runner.
Moved from forge/orchestrator.py 2026-07-25."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
    single-pass paths (harness eval_common, alma launch)."""
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

# Type aliases for the injected seam (documentation only; not enforced).
RunStageFn = Callable[[str, str, Dict[str, Any]], Awaitable[Optional[Exception]]]
ReadMetricsFn = Callable[[str, str], Dict[str, Any]]
SampleSeedFn = Callable[[str], Optional[str]]
StagesWriter = Callable[[str, Dict[str, Any]], None]


async def run_gauntlet(
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    coverage: str,
    smoke: bool,
    sample_seed_for: SampleSeedFn,
    run_stage_fn: RunStageFn,
    read_metrics_fn: ReadMetricsFn,
    stages_writer: Optional[StagesWriter] = None,
) -> Dict[str, Dict[str, Any]]:
    """Drive the per-benchmark staged gauntlet.

    Execution is injected via ``run_stage_fn(ds, stage_name, spec) ->
    Optional[Exception]`` (forge: container; baseline: in-process). The
    promotion / elimination / cost-accounting / telemetry / stages.json logic
    lives HERE (identical for forge and baselines). ``read_metrics_fn(ds,
    stage_name) -> Dict`` reads that stage's metrics (raw_score, score_max,
    tokens). ``sample_seed_for(ds) -> Optional[str]`` supplies the per-run seed,
    injected into every stage spec as ``spec['sample_seed']`` (constant across a
    run's stages so stage1 ⊂ stage2 ⊂ stage3 nesting holds); when it returns
    None the spec is left untouched. ``stages_writer(ds, summary)`` (optional)
    persists the per-benchmark stage summary (forge writes stages.json + copies
    the final-stage artifacts to the dataset root).

    Behavior mirrors forge's former inline ``evaluate_harness`` loop exactly:

    - ``smoke=True``: one ``sanity_check``-sized run per benchmark, no gating,
      telemetry ``stage=0.0`` / ``eliminated=False`` (stages_writer NOT called).
    - ``coverage='full'`` (progressive=False): a single ``('single', spec, None)``
      plan (no promotion gates), sized by the dataset's REQUIRED `single_stage`
      config block via ``resolve_sampling_plan`` — raises ``ValueError`` if that
      dataset has no `single_stage` block (no more automatic whole-split
      `full_wire_spec`; sizing is config-file only). Reached stage telemetry is
      ``FULL_STAGE`` (the "full" plan name is legacy and no longer emitted, but
      still mapped for back-compat).
    - ``coverage='sample'`` (default, progressive=True): the stage1 → stage2 →
      stage3 gauntlet from the dataset's `stages` block; after a gated stage
      the normalized score must be >= its threshold to advance, else the
      benchmark stops ("eliminated"). A crashed stage (a non-None
      ``run_stage_fn`` return) also eliminates and stops.

    Cost accounting sums tokens across ALL executed stages. Returns the same
    per-dataset metrics dict shape forge builds today (raw_score, score_max,
    per_user_stddev, tokens, stage, eliminated).
    """
    scores: Dict[str, Dict[str, Any]] = {}
    for ds, params in datasets_config.items():
        seed = sample_seed_for(ds)

        def _spec(base: Dict[str, Any]) -> Dict[str, Any]:
            # Copy so the caller's plan specs are never mutated; the per-run
            # seed rides along in every stage spec (constant → nesting holds).
            s = dict(base)
            if seed is not None:
                s["sample_seed"] = seed
            return s

        if smoke:
            # Sanity-size single run, no gating, artifacts at the dataset root.
            spec = _spec(stage_wire_spec(ds, params["stages"]["sanity_check"]))
            await run_stage_fn(ds, "sanity", spec)
            m = read_metrics_fn(ds, "sanity")
            m["stage"] = 0.0  # 0 = sanity-size smoke run (not a gauntlet tier)
            m["eliminated"] = False
            scores[ds] = m
            continue

        # ---- staged gauntlet (progressive=True) OR single pass (progressive=False,
        # sized by the REQUIRED `single_stage` block — raises if absent) ----
        plan: List[Tuple[str, Dict[str, Any], Optional[float]]] = resolve_sampling_plan(
            ds, params, progressive=(coverage != "full")
        )

        stage_summary: Dict[str, Any] = {}
        final_metrics: Dict[str, Any] = {}
        reached = ""
        eliminated = False
        for stage_name, base_spec, threshold in plan:
            spec = _spec(base_spec)
            crashed = await run_stage_fn(ds, stage_name, spec)
            m = read_metrics_fn(ds, stage_name)
            score_max = int(m.get("score_max", 10)) or 1
            normalized = float(m.get("raw_score", 0.0)) / score_max
            stage_summary[stage_name] = {
                "raw_score": m["raw_score"],
                "score_max": m["score_max"],
                "normalized": normalized,
                "threshold": threshold,
                "tokens": m["tokens"],
                "spec": spec,
            }
            # Cost accounting spans ALL executed stages, not just the last.
            # (Guarded so non-numeric token payloads — e.g. injected test fakes
            #  — don't blow up; forge's tokens are always ints, so this is a
            #  no-op for the real path.)
            if isinstance(m.get("tokens"), (int, float)):
                m["tokens"] += sum(
                    s["tokens"] for n, s in stage_summary.items() if n != stage_name
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

        stage_num = {
            "stage1": 1.0, "stage2": 2.0, "stage3": 3.0,
            "full": FULL_STAGE,     # legacy plan name (no other caller emits it anymore)
            "single": FULL_STAGE,   # progressive=false single pass — same tier as "full"
        }.get(reached, 0.0)
        final_metrics["stage"] = stage_num
        final_metrics["eliminated"] = eliminated
        if stages_writer is not None:
            stages_writer(
                ds, {"stages": stage_summary, "reached": reached, "eliminated": eliminated}
            )
        scores[ds] = final_metrics
    return scores


# ---------------------------------------------------------------------------
# evaluate_memo — the single evaluation entry point + the executor seam.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Executor:
    """HOW one stage of an evaluation runs — the ONE thing that differs by
    execution model (the trust boundary): a Singularity container per stage for
    forge (untrusted, model-generated harness code) vs a bare in-process
    ``run_all_users`` for the baselines (trusted hand-written code). It bundles
    the three run_gauntlet callbacks a caller must supply:

    - ``run_stage(ds, stage, spec) -> Optional[Exception]`` executes one stage
      (build the memo, run the workflow over the sampled users, write that
      stage's score.json/token_usage.json/traces), returning a non-None
      exception iff the stage crashed (which eliminates + stops the gauntlet).
    - ``read_metrics(ds, stage) -> {raw_score, score_max, tokens}`` reads that
      stage's metrics back (off disk for forge's container, from an in-memory
      dict for the in-process runners).
    - ``write_stages(ds, summary)`` (optional) persists the per-benchmark stage
      summary — stages.json at the out_dir root + the reached-stage artifacts
      copied up to the root.

    Everything else — promotion / elimination / sizing / seeding / cost
    accounting / stages.json shape — lives in run_gauntlet, identical for every
    execution model."""
    run_stage: RunStageFn
    read_metrics: ReadMetricsFn
    write_stages: Optional[StagesWriter] = None


async def evaluate_memo(
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    progressive: bool,
    executor: Executor,
    sample_seed_for: SampleSeedFn = lambda ds: None,
    smoke: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a memo across benchmarks — the SINGLE entry every context uses
    (forge search, forge.heldout, harness baselines, alma), so "evaluate a
    harness on a task" is one function regardless of who calls it.

    - ``progressive=True``  → the staged stage1→2→3 gauntlet (sizes from each
      dataset's ``stages`` block; promotion thresholds gate advancement).
    - ``progressive=False`` → ONE ('single', spec, None) pass sized by the
      REQUIRED ``single_stage`` block (raises ValueError if absent).
    - ``smoke=True`` → one sanity-size pass per benchmark, no gating.

    ``datasets_config`` carries the per-dataset sizing (``stages`` /
    ``single_stage``). HOW a stage executes is injected via ``executor`` (the
    container-vs-in-process seam); ``sample_seed_for(ds)`` supplies the per-run
    seed. Thin facade over run_gauntlet: maps progressive → coverage and forwards
    the executor's three callbacks. Returns run_gauntlet's per-dataset metrics
    dict {ds: {raw_score, score_max, stage, eliminated, tokens, ...}}."""
    return await run_gauntlet(
        datasets_config=datasets_config,
        coverage=("sample" if progressive else "full"),
        smoke=smoke,
        sample_seed_for=sample_seed_for,
        run_stage_fn=executor.run_stage,
        read_metrics_fn=executor.read_metrics,
        stages_writer=executor.write_stages,
    )
