"""Forge outer loop.

Per-harness flow (`mode=search` or `test`):

    propose → ensure_image
           → sanity_check_harness (small real-data run per dataset, if
                                   sanity.enabled)
               ├─ pass    → evaluate_harness (full run per dataset) → frontier
               └─ fail    → propose_with_fix(error_trace) → retry (max K times)
                             ├─ eventually pass → evaluate_harness
                             └─ all retries exhausted → skip eval,
                                 enter frontier with sanity_failed + score=0

For `mode=dev`: skip sanity entirely and run the evaluator once per
benchmark at `stages.sanity_check` sizes (dev IS the sanity-size verification).

Two ways to launch:

    # (a) config-first (recommended for long runs / reproducibility)
    python -m forge.orchestrator --config configs/search.yaml

    # (b) CLI-first (quick iteration; uniform params across all datasets)
    python -m forge.orchestrator --steps 3 --datasets dynamicmem,locomo \
        --mode dev

CLI flags always override the config file when both are given. See
`configs/smoke.yaml` and `configs/search.yaml` for the full config schema.

Outputs (per-run, under workspace/<run_id>/):
    harnesses/<int>_<hash>/<dataset>/         full-eval score + traces
    harnesses/<int>_<hash>/<dataset>/sanity/  small sanity-check score + traces
    harnesses/<int>_<hash>/harness.py         the harness code itself
    harnesses/<int>_<hash>/meta.json          parent_ids, description, content_hash, created_at
    frontier.json                             population snapshot (with sanity_status)
    forge/logs/orchestrator.log               rotating log (shared across runs)

Per-harness eval over multiple datasets is serial (one Singularity exec per
dataset). `Frontier.entries[i].objectives` stores both the mean ("accuracy")
and per-dataset scores ("accuracy_<dataset>"). v4: parent selection is
delegated to the proposer agent — the search loop has NO selection rule.
v5: each orchestrator invocation gets its own per-run dir (`--run-name`
or auto timestamp); harness dirs are renamed `<int>` → `<int>_<hash8>`
after sanity loop settles, before full eval.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from forge.env_builder import EnvBuildError, ensure_image
from forge.evaluator import run_evaluation
from forge.paths import (
    LOGS_DIR,
    SEEDS_DIR,
    ensure_dirs,
    paths,
)
from forge.prompts import PromptVersionError, load_template_module, resolve_version
from forge.proposer import propose, propose_with_fix
from forge.selection import Entry, Frontier

log = logging.getLogger("forge.orchestrator")


class ProposerInfraError(RuntimeError):
    """Raised when the proposer subprocess itself fails (rc!=0, timeout, auth).

    Distinct from harness-level failures (sanity_failed, env_build, eval crash)
    which the search loop continues past — those are CC's design mistakes that
    the search is supposed to learn from. ProposerInfraError means the proposer
    subprocess never completed (no harness.py written), so there's no signal
    to learn from and continuing would just generate a placeholder entry that
    pollutes the frontier and confuses resume logic.

    Triggers (a) cleanup of the empty harness dir, (b) abort of search_loop,
    (c) main()'s sys.exit(2) so wrapper scripts can detect.
    """


# ---------------------------------------------------------------------------
# Config resolution: defaults ← YAML file ← CLI overrides
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "steps": 1,
    # search — training/exploration on the search (training) split, full staged gauntlet
    # test   — held-out evaluation on the test split, full staged gauntlet
    # dev    — run the whole pipeline at stages.sanity_check sizes only (search split, no gauntlet, no sanity gate)
    "mode": "search",
    "model": "gpt-5-mini",
    "judge_model": "gpt-5-mini",
    "update_type": "all_at_once",
    "max_sample_concurrent": 3,
    # Cross-stage memory cache: persist each user's Phase-1 memory per
    # (harness, dataset) and reuse it at deeper gauntlet stages (nested
    # sampling makes it bit-for-bit reusable). Stage1..3 only; sanity/dev
    # never touch it. Disable via `memory_cache: false` or --no-memory-cache.
    "memory_cache": True,
    "proposer": {
        # Generic propose-time controls (shared by both agents).
        "max_turns": 80,
        "timeout_s": 25 * 60,
        # Per-agent subsections. Only the subsection matching cfg["agent"] is
        # consumed at propose-time; the other is ignored. Each agent has its
        # own model + agent-specific knobs. Fields set to null (None) fall back
        # to the agent's CLI default (no flag passed).
        "claude_code": {
            "model": "claude-opus-4-7",
            # low / medium / high / xhigh / max  (CC --effort)
            "effort": "medium",
            # Tools the agent CLI is told not to consider. Defense-in-depth
            # only — actual filesystem isolation is the Singularity bind list.
            # Default ["mcp__*"] keeps CC from listing host MCP servers (which
            # are unreachable anyway due to --containall, but pruning them
            # from the prompt avoids wasted turns).
            "disallowed_tools": ["mcp__*"],
        },
        "codex": {
            "model": "gpt-5.5",
            # low / medium / high  (codex -c model_reasoning_effort=...)
            "reasoning_effort": "medium",
            # No `disallowed_tools` field: codex's tool scope is controlled
            # by --sandbox mode (we use danger-full-access because the outer
            # Singularity sandbox is the real boundary), not per-tool blocking.
        },
    },
    "selection": {
        # tau is still consumed by Frontier.sample_parent if the held-out test
        # script needs a stochastic top-K pick. NOT used by search_loop —
        # parent selection is fully delegated to the proposer agent (v4).
        "tau": 0.5,
    },
    "propose": {
        # k harnesses proposed per outer-loop step. Default 2 follows the
        # Meta-Harness paper. Each is generated by an independent CC-SDK call.
        "k_per_step": 2,
    },
    "sanity": {
        "enabled": True,      # run the pre-eval sanity gate (mode=search/test; mode=dev always skips)
        "max_retries": 2,     # on sanity failure, call propose_with_fix up to this many times
    },
    "seed": {
        # When enabled, copy seeds/<source>/ → <run>/harnesses/0/ at startup
        # to give the proposer a known baseline to reference. Default is OFF
        # — CC writes the first harness from scratch with parent_ids=[].
        # Opt in via `seed: {enabled: true, source: <name>}` in YAML or the
        # `--seed` CLI flag.
        "enabled": False,
        "source": "no_memory",
    },
    "gpu": {
        # When enabled, evaluator runs `singularity exec --nv ...` so the
        # container can use the host's NVIDIA driver (libcuda.so) and any
        # GPUs visible at /dev/nvidia*. Default OFF for portability — turning
        # on couples runs to a host with a compatible NVIDIA driver. Opt in
        # via `gpu: {enabled: true}` in YAML or `--gpu` CLI flag.
        # Proposer container does NOT honor this (CC just makes API calls).
        "enabled": False,
    },
    # When True, search_loop's resume path scans `harnesses/` for dirs that
    # exist on disk but aren't in frontier (e.g. orchestrator killed mid-eval),
    # classifies them by file presence, and runs only the missing pipeline
    # stages before adding them to frontier. Disable via `--no-adopt-orphans`
    # if you suspect the orphan dirs are corrupt.
    "adopt_orphans": True,
    # Which coding-agent backend the proposer drives: "claude_code" (uses the
    # `claude` CLI in stream-json mode, default) or "codex" (OpenAI Codex CLI
    # in --json mode, auths via OPENAI_API_KEY).
    "agent": "claude_code",
    "prompts": {
        # Prompt template version stem under forge/prompts/templates/.
        # "latest" → read forge/prompts/templates/_default at startup. Pin to
        # an explicit stem (e.g. "20260518_2112_99812772") to lock a run to
        # a specific prompt revision — required for clean A/B comparisons.
        "version": "latest",
    },
    # `run_name` defaults to a timestamp at resolve time (so the default isn't
    # frozen at import time). `datasets` intentionally omitted from defaults.
}


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


def stage_wire_spec(ds: str, stage_params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a stage's benchmark-native size fields to the wire spec the
    container consumes: {"n_samples": int, ...family extras}. `threshold`
    is an orchestrator-side concern and is stripped."""
    family = _benchmark_family(ds)
    sample_field, extras = _FAMILY_FIELDS[family]
    spec: Dict[str, Any] = {"n_samples": int(stage_params[sample_field])}
    for f in extras:
        spec[f] = int(stage_params[f])
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
        # sanity_check and stage3 never gate
        if name in ("sanity_check", "stage3"):
            block.pop("threshold", None)
        thr = block.get("threshold")
        if thr is not None and not (0.0 <= float(thr) <= 1.0):
            raise ValueError(f"datasets.{ds}.stages.{name}.threshold must be in [0,1], got {thr}")

    # Monotonic non-decreasing sizes across stage1..3 (nesting depends on it).
    for field in (sample_field, *extras):
        seq = [int(stages[name][field]) for name in STAGE_ORDER]
        if any(b < a for a, b in zip(seq, seq[1:])):
            raise ValueError(
                f"datasets.{ds}.stages: {field} must be non-decreasing across "
                f"stage1..stage3 (got {seq}) — staged nesting depends on it"
            )


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """In-place deep merge of overlay into base (dicts only)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the effective run config from defaults, optional YAML, and CLI.

    Precedence (lowest → highest):
      1. DEFAULT_CONFIG (this module)
      2. YAML at --config path (if given)
      3. Explicit CLI args (only when not None)
    """
    cfg: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)

    if args.config is not None:
        path = Path(args.config)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if not isinstance(file_cfg, dict):
            raise ValueError(f"config file must be a mapping at top level: {path}")
        _deep_merge(cfg, file_cfg)

    if "status" in cfg:
        raise ValueError(
            "top-level `status:` was renamed to `mode:` "
            "(search / test / dev; the old `devtest` value is now `dev`). "
            "Update the YAML and re-run."
        )
    if "memory_dumps" in cfg:
        raise ValueError(
            "`memory_dumps:` was removed (2026-07-08) — the lossy post-Phase-1 "
            "memory dump mechanism is gone. To inspect a harness's built "
            "memory, unpickle harnesses/<id>/<ds>/memory_cache/*.pkl instead. "
            "Delete the field from the YAML and re-run."
        )
    if cfg.get("mode", "search") not in ("search", "test", "dev"):
        raise ValueError(
            f"mode must be one of search / test / dev, got {cfg.get('mode')!r} "
            "(the old `devtest` value is now `dev`)"
        )

    # Top-level CLI overrides (each applied only if explicitly provided)
    for key in (
        "steps", "mode", "model", "judge_model",
        "update_type", "max_sample_concurrent",
    ):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # Generic proposer overrides (apply to both agents).
    _proposer_map = {
        "proposer_max_turns": "max_turns",
        "proposer_timeout_s": "timeout_s",
    }
    for cli_key, cfg_key in _proposer_map.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            cfg["proposer"][cfg_key] = val
    # --proposer-model overrides the ACTIVE agent's model. We compute the
    # active agent here (CLI --agent > cfg.agent > DEFAULT_CONFIG.agent).
    if args.proposer_model is not None:
        active_agent = args.agent or cfg.get("agent", "claude_code")
        cfg["proposer"][active_agent]["model"] = args.proposer_model
    # disallowed_tools (claude_code-only; codex has no per-tool block).
    # Comma-separated. Empty string = allow everything.
    if args.proposer_disallowed_tools is not None:
        cfg["proposer"]["claude_code"]["disallowed_tools"] = [
            t.strip() for t in args.proposer_disallowed_tools.split(",") if t.strip()
        ]
    if args.tau is not None:
        cfg["selection"]["tau"] = args.tau
    if args.k_per_step is not None:
        cfg["propose"]["k_per_step"] = args.k_per_step
    if args.no_memory_cache:
        cfg["memory_cache"] = False
    if args.no_sanity:
        cfg["sanity"]["enabled"] = False
    if args.sanity_max_retries is not None:
        cfg["sanity"]["max_retries"] = args.sanity_max_retries
    if args.no_seed:
        cfg["seed"]["enabled"] = False
    if args.seed_source is not None:
        cfg["seed"]["source"] = args.seed_source
    if args.gpu:
        cfg["gpu"]["enabled"] = True
    if args.no_adopt_orphans:
        cfg["adopt_orphans"] = False
    if args.agent is not None:
        cfg["agent"] = args.agent
    if args.prompts_version is not None:
        cfg["prompts"]["version"] = args.prompts_version

    # run_name: CLI > YAML > timestamp default. Resolved here (not in
    # DEFAULT_CONFIG) so `from forge.orchestrator import DEFAULT_CONFIG`
    # doesn't freeze the timestamp at import time.
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    cfg.setdefault("run_name", _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Datasets
    cli_datasets = getattr(args, "datasets", None)
    if cli_datasets is not None:
        # CLI --datasets replaces the YAML datasets block entirely; each
        # benchmark gets its family's DEFAULT_STAGES (see that constant).
        ds_list = [d.strip() for d in cli_datasets.split(",") if d.strip()]
        cfg["datasets"] = {ds: {} for ds in ds_list}

    # Validation + stage-schema resolution
    if not cfg.get("datasets"):
        raise ValueError(
            "No datasets specified. Provide `datasets:` in --config YAML, "
            "or pass --datasets on the CLI."
        )
    for ds, params in cfg["datasets"].items():
        if not isinstance(params, dict):
            raise ValueError(
                f"datasets.{ds} must be a mapping, got {type(params).__name__}"
            )
        _resolve_dataset_stages(ds, params)
        # Per-dataset judge override; falls back to the global judge_model.
        # (DynamicMem pins the official TCE judge gpt-5.4-2026-03-05 in the
        # YAML; other benchmarks typically inherit the global value.)
        params.setdefault("judge_model", cfg["judge_model"])

    return cfg


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s"
)


def _setup_logging(verbose: bool) -> None:
    """Phase-1 logging: console + global rotating tape (run-agnostic).

    The per-run file handler is attached separately by `_attach_run_log` once
    `paths.set_run_id(...)` resolves the workspace path.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    root.addHandler(ch)
    # Global tape — every run appends here. Rotating to cap disk use.
    fh = RotatingFileHandler(
        LOGS_DIR / "orchestrator.log", maxBytes=5 * 1024 * 1024, backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(_LOG_FORMATTER)
    root.addHandler(fh)


def _attach_run_log() -> Path:
    """Phase-2 logging: per-run FileHandler at workspace/<run_id>/orchestrator.log.

    Must be called after `paths.set_run_id(...)`. Returns the log file path.
    """
    paths.workspace.mkdir(parents=True, exist_ok=True)
    log_path = paths.workspace / "orchestrator.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(_LOG_FORMATTER)
    logging.getLogger().addHandler(fh)
    return log_path


def _snapshot_view(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied cfg with the inactive proposer agent's subsection
    pruned, so the snapshot file + startup log line reflect what ACTUALLY ran.

    The live cfg keeps both `proposer.claude_code` and `proposer.codex` as a
    static catalog (DEFAULT_CONFIG provides both so `--agent <other>` resolves
    cleanly without a YAML edit). For human-facing views (snapshot YAML +
    startup log "run config:" dump) we drop the inactive one — otherwise
    readers are left guessing which agent's settings the run is actually using.
    """
    snap = copy.deepcopy(cfg)
    active = snap.get("agent", "claude_code")
    proposer = snap.get("proposer", {})
    for other in ("claude_code", "codex"):
        if other != active:
            proposer.pop(other, None)
    return snap


def _write_resolved_config(cfg: Dict[str, Any]) -> Path:
    """Snapshot the effective run config into workspace/<run_id>/config.yaml.

    Captures the dict AFTER defaults + YAML + CLI have been merged, so the
    file alone is enough to reproduce the run (regardless of whether the
    user passed a config file, plain CLI args, or both). Inactive agent's
    proposer subsection is dropped — see `_snapshot_view`.
    """
    paths.workspace.mkdir(parents=True, exist_ok=True)
    cfg_path = paths.workspace / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_snapshot_view(cfg), f, sort_keys=False, allow_unicode=True)
    return cfg_path


# ---------------------------------------------------------------------------
# Proposer config resolution
# ---------------------------------------------------------------------------

def _resolve_proposer_for_agent(
    cfg: Dict[str, Any], agent: str
) -> Tuple[str, Dict[str, Any]]:
    """Compute (model, agent_opts) for the active agent from cfg.proposer.

    Pure lookup against `cfg.proposer[agent]` — no fallback to a top-level
    `proposer.model`. Each agent's subsection is self-contained.

    `agent_opts` contains all per-agent fields the in-container script
    consumes when building the agent CLI argv:

      claude_code:
        effort           → --effort <level>
        disallowed_tools → --disallowed-tools <comma-sep>
      codex:
        reasoning_effort → -c model_reasoning_effort="<level>"
    """
    proposer_cfg = cfg.get("proposer", {}) or {}
    agent_cfg = proposer_cfg.get(agent)
    if agent_cfg is None:
        raise ValueError(
            f"No `proposer.{agent}` subsection in config. "
            f"Each agent needs its own model + opts under proposer.<agent>."
        )

    model = agent_cfg.get("model")
    if not model:
        raise ValueError(
            f"No model configured for agent={agent!r}: set proposer.{agent}.model "
            f"in the YAML config."
        )

    opts: Dict[str, Any] = {}
    if agent == "claude_code":
        if agent_cfg.get("effort") is not None:
            opts["effort"] = str(agent_cfg["effort"])
        # disallowed_tools is always passed (defaults to empty list).
        opts["disallowed_tools"] = list(agent_cfg.get("disallowed_tools") or [])
    elif agent == "codex":
        if agent_cfg.get("reasoning_effort") is not None:
            opts["reasoning_effort"] = str(agent_cfg["reasoning_effort"])

    return model, opts


# ---------------------------------------------------------------------------
# Id allocation
# ---------------------------------------------------------------------------

def _next_id(existing: List[str]) -> str:
    """Return the next integer id as a string. Tolerates v5 ids of the form
    `<int>_<hash>` — splits on `_` and takes the integer prefix."""
    nums: List[int] = []
    for id in existing:
        head = id.split("_", 1)[0]
        try:
            nums.append(int(head))
        except ValueError:
            continue
    return str(max(nums, default=-1) + 1)


# ---------------------------------------------------------------------------
# Content hashing — fingerprint of the harness code that gets evaluated.
# Hash computed AFTER any propose_with_fix retries settle so it reflects the
# code that actually runs in evaluate_harness.
# ---------------------------------------------------------------------------

def _compute_content_hash(harness_dir: Path) -> str:
    """Return sha256(harness.py + sorted helper .py + requirements.txt)[:16].

    Excludes meta.json (avoids self-reference loop), PROPOSAL_READY (sentinel),
    and any per-dataset / sanity / runs subdirectories (eval artifacts).
    """
    harness_py = harness_dir / "harness.py"
    if not harness_py.exists():
        raise FileNotFoundError(f"harness.py missing in {harness_dir}")
    parts = [harness_py.read_bytes()]
    for p in sorted(harness_dir.glob("*.py")):
        if p.name == "harness.py":
            continue
        parts.append(p.read_bytes())
    req = harness_dir / "requirements.txt"
    if req.exists():
        parts.append(req.read_bytes())
    return hashlib.sha256(b"\n".join(parts)).hexdigest()[:16]


def _finalize_harness_id(int_id: str, harness_dir: Path) -> Tuple[str, Path]:
    """Rename `harnesses/<int>` → `harnesses/<int>_<hash8>` and stamp meta.json.

    Idempotent for already-finalized dirs (e.g. seed copied from `seeds/`
    that may already include a content_hash). Returns `(final_id, new_dir)`.
    """
    # Already finalized? (e.g. user passed a pre-named dir; or seed already had hash)
    if "_" in harness_dir.name:
        return harness_dir.name, harness_dir

    full_hash = _compute_content_hash(harness_dir)
    short = full_hash[:8]
    final_id = f"{int_id}_{short}"
    final_dir = harness_dir.parent / final_id
    if final_dir.exists() and final_dir != harness_dir:
        # Two harnesses with the same int prefix shouldn't exist; if they do,
        # someone manually fiddled the workspace. Don't clobber.
        raise RuntimeError(
            f"target dir already exists: {final_dir} (refusing to overwrite)"
        )
    if final_dir != harness_dir:
        os.rename(harness_dir, final_dir)

    # Stamp meta.json with content_hash + created_at (preserve CC's parent_ids
    # and description; don't overwrite an existing content_hash if it matches).
    meta_path = final_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            log.warning(f"could not parse {meta_path}: {exc} (overwriting)")
            meta = {}
    meta.setdefault("parent_ids", [])
    meta.setdefault("description", "")
    meta["content_hash"] = full_hash
    meta.setdefault("created_at", _dt.datetime.now().isoformat(timespec="seconds"))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return final_id, final_dir


# ---------------------------------------------------------------------------
# Seed harness bootstrap.
# When cfg.seed.enabled, copy seeds/<source>/ → <run>/harnesses/0/ at startup.
# ---------------------------------------------------------------------------

def _bootstrap_seed(seed_source: str) -> Optional[str]:
    """Copy the seed library entry into this run's harnesses/0/.

    Returns the destination integer-id string (always "0") if a copy happened
    or the dir already existed; returns None if no seed library entry was
    found at `seeds/<seed_source>/` (caller should treat as "no seed").
    """
    src = SEEDS_DIR / seed_source
    if not src.exists():
        log.error(
            f"seed source not found: {src}. Run with --no-seed or pick a "
            f"valid --seed-source from {[p.name for p in SEEDS_DIR.iterdir()] if SEEDS_DIR.exists() else '[]'}"
        )
        return None
    dst = paths.harnesses_dir / "0"
    if dst.exists():
        log.info(f"seed already present at {dst}; skipping copy")
        return "0"
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    log.info(f"seed bootstrapped: {src} → {dst}")
    return "0"


# ---------------------------------------------------------------------------
# Per-dataset output helpers.
# Eval artifacts land at workspace/harnesses/<id>/<dataset>/...
# Sanity artifacts land at workspace/harnesses/<id>/<dataset>/sanity/...
# ---------------------------------------------------------------------------

def _dataset_dir(harness_dir: Path, dataset: str) -> Path:
    return harness_dir / dataset


def _sanity_dir(harness_dir: Path, dataset: str) -> Path:
    return harness_dir / dataset / "sanity"


def _read_dataset_metrics(harness_dir: Path, dataset: str, subdir: Optional[str] = None) -> Dict[str, Any]:
    """Read score.json + token_usage.json for one dataset, return a metrics dict.

    `subdir` selects a stage subdirectory (e.g. "stage2") under the dataset
    dir; None reads the dataset root (the final-stage copy).

    Keys:
      raw_score:        float, benchmark_overall_eval_score (native scale)
      score_max:        int, judge's max score (1 for all current benchmarks)
      per_user_stddev:  Optional[float], population stddev of per_user[*].reward;
                        None when fewer than 2 valid users (stddev undefined)
      tokens:           int, sum of total_tokens across all models in token_usage.json

    Defaults are chosen to make a missing/unreadable file behave like a 0-score
    completed eval (raw_score=0, score_max=10 for back-compat, no stddev, 0 tokens).
    """
    metrics: Dict[str, Any] = {
        "raw_score": 0.0,
        "score_max": 10,
        "per_user_stddev": None,
        "tokens": 0,
    }
    ds_dir = _dataset_dir(harness_dir, dataset)
    if subdir:
        ds_dir = ds_dir / subdir

    # score.json: raw_score + score_max + per_user_stddev
    score_json = ds_dir / "score.json"
    if score_json.exists():
        try:
            with score_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            bes = data.get("benchmark_eval_score", {})
            metrics["raw_score"] = float(bes.get("benchmark_overall_eval_score", 0.0))
            metrics["score_max"] = int(bes.get("score_max", 10))
            rewards = [
                float(v["reward"])
                for v in data.get("per_user", {}).values()
                if v.get("reward") is not None
            ]
            if len(rewards) >= 2:
                # Population stddev (ddof=0) — variance across the evaluated users.
                mean = sum(rewards) / len(rewards)
                metrics["per_user_stddev"] = (
                    sum((r - mean) ** 2 for r in rewards) / len(rewards)
                ) ** 0.5
        except Exception as exc:
            log.warning(f"could not parse {dataset} score.json for {harness_dir.name}: {exc}")

    # token_usage.json: sum total_tokens across all models
    tokens_json = ds_dir / "token_usage.json"
    if tokens_json.exists():
        try:
            with tokens_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            metrics["tokens"] = sum(
                int(model_usage.get("total_tokens", 0))
                for model_usage in data.values()
                if isinstance(model_usage, dict)
            )
        except Exception as exc:
            log.warning(f"could not parse {dataset} token_usage.json for {harness_dir.name}: {exc}")

    return metrics


def _write_failure(out_dir: Path, msg: str) -> None:
    """Write a stub score.json indicating orchestrator-level failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_eval_score": {
            "benchmark_overall_eval_score": 0.0,
            "benchmark_overall_eval_standard_deviation": 0.0,
        },
        "per_user": {},
        "invalid_users": [{"user_id": "orchestrator", "error": msg}],
    }
    with (out_dir / "score.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _publish_run_artifacts(run_dir: Path, dst_dir: Path) -> None:
    """Move evaluator outputs from the transient runs/ staging dir into the
    per-dataset target dir, then remove the staging dir.

    Why move (not copy): runs/ exists only because the container needs a
    writable /out separate from the read-only /harness bind. After the host
    has the artifacts, leaving them in runs/ doubles disk use and adds
    visual clutter. We `shutil.copy2 + rmtree` so the dst_dir always lands
    on the same filesystem regardless of where runs/ was bound.

    Includes `subprocess.log` (container-side Phase 1/2 progress) so users
    don't have to dig into runs/<...>/ to inspect what happened inside the
    container.

    dst_dir is a PERSISTENT location that may hold artifacts from a prior
    attempt (sanity retry N-1, orphan re-eval, resumed run). Those are
    removed FIRST — unconditionally — so that after publishing, dst_dir
    reflects exactly what THIS execution produced. Without this, a crashed
    attempt would inherit the previous attempt's score.json and the caller's
    "did we get a score?" check (and sanity-error collection, and stage
    promotion) would silently operate on stale results.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("score.json", "token_usage.json", "subprocess.log"):
        stale = dst_dir / name
        if stale.exists():
            try:
                stale.unlink()
            except OSError as exc:
                log.warning(f"publish: could not remove stale {stale}: {exc}")
    stale_traces = dst_dir / "traces"
    if stale_traces.exists():
        shutil.rmtree(stale_traces, ignore_errors=True)

    for name in ("score.json", "token_usage.json", "subprocess.log"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(str(src), str(dst_dir / name))
    for sub in ("traces",):
        src_dir = run_dir / sub
        if not src_dir.exists():
            continue
        dst_sub = dst_dir / sub
        dst_sub.mkdir(exist_ok=True)
        for p in src_dir.iterdir():
            shutil.copy2(str(p), str(dst_sub / p.name))
    # Drop the staging dir entirely. If a copy above failed, rmtree still
    # runs — the score.json that ends up in dst_dir remains the source of
    # truth, and duplicating staging on a partial failure isn't useful.
    shutil.rmtree(run_dir, ignore_errors=True)
    # Also rmdir the parent `runs/` if it's empty — keeps the workspace
    # listing clean when no eval is currently in flight. Silently a no-op
    # if a concurrent eval is using a sibling subdir.
    try:
        run_dir.parent.rmdir()
    except OSError:
        pass


def _build_objectives(
    per_dataset_metrics: Dict[str, Dict[str, Any]],
    harness_dir: Path,
) -> Dict[str, Any]:
    """Aggregate per-dataset metrics + harness-level attrs into a frontier objectives dict.

    Per-benchmark fields (one set per dataset, native scale preserved):
      - `accuracy_<dataset>`     raw judge score (all three benchmarks 0-1:
                                 DynamicMem TCE holistic, LoCoMo/LME binary)
      - `score_max_<dataset>`    judge's max — so consumers can normalize themselves
      - `robustness_<dataset>`   stddev of per-user reward (lower = more reliable);
                                 omitted when fewer than 2 valid users (stddev undefined)

    Harness-level fields (single value):
      - `accuracy`               mean of (raw / score_max) across datasets ∈ [0, 1].
                                 Equal weight per benchmark regardless of judge scale.
      - `code_length`            bytes of harness.py (rough simplicity proxy; lower = simpler)
      - `tokens_total`           sum of total_tokens across all datasets and models
                                 (rough $-cost proxy; lower = cheaper)

    These extra fields are **telemetry only** — `Frontier.OBJECTIVES` is still
    `("accuracy",)`, so selection algorithms ignore them. They surface to the
    proposer agent (which reads frontier.json directly) so CC can factor in
    "same accuracy, prefer simpler / cheaper / more reliable" when picking priors.
    """
    out: Dict[str, Any] = {}
    if not per_dataset_metrics:
        out["accuracy"] = 0.0
    else:
        normalized: List[float] = []
        total_tokens = 0
        for ds, m in per_dataset_metrics.items():
            raw = float(m.get("raw_score", 0.0))
            score_max = int(m.get("score_max", 10))
            out[f"accuracy_{ds}"] = raw
            out[f"score_max_{ds}"] = float(score_max)
            stddev = m.get("per_user_stddev")
            if stddev is not None:
                out[f"robustness_{ds}"] = float(stddev)
            # Staged-evaluation telemetry: highest stage this benchmark
            # reached (1/2/3; 0 = sanity-size dev run). `accuracy_<ds>`
            # is the score AT that stage — compare candidates at the same
            # stage_<ds> when reading frontier.json.
            if m.get("stage") is not None:
                out[f"stage_{ds}"] = float(m["stage"])
            normalized.append(raw / score_max if score_max > 0 else 0.0)
            total_tokens += int(m.get("tokens", 0))
        out["accuracy"] = sum(normalized) / len(normalized)  # ∈ [0, 1]
        out["tokens_total"] = total_tokens

    # Harness-level: code_length (best-effort; missing harness.py from a crashed
    # propose just means we record 0 rather than crashing).
    try:
        out["code_length"] = (harness_dir / "harness.py").stat().st_size
    except OSError:
        out["code_length"] = 0

    return out


# ---------------------------------------------------------------------------
# Sanity check — small real-data run per dataset, ensures harness runs without crashing
# ---------------------------------------------------------------------------

def _collect_sanity_errors(
    harness_dir: Path, datasets_config: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str]:
    """Inspect sanity score.json for each dataset. Return (passed, error_trace).

    Harness fails sanity if ANY dataset has invalid_users non-empty OR any
    per_user entry has non-empty failure_info.
    """
    errors: List[str] = []
    for ds in datasets_config:
        score_path = _sanity_dir(harness_dir, ds) / "score.json"
        if not score_path.exists():
            errors.append(f"[{ds}] sanity score.json missing (singularity likely didn't launch)")
            continue
        try:
            with score_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            errors.append(f"[{ds}] failed to parse sanity score.json: {exc}")
            continue
        invalid = data.get("invalid_users", [])
        for iu in invalid:
            errors.append(f"[{ds}] invalid user {iu.get('user_id', '?')}: {iu.get('error', '')}")
        for uid, v in data.get("per_user", {}).items():
            fi = v.get("failure_info")
            if fi:
                errors.append(f"[{ds}] user {uid} partial failure: {fi}")
    if not errors:
        return True, ""
    return False, "\n".join(errors)


async def sanity_check_harness(
    harness_id: str,
    image_path: Path,
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    split: str,   # search | test — which benchmark split (mode=dev never reaches here)
    model: str,
    judge_model: str,
    update_type: str,
    max_sample_concurrent: int,
    gpu: bool = False,
) -> Tuple[bool, str]:
    """Run a sanity_check-sized evaluation on every dataset (wire: --stage sanity).

    Returns (passed, error_trace). Artifacts go to `<harness>/<dataset>/sanity/`.
    """
    harness_dir = paths.harnesses_dir / harness_id

    for ds, params in datasets_config.items():
        sanity_dst = _sanity_dir(harness_dir, ds)
        run_dir = paths.runs_dir / f"{harness_id}_{int(time.time())}_{ds}_sanity"
        crashed: Optional[Exception] = None
        try:
            await run_evaluation(
                harness_dir=harness_dir,
                image_path=image_path,
                out_dir=run_dir,
                dataset=ds,
                split=split,
                stage="sanity",
                stage_spec=stage_wire_spec(ds, params["stages"]["sanity_check"]),
                model=model,
                judge_model=params.get("judge_model", judge_model),
                update_type=update_type,
                max_sample_concurrent=max_sample_concurrent,
                gpu=gpu,
            )
        except Exception as exc:
            log.error(f"sanity crashed for {harness_id} [{ds}]: {exc}")
            crashed = exc
        # Always publish + cleanup, even on crash — the container may have
        # written subprocess.log / partial score.json that's useful for retry.
        _publish_run_artifacts(run_dir, sanity_dst)
        if not (sanity_dst / "score.json").exists():
            _write_failure(sanity_dst, f"sanity_crashed: {crashed}" if crashed else "no score.json produced")

    return _collect_sanity_errors(harness_dir, datasets_config)


# ---------------------------------------------------------------------------
# Full eval — one harness across every dataset (serial).
# ---------------------------------------------------------------------------

async def evaluate_harness(
    harness_id: str,
    image_path: Path,
    *,
    datasets_config: Dict[str, Dict[str, Any]],
    mode: str,                    # search | test | dev
    model: str,
    judge_model: str,
    update_type: str,
    max_sample_concurrent: int,
    memory_cache: bool = True,
    gpu: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run the STAGED evaluation gauntlet on every dataset (serial).

    Per benchmark (independent promotion): run stage1 → stage2 → stage3,
    publishing each stage's artifacts to `harnesses/<id>/<ds>/<stage>/`.
    After a gated stage (stage1/stage2), the normalized score must be
    >= that stage's threshold to advance; otherwise the benchmark stops
    there ("eliminated"). The final (highest-reached) stage's score.json +
    token_usage.json are copied to the dataset root for browsing/back-compat,
    and a `stages.json` summary is written alongside.

    `mode=dev` bypasses the gauntlet entirely: one sanity_check-sized
    run per benchmark on the search split, published at the dataset root.

    Returns per-dataset metrics dicts (raw_score, score_max, per_user_stddev,
    tokens, stage, eliminated) — consumed by `_build_objectives`.
    """
    harness_dir = paths.harnesses_dir / harness_id

    scores: Dict[str, Dict[str, Any]] = {}
    for ds, params in datasets_config.items():
        ds_judge = params.get("judge_model", judge_model)
        dst_root = _dataset_dir(harness_dir, ds)

        async def _run_stage(stage_name: str, spec: Dict[str, Any], dst: Path,
                             split: str) -> Optional[Exception]:
            run_dir = paths.runs_dir / f"{harness_id}_{int(time.time())}_{ds}_{stage_name}"
            # Cross-stage memory cache: gauntlet tiers only (launch.py gates
            # on the stage name too; sanity/dev execs never get the mount).
            memcache_dir: Optional[Path] = None
            if memory_cache and stage_name.startswith("stage"):
                memcache_dir = dst_root / "memory_cache"
                memcache_dir.mkdir(parents=True, exist_ok=True)
            crashed: Optional[Exception] = None
            try:
                await run_evaluation(
                    harness_dir=harness_dir,
                    image_path=image_path,
                    out_dir=run_dir,
                    dataset=ds,
                    split=split,
                    stage=stage_name,
                    stage_spec=spec,
                    model=model,
                    judge_model=ds_judge,
                    update_type=update_type,
                    max_sample_concurrent=max_sample_concurrent,
                    memcache_dir=memcache_dir,
                    gpu=gpu,
                )
            except Exception as exc:
                log.error(f"evaluation crashed for {harness_id} [{ds}/{stage_name}]: {exc}")
                crashed = exc
            # Always publish + cleanup. If the container died mid-write,
            # subprocess.log + any partial score.json are still useful for triage.
            _publish_run_artifacts(run_dir, dst)
            if not (dst / "score.json").exists():
                _write_failure(dst, f"eval_crashed: {crashed}" if crashed else "no score.json produced")
                if crashed is None:
                    # Container exited without a score (rc!=0, OOM-kill, ...).
                    # Surface it as a crash so stages.json records it and the
                    # gauntlet stops here — a scoreless run must never be
                    # indistinguishable from a genuine 0-score run.
                    crashed = RuntimeError(
                        f"container produced no score.json [{ds}/{stage_name}]"
                    )
            return crashed

        if mode == "dev":
            # Sanity-size single run, no gating, artifacts at the dataset root.
            spec = stage_wire_spec(ds, params["stages"]["sanity_check"])
            await _run_stage("sanity", spec, dst_root, "search")
            m = _read_dataset_metrics(harness_dir, ds)
            m["stage"] = 0.0  # 0 = sanity-size dev run (not a gauntlet tier)
            m["eliminated"] = False
            scores[ds] = m
            continue

        # ---- staged gauntlet (mode = search | test; mode IS the split) ----
        plan = stage_plan(ds, params)
        stage_summary: Dict[str, Any] = {}
        final_metrics: Dict[str, Any] = {}
        reached = ""
        eliminated = False
        for stage_name, spec, threshold in plan:
            crashed = await _run_stage(stage_name, spec, dst_root / stage_name, mode)
            m = _read_dataset_metrics(harness_dir, ds, subdir=stage_name)
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
                log.info(
                    f"{harness_id} [{ds}] ELIMINATED at {stage_name}: "
                    f"{normalized:.3f} < threshold {threshold}"
                )
                break
            if threshold is not None:
                log.info(
                    f"{harness_id} [{ds}] promoted past {stage_name}: "
                    f"{normalized:.3f} >= {threshold}"
                )

        # Final-stage artifacts to the dataset root (CC browsing + the
        # frontier reader default path).
        for fname in ("score.json", "token_usage.json"):
            src = dst_root / reached / fname
            if src.exists():
                shutil.copy2(src, dst_root / fname)
        with (dst_root / "stages.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"reached": reached, "eliminated": eliminated, "stages": stage_summary},
                f, indent=2, ensure_ascii=False,
            )

        final_metrics["stage"] = float(STAGE_ORDER.index(reached) + 1)
        final_metrics["eliminated"] = eliminated
        scores[ds] = final_metrics

    return scores


# ---------------------------------------------------------------------------
# One harness from propose through eval (or sanity-skip). Returns
# (per_dataset_scores, sanity_status).
# ---------------------------------------------------------------------------

async def propose_eval_one(
    cfg: Dict[str, Any],
    new_id: str,
    *,
    is_seed: bool = False,
) -> Tuple[str, Dict[str, float], str, List[str]]:
    """End-to-end for one harness (seed or proposed).

    Returns (final_id, per_dataset_scores, sanity_status, parent_ids):
      - final_id: harness id after content-hash rename (e.g. "3_a1b2c3d4");
        falls back to the integer id if no harness.py was produced
        (proposer-failure path).
      - sanity_status: "passed" / "failed" / "skipped" / "passed_on_retry_N"
        (also "proposer_failed" / "env_build_failed").
      - parent_ids: list of harness ids the proposer drew from
        (empty for the seed; populated from `meta.json::parent_ids`).

    The harness directory is renamed `harnesses/<int>` →
    `harnesses/<int>_<hash8>` after the sanity-retry loop settles, so the
    suffix reflects the code that's actually evaluated.
    """
    datasets_config = cfg["datasets"]
    dataset_names = list(datasets_config.keys())
    mode = cfg["mode"]   # search | test | dev

    # Sanity is skipped entirely when mode=dev (dev IS a sanity-size quick
    # check by definition; running another sanity layer is redundant).
    # For search/test, sanity respects cfg["sanity"]["enabled"].
    sanity_enabled = (mode != "dev") and cfg["sanity"]["enabled"]

    harness_dir = paths.harnesses_dir / new_id
    final_id = new_id  # may get renamed to f"{new_id}_{hash8}" below

    def _settle_path() -> None:
        """Rename harness_dir → <int>_<hash8>. Updates outer-scope vars."""
        nonlocal harness_dir, final_id
        if (harness_dir / "harness.py").exists() and "_" not in harness_dir.name:
            final_id, harness_dir = _finalize_harness_id(new_id, harness_dir)

    # Propose (unless seed). No parent_id passed — the agent browses the
    # workspace itself and records its chosen prior(s) in meta.json::parent_ids.
    if not is_seed:
        agent = cfg["agent"]
        agent_model, agent_opts = _resolve_proposer_for_agent(cfg, agent)
        try:
            await propose(
                new_id=new_id,
                model=agent_model,
                max_turns=cfg["proposer"]["max_turns"],
                timeout_s=cfg["proposer"]["timeout_s"],
                sanity_enabled=sanity_enabled,
                active_datasets=list(cfg.get("datasets", {}).keys()),
                update_type=cfg["update_type"],
                agent=agent,
                agent_opts=agent_opts,
                prompts_version=cfg["prompts"]["version"],
            )
        except (TimeoutError, RuntimeError) as exc:
            # propose() failed — most likely subprocess exited rc!=0 (auth 401,
            # SDK timeout, or other infra error). harness.py was almost
            # certainly never written. Clean up the empty harness dir so resume
            # isn't polluted, then propagate ProposerInfraError to abort the
            # search loop. main() catches it and sys.exit(2).
            log.error(f"proposer failed for {new_id}: {exc}")
            if harness_dir.exists() and not (harness_dir / "harness.py").exists():
                shutil.rmtree(harness_dir, ignore_errors=True)
                log.info(f"cleaned up empty harness dir {harness_dir}")
            raise ProposerInfraError(
                f"propose() failed for harness {new_id}: {exc}"
            ) from exc

    # Build (or look up cached) env image. We do NOT host-side-validate the
    # harness anymore — that used to live in `forge.contract.validate` but
    # required host venv to mirror the container's package list (issue #6).
    # Now: validation happens naturally inside the container at sanity time
    # (or at the first gauntlet stage when sanity is disabled) via
    # launch.py::_load_harness_class.
    try:
        image_path = await ensure_image(harness_dir)
    except EnvBuildError as exc:
        log.error(f"env build failed for {new_id}: {exc}")
        _settle_path()
        for ds in dataset_names:
            _write_failure(_dataset_dir(harness_dir, ds), f"env_build_failed: {exc}")
        return final_id, {ds: {"raw_score": 0.0, "score_max": 1, "per_user_stddev": None, "tokens": 0} for ds in dataset_names}, "env_build_failed", _read_parent_ids(harness_dir)

    # Sanity retry loop (only when sanity_enabled + not seed — we trust the
    # seed harness to be clean; running sanity on it just wastes time.). All
    # operations here use the integer-only path; rename happens after the
    # loop settles.
    sanity_status = "skipped"
    if sanity_enabled and not is_seed:
        max_retries = int(cfg["sanity"]["max_retries"])
        for attempt in range(max_retries + 1):
            passed, error_trace = await sanity_check_harness(
                new_id, image_path,
                datasets_config=datasets_config,
                split=cfg["mode"],   # sanity only runs at mode search/test
                model=cfg["model"], judge_model=cfg["judge_model"],
                update_type=cfg["update_type"],
                max_sample_concurrent=cfg["max_sample_concurrent"],
                gpu=cfg["gpu"]["enabled"],
            )
            if passed:
                sanity_status = "passed" if attempt == 0 else f"passed_on_retry_{attempt}"
                log.info(f"sanity OK for {new_id} (attempt {attempt}/{max_retries})")
                break
            log.warning(
                f"sanity FAILED for {new_id} (attempt {attempt}/{max_retries}):\n{error_trace}"
            )
            if attempt >= max_retries:
                sanity_status = "failed"
                break
            # Retry: ask the agent to fix its own harness based on the error trace.
            agent = cfg["agent"]
            agent_model, agent_opts = _resolve_proposer_for_agent(cfg, agent)
            try:
                await propose_with_fix(
                    new_id=new_id,
                    error_trace=error_trace,
                    model=agent_model,
                    max_turns=cfg["proposer"]["max_turns"],
                    timeout_s=cfg["proposer"]["timeout_s"],
                    sanity_enabled=sanity_enabled,
                    active_datasets=list(cfg.get("datasets", {}).keys()),
                    update_type=cfg["update_type"],
                    agent=agent,
                    agent_opts=agent_opts,
                    prompts_version=cfg["prompts"]["version"],
                )
            except (TimeoutError, RuntimeError) as exc:
                # propose_with_fix subprocess died — same infra failure mode as
                # propose() above. harness.py from the original propose still
                # exists (don't delete the dir), but we abort the loop because
                # propose-side infra is unavailable. Resume picks up cleanly.
                log.error(f"propose_with_fix failed for {new_id}: {exc}")
                raise ProposerInfraError(
                    f"propose_with_fix() failed for harness {new_id}: {exc}"
                ) from exc
            # CC may have edited requirements.txt during the fix → re-resolve
            # the container image (cached unless requirements.txt actually
            # changed).
            try:
                image_path = await ensure_image(harness_dir)
            except EnvBuildError as exc:
                log.error(f"env rebuild after fix failed for {new_id}: {exc}")
                sanity_status = "failed"
                break

        if sanity_status == "failed":
            log.error(f"sanity_failed for {new_id}; skipping full eval")
            _settle_path()
            for ds in dataset_names:
                _write_failure(
                    _dataset_dir(harness_dir, ds),
                    "sanity_failed_after_retries",
                )
            return final_id, {ds: {"raw_score": 0.0, "score_max": 1, "per_user_stddev": None, "tokens": 0} for ds in dataset_names}, "failed", _read_parent_ids(harness_dir)

    # Sanity loop settled (or skipped). Compute hash on the post-fix code,
    # rename `<int>` → `<int>_<hash8>`, then run full eval against the new dir.
    _settle_path()

    per_ds = await evaluate_harness(
        final_id, image_path,
        datasets_config=datasets_config,
        mode=mode,
        model=cfg["model"], judge_model=cfg["judge_model"],
        update_type=cfg["update_type"],
        max_sample_concurrent=cfg["max_sample_concurrent"],
        memory_cache=cfg.get("memory_cache", True),
        gpu=cfg["gpu"]["enabled"],
    )
    return final_id, per_ds, sanity_status, _read_parent_ids(harness_dir)


# ---------------------------------------------------------------------------
# Outer loop
# ---------------------------------------------------------------------------

async def search_loop(cfg: Dict[str, Any]) -> None:
    ensure_dirs()
    frontier = Frontier.load(paths.frontier_path)

    # Resume cleanup: drop any polluting entries from a prior partial run that
    # used the old "catch + placeholder + continue" error handling. After this
    # call, every remaining frontier entry has content_hash != None (real
    # successful or sanity-failed harness). Persisted immediately so the
    # cleanup is reflected on disk even if the rest of search_loop aborts.
    if _cleanup_polluting_entries(frontier) > 0:
        frontier.save(paths.frontier_path)

    # Resume orphan adoption: re-attach harness dirs that exist on disk but
    # aren't in frontier (e.g. orchestrator killed mid-eval). Runs BEFORE
    # seed-bootstrap so an orphaned 0_<hash> seed is auto-detected and the
    # bootstrap's `seed_already_in_frontier` check below sees it.
    if cfg.get("adopt_orphans", True):
        n_adopted = await _adopt_orphan_dirs(cfg, frontier)
        if n_adopted:
            log.info(f"adoption: re-attached {n_adopted} orphan harness(es) from disk")

    datasets_config: Dict[str, Dict[str, Any]] = cfg["datasets"]
    dataset_names = list(datasets_config.keys())

    # Seed harness bootstrap (v5): copy seeds/<source>/ → harnesses/0/ when
    # enabled. Then evaluate seed and rename → 0_<hash>. Skips the seed entirely
    # when seed.enabled=false, so CC's first proposal becomes the first entry.
    seed_already_in_frontier = any(e.id.split("_", 1)[0] == "0" for e in frontier.all_entries())
    if cfg["seed"]["enabled"] and not seed_already_in_frontier:
        seed_int_id = _bootstrap_seed(cfg["seed"]["source"])
        if seed_int_id is None:
            log.error("seed bootstrap failed; aborting search")
            return
        log.info(f"evaluating seed harness on {dataset_names} ...")
        seed_final_id, per_ds, sanity_status, parent_ids = await propose_eval_one(
            cfg, new_id=seed_int_id, is_seed=True,
        )
        seed_dir = paths.harnesses_dir / seed_final_id
        objectives = _build_objectives(per_ds, seed_dir)
        seed_meta = _read_meta(seed_dir)
        entry = Entry(
            id=seed_final_id, objectives=objectives, parent_ids=parent_ids,
            content_hash=seed_meta.get("content_hash"),
            created_at=seed_meta.get("created_at"),
        )
        frontier.add(entry)
        _persist_sanity_status(entry.id, sanity_status)
        frontier.save(paths.frontier_path)
        log.info(
            f"seed {seed_final_id}: mean={objectives['accuracy']:.3f} "
            + ", ".join(
                f"{ds}={per_ds.get(ds, {}).get('raw_score', 0):.3f}"
                for ds in dataset_names
            )
            + f"  [sanity={sanity_status}]"
        )
    elif not cfg["seed"]["enabled"] and not seed_already_in_frontier:
        log.info("seed disabled — first proposed harness will have no prior to reference")

    # Search loop. v4: no parent selection — for each step, propose k harnesses
    # back-to-back; each call lets CC browse the workspace and pick its own
    # priors. v5: dir is renamed to <int>_<hash> after sanity settles.
    #
    # Resume semantics (2026-04-28): cfg["steps"] is the TOTAL number of search
    # steps across all invocations of this --run-name. We compute "proposals
    # already done" from the frontier (post-cleanup, every entry is real) and
    # only run the remaining count. Empty workspace → done=0 → fresh run with
    # full budget. Same workspace re-launched → done=N → only `total - N` more.
    k = int(cfg["propose"]["k_per_step"])
    total = cfg["steps"] * k
    done = len(frontier.all_entries())
    # Seed (if bootstrapped) sits at id "0_<hash>" — it's a baseline reference,
    # not a "search proposal". Subtract it from `done` so it doesn't eat into
    # the proposal budget.
    if cfg["seed"]["enabled"] and any(
        e.id.startswith("0_") for e in frontier.all_entries()
    ):
        done = max(0, done - 1)

    if done >= total:
        log.info(
            f"all {total} proposals already complete in this workspace; nothing to do"
        )
        return

    log.info(
        f"{'resuming' if done > 0 else 'starting'} search: {done}/{total} "
        f"proposals already done; running {total - done} more"
    )

    for n in range(done, total):
        step = n // k + 1
        j = n % k + 1
        int_id = _next_id([e.id for e in frontier.all_entries()])
        log.info(f"[step {step}/{cfg['steps']}] candidate {j}/{k} → int_id={int_id}")
        try:
            final_id, per_ds, sanity_status, parent_ids = await propose_eval_one(
                cfg, new_id=int_id, is_seed=False,
            )
        except ProposerInfraError as exc:
            log.error(
                f"[step {step}/{cfg['steps']}] propose subprocess failed for "
                f"{int_id}: {exc}\n"
                f"  → search aborted to avoid generating placeholder entries.\n"
                f"  → Resume with the SAME --run-name (total {cfg['steps']} steps; "
                f"{n} of {total} proposals done before this attempt's failure)."
            )
            raise   # propagate to main() for sys.exit(2)

        new_dir = paths.harnesses_dir / final_id
        objectives = _build_objectives(per_ds, new_dir)
        new_meta = _read_meta(new_dir)
        entry = Entry(
            id=final_id, objectives=objectives, parent_ids=parent_ids,
            content_hash=new_meta.get("content_hash"),
            created_at=new_meta.get("created_at"),
        )
        frontier.add(entry)
        _persist_sanity_status(entry.id, sanity_status)
        frontier.save(paths.frontier_path)
        log.info(
            f"[step {step}/{j}] {final_id} mean={objectives['accuracy']:.3f} "
            + ", ".join(
                f"{ds}={per_ds.get(ds, {}).get('raw_score', 0):.3f}"
                for ds in dataset_names
            )
            + f"  [sanity={sanity_status}]  parent_ids={parent_ids}"
        )


# ---------------------------------------------------------------------------
# Resume orphan adoption: re-attach harness dirs that exist on disk but
# aren't in frontier (e.g. orchestrator killed mid-eval). Classify each by
# file presence and run ONLY the missing pipeline stages.
#
# Cases handled (Case B — pre-sanity orphans — is intentionally discarded
# rather than re-sanity'd; that requires extracting propose_eval_one's
# sanity loop into a helper, deferred):
#
#   complete       hash-suffix dir + every <ds>/score.json present  →
#                  build Entry from on-disk metrics, no eval
#   sanity_passed  hash-suffix dir + every <ds>/sanity/score.json   →
#                  call evaluate_harness for the missing full eval
#   sanity_failed  hash-suffix dir + every <ds>/score.json contains
#                  the "sanity_failed_after_retries" sentinel       →
#                  build all-zero Entry, no eval
#   incomplete     anything else (no harness.py, no hash suffix,
#                  partial sanity, etc.)                            →
#                  rm -rf the dir + warn
# ---------------------------------------------------------------------------

# Matches a valid harness dir name: integer (pre-finalize) or integer_hash8
# (post-finalize). Anything else (e.g. user-created `notes/`, transient
# scratch files) is filtered out by _scan_orphan_dirs.
_HARNESS_DIR_RX = re.compile(r"^\d+(_[0-9a-f]{8})?$")


def _scan_orphan_dirs(frontier: Frontier) -> List[Path]:
    """Return harness dirs on disk whose int prefix isn't in `frontier`,
    sorted by integer prefix ascending so adoption runs in id order."""
    if not paths.harnesses_dir.exists():
        return []
    in_frontier_int_prefixes = {
        e.id.split("_", 1)[0] for e in frontier.all_entries()
    }
    orphans: List[Tuple[int, Path]] = []
    for child in paths.harnesses_dir.iterdir():
        if not child.is_dir():
            continue
        if not _HARNESS_DIR_RX.match(child.name):
            continue
        int_prefix = child.name.split("_", 1)[0]
        if int_prefix in in_frontier_int_prefixes:
            continue
        try:
            orphans.append((int(int_prefix), child))
        except ValueError:
            continue
    orphans.sort(key=lambda t: t[0])
    return [path for _, path in orphans]


def _classify_orphan(harness_dir: Path, dataset_names: List[str]) -> str:
    """Return 'complete' | 'sanity_passed' | 'sanity_failed' | 'incomplete'.

    Decision tree (in order):
      - no harness.py                                  → 'incomplete'
      - dir name has no underscore (Case B)            → 'incomplete'
      - meta.json missing or content_hash missing      → 'incomplete'
      - every <ds>/sanity/score.json present:
          if every <ds>/score.json present             → 'complete'
          else                                         → 'sanity_passed'
      - no <ds>/sanity/ exists for any ds AND every
        <ds>/score.json contains the sanity-failed
        sentinel                                       → 'sanity_failed'
      - else (partial / mixed-state)                   → 'incomplete'
    """
    if not (harness_dir / "harness.py").exists():
        return "incomplete"
    if "_" not in harness_dir.name:
        return "incomplete"
    meta = _read_meta(harness_dir)
    if not meta.get("content_hash"):
        return "incomplete"

    sanity_score_paths = [
        _sanity_dir(harness_dir, ds) / "score.json" for ds in dataset_names
    ]
    full_score_paths = [
        _dataset_dir(harness_dir, ds) / "score.json" for ds in dataset_names
    ]
    has_all_sanity = all(p.exists() for p in sanity_score_paths)
    has_all_full = all(p.exists() for p in full_score_paths)
    has_no_sanity_dirs = not any(
        _sanity_dir(harness_dir, ds).exists() for ds in dataset_names
    )

    if has_all_sanity:
        return "complete" if has_all_full else "sanity_passed"

    # Case E: no sanity dirs at all + all full eval score.jsons exist with the
    # "sanity_failed_after_retries" marker that propose_eval_one writes via
    # _write_failure when the sanity retry loop exhausts.
    if has_no_sanity_dirs and has_all_full:
        for p in full_score_paths:
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return "incomplete"
            invalid = data.get("invalid_users", [])
            if not any(
                isinstance(u, dict) and u.get("error") == "sanity_failed_after_retries"
                for u in invalid
            ):
                return "incomplete"
        return "sanity_failed"

    return "incomplete"


def _build_adopted_entry(harness_dir: Path, per_ds: Dict[str, Dict[str, Any]]) -> Entry:
    """Construct an Entry from per-dataset metrics + the dir's meta.json.

    Used by all three "successful adoption" paths (complete / sanity_passed /
    sanity_failed) — the only difference is whether we ran evaluate_harness
    to get per_ds or read it straight from existing score.jsons.
    """
    meta = _read_meta(harness_dir)
    return Entry(
        id=harness_dir.name,
        objectives=_build_objectives(per_ds, harness_dir),
        parent_ids=_read_parent_ids(harness_dir),
        content_hash=meta.get("content_hash"),
        created_at=meta.get("created_at"),
    )


async def _adopt_orphan(
    cfg: Dict[str, Any], harness_dir: Path, frontier: Frontier
) -> Optional[Entry]:
    """Re-enter the pipeline at the right stage for one orphan.

    Returns the new Entry (already added to frontier — caller batches the
    save) or None if the orphan was discarded.
    """
    dataset_names = list(cfg["datasets"].keys())
    kind = _classify_orphan(harness_dir, dataset_names)
    name = harness_dir.name

    if kind == "incomplete":
        log.warning(
            f"adoption: discarded incomplete orphan harnesses/{name} "
            f"(missing harness.py / hash suffix / content_hash, or partial sanity/eval state)"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None

    # Defensive content_hash check (only meaningful for hash-suffixed dirs,
    # which all surviving cases here are). Detects post-finalize edits to
    # harness.py — would mean the eval would run different code than the
    # hash suffix advertises.
    expected_hash8 = name.split("_", 1)[1]
    try:
        actual_hash8 = _compute_content_hash(harness_dir)[:8]
    except Exception as exc:
        log.warning(
            f"adoption: discarded harnesses/{name} — could not "
            f"compute content_hash: {exc}"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None
    if actual_hash8 != expected_hash8:
        log.warning(
            f"adoption: discarded harnesses/{name} — content_hash mismatch "
            f"(computed={actual_hash8}, dir suffix={expected_hash8}); "
            f"harness.py was modified post-finalize"
        )
        shutil.rmtree(harness_dir, ignore_errors=True)
        return None

    # Sanity-failed: synthetic 0-scores already on disk; just build the Entry.
    if kind == "sanity_failed":
        log.info(f"adoption: harnesses/{name} → sanity_failed (no eval needed)")
        per_ds = {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names}
        entry = _build_adopted_entry(harness_dir, per_ds)
        frontier.add(entry)
        _persist_sanity_status(name, "adopted_failed")
        return entry

    # Fully complete: just build the Entry from on-disk score.jsons.
    if kind == "complete":
        log.info(f"adoption: harnesses/{name} → complete (no eval needed)")
        per_ds = {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names}
        entry = _build_adopted_entry(harness_dir, per_ds)
        frontier.add(entry)
        _persist_sanity_status(name, "adopted_complete")
        return entry

    # Sanity passed but full eval missing/partial: run evaluate_harness only.
    log.info(
        f"adoption: harnesses/{name} → sanity_passed; running evaluate_harness "
        f"on {dataset_names}"
    )
    try:
        image_path = await ensure_image(harness_dir)
    except EnvBuildError as exc:
        log.error(
            f"adoption: env build failed for harnesses/{name}: {exc}; "
            f"adopting as failed Entry with all-zero objectives"
        )
        for ds in dataset_names:
            _write_failure(_dataset_dir(harness_dir, ds), f"env_build_failed: {exc}")
        per_ds = {ds: _read_dataset_metrics(harness_dir, ds) for ds in dataset_names}
        entry = _build_adopted_entry(harness_dir, per_ds)
        frontier.add(entry)
        _persist_sanity_status(name, "adopted_env_build_failed")
        return entry

    per_ds = await evaluate_harness(
        name, image_path,
        datasets_config=cfg["datasets"],
        mode=cfg["mode"],
        model=cfg["model"], judge_model=cfg["judge_model"],
        update_type=cfg["update_type"],
        max_sample_concurrent=cfg["max_sample_concurrent"],
        memory_cache=cfg.get("memory_cache", True),
        gpu=cfg["gpu"]["enabled"],
    )
    entry = _build_adopted_entry(harness_dir, per_ds)
    frontier.add(entry)
    _persist_sanity_status(name, "adopted_passed")
    return entry


async def _adopt_orphan_dirs(cfg: Dict[str, Any], frontier: Frontier) -> int:
    """Top-level: scan + classify + adopt orphan harness dirs in id order.

    Saves the frontier once at the end if any adoption succeeded. Returns
    the count successfully adopted (incomplete/discarded don't count).
    """
    orphans = _scan_orphan_dirs(frontier)
    if not orphans:
        return 0
    log.info(
        f"adoption: scanning harnesses/ for orphans not in frontier — "
        f"found {len(orphans)} candidate(s): {[p.name for p in orphans]}"
    )
    adopted = 0
    for harness_dir in orphans:
        try:
            entry = await _adopt_orphan(cfg, harness_dir, frontier)
        except Exception as exc:
            log.error(
                f"adoption: unexpected error adopting harnesses/{harness_dir.name}: "
                f"{exc}; leaving dir as-is"
            )
            continue
        if entry is not None:
            adopted += 1
            log.info(
                f"adoption: re-attached {entry.id} "
                f"accuracy={entry.objectives.get('accuracy', 0):.3f}"
            )
    if adopted:
        frontier.save(paths.frontier_path)
    return adopted


# ---------------------------------------------------------------------------
# Resume cleanup: drop polluting placeholder entries from prior partial runs.
# Only `propose()`-failure entries (content_hash=None, integer-only id, empty
# dir) are removed. Sanity-failed harnesses (have hash + harness.py) are kept
# as legitimate "score=0 because design didn't pass sanity" data points.
# ---------------------------------------------------------------------------

def _cleanup_polluting_entries(frontier: Frontier) -> int:
    """Filter out entries with content_hash=None and rm -rf their empty dirs.

    Returns the number of entries removed (0 if frontier is clean). When
    nonzero, caller should `frontier.save(...)` to persist the cleanup.
    """
    polluting_ids = {e.id for e in frontier.all_entries() if e.content_hash is None}
    if not polluting_ids:
        return 0
    n_dirs_removed = 0
    for entry_id in polluting_ids:
        # Only touch integer-only dirs (truly empty, just sanity_status.txt).
        # Hash-suffixed dirs shouldn't reach this branch in normal flow (their
        # _finalize_harness_id sets content_hash) but defensively skip them.
        if "_" not in entry_id:
            harness_dir = paths.harnesses_dir / entry_id
            if harness_dir.exists() and not (harness_dir / "harness.py").exists():
                shutil.rmtree(harness_dir, ignore_errors=True)
                n_dirs_removed += 1
    n_removed = frontier.remove_by_ids(polluting_ids)
    log.info(
        f"cleanup: removed {n_removed} polluting frontier entries, "
        f"{n_dirs_removed} empty harness dirs"
    )
    return n_removed


# ---------------------------------------------------------------------------
# sanity_status is orthogonal to selection; we keep it as a sidecar file
# so we don't have to plumb it through the Frontier dataclass yet.
# ---------------------------------------------------------------------------

def _persist_sanity_status(harness_id: str, status: str) -> None:
    ds_dir = paths.harnesses_dir / harness_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    try:
        with (ds_dir / "sanity_status.txt").open("w", encoding="utf-8") as f:
            f.write(status + "\n")
    except OSError as exc:
        log.warning(f"could not write sanity_status for {harness_id}: {exc}")


def _read_meta(harness_dir: Path) -> Dict[str, Any]:
    """Read meta.json (returns {} if missing/invalid)."""
    meta_path = harness_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"could not parse meta.json for {harness_dir.name}: {exc}")
        return {}


def _read_parent_ids(harness_dir: Path) -> List[str]:
    """Read `meta.json::parent_ids` (or fall back to old `parent_id`).

    Returns:
        List of harness ids that the proposer says it drew from. Empty list
        if meta.json is missing or doesn't list any (e.g. seed harness).
    """
    meta = _read_meta(harness_dir)
    if not meta:
        return []
    raw = meta.get("parent_ids")
    if raw is None:
        # Legacy single-parent schema
        legacy = meta.get("parent_id")
        if legacy is None or legacy == "":
            return []
        return [str(legacy)]
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    log.warning(
        f"meta.json::parent_ids in {harness_dir.name} has unexpected type "
        f"{type(raw).__name__}; ignoring"
    )
    return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: avoid surprising prefix matching. Without this,
    # `--mode check` would be parsed as `--model check` (since "mode" is a
    # prefix of "model"), silently using "check" as the OpenAI model name
    # rather than erroring as "unrecognized argument".
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file. CLI flags override its values.",
    )
    # Each override defaults to None so we can detect "explicitly set" vs "use config/default"
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset names. If given, replaces any YAML "
             "`datasets:` block; each benchmark gets its family's default "
             "`stages` sizes/thresholds (customize via YAML for non-defaults).",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["search", "test", "dev"],
        help="search = training/exploration on the search (training) split; "
             "test = held-out evaluation on the test split; "
             "dev = run the whole pipeline at stages.sanity_check sizes "
             "(search split, no gauntlet, no sanity gate).",
    )

    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--update-type", default=None,
                        choices=["all_at_once", "chunked", "sequential"])
    parser.add_argument("--max-sample-concurrent", type=int, default=None)

    parser.add_argument("--proposer-model", default=None)
    parser.add_argument("--proposer-max-turns", type=int, default=None)
    parser.add_argument("--proposer-timeout-s", type=int, default=None)
    parser.add_argument(
        "--proposer-disallowed-tools", default=None,
        help='Comma-separated list of tools the proposer cannot use (default ["mcp__*"]). '
             'Pass "" to allow all tools. Examples: "mcp__*", "Bash,WebFetch,mcp__*".',
    )

    parser.add_argument("--tau", type=float, default=None,
                        help="Softmax temperature for Frontier.sample_parent (held-out test only)")

    parser.add_argument("--k-per-step", type=int, default=None,
                        help="Harnesses proposed per outer-loop step (default 2)")

    parser.add_argument("--no-memory-cache", action="store_true",
                        help="Disable the cross-stage memory cache "
                             "(harnesses rebuild Phase-1 memory at every stage)")
    parser.add_argument("--no-sanity", action="store_true",
                        help="Disable the pre-eval sanity gate (mode=search/test; dev always skips it)")
    parser.add_argument("--sanity-max-retries", type=int, default=None,
                        help="Max propose_with_fix retries when sanity fails (default 2)")

    parser.add_argument("--run-name", default=None,
                        help="Per-run workspace dir under workspace/. "
                             "Defaults to a timestamp (YYYYMMDD_HHMMSS).")
    parser.add_argument("--no-seed", action="store_true",
                        help="Skip copying the seed harness — first proposed "
                             "harness has no prior to reference.")
    parser.add_argument("--seed-source", default=None,
                        help="Which subdir under seeds/ to copy as harness 0 "
                             "(default 'no_memory').")

    parser.add_argument("--gpu", action="store_true",
                        help="Pass --nv to evaluator's singularity exec so the "
                             "container can use the host's NVIDIA driver. "
                             "Default off (CPU-only).")
    parser.add_argument("--no-adopt-orphans", action="store_true",
                        help="Skip orphan-harness-dir adoption pass on resume. "
                             "Default ON: dirs on disk but not in frontier are "
                             "re-attached (only the missing pipeline stages re-run). "
                             "Pass this to force a clean restart instead.")
    parser.add_argument("--agent", default=None,
                        choices=["claude_code", "codex"],
                        help="Coding-agent backend the proposer drives. "
                             "claude_code (default) shells out to the `claude` CLI; "
                             "codex shells out to OpenAI Codex CLI (`codex exec`). "
                             "codex auth = OPENAI_API_KEY env var.")
    parser.add_argument("--prompts-version", default=None,
                        help="Prompt template version stem under "
                             "forge/prompts/templates/ (format "
                             "YYYYMMDD_HHMM_<hash8>). Pass 'latest' or omit to "
                             "use forge/prompts/templates/_default. Pin to an "
                             "explicit stem for A/B comparisons of prompt revisions.")

    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    _setup_logging(args.verbose)
    cfg = _resolve_config(args)
    paths.set_run_id(cfg["run_name"])
    run_log = _attach_run_log()
    try:
        resolved_prompts_version = resolve_version(cfg["prompts"]["version"])
        # Load now so missing file / missing exports surface at startup, not
        # halfway through the search loop. Result is cached in loader._MODULE_CACHE.
        load_template_module(resolved_prompts_version)
    except PromptVersionError as exc:
        log.error(f"prompts: {exc}")
        sys.exit(2)
    log.info(
        f"prompts version: {resolved_prompts_version} "
        f"(cfg.prompts.version={cfg['prompts']['version']!r})"
    )
    # PIN the resolved stem for the whole run (before the config snapshot,
    # so config.yaml records it too). Without this, "latest" is re-resolved
    # against templates/_default on EVERY propose call — running
    # `bump_prompt.sh finalize --set-default` while a long search is in
    # flight would silently switch prompt versions mid-run while the startup
    # log and config.yaml snapshot still claim the old stem (breaking the
    # A/B integrity the versioning system exists for).
    cfg["prompts"]["version"] = resolved_prompts_version
    cfg_snapshot = _write_resolved_config(cfg)
    log.info(
        f"run config: {json.dumps(_snapshot_view(cfg), default=str)}"
    )
    log.info(f"workspace: {paths.workspace}")
    log.info(f"run log: {run_log}")
    log.info(f"config snapshot: {cfg_snapshot}")
    try:
        asyncio.run(search_loop(cfg))
    except ProposerInfraError as exc:
        # search_loop already logged the actionable message. Exit with a
        # distinct code so wrapper scripts can detect "abort due to propose
        # infra failure" (rc=2) versus other errors (rc=1, generic crashes).
        log.error(f"search aborted: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
