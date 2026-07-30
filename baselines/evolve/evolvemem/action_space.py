"""EvolveMem's evolvable retrieval-configuration action space (θ).

The paper (EvolveMem, §3.2) exposes the FULL retrieval configuration as a
structured action space optimized by the diagnosis loop:

    θ = (k_sem, k_kw, k_str, B_ctx, mode, {w_v}, α, {θ_c}_{c∈C})

This module is the single source of truth for that space: defaults, bounds,
clamping, (de)serialization, and the random perturbation used by the
explore-on-stagnation branch. `memo_evolvemem.py` consumes a config dict at
instance construction; `evolution.py` mutates it between rounds.

Two pragmatic extensions over the paper's θ (both documented in README.md):
  - `per_category` implements the paper's per-category overrides θ_c with
    regex query-matching (the benchmarks here don't ship a category oracle
    at retrieve time, so categories are pattern-defined and the diagnosis
    LLM can invent/refine the patterns).
  - `extras` is the self-expansion surface: the diagnosis LLM may propose
    dimensions not in the original space. Keys the memo implements are
    listed in EXTRA_HOOKS; unknown keys are stored + logged but inert
    (honest bound on "entirely new configuration dimensions").
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FUSION_MODES = ("sum", "weighted_sum", "rrf")
ANSWER_STYLES = ("none", "concise", "explanatory", "verifying", "inferential")
EXTRACTION_MODES = ("raw", "llm")

# Answer-generation style α is realized as a guidance sentence placed in the
# retrieved dict (the QA agent sees the dict verbatim); "none" omits it so
# the default configuration is bit-identical to a plain retrieval baseline.
ANSWER_STYLE_TEXT = {
    "concise": "Answer as concisely as possible using only the retrieved memories.",
    "explanatory": "Answer and briefly explain which retrieved memories support it.",
    "verifying": "Cross-check the answer against every retrieved memory before responding.",
    "inferential": "If no memory states the answer directly, infer it by combining the retrieved memories.",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    # --- multi-view candidate generation ---
    "k_sem": 10,          # semantic (dense-embedding) view top-k
    "k_kw": 10,           # lexical (BM25) view top-k
    "k_str": 5,           # structured-metadata (entity/keyword filter) view top-k
    # --- fusion ---
    "fusion_mode": "rrf",             # sum | weighted_sum | rrf
    "w_sem": 1.0,                      # per-view weights (weighted_sum mode)
    "w_kw": 1.0,
    "w_str": 0.5,
    "rrf_k": 60,                       # RRF smoothing constant
    "lambda_importance": 0.1,          # λ_ι — importance bonus weight
    "lambda_recency": 0.1,             # λ_r — recency bonus weight
    "recency_halflife_days": 90.0,     # recency factor half-life
    # --- context budget / answer ---
    "b_ctx": 12,                       # max memories passed to the QA agent
    "answer_style": "none",            # α — none | concise | explanatory | verifying | inferential
    # --- query augmentation toggles ---
    "entity_swap": False,              # strip person names, re-search by topic, union
    "query_decomposition": False,      # LLM multi-hop split + RRF merge
    "decomposition_model": "gpt-5-mini",
    # --- memory store (§3.1 knobs the diagnosis loop may also touch) ---
    "extraction_mode": "raw",          # raw (heuristic units) | llm (typed extraction)
    "extraction_model": "gpt-5-mini",
    "extraction_window": 40,           # items per LLM-extraction window
    "unit_granularity": "auto",        # auto | item | session
    "dedup_tau": 0.9,                  # Jaccard threshold τ_J for consolidation dedup
    "importance_decay": 0.002,         # α_d per day
    "importance_floor": 0.1,           # ι_min
    # --- per-category overrides θ_c: first regex match on the query wins ---
    # e.g. [{"name": "temporal", "pattern": "(?i)when|last|recent",
    #        "overrides": {"lambda_recency": 0.4}}]
    "per_category": [],
    # --- self-expansion surface (see EXTRA_HOOKS) ---
    "extras": {},
}

# (lo, hi) bounds for numeric dimensions — clamp_config projects onto these
# before any proposed value is applied (paper: "every dimension is clamped
# to a safe range").
BOUNDS: Dict[str, Tuple[float, float]] = {
    "k_sem": (0, 50),
    "k_kw": (0, 50),
    "k_str": (0, 30),
    "w_sem": (0.0, 5.0),
    "w_kw": (0.0, 5.0),
    "w_str": (0.0, 5.0),
    "rrf_k": (1, 200),
    "lambda_importance": (0.0, 1.0),
    "lambda_recency": (0.0, 1.0),
    "recency_halflife_days": (1.0, 3650.0),
    "b_ctx": (1, 60),
    "extraction_window": (5, 200),
    "dedup_tau": (0.5, 1.0),
    "importance_decay": (0.0, 0.1),
    "importance_floor": (0.0, 1.0),
}

INT_KEYS = {"k_sem", "k_kw", "k_str", "rrf_k", "b_ctx", "extraction_window"}
BOOL_KEYS = {"entity_swap", "query_decomposition"}
ENUM_KEYS = {
    "fusion_mode": FUSION_MODES,
    "answer_style": ANSWER_STYLES,
    "extraction_mode": EXTRACTION_MODES,
    "unit_granularity": ("auto", "item", "session"),
}

# `extras` keys the memo actually reads. The diagnosis LLM may propose any
# key; only these change behavior (unknown ones are stored + logged).
EXTRA_HOOKS = {
    "min_token_len": "int — drop BM25/keyword tokens shorter than this (default 2)",
    "stopword_extra": "list[str] — additional stopwords for keyword extraction",
    "dedup_scope": "'window'|'global' — dedup against recent 500 units or all (default window)",
    "session_join": "int — merge this many consecutive messages per unit for session data (default 4)",
    "query_expand_keywords": "int — add top-N keywords from top-ranked semantic hits to the BM25 query (0 = off)",
}


def clamp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Project a (possibly LLM-proposed) config onto the valid space.

    Unknown top-level keys are moved under `extras` rather than dropped, so
    a proposed new dimension is preserved for inspection even when inert.
    """
    out = copy.deepcopy(DEFAULT_CONFIG)
    extras = dict(out["extras"])
    for key, val in (cfg or {}).items():
        if key == "extras":
            if isinstance(val, dict):
                extras.update(val)
            continue
        if key == "per_category":
            out["per_category"] = _clamp_per_category(val)
            continue
        if key not in DEFAULT_CONFIG:
            extras[key] = val
            continue
        out[key] = _clamp_value(key, val, fallback=out[key])
    out["extras"] = extras
    return out


def _clamp_value(key: str, val: Any, fallback: Any) -> Any:
    if key in ENUM_KEYS:
        return val if val in ENUM_KEYS[key] else fallback
    if key in BOOL_KEYS:
        return bool(val)
    if key in BOUNDS:
        try:
            num = float(val)
        except (TypeError, ValueError):
            return fallback
        lo, hi = BOUNDS[key]
        num = min(max(num, lo), hi)
        return int(round(num)) if key in INT_KEYS else num
    if isinstance(fallback, str):
        return str(val)
    return val


def _clamp_per_category(val: Any) -> List[Dict[str, Any]]:
    """Validate per-category override entries; silently drop malformed ones."""
    import re
    out: List[Dict[str, Any]] = []
    if not isinstance(val, list):
        return out
    for entry in val[:8]:  # cap category count
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        overrides = entry.get("overrides")
        if not isinstance(pattern, str) or not isinstance(overrides, dict):
            continue
        try:
            re.compile(pattern)
        except re.error:
            continue
        clean = {
            k: _clamp_value(k, v, fallback=DEFAULT_CONFIG.get(k))
            for k, v in overrides.items()
            if k in DEFAULT_CONFIG and k not in ("per_category", "extras")
        }
        if clean:
            out.append({
                "name": str(entry.get("name", f"cat{len(out)}")),
                "pattern": pattern,
                "overrides": clean,
            })
    return out


def apply_adjustments(cfg: Dict[str, Any], adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """θ ⊕ Δθ — apply a diagnosis proposal, then clamp.

    Each adjustment: {"parameter": <dotted key>, "new_value": <any>}.
    Supported paths: top-level keys, "extras.<name>", and "per_category"
    (whole-list replacement).
    """
    out = copy.deepcopy(cfg)
    for adj in adjustments or []:
        param = str(adj.get("parameter", ""))
        value = adj.get("new_value")
        if not param:
            continue
        if param.startswith("extras."):
            out.setdefault("extras", {})[param[len("extras."):]] = value
        else:
            out[param] = value
    return clamp_config(out)


def perturb_config(cfg: Dict[str, Any], seed: int, scale: float = 0.25) -> Dict[str, Any]:
    """η_exp — random perturbation of numeric dimensions to escape local
    optima (explore-on-stagnation branch). Each bounded numeric dimension is
    jittered by ±scale of its range with prob 0.5; one enum dimension is
    re-rolled with prob 0.3."""
    rng = random.Random(seed)
    out = copy.deepcopy(cfg)
    for key, (lo, hi) in BOUNDS.items():
        if rng.random() < 0.5:
            span = (hi - lo) * scale
            jittered = float(out.get(key, DEFAULT_CONFIG[key])) + rng.uniform(-span, span)
            out[key] = jittered
    if rng.random() < 0.3:
        enum_key = rng.choice(list(ENUM_KEYS))
        out[enum_key] = rng.choice(list(ENUM_KEYS[enum_key]))
    return clamp_config(out)


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    """Load + clamp a config file; missing path → clamped defaults."""
    if path is None or not Path(path).exists():
        return clamp_config({})
    with Path(path).open(encoding="utf-8") as f:
        return clamp_config(json.load(f))


def weak_initial_config() -> Dict[str, Any]:
    """Official weak_initial_config, mapped onto this space: a deliberately
    under-powered BM25-only starting point (semantic + structured views off,
    tight context, no augmentation, no recency shaping) so evolution has
    room to climb — mirrors upstream's weak start ("we want the
    evolved-minus-static delta to be the headline"). With only the keyword
    view active the fusion mode is immaterial; weighted_sum with w_kw=1
    reproduces upstream's keyword_only."""
    return clamp_config({
        "k_sem": 0,
        "k_kw": 5,
        "k_str": 0,
        "b_ctx": 8,
        "fusion_mode": "weighted_sum",
        "w_sem": 0.0, "w_kw": 1.0, "w_str": 0.0,
        "lambda_importance": 0.0,
        "lambda_recency": 0.0,
        "entity_swap": False,
        "query_decomposition": False,
        "answer_style": "concise",
    })
