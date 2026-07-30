"""EvolveMem's action space θ over the REAL SimpleMem substrate.

The paper's actual setup: the evolvable configuration is SimpleMem's own
retrieval/build knobs. This module mirrors `action_space.py`'s function
surface (clamp_config / apply_adjustments / perturb_config / save/load)
over the substrate's native dimensions, so `evolution.py`'s guard loop and
`run_main.py` can drive either space through the same interface (the space
module is passed in — see run_main's --substrate flag).

Dimensions (all consumed by simplemem/memo.py via $EVOLVEMEM_CONFIG):
  build     window_size
  retrieve  semantic_top_k / keyword_top_k / structured_top_k
            enable_planning / enable_reflection / max_reflection_rounds

Deliberately NOT exposed: internal LLM + embedding models (cost/fidelity
anchors), parallelism knobs (throughput, not quality — evolving them would
just game the delay axis without touching retrieval behavior).
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG: Dict[str, Any] = {
    # --- build (§3.1) ---
    "window_size": 40,             # dialogues per compression window
    # --- retrieval views (§3.3) ---
    "semantic_top_k": 25,
    "keyword_top_k": 5,
    "structured_top_k": 5,
    # --- retrieval planning / reflection (§3.3) ---
    "enable_planning": True,
    "enable_reflection": True,
    "max_reflection_rounds": 2,
}

BOUNDS: Dict[str, Tuple[float, float]] = {
    "window_size": (5, 120),
    "semantic_top_k": (1, 80),
    "keyword_top_k": (0, 40),
    "structured_top_k": (0, 40),
    "max_reflection_rounds": (0, 4),
}

INT_KEYS = set(BOUNDS)
BOOL_KEYS = {"enable_planning", "enable_reflection"}


def clamp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Project a (possibly LLM-proposed) config onto the valid space.
    Unknown keys are preserved under `extras` for inspection but inert
    (the substrate adapter only reads the known dimensions)."""
    out = copy.deepcopy(DEFAULT_CONFIG)
    extras: Dict[str, Any] = {}
    for key, val in (cfg or {}).items():
        if key == "extras":
            if isinstance(val, dict):
                extras.update(val)
            continue
        if key not in DEFAULT_CONFIG:
            extras[key] = val
            continue
        if key in BOOL_KEYS:
            out[key] = bool(val)
            continue
        try:
            num = float(val)
        except (TypeError, ValueError):
            continue
        lo, hi = BOUNDS[key]
        out[key] = int(round(min(max(num, lo), hi)))
    if extras:
        out["extras"] = extras
    return out


def apply_adjustments(cfg: Dict[str, Any], adjustments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """θ ⊕ Δθ — apply a diagnosis proposal, then clamp."""
    out = copy.deepcopy(cfg)
    for adj in adjustments or []:
        param = str(adj.get("parameter", ""))
        if not param:
            continue
        if param.startswith("extras."):
            out.setdefault("extras", {})[param[len("extras."):]] = adj.get("new_value")
        else:
            out[param] = adj.get("new_value")
    return clamp_config(out)


def perturb_config(cfg: Dict[str, Any], seed: int, scale: float = 0.25) -> Dict[str, Any]:
    """η_exp — random perturbation to escape local optima."""
    rng = random.Random(seed)
    out = copy.deepcopy(cfg)
    for key, (lo, hi) in BOUNDS.items():
        if rng.random() < 0.5:
            span = (hi - lo) * scale
            out[key] = float(out.get(key, DEFAULT_CONFIG[key])) + rng.uniform(-span, span)
    for key in BOOL_KEYS:
        if rng.random() < 0.2:
            out[key] = not bool(out.get(key, DEFAULT_CONFIG[key]))
    return clamp_config(out)


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return clamp_config({})
    with Path(path).open(encoding="utf-8") as f:
        return clamp_config(json.load(f))


# ---------------------------------------------------------------------------
# Diagnosis support — space description + substrate-specific rubric consumed
# by diagnosis.diagnose(space=...).
# ---------------------------------------------------------------------------

def space_description() -> Dict[str, Any]:
    return {
        "dimensions": dict(DEFAULT_CONFIG),
        "numeric_bounds": {k: list(v) for k, v in BOUNDS.items()},
        "booleans": sorted(BOOL_KEYS),
        "note": ("θ over the REAL SimpleMem substrate: window compression size, "
                 "the three retrieval-view top-ks, and the LLM planning/"
                 "reflection controls. No per-category overrides and no extras "
                 "hooks exist on this substrate."),
    }


RUBRIC = """\
Common failure patterns to check for (SimpleMem substrate — diagnose from
evidence, not intuition):
- MISSING GOLD ENTRY: the fact was never surfaced. Levers: semantic_top_k /
  keyword_top_k up; enable_planning on (multi-query expansion widens recall).
- NOISE DILUTION: gold retrieved but buried in a large context. Levers:
  semantic_top_k down; keyword/structured_top_k down.
- MULTI-HOP MISS: answer needs several entries; only one hop surfaced.
  Levers: enable_reflection on / max_reflection_rounds up (adequacy check
  issues follow-up queries); enable_planning on.
- TEMPORAL/SYMBOLIC MISS: time- or person-constrained questions failing.
  Levers: structured_top_k up (symbolic-layer filtering does the work here).
- COMPRESSION LOSS: the memory entries themselves lack the detail (window
  compression dropped it). Levers: window_size down (smaller windows → more,
  finer entries). This is a BUILD-side change — expensive to re-evaluate.
- COST WASTE: score fine but reflection/planning burning calls without
  changing outcomes. Levers: max_reflection_rounds down, enable_planning off.
"""
