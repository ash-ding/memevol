"""Shared config resolution for the baselines (and the reusable deep_merge
primitive forge also uses). Baseline-agnostic: imports only stdlib + PyYAML,
NEVER forge. Pattern mirrors forge's DEFAULT_CONFIG ← YAML ← CLI precedence.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """In-place recursive merge of overlay into base (dicts only)."""
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def resolve_config(
    defaults: Dict[str, Any],
    config_path: Optional[str],
    cli_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Effective config, precedence lowest→highest:
      1. deepcopy(defaults)
      2. YAML at config_path (if given) deep-merged in (must be a mapping)
      3. cli_overrides — each key applied ONLY when its value is not None
    """
    cfg = copy.deepcopy(defaults)
    if config_path is not None:
        file_cfg = load_config_file(config_path)
        deep_merge(cfg, file_cfg)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Strict-config completeness validation — used by baselines' run.py (Task 2)
# and forge's _resolve_config (Task 3) to reject configs that omit required
# keys instead of silently falling back to a default. A `null` YAML value
# still counts as "provided" (the operator explicitly opted into the
# default) — only an ABSENT key is a completeness violation.
# ---------------------------------------------------------------------------


class ConfigCompletenessError(ValueError):
    """Raised in strict-config mode when the config file omits a required key."""


def load_config_file(path):
    """safe_load a YAML config to a mapping (shared by resolve_config + strict)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config file must be a mapping at top level: {p}")
    return cfg


def strict_on(config_path, resolved_cfg):
    """Strict completeness enforcement is active iff a config file was given AND
    strict_config (default True) was not turned off."""
    return config_path is not None and bool(resolved_cfg.get("strict_config", True))


def provided_keys(file_cfg, cli_overrides):
    """Top-level keys the operator actually supplied: YAML keys ∪ CLI keys whose
    value is not None."""
    return set(file_cfg) | {k for k, v in cli_overrides.items() if v is not None}


def _completeness_msg(context, missing):
    return (
        f"{context}: config is missing required key(s): {sorted(missing)}. Every "
        f"parameter must be listed explicitly (a null value is allowed) — this is "
        f"strict-config mode. Set strict_config: false (or pass --no-strict-config) "
        f"to disable."
    )


def require_present_keys(provided, required, context):
    missing = set(required) - set(provided)
    if missing:
        raise ConfigCompletenessError(_completeness_msg(context, missing))


REQUIRED = object()   # schema sentinel: a required leaf key
_MISSING = object()   # walk sentinel: key absent from the provided tree


class Cond:
    """Schema node applied only when predicate(resolved_cfg) is truthy."""
    def __init__(self, predicate, subschema):
        self.predicate = predicate
        self.subschema = subschema


def _walk_schema(node, provided_node, resolved_cfg, path, missing):
    if isinstance(node, Cond):
        if node.predicate(resolved_cfg):
            _walk_schema(node.subschema, provided_node, resolved_cfg, path, missing)
        return
    if node is REQUIRED:
        if provided_node is _MISSING:
            missing.append(".".join(path))
        return
    if isinstance(node, dict):
        pv = provided_node if isinstance(provided_node, dict) else {}
        for k, sub in node.items():
            _walk_schema(sub, pv.get(k, _MISSING), resolved_cfg, path + [k], missing)


def require_schema(provided_tree, schema, resolved_cfg, context):
    """Validate a nested `provided_tree` (raw YAML ∪ CLI-provided paths) against
    `schema` (nested dict of REQUIRED / Cond / sub-dicts). Collects ALL missing
    paths and raises ConfigCompletenessError once. `resolved_cfg` drives Cond
    predicates."""
    missing = []
    _walk_schema(schema, provided_tree, resolved_cfg, [], missing)
    if missing:
        raise ConfigCompletenessError(_completeness_msg(context, missing))


def missing_schema_paths(provided_tree, schema, resolved_cfg):
    """Like require_schema but returns the missing-path list instead of raising —
    for callers that combine it with extra checks before one raise."""
    missing = []
    _walk_schema(schema, provided_tree, resolved_cfg, [], missing)
    return missing


def raise_completeness(context, missing):
    """Raise ConfigCompletenessError with the standard message for `missing`
    paths (public entry so callers needn't touch the private formatter)."""
    if missing:
        raise ConfigCompletenessError(_completeness_msg(context, missing))
