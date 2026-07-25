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
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if not isinstance(file_cfg, dict):
            raise ValueError(f"config file must be a mapping at top level: {path}")
        deep_merge(cfg, file_cfg)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg
