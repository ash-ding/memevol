"""Harness validator (host-side, optional / external use).

A candidate is a directory under `workspace/harnesses/<id>/` containing:
  - `harness.py` (REQUIRED) — defines a `MemoStructure` subclass
  - `requirements.txt` (OPTIONAL) — pip deps layered on eval-base
  - `meta.json` (OPTIONAL) — {"parent_ids": [...], "description": ...}
  - helper `.py` files (OPTIONAL) — importable from harness.py via sys.path

Validation = dynamic import + subclass check. No execution beyond import.

⚠ As of 2026-04-25 the orchestrator no longer calls this on the host.
It used to live in the propose→eval pipeline as a "is the file even
loadable?" pre-check before paying for Singularity startup, but that
required `baselines/venv/` to mirror the container's package list — when
a CC-proposed harness imported `rank_bm25` (in container, not in host venv)
this validator would false-fail the harness despite it being correct.

Today, validation happens inside the container as the first step of
`launch.py::_load_harness_class` (called by both sanity and full eval).
This module is kept for external scripts, CI hooks, or interactive dev
that want to syntax-check a harness without spinning up Singularity. Be
aware that an `ImportError` here may simply mean a base-image package is
missing from `baselines/venv/`.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Type

from common.harness_base import MemoStructure

REQUIRED_FILE = "harness.py"


class HarnessError(Exception):
    """Raised when a candidate directory fails validation."""


def load_harness_class(harness_dir: Path) -> Type[MemoStructure]:
    """Import harness.py and return its MemoStructure subclass."""
    harness_py = harness_dir / REQUIRED_FILE
    if not harness_py.exists():
        raise HarnessError(f"Missing {REQUIRED_FILE} in {harness_dir}")

    dir_str = str(harness_dir)
    if dir_str not in sys.path:
        sys.path.insert(0, dir_str)

    mod_name = f"forge_harness_{harness_dir.name}"
    spec = importlib.util.spec_from_file_location(mod_name, str(harness_py))
    if spec is None or spec.loader is None:
        raise HarnessError(f"Cannot create module spec for {harness_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise HarnessError(f"Import failed: {type(exc).__name__}: {exc}") from exc

    candidates = [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        # select the class defined in this harness file; imported bases like
        # forge.harness_base.MemoStructure are no longer abstract, so an
        # isabstract filter would wrongly match them.
        if issubclass(obj, MemoStructure) and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise HarnessError(f"No MemoStructure subclass found in {harness_py}")
    return candidates[0]


def validate(harness_dir: Path) -> None:
    """Raise HarnessError if the candidate is invalid; return None if OK."""
    load_harness_class(harness_dir)
