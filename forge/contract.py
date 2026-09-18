"""Harness validator (host-side, optional / external use).

A candidate is a directory under `workspace/harnesses/<id>/` containing:
  - `memo.py` (REQUIRED) — defines a `MemoClass` subclass
  - `requirements.txt` (OPTIONAL) — pip deps layered on eval-base
  - `meta.json` (OPTIONAL) — {"parent_ids": [...], "description": ...}
  - `src/` (OPTIONAL) — the implementation, organised however the harness
    likes; importable from memo.py, which has its own dir on sys.path

Validation = dynamic import + subclass check. No execution beyond import.

⚠ As of 2026-04-25 the orchestrator no longer calls this on the host.
It used to live in the propose→eval pipeline as a "is the file even
loadable?" pre-check before paying for Singularity startup, but that
required the host env to mirror the container's package list — when
a CC-proposed harness imported `rank_bm25` (in container, not in the host
env) this validator would false-fail the harness despite it being correct.

Today, validation happens inside the container as the first step of
`launch.py::_load_harness_class` (called by both sanity and full eval).
This module is kept for external scripts, CI hooks, or interactive dev
that want to syntax-check a harness without spinning up Singularity. Be
aware that an `ImportError` here may simply mean a base-image package is
missing from the host env (2026-08-05: the host env is a `uv sync`'d
project — `.venv/` built from the repo-root `pyproject.toml` / `uv.lock`).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Type

from common.memo_class import MemoClass

from forge.paths import ENTRY_FILE, LEGACY_ENTRY_FILE, entry_file

REQUIRED_FILE = ENTRY_FILE


class HarnessError(Exception):
    """Raised when a candidate directory fails validation."""


def load_harness_class(harness_dir: Path) -> Type[MemoClass]:
    """Import the harness's interface file and return its MemoClass subclass."""
    harness_py = entry_file(harness_dir)
    if harness_py is None:
        raise HarnessError(
            f"Missing {REQUIRED_FILE} in {harness_dir} "
            f"(also accepted for older harnesses: {LEGACY_ENTRY_FILE})")

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

    from common.memo_select import select_memo_class
    # Only classes DEFINED in this harness file (not imported bases), concrete first.
    defined = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
               if obj.__module__ == module.__name__]
    try:
        return select_memo_class(defined, str(harness_py))
    except TypeError as exc:
        raise HarnessError(str(exc)) from exc


def validate(harness_dir: Path) -> None:
    """Raise HarnessError if the candidate is invalid; return None if OK."""
    load_harness_class(harness_dir)
