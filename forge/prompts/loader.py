"""Prompt template loading + version resolution.

A "version stem" is a filename under `forge/prompts/templates/`, without the
`.py` extension — `<YYYYMMDD>_<HHMM>_<8 hex chars>`. The special name
`"latest"` (or `None` / empty string) resolves to whatever stem is written
in `forge/prompts/templates/_default`.

Once loaded, a template module is cached by stem — subsequent calls within
the same process are free.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional


class PromptVersionError(RuntimeError):
    """Raised when a requested prompt version is missing, malformed, or
    inconsistent. Always actionable — explains exactly what's wrong."""


_HERE = Path(__file__).resolve().parent
_TEMPLATES_DIR = _HERE / "templates"
_DEFAULT_POINTER = _TEMPLATES_DIR / "_default"

# Filename stem regex: YYYYMMDD_HHMM_hash8 (UTC date, hour:minute, 8 hex)
_VERSION_STEM_RX = re.compile(r"^\d{8}_\d{4}_[0-9a-f]{8}$")

# Required exports on every template module — checked at load time so a
# malformed template fails fast with a clear message instead of crashing
# the renderer with an AttributeError somewhere deep.
_REQUIRED_EXPORTS = (
    "PROMPT_VERSION",
    "SYSTEM_TEMPLATE",
    "SANITY_ON_SUBS",
    "SANITY_OFF_SUBS",
    "DATASET_INFO",
    "DATASET_RENDER_ORDER",
    "TASK_PROMPT_TEMPLATE",
    "FIX_PROMPT_TEMPLATE",
)


_MODULE_CACHE: Dict[str, ModuleType] = {}


def resolve_version(version: Optional[str]) -> str:
    """Resolve `None` / `"latest"` / `""` to the current default stem.

    Otherwise return `version` unchanged after validating its shape.

    Raises:
      PromptVersionError if the default pointer is missing/empty, or the
      requested name doesn't match the version-stem regex.
    """
    if version in (None, "", "latest"):
        if not _DEFAULT_POINTER.exists():
            raise PromptVersionError(
                f"prompts: default pointer missing at {_DEFAULT_POINTER}. "
                f"Either set cfg.prompts.version to an explicit stem, or "
                f"create the _default file with the active stem name."
            )
        stem = _DEFAULT_POINTER.read_text(encoding="utf-8").strip()
        if not stem:
            raise PromptVersionError(
                f"prompts: default pointer {_DEFAULT_POINTER} is empty"
            )
        if not _VERSION_STEM_RX.match(stem):
            raise PromptVersionError(
                f"prompts: default pointer contents {stem!r} doesn't look "
                f"like a version stem (expected YYYYMMDD_HHMM_hash8)"
            )
        return stem
    if not _VERSION_STEM_RX.match(version):
        raise PromptVersionError(
            f"prompts: version {version!r} doesn't match the expected "
            f"YYYYMMDD_HHMM_hash8 pattern"
        )
    return version


def load_template_module(version: Optional[str]) -> ModuleType:
    """Import the template module for a given (or default-resolved) version.

    Caches by stem, so repeated calls within a process are free.

    Raises PromptVersionError on any of:
      - version stem doesn't resolve to a file under templates/
      - module's PROMPT_VERSION attribute doesn't equal the stem
      - module is missing any of `_REQUIRED_EXPORTS`
    """
    stem = resolve_version(version)
    if stem in _MODULE_CACHE:
        return _MODULE_CACHE[stem]

    file_path = _TEMPLATES_DIR / f"{stem}.py"
    if not file_path.exists():
        available = sorted(
            p.stem for p in _TEMPLATES_DIR.glob("*.py")
            if not p.stem.startswith("_")
        )
        raise PromptVersionError(
            f"prompts: template {stem!r} not found at {file_path}. "
            f"Available: {available}"
        )

    module = importlib.import_module(f"forge.prompts.templates.{stem}")

    declared = getattr(module, "PROMPT_VERSION", None)
    if declared != stem:
        raise PromptVersionError(
            f"prompts: template {stem}.py declares PROMPT_VERSION={declared!r}, "
            f"but the filename stem is {stem!r}. Fix the file so they match."
        )

    missing = [name for name in _REQUIRED_EXPORTS if not hasattr(module, name)]
    if missing:
        raise PromptVersionError(
            f"prompts: template {stem}.py is missing required exports: "
            f"{missing}. See forge/prompts/loader.py::_REQUIRED_EXPORTS."
        )

    _MODULE_CACHE[stem] = module
    return module
