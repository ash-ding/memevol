"""Prompt part loading + version resolution.

A prompt version is a set of files under `forge/prompts/parts/`, all sharing
one version stem — `<YYYYMMDD>_<HHMM>_<8 hex chars>`:

    evolving_<stem>.md     HOW TO EVOLVE: the loop, the workspace layout, the
                           harness contract, the rules. Carries the
                           `<<DESIGN_KNOWLEDGE_BLOCK>>` sentinel.
    design_<stem>.md       HOW TO DESIGN A MEMORY SYSTEM: the search
                           direction. Substituted into that sentinel.
    task_<stem>.md         the per-iteration task prompt
    fix_<stem>.md          the post-sanity-failure fix prompt
    fragments_<stem>.py    keyed substitution data (see that file's docstring)

The special name `"latest"` (or `None` / empty string) resolves to whatever
stem is written in `forge/prompts/parts/_default`.

The two prose halves are separate files on purpose: `evolving_` is mechanical
and follows the loop's own file organization, while `design_` is the part that
should eventually be derived from evidence rather than hand-written. Swapping
`design_` alone is therefore a supported, one-file change.

Once loaded, a version's parts are cached by stem — subsequent calls within
the same process are free.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromptVersionError(RuntimeError):
    """Raised when a requested prompt version is missing, malformed, or
    inconsistent. Always actionable — explains exactly what's wrong."""


_HERE = Path(__file__).resolve().parent
_PARTS_DIR = _HERE / "parts"
_DEFAULT_POINTER = _PARTS_DIR / "_default"

# Version stem: YYYYMMDD_HHMM_hash8 (UTC date, hour:minute, 8 hex)
_VERSION_STEM_RX = re.compile(r"^\d{8}_\d{4}_[0-9a-f]{8}$")

# Where the design half is spliced into the evolving half.
DESIGN_SENTINEL = "<<DESIGN_KNOWLEDGE_BLOCK>>\n"

# Exports every fragments_<stem>.py must provide — checked at load time so a
# malformed fragments file fails fast with a clear message instead of crashing
# the renderer with an AttributeError somewhere deep.
_REQUIRED_FRAGMENTS = (
    "OBJECTIVE_AXES_SUBS",
    "SANITY_ON_SUBS",
    "SANITY_OFF_SUBS",
    "DATASET_INFO",
    "DATASET_RENDER_ORDER",
)


@dataclass(frozen=True)
class PromptParts:
    """One prompt version, assembled. Attribute names match what the renderer
    reads, so the renderer stays version-agnostic."""

    PROMPT_VERSION: str
    SYSTEM_TEMPLATE: str
    TASK_PROMPT_TEMPLATE: str
    FIX_PROMPT_TEMPLATE: str
    OBJECTIVE_AXES_SUBS: Dict[Any, str]
    SANITY_ON_SUBS: Dict[str, str]
    SANITY_OFF_SUBS: Dict[str, str]
    DATASET_INFO: Dict[str, Dict[str, str]]
    DATASET_RENDER_ORDER: List[str]


_PARTS_CACHE: Dict[str, PromptParts] = {}


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


def _available_versions() -> List[str]:
    return sorted(
        p.name[len("evolving_"):-len(".md")]
        for p in _PARTS_DIR.glob("evolving_*.md")
    )


def _read_part(stem: str, kind: str) -> str:
    path = _PARTS_DIR / f"{kind}_{stem}.md"
    if not path.exists():
        raise PromptVersionError(
            f"prompts: {kind} part for version {stem!r} not found at {path}. "
            f"Available versions: {_available_versions()}"
        )
    return path.read_text(encoding="utf-8")


def load_prompt_parts(version: Optional[str]) -> PromptParts:
    """Assemble the parts for a given (or default-resolved) version.

    Caches by stem, so repeated calls within a process are free.

    Raises PromptVersionError on any of:
      - a part file for the stem is missing
      - the evolving half does not carry exactly one design sentinel
      - the fragments module declares a mismatched PROMPT_VERSION
      - the fragments module is missing any of `_REQUIRED_FRAGMENTS`
    """
    stem = resolve_version(version)
    if stem in _PARTS_CACHE:
        return _PARTS_CACHE[stem]

    evolving = _read_part(stem, "evolving")
    design = _read_part(stem, "design")

    n = evolving.count(DESIGN_SENTINEL)
    if n != 1:
        raise PromptVersionError(
            f"prompts: evolving_{stem}.md carries the design sentinel "
            f"{DESIGN_SENTINEL.strip()} {n} times; expected exactly 1. "
            f"Without it the memory-system design knowledge is dropped."
        )
    system_template = evolving.replace(DESIGN_SENTINEL, design)

    try:
        fragments = importlib.import_module(f"forge.prompts.parts.fragments_{stem}")
    except ModuleNotFoundError as exc:
        raise PromptVersionError(
            f"prompts: fragments module for version {stem!r} not importable "
            f"({exc}). Expected {_PARTS_DIR / f'fragments_{stem}.py'}. "
            f"Available versions: {_available_versions()}"
        ) from exc

    declared = getattr(fragments, "PROMPT_VERSION", None)
    if declared != stem:
        raise PromptVersionError(
            f"prompts: fragments_{stem}.py declares PROMPT_VERSION={declared!r}, "
            f"but the filename stem is {stem!r}. Fix the file so they match."
        )

    missing = [name for name in _REQUIRED_FRAGMENTS if not hasattr(fragments, name)]
    if missing:
        raise PromptVersionError(
            f"prompts: fragments_{stem}.py is missing required exports: "
            f"{missing}. See forge/prompts/loader.py::_REQUIRED_FRAGMENTS."
        )

    parts = PromptParts(
        PROMPT_VERSION=stem,
        SYSTEM_TEMPLATE=system_template,
        TASK_PROMPT_TEMPLATE=_read_part(stem, "task"),
        FIX_PROMPT_TEMPLATE=_read_part(stem, "fix"),
        OBJECTIVE_AXES_SUBS=fragments.OBJECTIVE_AXES_SUBS,
        SANITY_ON_SUBS=fragments.SANITY_ON_SUBS,
        SANITY_OFF_SUBS=fragments.SANITY_OFF_SUBS,
        DATASET_INFO=fragments.DATASET_INFO,
        DATASET_RENDER_ORDER=fragments.DATASET_RENDER_ORDER,
    )
    _PARTS_CACHE[stem] = parts
    return parts
