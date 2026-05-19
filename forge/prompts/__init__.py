"""Versioned prompt templates for the forge proposer.

Public API (back-compat with the pre-package `forge/prompts.py`):
  - build_proposer_system(*, sanity_enabled, active_datasets, update_type, version)
  - proposer_task_prompt(new_dir_rel, *, version)
  - proposer_fix_prompt(new_dir_rel, error_trace, *, version)
  - PromptVersionError
  - resolve_version(version)        — exposed for orchestrator startup banner
  - load_template_module(version)   — exposed for tests / introspection

Template versions live under `forge/prompts/templates/<YYYYMMDD>_<HHMM>_<hash8>.py`,
with `forge/prompts/templates/_default` pointing at the current default.
"""

from .loader import (
    PromptVersionError,
    load_template_module,
    resolve_version,
)
from .renderer import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)

__all__ = [
    "PromptVersionError",
    "build_proposer_system",
    "load_template_module",
    "proposer_fix_prompt",
    "proposer_task_prompt",
    "resolve_version",
]
