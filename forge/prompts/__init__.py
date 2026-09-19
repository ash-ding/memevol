"""Versioned prompt parts for the forge proposer.

A prompt version is four Markdown files plus one Python fragments module under
`forge/prompts/parts/`, sharing a `<YYYYMMDD>_<HHMM>_<hash8>` stem. The two
prose halves are deliberately separate:

  - `evolving_<stem>.md` — how to run the search: workspace layout, the
    two-phase protocol, the harness contract, the rules.
  - `design_<stem>.md`   — how to design a good memory system: the search
    direction. Spliced into the evolving half at
    `<<DESIGN_KNOWLEDGE_BLOCK>>`.

Splitting them means the design half can be replaced on its own — by hand, or
eventually by something derived from execution evidence — without touching the
mechanical half.

`forge/prompts/old/` holds the pre-split single-file templates, kept for
reference; nothing loads them.

Public API:
  - build_proposer_system(*, sanity_enabled, active_datasets, metrics, version)
  - proposer_task_prompt(new_dir_rel, *, version)
  - proposer_fix_prompt(new_dir_rel, error_trace, *, version)
  - PromptVersionError
  - resolve_version(version)      — exposed for orchestrator startup banner
  - load_prompt_parts(version)    — exposed for tests / introspection
  - PromptParts                   — what load_prompt_parts returns
"""

from .loader import (
    PromptParts,
    PromptVersionError,
    load_prompt_parts,
    resolve_version,
)
from .renderer import (
    build_proposer_system,
    proposer_fix_prompt,
    proposer_task_prompt,
)

__all__ = [
    "PromptParts",
    "PromptVersionError",
    "build_proposer_system",
    "load_prompt_parts",
    "proposer_fix_prompt",
    "proposer_task_prompt",
    "resolve_version",
]
