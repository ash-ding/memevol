"""What a harness IS, on disk — shared by the orchestrator and the seed cache.

Split out of `forge.orchestrator` because the cache needed these three
helpers and the orchestrator runs as `python -m forge.orchestrator`, i.e. as
`__main__`. Importing `forge.orchestrator` from inside it therefore executed
the whole 2600-line module a SECOND time, under its real name — and a
re-execution that raced an import left the cache failing with
`cannot import name 'load_prompt_parts' from 'forge.prompts'`, swallowed into
a warning while the run reported success and the seed silently went missing.
Nothing here imports the orchestrator, so that cannot happen again.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import List, Optional

from benchmarks.registry import DATASETS
from forge.paths import ENTRY_FILE, entry_file


#: Bookkeeping a harness dir accumulates that is NOT its code: eval artifacts
#: (one dir per dataset, plus the evaluator's transient runs/), the cache's own
#: index, and status files the pipeline writes next to the code. Everything
#: else in the directory IS the harness — a proposal is usually just
#: memo.py, but a packaged baseline (tools/package_baseline.py) brings a
#: whole package tree with it.
_NON_CODE_DIRS = frozenset({"runs", "evals", "__pycache__", ".git"}) | frozenset(DATASETS)
_NON_CODE_FILES = frozenset({
    "meta.json",            # carries the hash itself — self-reference
    "PROPOSAL_READY",       # proposer sentinel
    "sanity_status.txt", "build.log", "index.json",
})


def _code_files(harness_dir: Path) -> List[Path]:
    """Every file that makes up the harness, sorted, relative order stable."""
    out: List[Path] = []
    for path in sorted(harness_dir.rglob("*")):
        if not path.is_file() or path.name in _NON_CODE_FILES:
            continue
        rel = path.relative_to(harness_dir)
        if any(part in _NON_CODE_DIRS for part in rel.parts[:-1]):
            continue
        out.append(path)
    return out


def _compute_content_hash(harness_dir: Path) -> str:
    """sha256 over the harness's code — every file, at its path — [:16].

    Paths are hashed alongside contents: moving a file is a different harness,
    and two packaged baselines that share a memo.py must not collide because
    their top-level files happen to match.

    Subdirectories count (they did not before 2026-09-18): a packaged baseline
    keeps its method under `baselines/harness/<name>/`, so a hash over
    top-level files alone would give every packaged baseline with the same
    requirements the same identity.
    """
    if entry_file(harness_dir) is None:
        raise FileNotFoundError(f"{ENTRY_FILE} missing in {harness_dir}")
    h = hashlib.sha256()
    for path in _code_files(harness_dir):
        h.update(str(path.relative_to(harness_dir)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\n")
    return h.hexdigest()[:16]



def _copy_harness_code(src: Path, dst: Path) -> None:
    """Copy a harness's code from `src` into `dst`, leaving results behind.

    Mirrors `_code_files` — plus `meta.json`, which is not hashed but IS part
    of what a harness is (its parents and description).
    """
    dst.mkdir(parents=True, exist_ok=True)
    files = _code_files(src)
    meta = src / "meta.json"
    if meta.is_file():
        files.append(meta)
    for path in files:
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
