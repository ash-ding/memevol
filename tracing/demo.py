"""Standalone driver/demo — exercises the tracer in complete isolation.

Runnable as ``python -m tracing`` (or ``python -m tracing.demo``). It builds a
lightweight in-process fake memo (an amem-style ``note_list`` shape, so a real
shape adapter is used — no eval, no network, no heavy deps), calls
:func:`tracing.wrap_memo` ITSELF (the wrapping is performed here, NOT by
``common/workflow.py``), drives a scripted BUILD followed by a couple of
read-only RETRIEVE calls plus a ``mark()`` checkpoint, and produces a sample
per-user git repo under a temp/``workspace`` dir. It then prints the resulting
``git log`` — the memory-evolution timeline a human (or the smoke test) can
``git log`` / ``git diff`` / ``git blame``.

Because the demo owns the wrapping, the entire hook-boundary tracing path is
developable and verifiable without touching ``common/`` at all.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import tracing
from tracing.adapters import NoteListAdapter, register_adapter
from tracing.git_store import GitStore
from tracing.triggers import _trace_root, _user_key

# ---------------------------------------------------------------------------
# A tiny in-process fake memo mimicking amem's note_list state shape.
# ---------------------------------------------------------------------------


class _DemoNote:
    def __init__(self, note_id: str, content: str, keywords: Optional[List[str]] = None):
        self.id = note_id
        self.content = content
        self.keywords = keywords or []


class _DemoSystem:
    def __init__(self) -> None:
        self.memories: Dict[str, _DemoNote] = {}


class DemoMemo:
    """Fake MemoClass-shaped object (note_list shape). No network, no deps."""

    def __init__(self) -> None:
        self._system = _DemoSystem()

    async def build_memory_from_data(self, recorder: Any) -> None:
        facts = (recorder.init or {}).get("facts", [])
        for i, text in enumerate(facts):
            note_id = f"n{i}"
            self._system.memories[note_id] = _DemoNote(
                note_id, text, keywords=["demo"])

    async def retrieve_memory_for_query(self, recorder: Any) -> Dict:
        # Strictly read-only: just surface the current note contents.
        return {"retrieved": [n.content for n in self._system.memories.values()]}

    async def use_memory_to_answer(self, recorder: Any, retrieved: Dict,
                                   prompt: str) -> Optional[str]:
        return "demo-answer"


class _DemoRecorder:
    """Minimal recorder stand-in (only ``.init`` is read by the fake memo)."""

    def __init__(self, init: Optional[Dict[str, Any]] = None):
        self.init: Dict[str, Any] = init or {}


# Opt the demo class into the known note_list shape via the public extension
# point (no registry-literal edit, no harness touched).
register_adapter("DemoMemo", NoteListAdapter())


async def _drive(memo: Any) -> None:
    # BUILD — ingests three notes (a state change -> one commit).
    build_rec = _DemoRecorder({"facts": [
        "Alice lives in Paris.",
        "Alice adopted a cat named Mochi.",
        "Alice started learning the cello.",
    ]})
    await memo.build_memory_from_data(build_rec)

    # RETRIEVE x2 — read-only, so commit-on-change produces NO new commit.
    q = _DemoRecorder({"query": "Where does Alice live?"})
    retrieved = await memo.retrieve_memory_for_query(q)
    await memo.use_memory_to_answer(q, retrieved, "Where does Alice live?")
    await memo.retrieve_memory_for_query(_DemoRecorder({"query": "What pet?"}))

    # Rewrite a note in place, then force an opt-in semantic checkpoint — the
    # mark() trigger source captures the change (an in-place rewrite of an
    # existing item -> pattern=link/update, trigger=mark).
    memo._system.memories["n1"] = _DemoNote(
        "n1", "Alice adopted two cats, Mochi and Kiki.", keywords=["demo", "pets"])
    tracing.mark("post-consolidation")


def run_demo(trace_root: Optional[Path] = None, *, run_id: str = "demo-run",
             user_dir: str = "demo_user") -> Dict[str, Any]:
    """Run the scripted demo and return a summary dict:
    ``{"repo_dir": Path, "log": [str, ...]}``.

    ``trace_root`` overrides where the per-user git repo is written (defaults to
    a fresh temp dir). Tracing is enabled for the duration and the environment
    is restored afterwards, so calling this never leaks global state.
    """
    prev_enabled = os.environ.get("MEMEVOL_TRACE")
    prev_root = os.environ.get("MEMEVOL_TRACE_ROOT")
    root = Path(trace_root) if trace_root is not None else Path(
        tempfile.mkdtemp(prefix="memevol-trace-demo-"))
    os.environ["MEMEVOL_TRACE"] = "1"
    os.environ["MEMEVOL_TRACE_ROOT"] = str(root)
    try:
        memo = DemoMemo()
        wrapped = tracing.wrap_memo(
            memo, run_id=run_id, user_dir=user_dir, harness_name="DemoMemo")
        asyncio.run(_drive(wrapped))

        repo_dir = _trace_root() / run_id / "traces" / "git" / _user_key(user_dir)
        log = GitStore(repo_dir).log_lines()
        return {"repo_dir": repo_dir, "log": log}
    finally:
        _restore_env("MEMEVOL_TRACE", prev_enabled)
        _restore_env("MEMEVOL_TRACE_ROOT", prev_root)


def _restore_env(key: str, value: Optional[str]) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def main() -> int:
    result = run_demo()
    repo_dir = result["repo_dir"]
    log = result["log"]
    print(f"[tracing demo] per-user git repo: {repo_dir}")
    print(f"[tracing demo] {len(log)} commit(s) — memory-evolution timeline:\n")
    for line in log:
        print("  " + line)
    print("\nTry it yourself:")
    print(f"  git -C {repo_dir} log --stat")
    print(f"  git -C {repo_dir} log --format='%s%n%(trailers)'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
