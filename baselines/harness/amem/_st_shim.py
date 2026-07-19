"""Import sentence-transformers without tripping over memevol's `datasets/`.

ST 5.x collides with memevol's benchmark package `datasets/` (same top-level
name) at TWO points; both are handled here:

  * IMPORT time — `import sentence_transformers` eagerly imports HuggingFace
    `datasets` (model_card.py does `from datasets import Dataset` at module
    load). `ensure_sentence_transformers()` imports ST with the project root
    off sys.path and memevol's `datasets*` modules popped, so `datasets`
    resolves to the installed HF library, then restores memevol's view.

  * CALL time — `SentenceTransformer.__init__` -> `model_card.get_versions()`
    does `from datasets import __version__` when the embedder is CONSTRUCTED
    (long after import, when memevol's `datasets` is the ambient view again).
    `hf_datasets_active()` swaps HF `datasets` back in for the duration of that
    construction and restores memevol's view on exit. A-mem constructs its
    embedder lazily, so its construction site
    (baselines/harness/amem/memo.py::_ensure_system) runs inside this manager.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# HuggingFace `datasets*` module objects, captured the first time we import with
# the project root off sys.path. Cached so hf_datasets_active() can re-activate
# them without re-importing. None until the first capture.
_hf_datasets_mods = None


def _memevol_datasets_in_sys_modules():
    return [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]


def _run_with_hf_datasets(action):
    """Run `action()` with the project root removed from sys.path and any
    memevol `datasets*` modules popped, so `datasets` resolves to the installed
    HuggingFace library. Restore memevol's view afterward (even on error).
    Return a dict of the HF `datasets*` modules present after `action` ran
    (the module objects survive removal from sys.modules via this reference)."""
    saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _PROJECT_ROOT]
    saved_mods = {name: sys.modules.pop(name) for name in _memevol_datasets_in_sys_modules()}
    try:
        action()
        return {name: sys.modules[name] for name in _memevol_datasets_in_sys_modules()}
    finally:
        sys.path[:] = saved_path
        for name in _memevol_datasets_in_sys_modules():
            del sys.modules[name]
        sys.modules.update(saved_mods)


def ensure_sentence_transformers() -> None:
    """Idempotently import sentence-transformers (IMPORT-time collision). See
    the module docstring."""
    global _hf_datasets_mods
    if "sentence_transformers" in sys.modules:
        return

    def _import_st():
        import sentence_transformers  # noqa: F401

    _hf_datasets_mods = _run_with_hf_datasets(_import_st)


def _ensure_hf_datasets_cached() -> None:
    """Populate _hf_datasets_mods if ensure_sentence_transformers() never ran
    (defensive — in practice ST import already captured HF datasets)."""
    global _hf_datasets_mods
    if _hf_datasets_mods:
        return

    def _import_ds():
        import datasets  # noqa: F401  — the HuggingFace library

    _hf_datasets_mods = _run_with_hf_datasets(_import_ds)


@contextlib.contextmanager
def hf_datasets_active():
    """Temporarily make `import datasets` resolve to the HuggingFace library
    instead of memevol's benchmark package, for sentence-transformers code that
    imports `datasets` LAZILY at call time (SentenceTransformer construction ->
    model_card.get_versions() -> `from datasets import __version__`). memevol's
    `datasets` view is restored on exit. Do NOT run memevol-`datasets` imports
    (e.g. `datasets.locomo.env`) inside this block — they would resolve to HF.
    Callers MUST NOT `await` inside this block: the sys.modules swap is
    process-global and would corrupt a concurrently-scheduled coroutine's
    `datasets` view (safe today because every memo hook body is await-free)."""
    _ensure_hf_datasets_cached()
    saved = {name: sys.modules[name] for name in _memevol_datasets_in_sys_modules()}
    for name in list(saved):
        del sys.modules[name]
    sys.modules.update(_hf_datasets_mods)
    try:
        yield
    finally:
        for name in _memevol_datasets_in_sys_modules():
            del sys.modules[name]
        sys.modules.update(saved)
