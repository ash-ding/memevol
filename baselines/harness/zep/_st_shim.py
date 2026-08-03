"""sentence-transformers / vendored-graphiti bootstrap for the zep baseline.

Three integration concerns, all handled here (NO edits to vendored graphiti_core):

  1. VENDOR PATH — `import graphiti_core` must resolve to the byte-identical
     copy under `vendor/`, not any pip-installed graphiti-core.
     `ensure_vendor_on_path()` prepends `vendor/` to sys.path[0].

  2. `datasets` SHADOW — memevol's benchmark package `datasets/` collides with
     the HuggingFace `datasets` library that sentence-transformers imports (BGE-m3
     embedder + BGE reranker both build on sentence-transformers). Transplanted
     VERBATIM from baselines/harness/amem/_st_shim.py:
       * IMPORT time — `import sentence_transformers` eagerly imports HF
         `datasets`; `ensure_sentence_transformers()` imports ST with the project
         root off sys.path so `datasets` resolves to the HF library.
       * CALL time — `SentenceTransformer.__init__` / `CrossEncoder.__init__` ->
         `model_card.get_versions()` does `from datasets import __version__` at
         CONSTRUCTION; `hf_datasets_active()` swaps HF `datasets` back in for the
         duration of that construction. Model construction is SYNC (no await
         inside the manager — the process-global swap stays coroutine-safe);
         `.encode()` / `.predict()` at query time need no shim.

  3. SHARED MODEL CACHE — a fresh ZepMemo is created per user; without a cache
     every user reloads BGE-m3 (~2GB) + BGE-reranker weights. `get_bge_embedder`
     / `get_bge_reranker` construct each model ONCE per process (under
     `hf_datasets_active()`) and hand back the shared instance.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor"
_PROJECT_ROOT = _HERE.parents[2]   # baselines/harness/zep -> repo root

# HuggingFace `datasets*` module objects, captured the first time we import with
# the project root off sys.path. None until the first capture.
_hf_datasets_mods = None


def ensure_vendor_on_path() -> None:
    """Prepend vendor/ so `import graphiti_core` hits the vendored, byte-identical
    copy rather than any pip-installed graphiti-core. Idempotent."""
    p = str(_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _memevol_datasets_in_sys_modules():
    return [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]


def _run_with_hf_datasets(action):
    """Run `action()` with the project root removed from sys.path and any memevol
    `datasets*` modules popped, so `datasets` resolves to the installed
    HuggingFace library. Restore memevol's view afterward (even on error)."""
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
    """Idempotently import sentence-transformers (IMPORT-time `datasets`
    collision). MUST run before importing graphiti_core.cross_encoder.
    bge_reranker_client (which imports sentence_transformers at module load)."""
    global _hf_datasets_mods
    if "sentence_transformers" in sys.modules:
        return

    def _import_st():
        import sentence_transformers  # noqa: F401

    _hf_datasets_mods = _run_with_hf_datasets(_import_st)


def _ensure_hf_datasets_cached() -> None:
    global _hf_datasets_mods
    if _hf_datasets_mods:
        return

    def _import_ds():
        import datasets  # noqa: F401  — the HuggingFace library

    _hf_datasets_mods = _run_with_hf_datasets(_import_ds)


@contextlib.contextmanager
def hf_datasets_active():
    """Make `import datasets` resolve to the HuggingFace library for the duration
    of a SentenceTransformer / CrossEncoder construction (which lazily does
    `from datasets import __version__`). memevol's `datasets` view is restored on
    exit. Do NOT `await` inside this block (process-global sys.modules swap)."""
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


# --- Shared, process-wide model cache (construct once, reuse across users) ---
_embedder_cache: dict = {}
_reranker_cache: dict = {}


def get_bge_embedder(model_name: str, device: str | None = None):
    """Cached SentenceTransformer for the BGE-m3 embedder. Constructed once per
    (model_name, device) under hf_datasets_active()."""
    ensure_sentence_transformers()
    key = (model_name, device)
    if key not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        with hf_datasets_active():
            _embedder_cache[key] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[key]


def get_bge_reranker(model_name: str, device: str | None = None):
    """Cached CrossEncoder for the BGE reranker. Constructed once per
    (model_name, device) under hf_datasets_active()."""
    ensure_sentence_transformers()
    key = (model_name, device)
    if key not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        with hf_datasets_active():
            _reranker_cache[key] = CrossEncoder(model_name, device=device)
    return _reranker_cache[key]
