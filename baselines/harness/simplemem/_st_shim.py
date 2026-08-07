"""Import-environment shims for the vendored SimpleMem text pipeline.

Three concerns, all integration-only (SimpleMem's vendored code stays
byte-identical):

  1. SRC PATH — ``ensure_simplemem_importable()`` puts ``src/`` on
     ``sys.path`` so SimpleMem's absolute imports (``from simplemem.core...``)
     resolve to our vendored copy under baselines/harness/simplemem/src/.

  2. HuggingFace ``datasets`` vs memevol's ``benchmarks/`` — SimpleMem's import
     chain pulls in ``lancedb`` (``core/database`` init does ``from datasets
     import Dataset`` at import) and, at embedder construction, sentence-
     transformers (``model_card`` does ``from datasets import __version__``).
     memevol ships a benchmark package ALSO named ``benchmarks/`` that shadows the
     HF library on ``sys.path``. We make ``datasets`` resolve to HF for those
     spots.

     CRITICAL: HF ``datasets`` registers PROCESS-GLOBAL pyarrow extension types
     at import (``pa.register_extension_type(Array2DExtensionType...)``). Importing
     it twice raises ``ArrowKeyError: ... already defined``. So HF ``datasets`` is
     imported EXACTLY ONCE (``_capture_hf_datasets``); every "activation"
     thereafter merely swaps the CACHED module objects in/out of ``sys.modules``
     — it never re-imports.

  3. SHARED EMBEDDER — SimpleMem constructs a fresh ``EmbeddingModel`` (a
     ~0.6B Qwen3 sentence-transformer) inside EVERY ``SimpleMemSystem`` (i.e.
     per user/sample). ``install_embedding_cache()`` memoizes the underlying
     ``SentenceTransformer(model_path, **kwargs)`` by model path so the weights
     load ONCE per process and every user shares them. Encoding is read-only on
     the model, so sharing is safe across the workflow's concurrent samples.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _HARNESS_DIR / "src"
_PROJECT_ROOT = _HARNESS_DIR.parents[2]   # baselines/harness/simplemem -> repo root

# HuggingFace `datasets*` module objects, captured ONCE by _capture_hf_datasets().
# None until that first (and only) import. Reused for every hf_datasets_active().
_hf_datasets_mods = None


def ensure_simplemem_importable() -> None:
    """Idempotently put the vendored ``simplemem`` package on sys.path so its
    absolute internal imports resolve to our copy."""
    p = str(_SRC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _memevol_datasets_in_sys_modules():
    return [m for m in list(sys.modules) if m == "datasets" or m.startswith("datasets.")]


def _capture_hf_datasets() -> None:
    """Import the HuggingFace ``datasets`` library EXACTLY ONCE and cache its
    module objects in ``_hf_datasets_mods``. Idempotent — a no-op once captured.

    The import runs with the project root off ``sys.path`` and memevol's
    ``datasets*`` modules popped, so ``import datasets`` resolves to the installed
    HF library. Because datasets registers process-global pyarrow extension types
    at import, this MUST be the only place datasets is ever imported; callers
    activate it via cached-object swaps (``hf_datasets_active``), never re-import.
    memevol's ``datasets`` view is restored before returning (the HF module
    objects survive removal from ``sys.modules`` via the cached reference)."""
    global _hf_datasets_mods
    if _hf_datasets_mods is not None:
        return
    saved_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _PROJECT_ROOT]
    saved_mods = {name: sys.modules.pop(name) for name in _memevol_datasets_in_sys_modules()}
    try:
        import datasets  # noqa: F401  — the HuggingFace library; the ONE import
        _hf_datasets_mods = {name: sys.modules[name] for name in _memevol_datasets_in_sys_modules()}
    finally:
        sys.path[:] = saved_path
        for name in _memevol_datasets_in_sys_modules():
            del sys.modules[name]
        sys.modules.update(saved_mods)


@contextlib.contextmanager
def hf_datasets_active():
    """Temporarily make ``import datasets`` resolve to the (already-imported,
    cached) HuggingFace library instead of memevol's benchmark package. Used for
    any code that does ``from datasets import ...`` — lancedb at simplemem-import
    time, sentence-transformers at ``SentenceTransformer`` construction. Swaps
    CACHED module objects only (never re-imports → no pyarrow re-registration).
    memevol's ``datasets`` view is restored on exit.

    Do NOT run memevol-``datasets`` imports (e.g. ``benchmarks.locomo.env``) inside
    this block — they would resolve to HF. Callers MUST NOT ``await`` inside: the
    sys.modules swap is process-global and would corrupt a concurrently-scheduled
    coroutine's ``datasets`` view (safe today — every memo hook body is
    await-free around these swaps)."""
    _capture_hf_datasets()
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


def ensure_sentence_transformers() -> None:
    """Idempotently import sentence-transformers with HF ``datasets`` active (its
    ``model_card`` may do ``from datasets import ...`` at import in some
    versions)."""
    if "sentence_transformers" in sys.modules:
        return
    with hf_datasets_active():
        import sentence_transformers  # noqa: F401


def import_simplemem_system():
    """Import the vendored SimpleMem text pipeline with HF ``datasets`` active,
    and return ``(SimpleMemSystem, Dialogue, MemoryEntry)``.

    The import chain pulls in ``lancedb`` (``simplemem.core.database`` →
    ``vector_store_backend`` → ``import lancedb``), whose init does ``from
    datasets import Dataset`` at import. Under ``hf_datasets_active`` the cached HF
    ``datasets`` is already in ``sys.modules``, so that resolves to HF with no
    re-import. memevol's ``datasets`` view is restored on exit, so memo.py can
    still ``from benchmarks.locomo.env import ...`` afterward. Requires the HF
    ``datasets`` package to be installed."""
    ensure_simplemem_importable()
    holder = {}
    with hf_datasets_active():
        from simplemem.text.system import SimpleMemSystem
        from simplemem.core.models.memory_entry import Dialogue, MemoryEntry
        holder.update(SimpleMemSystem=SimpleMemSystem, Dialogue=Dialogue, MemoryEntry=MemoryEntry)
    return holder["SimpleMemSystem"], holder["Dialogue"], holder["MemoryEntry"]


# --- Shared embedder cache --------------------------------------------------

_st_model_cache: dict = {}
_embedding_cache_installed = False


def install_embedding_cache() -> None:
    """Monkeypatch ``sentence_transformers.SentenceTransformer`` with a
    memoizing factory so the (heavy) embedding weights load once per process and
    are shared across every per-user ``SimpleMemSystem``. Idempotent. Leaves
    SimpleMem's vendored ``EmbeddingModel`` code untouched — it just receives a
    cached model object from the patched constructor.

    Keyed on the resolved model path (first positional arg or ``model_name_or_path``
    kwarg). Different embedding models (e.g. a MiniLM fallback) get distinct
    cache slots. If construction fails it is NOT cached (so a transient failure
    can be retried)."""
    global _embedding_cache_installed
    if _embedding_cache_installed:
        return
    ensure_sentence_transformers()
    import sentence_transformers as _st

    _real_ctor = _st.SentenceTransformer

    def _cached(*args, **kwargs):
        key = args[0] if args else kwargs.get("model_name_or_path")
        cached = _st_model_cache.get(key)
        if cached is not None:
            return cached
        with hf_datasets_active():   # construction triggers the lazy `datasets` import
            model = _real_ctor(*args, **kwargs)
        if key is not None:
            _st_model_cache[key] = model
        return model

    # Preserve the original on the wrapper for anyone who needs the true class.
    _cached._real_sentence_transformer = _real_ctor  # type: ignore[attr-defined]
    _st.SentenceTransformer = _cached
    _embedding_cache_installed = True
