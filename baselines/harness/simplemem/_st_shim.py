"""Import-environment shims for the vendored SimpleMem text pipeline.

Two concerns, all integration-only (SimpleMem's vendored code stays
byte-identical):

  1. SRC PATH — ``ensure_simplemem_importable()`` puts ``src/`` on
     ``sys.path`` so SimpleMem's absolute imports (``from simplemem.core...``)
     resolve to our vendored copy under baselines/harness/simplemem/src/.

  2. SHARED EMBEDDER — SimpleMem constructs a fresh ``EmbeddingModel`` (a
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



def ensure_simplemem_importable() -> None:
    """Idempotently put the vendored ``simplemem`` package on sys.path so its
    absolute internal imports resolve to our copy."""
    p = str(_SRC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_sentence_transformers() -> None:
    """Idempotently import sentence-transformers. (Historically this had to
    dodge memevol's top-level `datasets/` package, which shadowed the HF
    `datasets` library that ST imports; the package was renamed to
    `benchmarks/` in 2026-08, so a plain import is correct now.)"""
    if "sentence_transformers" not in sys.modules:
        import sentence_transformers  # noqa: F401


def import_simplemem_system():
    """Import the vendored SimpleMem text pipeline with HF ``datasets`` active,
    and return ``(SimpleMemSystem, Dialogue, MemoryEntry)``.

    The import chain pulls in ``lancedb`` (``simplemem.core.database`` →
    ``vector_store_backend`` → ``import lancedb``), whose init does ``from
    datasets import Dataset`` at import — resolved against the installed HF
    ``datasets`` is already in ``sys.modules``, so that resolves to HF with no
    re-import. memevol's ``datasets`` view is restored on exit, so memo.py can
    still ``from benchmarks.locomo.env import ...`` afterward. Requires the HF
    ``datasets`` package to be installed."""
    ensure_simplemem_importable()
    holder = {}
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
        model = _real_ctor(*args, **kwargs)
        if key is not None:
            _st_model_cache[key] = model
        return model

    # Preserve the original on the wrapper for anyone who needs the true class.
    _cached._real_sentence_transformer = _real_ctor  # type: ignore[attr-defined]
    _st.SentenceTransformer = _cached
    _embedding_cache_installed = True
