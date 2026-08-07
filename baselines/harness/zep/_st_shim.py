"""sentence-transformers / vendored-graphiti bootstrap for the zep baseline.

Two integration concerns, all handled here (NO edits to vendored graphiti_core):

  1. SRC PATH — `import graphiti_core` must resolve to the byte-identical
     copy under `src/`, not any pip-installed graphiti-core.
     `ensure_src_on_path()` prepends `src/` to sys.path[0].

  2. SHARED MODEL CACHE — a fresh ZepMemo is created per user; without a cache
     every user reloads BGE-m3 (~2GB) + BGE-reranker weights. `get_bge_embedder`
     / `get_bge_reranker` construct each model ONCE per process (under
     and hand back the shared instance.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
_PROJECT_ROOT = _HERE.parents[2]   # baselines/harness/zep -> repo root


def ensure_src_on_path() -> None:
    """Prepend src/ so `import graphiti_core` hits the vendored, byte-identical
    copy rather than any pip-installed graphiti-core. Idempotent."""
    p = str(_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_sentence_transformers() -> None:
    """Idempotently import sentence-transformers. (Historically this had to
    dodge memevol's top-level `datasets/` package, which shadowed the HF
    `datasets` library that ST imports; the package was renamed to
    `benchmarks/` in 2026-08, so a plain import is correct now.)"""
    if "sentence_transformers" not in sys.modules:
        import sentence_transformers  # noqa: F401


def get_bge_embedder(model_name: str, device: str | None = None):
    """Cached SentenceTransformer for the BGE-m3 embedder. Constructed once per
    (model_name, device)."""
    ensure_sentence_transformers()
    key = (model_name, device)
    if key not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        _embedder_cache[key] = SentenceTransformer(model_name, device=device)
    return _embedder_cache[key]


def get_bge_reranker(model_name: str, device: str | None = None):
    """Cached CrossEncoder for the BGE reranker. Constructed once per
    (model_name, device)."""
    ensure_sentence_transformers()
    key = (model_name, device)
    if key not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        _reranker_cache[key] = CrossEncoder(model_name, device=device)
    return _reranker_cache[key]
