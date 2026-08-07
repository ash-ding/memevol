"""Import-environment shims for the vendored LightMem text pipeline.

Two concerns, all integration-only (LightMem's vendored code stays
byte-identical):

  1. SRC PATH — ``ensure_lightmem_importable()`` puts ``src/`` on
     ``sys.path`` so LightMem's absolute imports (``from lightmem.memory...``)
     resolve to our vendored copy under baselines/harness/lightmem/src/.

  2. SHARED EMBEDDER — LightMem constructs a fresh ``SentenceTransformer``
     inside EVERY ``LightMemory`` (i.e. per user/sample) via its
     ``TextEmbedderHuggingface``. ``install_embedding_cache()`` memoizes the
     underlying ``SentenceTransformer(model_path, **kwargs)`` by model path so
     the weights load ONCE per process and every user shares them. Encoding is
     read-only on the model, so sharing is safe across concurrent samples.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _HARNESS_DIR / "src"
_PROJECT_ROOT = _HARNESS_DIR.parents[2]   # baselines/harness/lightmem -> repo root



def ensure_lightmem_importable() -> None:
    """Idempotently put the vendored ``lightmem`` package on sys.path so its
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


def import_lightmemory():
    """Import the vendored LightMem pipeline with HF ``datasets`` active, and
    return the ``LightMemory`` class. Imported here purely
    defensively (LightMem's import chain does not touch ``datasets`` until an
    embedder is constructed, but the swap is cheap and keeps the rule uniform).
    memevol's ``datasets`` view is restored on exit, so memo.py can still
    ``from benchmarks.locomo.env import ...`` afterward."""
    ensure_lightmem_importable()
    holder = {}
    from lightmem.memory.lightmem import LightMemory
    holder["LightMemory"] = LightMemory
    return holder["LightMemory"]


# --- Eager attention for the LLMlingua-2 model ------------------------------

_eager_attention_installed = False


def install_eager_attention() -> None:
    """Force ``attn_implementation="eager"`` for the LLMlingua-2 model that
    LightMem's topic segmenter reuses.

    The segmenter reads per-layer ``outputs.attentions`` off that model
    (``attentions[8..11]``). transformers >= 5 defaults to the ``sdpa`` attention
    backend, which returns an EMPTY attentions tuple for ``output_attentions=True``
    (verified in-env: sdpa/default → len 0, eager → len 12), so the segmenter
    raises ``IndexError: tuple index out of range``. llmlingua loads the model via
    ``AutoModelForTokenClassification.from_pretrained`` without specifying an
    implementation; wrap that classmethod to default to eager (callers passing an
    explicit ``attn_implementation`` still win). Idempotent.

    Integration-only: the vendored LightMem + llmlingua code is untouched, and
    eager attention is numerically identical to sdpa (just unfused). Surgical —
    only token-classification models are affected (the LLMlingua-2 model); the
    sentence-transformer embedder loads via a different class and is unaffected."""
    global _eager_attention_installed
    if _eager_attention_installed:
        return
    import transformers
    cls = transformers.AutoModelForTokenClassification
    _real = cls.from_pretrained   # bound classmethod (cls already baked in)

    def _patched(*args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return _real(*args, **kwargs)

    cls.from_pretrained = staticmethod(_patched)
    _eager_attention_installed = True


# --- Shared embedder cache --------------------------------------------------

_st_model_cache: dict = {}
_embedding_cache_installed = False


def install_embedding_cache() -> None:
    """Monkeypatch ``sentence_transformers.SentenceTransformer`` with a
    memoizing factory so the (heavy) embedding weights load once per process and
    are shared across every per-user ``LightMemory``. Idempotent. Leaves
    LightMem's vendored ``TextEmbedderHuggingface`` untouched — it just receives
    a cached model object from the patched constructor.

    Keyed on the resolved model path (first positional arg or ``model_name_or_path``
    kwarg). Different embedding models get distinct cache slots. If construction
    fails it is NOT cached (so a transient failure can be retried)."""
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
