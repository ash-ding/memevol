"""Import-environment shims for the vendored LightMem text pipeline.

Three concerns, all integration-only (LightMem's vendored code stays
byte-identical):

  1. SRC PATH — ``ensure_lightmem_importable()`` puts ``src/`` on
     ``sys.path`` so LightMem's absolute imports (``from lightmem.memory...``)
     resolve to our vendored copy under baselines/harness/lightmem/src/.

  2. HuggingFace ``datasets`` vs memevol's ``benchmarks/`` — constructing
     LightMem's HuggingFace text embedder builds a ``SentenceTransformer``,
     whose ``model_card`` does ``from datasets import __version__`` at
     construction time. memevol ships a benchmark package ALSO named
     ``benchmarks/`` that shadows the HF library on ``sys.path``. We make
     ``datasets`` resolve to HF for that construction.

     CRITICAL: HF ``datasets`` registers PROCESS-GLOBAL pyarrow extension types
     at import (``pa.register_extension_type(...)``). Importing it twice raises
     ``ArrowKeyError: ... already defined``. So HF ``datasets`` is imported
     EXACTLY ONCE (``_capture_hf_datasets``); every "activation" thereafter
     merely swaps the CACHED module objects in/out of ``sys.modules`` — it never
     re-imports. (Same class of workaround as the amem / simplemem baselines.)

  3. SHARED EMBEDDER — LightMem constructs a fresh ``SentenceTransformer``
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

# HuggingFace `datasets*` module objects, captured ONCE by _capture_hf_datasets().
# None until that first (and only) import. Reused for every hf_datasets_active().
_hf_datasets_mods = None


def ensure_lightmem_importable() -> None:
    """Idempotently put the vendored ``lightmem`` package on sys.path so its
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
    any code that does ``from datasets import ...`` — sentence-transformers at
    ``SentenceTransformer`` construction. Swaps CACHED module objects only (never
    re-imports → no pyarrow re-registration). memevol's ``datasets`` view is
    restored on exit.

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


def import_lightmemory():
    """Import the vendored LightMem pipeline with HF ``datasets`` active, and
    return the ``LightMemory`` class. Imported under ``hf_datasets_active`` purely
    defensively (LightMem's import chain does not touch ``datasets`` until an
    embedder is constructed, but the swap is cheap and keeps the rule uniform).
    memevol's ``datasets`` view is restored on exit, so memo.py can still
    ``from benchmarks.locomo.env import ...`` afterward."""
    ensure_lightmem_importable()
    holder = {}
    with hf_datasets_active():
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
    with hf_datasets_active():
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
        with hf_datasets_active():   # construction triggers the lazy `datasets` import
            model = _real_ctor(*args, **kwargs)
        if key is not None:
            _st_model_cache[key] = model
        return model

    # Preserve the original on the wrapper for anyone who needs the true class.
    _cached._real_sentence_transformer = _real_ctor  # type: ignore[attr-defined]
    _st.SentenceTransformer = _cached
    _embedding_cache_installed = True
