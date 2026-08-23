"""Name the local models a run used — the compute that can never be a token.

Two models in the fleet are local, have no API equivalent, and produce no
usage object, so they can never appear in token accounting:

  LLMlingua-2 (lightmem)          a prompt COMPRESSOR — a BERT token
                                  classifier scoring each token for retention
                                  before write. ~180M, one forward pass per
                                  ingested text.
  bge-reranker-v2-m3 (zep)        a CROSS-ENCODER reranker — it scores
                                  (query, doc) PAIRS, so k forward passes per
                                  query over the top-k candidates. ~570M, the
                                  heaviest per-query cost in the fleet.

The local EMBEDDERS (all-MiniLM-L6-v2, BAAI/bge-m3, Qwen3-Embedding-0.6B) are
in the same category.

This is a permanent hole in cost comparability, not a bug to fix: token counts
systematically understate the baselines that do local compute, and the gap is
largest exactly where the compute is heaviest (zep runs bge-m3 AND the
cross-encoder). The mitigation is disclosure — a cost figure must never be
read as complete when it is not — so every run record names what ran, and
`common.tokens` records per-phase wall-clock, the only uniform proxy that
covers both API and local work.

Registration happens at the LIBRARY boundary (`install()`), for the same
reason `common/openai_usage.py` patches the SDK boundary: the models are
constructed deep inside vendored `src/` trees that are byte-identical to
upstream and must not be edited.

It patches the constructor METHOD on the real class rather than rebinding the
class name, so it composes with the per-baseline `_st_shim` memoizing
factories in either install order — but finding that class takes care, since
those factories rebind the package attribute to a plain function (see
`_sentence_transformer_class`).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from common.logger import get_logger

log = get_logger("main")

_lock = threading.Lock()
_models: Dict[str, Dict[str, Any]] = {}
_installed = False


def register(name: str, role: str, device: Optional[str] = None,
             library: str = "") -> None:
    """Record that a local model was loaded. Idempotent per (name, role)."""
    if not name:
        return
    key = f"{name}::{role}"
    with _lock:
        entry = _models.get(key)
        if entry is None:
            _models[key] = {"name": str(name), "role": role,
                            "device": str(device) if device else "unknown",
                            "library": library, "instances": 1}
            log.info(f"[local-model] {role}: {name} on {device or 'unknown'}")
        else:
            entry["instances"] += 1
            if entry["device"] == "unknown" and device:
                entry["device"] = str(device)


def summary() -> List[Dict[str, Any]]:
    """Every local model this process loaded, sorted for stable artifacts."""
    with _lock:
        return sorted((dict(v) for v in _models.values()),
                      key=lambda m: (m["role"], m["name"]))


def reset() -> None:
    """Drop the registry (tests)."""
    with _lock:
        _models.clear()


def _device_of(model: Any) -> Optional[str]:
    try:
        dev = getattr(model, "device", None)
        if dev is None:
            params = getattr(model, "parameters", None)
            if callable(params):
                dev = next(params()).device
        return str(dev) if dev is not None else None
    except Exception:
        return None


def _sentence_transformer_class() -> Any:
    """The REAL `SentenceTransformer` class, however the package attribute has
    been rebound.

    `sentence_transformers.SentenceTransformer` is NOT reliably the class by
    the time this runs: lightmem's and simplemem's `_st_shim` replace that
    attribute with a memoizing FACTORY FUNCTION at memo.py import time, which
    is before `evaluate_memo` installs this. Patching `__init__` on a function
    object silently does nothing — which is exactly what happened on the first
    real simplemem/LoCoMo run, whose run record reported no local models while
    Qwen3-Embedding-0.6B was loaded and running.

    Resolution order: unwrap the shim's stashed original, then the package
    attribute if it really is a class, then the defining module (new path
    first — `sentence_transformers.SentenceTransformer` is itself deprecated).
    """
    import sentence_transformers as _st

    candidate = getattr(_st, "SentenceTransformer", None)
    # The shims stash the original on the wrapper for exactly this purpose.
    unwrapped = getattr(candidate, "_real_sentence_transformer", None)
    if isinstance(unwrapped, type):
        return unwrapped
    if isinstance(candidate, type):
        return candidate
    for module_path in ("sentence_transformers.sentence_transformer.model",
                        "sentence_transformers.SentenceTransformer"):
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, "SentenceTransformer", None)
            if isinstance(cls, type):
                return cls
        except ImportError:
            continue
    raise RuntimeError(
        "could not resolve the real SentenceTransformer class "
        f"(package attribute is {type(candidate).__name__})"
    )


def install() -> bool:
    """Patch the local-model constructors to self-register. Idempotent.

    Returns True if anything was patched. A baseline with no local models
    installs this too — it is a no-op there, and calling it unconditionally
    from every memo.py is what keeps the run record honest by default.
    """
    global _installed
    if _installed:
        return False
    _installed = True
    patched = False

    # --- sentence-transformers: every local embedder in the fleet ---
    try:
        cls = _sentence_transformer_class()
        real_init = cls.__init__
        if not getattr(real_init, "__wrapped_by_memevol__", False):
            def _init(self, *args, **kwargs):
                real_init(self, *args, **kwargs)
                name = (args[0] if args else kwargs.get("model_name_or_path")) or ""
                register(str(name), role="embedder", device=_device_of(self),
                         library="sentence-transformers")

            _init.__wrapped_by_memevol__ = True   # type: ignore[attr-defined]
            cls.__init__ = _init
            patched = True
    except Exception as exc:
        # LOUD, not debug: a silent failure here produces an empty
        # `local_models` in the run record, which reads as "nothing local ran"
        # — the exact false-completeness this module exists to prevent.
        log.warning(
            f"[local-model] could not track sentence-transformers ({exc!r}); "
            f"local embedders will be MISSING from the run record"
        )

    # --- transformers: LLMlingua-2 (token classifier) and cross-encoders ---
    try:
        import transformers

        for cls_name, role in (("AutoModelForTokenClassification", "compressor"),
                               ("AutoModelForSequenceClassification", "reranker")):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            real = cls.from_pretrained
            if getattr(real, "__wrapped_by_memevol__", False):
                continue

            def _from_pretrained(*args, _real=real, _role=role, **kwargs):
                model = _real(*args, **kwargs)
                name = (args[0] if args else kwargs.get("pretrained_model_name_or_path")) or ""
                register(str(name), role=_role, device=_device_of(model),
                         library="transformers")
                return model

            _from_pretrained.__wrapped_by_memevol__ = True  # type: ignore[attr-defined]
            cls.from_pretrained = staticmethod(_from_pretrained)
            patched = True
    except Exception as exc:
        log.debug(f"[local-model] transformers not tracked: {exc!r}")

    return patched
