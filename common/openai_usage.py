"""Capture the OpenAI traffic of vendored baseline code, without editing it.

Every harness baseline drives its own LLM calls through the `openai` Python
SDK from inside `baselines/harness/*/src/` — zep (7 files), lightmem (5),
memoryos, simplemem, amem (a lazy import inside `OpenAIController.__init__`),
and hipporag2/mem0 through venv-installed packages. None of that traffic
reached `common.tokens`, which is most of what a memory system actually costs:
amem alone spends 2 gpt-4o-mini calls per ingested note, and build is the
dominant phase.

Why a shim rather than rewriting the call sites
-----------------------------------------------
The method source under `baselines/harness/*/src/` is vendored BYTE-IDENTICAL
to upstream, and each README ships a `diff -r` command asserting it. Editing
`src/graphiti_core/`, `src/simplemem/`, `src/memoryos/`, `src/lightmem/` or
`src/memory_layer.py` to swap in our client would break that guarantee and
every faithfulness claim built on it. So we patch at the SDK boundary — the
same pattern the repo already uses in the `_st_shim.py` files — and install it
from `memo.py`, which is integration code, not vendored source.

Why the RESOURCE METHODS, not the client constructors
-----------------------------------------------------
Patching `openai.OpenAI` / `openai.AsyncOpenAI` only works if it happens
before the vendored module runs `from openai import OpenAI`, since that binds
the name at import time — and most of these modules do exactly that at module
level. Patching `Completions.create` (and friends) on the resource CLASSES is
import-order independent, catches clients that were already constructed, and
covers the Azure client classes too (they share these resource classes).

Double counting is avoided by tagging the clients `common.llm` builds: usage
from those is reported by `common.llm` itself, so the shim skips them.

Not captured
------------
Streamed responses carry no usage block unless the caller asks for one
(`stream_options={"include_usage": True}`), so a streaming call is counted as
a CALL with zero tokens and warned about once. Local models (LLMlingua-2, the
bge cross-encoder, the local embedders) are not API calls at all and can never
appear here — see `common/tokens.py`.
"""
from __future__ import annotations

import functools
from typing import Any, Optional

from common.logger import get_logger

log = get_logger("main")

_installed = False
_stream_warned = False

#: Attribute stamped on clients built by `common.llm`, whose usage is already
#: reported there. Read by the wrappers to skip double counting.
OWNED_CLIENT_ATTR = "_memevol_reports_usage"


def _is_ours(resource: Any) -> bool:
    """True when this resource belongs to a client common.llm already tracks."""
    return bool(getattr(getattr(resource, "_client", None), OWNED_CLIENT_ATTR, False))


def _record(resource: Any, response: Any, requested_model: Optional[str],
            streaming: bool) -> None:
    """Report one completed SDK call to the global tracker. Never raises —
    accounting must not be able to fail a baseline's evaluation."""
    try:
        from common.tokens import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is None:
            return
        if streaming:
            global _stream_warned
            if not _stream_warned:
                _stream_warned = True
                log.warning(
                    "[openai-usage] a streamed call was made without "
                    "stream_options={'include_usage': True}; it is counted as a "
                    "call with 0 tokens (the API returns no usage block)"
                )
            usage = None
        else:
            usage = getattr(response, "usage", None)
        # The server's echoed model is the ground truth (it resolves aliases
        # and deployment names); fall back to what the caller asked for.
        model = getattr(response, "model", None) or requested_model or "unknown"
        GLOBAL_TOKEN_TRACKER.update(model_name=str(model), usage=usage)
    except Exception as exc:      # pragma: no cover - defensive
        log.debug(f"[openai-usage] failed to record usage: {exc!r}")


def _wrap_sync(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        response = method(self, *args, **kwargs)
        if not _is_ours(self):
            _record(self, response, kwargs.get("model"), bool(kwargs.get("stream")))
        return response
    wrapper.__wrapped_by_memevol__ = True    # type: ignore[attr-defined]
    return wrapper


def _wrap_async(method):
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        response = await method(self, *args, **kwargs)
        if not _is_ours(self):
            _record(self, response, kwargs.get("model"), bool(kwargs.get("stream")))
        return response
    wrapper.__wrapped_by_memevol__ = True    # type: ignore[attr-defined]
    return wrapper


def _patch(cls, is_async: bool) -> bool:
    method = getattr(cls, "create", None)
    if method is None or getattr(method, "__wrapped_by_memevol__", False):
        return False
    cls.create = _wrap_async(method) if is_async else _wrap_sync(method)
    return True


def install() -> bool:
    """Patch the OpenAI SDK's create() methods to report usage. Idempotent.

    Call from a baseline's `memo.py` BEFORE the vendored system is
    constructed. Returns True if anything was patched (False when already
    installed, or when the `openai` SDK is not importable — a baseline running
    on a non-OpenAI stack is not an error).
    """
    global _installed
    if _installed:
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        log.debug("[openai-usage] openai SDK not installed; nothing to patch")
        return False

    patched = []
    # Chat completions + embeddings are what the vendored systems use; the
    # Responses API is patched too so a future upstream bump is covered.
    # Each entry is optional: SDK versions move these around, and a missing
    # one must degrade to "not captured", never to an import error.
    targets = (
        ("openai.resources.chat.completions", "Completions", False),
        ("openai.resources.chat.completions", "AsyncCompletions", True),
        ("openai.resources.embeddings", "Embeddings", False),
        ("openai.resources.embeddings", "AsyncEmbeddings", True),
        ("openai.resources.responses", "Responses", False),
        ("openai.resources.responses", "AsyncResponses", True),
    )
    for module_path, cls_name, is_async in targets:
        try:
            import importlib
            module = importlib.import_module(module_path)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError):
            continue
        if _patch(cls, is_async):
            patched.append(f"{cls_name}")

    _installed = True
    if patched:
        log.info(f"[openai-usage] tracking SDK calls via {', '.join(patched)}")
    else:
        log.warning(
            "[openai-usage] found no OpenAI resource classes to patch — this "
            "baseline's internal LLM calls will NOT appear in token_usage.json"
        )
    return bool(patched)
