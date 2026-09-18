"""Shared model-configuration layer for the harness baselines (issue #26).

Every model a harness baseline touches — internal LLM, embedder, reranker,
compressor — must be a *configurable parameter*, so the fleet can be run in two
arms:

  faithful arm — each method's own published models (the defaults; what the
                 READMEs make faithfulness claims about);
  unified arm  — one GPT LLM + one API embedder everywhere, which is the arm
                 that is comparable, like-for-like, against the main method.

**WRAP, DO NOT REWRITE.** Each baseline's method source under ``src/`` is
vendored byte-identical and every README asserts it. So both levers here act at
a *boundary* the vendored code already goes through, from integration code
(``memo.py``), with zero edits under ``src/``:

  1. :func:`get_embedder` — one cached factory that returns either a real
     sentence-transformer or an API-backed stand-in with the same interface.
     Baselines whose vendored code constructs its own embedder reach it through
     :func:`install_embedder_factory`, which patches the
     ``sentence_transformers`` constructor they all funnel through.
  2. :func:`install_openai_param_normalisation` patches the OpenAI SDK's
     ``chat.completions.create`` so requests that hardcode sampling params the
     gpt-5 family rejects (``temperature``, ``max_tokens``) still work. Without
     it, five of the seven baselines cannot run a gpt-5 model AT ALL — their
     vendored clients send those params unconditionally.

Both patches are process-global and idempotent, and must be installed BEFORE the
vendored package is imported, because the vendored modules bind their names at
their own import time.

Issue #25 (unified LLM accounting) intercepts at the same
``chat.completions.create`` seam; that is deliberate — one interception point
gets both the param normalisation and the token accounting.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Device resolution
# --------------------------------------------------------------------------
#
# lightmem (embedding_device, llmlingua_device) and zep (device) used to
# default to a hardcoded "cuda" and therefore crashed outright on a CPU-only
# box. Their configs now default to null and route through here.


def resolve_device(value: Optional[str] = None) -> str:
    """Resolve a config device value to a concrete torch device string.

    ``None`` / ``""`` / ``"auto"`` → ``"cuda"`` when a CUDA device is actually
    visible, else ``"cpu"``. Any other value ("cpu", "cuda:1", "mps", ...) is
    passed through untouched, so an operator can still pin a device.

    A missing/broken torch resolves to ``"cpu"`` rather than raising: the
    caller only wants somewhere to put a model.
    """
    if value not in (None, "", "auto"):
        return str(value)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------
# API vs local embedding models
# --------------------------------------------------------------------------

# OpenAI's embedding models are exactly the `text-embedding-*` family. This is
# the same rule hipporag2 already used inline (`"text-embedding" not in
# embedding` ⇒ local), lifted here so all seven baselines classify identically.
_API_EMBEDDING_PREFIX = "text-embedding"

# Native output widths, used to answer `get_sentence_embedding_dimension()`
# without paying for a probe request. Unknown models fall back to probing.
_API_EMBEDDING_DIMS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def is_api_embedding_model(model_name: Optional[str]) -> bool:
    """True when `model_name` names an OpenAI API embedding model.

    Used to decide, per baseline, whether the embedder is a local
    sentence-transformer or an :class:`APIEmbedder`. Note the DIMENSION
    COUPLING this implies: `text-embedding-3-small` is 1536-dim where the local
    defaults are 384 (MiniLM) / 1024 (bge-m3, Qwen3), so any switch must carry
    the baseline's dimension knob with it, and invalidates vector indexes built
    at the old width.
    """
    return str(model_name or "").startswith(_API_EMBEDDING_PREFIX)


def api_embedding_dims(model_name: str) -> int:
    """Native output width of an OpenAI API embedding model.

    Used where a baseline must be TOLD the width explicitly (lightmem sizes its
    Qdrant collection from it; zep's Graphiti truncates vectors to it). Raises
    for a model outside the table rather than guessing — a wrong width either
    fails hard or silently truncates.
    """
    if model_name not in _API_EMBEDDING_DIMS:
        raise ValueError(
            f"unknown API embedding model {model_name!r}; known: {sorted(_API_EMBEDDING_DIMS)}"
        )
    return _API_EMBEDDING_DIMS[model_name]


class APIEmbedder:
    """An OpenAI API embedder wearing the `sentence_transformers` interface.

    This is the Tier-3 adapter: amem, memoryos and simplemem construct a
    SentenceTransformer *themselves* inside vendored code and then call
    ``.encode(...)`` on it, so the only way to give them an API embedder
    without editing ``src/`` is to hand them an object of the same shape.

    Requests go through :class:`common.llm.Embedding`, which brings the repo's
    retry kernel, the global concurrency gate, request chunking (item + token
    budgets) and TokenTracker reporting — so unlike the local embedders, an API
    embedder's cost IS visible in the token accounting.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        from common.llm import Embedding

        self.model_name = model_name
        self._embedding = Embedding(model=model_name)
        self._dim: Optional[int] = _API_EMBEDDING_DIMS.get(model_name)

    def get_sentence_embedding_dimension(self) -> int:
        """Native output width. Probes with a one-token request only for models
        outside `_API_EMBEDDING_DIMS` (result memoized)."""
        if self._dim is None:
            self._dim = len(self._embedding([" "])[0])
        return self._dim

    def get_config_dict(self) -> Dict[str, Any]:
        # amem reads `get_config_dict()['model_name']` when it serialises a
        # retriever's state.
        return {"model_name": self.model_name}

    def encode(
        self,
        sentences,
        *,
        normalize_embeddings: bool = False,
        **_ignored: Any,
    ):
        """Embed one string or a list of strings, returning a numpy array.

        Mirrors sentence-transformers' shape convention: a single string in
        gives a 1-D vector, a list gives a 2-D (N, D) array. Keyword arguments
        the API has no equivalent for (``show_progress_bar``, ``batch_size``,
        ``prompt_name``, ``convert_to_numpy``, ``device``, ...) are accepted and
        ignored — the caller is vendored code we cannot change.

        `normalize_embeddings` is honoured but is normally a no-op: OpenAI's
        `text-embedding-3-*` already return unit-length vectors.
        """
        import numpy as np

        single = isinstance(sentences, str)
        texts: List[str] = [sentences] if single else [str(s) for s in sentences]
        if not texts:
            return np.zeros((0, 0), dtype="float32")

        out = np.asarray(self._embedding(texts), dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms != 0)
        return out[0] if single else out


# --------------------------------------------------------------------------
# The embedder factory
# --------------------------------------------------------------------------
#
# A fresh MemoClass is built PER USER, so without memoization the weights (up to
# ~0.6B for simplemem's Qwen3) reload for every user in the split.
#
# Three baselines cannot be handed an embedder directly — their vendored code
# constructs one itself, with no injection point:
#
#   amem      SimpleEmbeddingRetriever.__init__ does `SentenceTransformer(model_name)`
#   simplemem EmbeddingModel.__init__ dispatches on `model_name.startswith("qwen3")`
#             and constructs the SentenceTransformer itself
#   lightmem  TextEmbedderHuggingface.__init__ does
#             `SentenceTransformer(config.model, **config.model_kwargs)`
#
# For all three the CONFIGURED name is also the name the vendored code requests,
# so patching that shared constructor (`install_embedder_factory`) is enough:
# the factory dispatches on the requested name.
#
# memoryos is the exception — its `get_embedding()` carries the model name as a
# DEFAULT ARGUMENT, so the requested name is never the configured one. Rather
# than give this module a process-wide override mechanism for one baseline,
# memoryos calls `get_embedder()` directly and seeds its own vendored model
# cache (see memoryos/memo.py). It needs no constructor patch at all.
#
# zep needs none of this: Graphiti accepts an injected EmbedderClient.

_model_cache: Dict[Any, Any] = {}
_cache_lock = threading.Lock()
_factory_installed = False


def get_embedder(model_name: str, device: Optional[str] = None, *args: Any, **kwargs: Any):
    """Cached, dispatching embedder factory — the one entry point.

    API model name → a shared :class:`APIEmbedder`. Anything else → a real
    SentenceTransformer, constructed once per (name, device) with the caller's
    remaining constructor arguments forwarded intact (simplemem's Qwen3 path
    passes ``trust_remote_code`` / ``model_kwargs`` / ``tokenizer_kwargs``;
    dropping those would silently change how the model loads).

    The cache key deliberately ignores those extra arguments: two callers asking
    for the same model on the same device get the same object, which is the
    point. A failed construction is NOT cached, so a transient failure can be
    retried.
    """
    key = (model_name, device)
    with _cache_lock:
        cached = _model_cache.get(key)
    if cached is not None:
        return cached

    # Users' hooks run on worker threads, so the first calls can arrive
    # together; loading under the shared lock loads each model once instead of
    # once per thread (several copies of a large local embedder can exhaust the
    # GPU) and never alongside another model load (see concurrency.py).
    from baselines.harness.concurrency import model_load_lock
    with model_load_lock:
        with _cache_lock:
            cached = _model_cache.get(key)
        if cached is not None:
            return cached

        if is_api_embedding_model(model_name):
            # Local-loading arguments have no meaning for an API embedder.
            model = APIEmbedder(model_name)
        else:
            import sentence_transformers as _st

            # Resolve the GENUINE class: `install_embedder_factory` may have already
            # replaced the attribute with the factory that called us, and building
            # through that would recurse forever.
            real = getattr(_st.SentenceTransformer, "_real_sentence_transformer",
                           _st.SentenceTransformer)
            model = real(model_name, *args, device=device, **kwargs)
            # Shared by every user of the process, and those users' hooks run on
            # worker threads: one encode at a time (see concurrency.serialize_calls).
            from baselines.harness.concurrency import serialize_calls
            serialize_calls(model, "encode")

        with _cache_lock:
            _model_cache[key] = model
        return model


def install_embedder_factory() -> None:
    """Route ``sentence_transformers.SentenceTransformer(...)`` through
    :func:`get_embedder`.

    Idempotent. **Must run before the vendored package is imported** — the
    vendored modules do `from sentence_transformers import SentenceTransformer`
    at their own import time, which binds the name once and for all.
    """
    global _factory_installed
    if _factory_installed:
        return
    import sentence_transformers as _st

    real = _st.SentenceTransformer

    def _factory(*args, **kwargs):
        kwargs = dict(kwargs)
        if args:
            requested, rest = args[0], args[1:]
        else:
            requested, rest = kwargs.pop("model_name_or_path", None), ()
        if requested is None:          # nothing to key on — build it raw
            return real(*args, **kwargs)
        device = kwargs.pop("device", None)
        return get_embedder(str(requested), device, *rest, **kwargs)

    # Keep the original reachable — get_embedder reads it back to avoid
    # recursing through this factory.
    _factory._real_sentence_transformer = real  # type: ignore[attr-defined]
    _st.SentenceTransformer = _factory
    _factory_installed = True


# --------------------------------------------------------------------------
# OpenAI sampling-param normalisation
# --------------------------------------------------------------------------
#
# Five baselines hardcode sampling params their vendored clients always send:
#   amem       temperature=0.7 AND max_tokens=1000  (OpenAIController)
#   lightmem   temperature + max_tokens from its manager config
#   simplemem  temperature=0.1..0.3                 (LLMClient)
#   memoryos   temperature + max_tokens             (utils.chat_completion)
#   zep        max_tokens (graphiti already drops temperature for gpt-5)
#
# The gpt-5 family rejects all of them (temperature must be the default; the cap
# is `max_completion_tokens`), so "point every baseline at one GPT model" fails
# on 5/7 no matter how many config keys exist. Normalising at the SDK boundary
# fixes that without touching src/.

# Mirrors common/llm.py::_REASONING_MODEL_PREFIXES — keep the two in step.
_RESTRICTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Params the reasoning families reject outright — dropped, `max_tokens`
# included (see normalise_chat_params for why renaming it was worse).
_DROPPED_PARAMS = ("temperature", "top_p", "presence_penalty", "frequency_penalty",
                   "max_tokens")

#: Smallest completion budget we will let a vendored caller impose on a
#: reasoning model. The budget covers REASONING as well as the answer, and the
#: vendored numbers were all chosen for 4-series models: the caps found under
#: `baselines/harness/*/src/` run from 10 to 2048, and mem0's own GPT-5 mapping
#: sends 2000. Measured on gpt-5-mini: at 2000 an extraction call spends the
#: whole budget thinking and returns EMPTY content (or, on a 30 k-char prompt,
#: never returns at all); with room to finish it answers in seconds. Matches
#: `common.llm.Agent`'s own default, so vendored and first-party calls get the
#: same headroom.
REASONING_MIN_COMPLETION_TOKENS = 16384

#: Per-request read timeout for a vendored call that sets none.
#:
#: The vendored clients build their own `OpenAI(api_key=..., base_url=...)` and
#: pass no timeout, so they inherit the SDK's defaults: a 600 s read timeout
#: and 2 retries. One request that does not come back therefore costs 1800 s
#: and takes the user down with it — measured to the second, twice: 1802 s of
#: build phase with a single embedding call recorded and no chat completion at
#: all.
#:
#: The bound does not make a bad provider window succeed — one measured window
#: stalled this same call at 150 s, 180 s and 900 s alike — it makes the run
#: fail in minutes instead of half an hour, and lets a one-off stall be retried
#: while the budget still allows it.
#:
#: 180 s is ~8x the slowest real call measured on mem0's extraction prompt
#: (33.6 k-char system prompt + a conversation session): gpt-5-mini at low
#: effort 15-23 s, at minimal 5-6 s, gpt-4.1-mini 2-3 s, and a whole 19-session
#: build averaged 26 s per session. It is deliberately not tighter: `common.llm`
#: documents that server-side queueing legitimately pushes latency up under
#: concurrency, and a client timeout tight enough to fire on queueing causes a
#: retry storm.
VENDORED_REQUEST_TIMEOUT_SECONDS = 180.0

#: How many chat completions a vendored baseline may have in flight at once.
#:
#: `common.llm` gates its own calls globally (MEMEVOL_LLM_MAX_CONCURRENT); the
#: vendored clients have no gate at all, and several fan out hard —
#: simplemem's packaged config runs 16 parallel workers, lightmem and mem0 use
#: thread pools of their own. Measured on simplemem's real build, 16 in
#: flight: the first batch took 211 s and 213 s for calls that take seconds on
#: their own, and everything behind them queued. With a 180 s request budget
#: that batch times out and retries, which adds load — the retry storm
#: `common.llm` warns about, from the one direction that had no brake.
#:
#: The gate is also what makes the 180 s budget honest: the clock starts when
#: the request is actually issued, so time spent waiting for a slot is not
#: charged against it.
VENDORED_MAX_CONCURRENT = 8

_sync_gate = threading.BoundedSemaphore(VENDORED_MAX_CONCURRENT)
_async_gates: "Dict[int, Any]" = {}
_async_gates_lock = threading.Lock()


def _async_gate():
    """One asyncio.Semaphore per running loop (they are not shareable)."""
    import asyncio
    loop = asyncio.get_running_loop()
    with _async_gates_lock:
        gate = _async_gates.get(id(loop))
        if gate is None:
            gate = asyncio.Semaphore(VENDORED_MAX_CONCURRENT)
            _async_gates[id(loop)] = gate
        return gate


_params_patched = False


def _is_restricted_model(model: Optional[str]) -> bool:
    return any(str(model or "").startswith(p) for p in _RESTRICTED_MODEL_PREFIXES)


def normalise_chat_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite one `chat.completions.create` kwarg dict for the target model.

    One rewrite applies to every model: a call that set no ``timeout`` gets
    ``VENDORED_REQUEST_TIMEOUT_SECONDS`` instead of the SDK's 600 s default,
    so a stalled request is retried in minutes rather than costing 1800 s and
    the user with it. See that constant for the measurement.

    Two more are no-ops on the 4-series models the faithful arm uses:

      * a repo-convention ``"model/effort"`` suffix (e.g. ``gpt-5-mini/low``)
        is split into ``model`` + ``reasoning_effort``. Vendored clients pass
        the configured string straight through and would 400 on the suffix;
      * for a reasoning model, ``temperature``/``top_p``/the penalties AND
        ``max_tokens`` are dropped, and a completion cap below
        ``REASONING_MIN_COMPLETION_TOKENS`` is raised to it.

    `max_tokens` used to be RENAMED to `max_completion_tokens`, on the
    reasoning that the caller means to cap the generation. That was wrong, and
    expensively so: on a reasoning model the cap covers reasoning AND output
    together, so a vendored value chosen for a 4-series model (mem0 sends
    2000) leaves nothing for the answer. Measured against gpt-5-mini with a
    30 k-char extraction prompt and `response_format=json_object`: capped, the
    request never returned (300 s timeout, and 30 min of retries in a real
    run); with the cap dropped it answered in 4.2 s. A cap written for a
    non-reasoning model has no equivalent meaning here, so the honest
    translation is to drop it — a caller who really wants one can pass
    `max_completion_tokens` itself, which is left untouched.

    Returns a new dict; the caller's is left alone.
    """
    out = dict(kwargs)
    model = out.get("model")

    # Bound the wait before anything model-specific: the vendored clients pass
    # no timeout at all, and the SDK's default is long enough that one stuck
    # request ends the run.
    if "timeout" not in out or out["timeout"] is None:
        out["timeout"] = VENDORED_REQUEST_TIMEOUT_SECONDS

    if isinstance(model, str) and "/" in model:
        base, effort = model.split("/", 1)
        out["model"] = model = base
        if _is_restricted_model(base):
            out.setdefault("reasoning_effort", effort)

    if not _is_restricted_model(model):
        return out

    for name in _DROPPED_PARAMS:
        out.pop(name, None)

    # A cap the vendored code set ITSELF (mem0 maps its max_tokens to this for
    # the GPT-5 family) is kept, but never below the floor: a 4-series budget
    # leaves a reasoning model nothing to answer with.
    cap = out.get("max_completion_tokens")
    if isinstance(cap, int) and cap < REASONING_MIN_COMPLETION_TOKENS:
        out["max_completion_tokens"] = REASONING_MIN_COMPLETION_TOKENS
    return out


def install_openai_param_normalisation() -> None:
    """Patch the OpenAI SDK so every `chat.completions.create` is normalised
    and gated.

    Patched on the resource CLASSES rather than on client instances, because
    the vendored code constructs its own ``OpenAI(...)`` clients at arbitrary
    times — this catches all of them, including ones built before this call.
    Idempotent; a no-op if the SDK layout is unrecognised.
    """
    global _params_patched
    if _params_patched:
        return
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return

    def _wrap(cls, is_async: bool):
        real = cls.create

        if is_async:
            # A genuine `async def`, not a sync function returning the
            # coroutine: the SDK's own method is a coroutine function, and
            # callers (and `inspect.iscoroutinefunction`) may rely on that.
            async def create(self, *args, **kwargs):
                params = normalise_chat_params(kwargs)
                async with _async_gate():
                    return await real(self, *args, **params)
        else:
            def create(self, *args, **kwargs):
                params = normalise_chat_params(kwargs)
                with _sync_gate:
                    return real(self, *args, **params)

        create._real_create = real  # type: ignore[attr-defined]
        cls.create = create

    for cls, is_async in ((Completions, False), (AsyncCompletions, True)):
        if not hasattr(cls.create, "_real_create"):
            _wrap(cls, is_async)
    _params_patched = True
