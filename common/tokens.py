"""LLM cost accounting — call counts and token usage, attributed to a PHASE.

Every number here answers one question: what did it cost to run this memory
system? The tracker is keyed by ``(model, phase)`` so the cost of *having* the
memory (build) is separable from the cost of *answering* with it (answer) and
from evaluation overhead (judge), which is never part of a cost claim.

Phases
------
``build``     inside ``build_memory_from_data``
``retrieve``  inside ``retrieve_memory_for_query`` (query rewriting, planning,
              reranking — memory-system work, but not build)
``answer``    the benchmark's shared QA agent (the only answerer there is)
``judge``     ``common.metric.Judge`` and the benchmark judges — evaluation
              overhead, excluded from any cost claim
``other``     the fallback when no phase is in scope (harness setup, ad-hoc
              scripts, anything calling common.llm outside the workflow)

The current phase lives in a ``ContextVar``, NOT a module-level global: users
run concurrently under ``max_sample_concurrent``, so a plain mutable "current
phase" would be raced across asyncio tasks and mislabel tokens. A ContextVar
set by ``common/workflow.py`` around each hook call propagates correctly into
tasks spawned inside it (and into ``asyncio.to_thread``, which copies the
context; a raw ``ThreadPoolExecutor.submit`` does NOT — see
``common.llm.Embedding.__call__``, which copies the context explicitly).

Thread safety
-------------
``update`` is sync and guarded by a ``threading.Lock``. It must be callable
from a plain worker thread because the vendored baseline systems drive the
*synchronous* OpenAI SDK; ``aupdate`` is the async-compatible spelling and
simply delegates. Both are cheap (a handful of integer adds).

Coverage caveat
---------------
Only API calls produce a usage object. Local models — LLMlingua-2 (lightmem's
prompt compressor), bge-reranker-v2-m3 (zep's cross-encoder), and the local
embedders (all-MiniLM-L6-v2, BAAI/bge-m3, Qwen3-Embedding-0.6B) — are real
compute that can NEVER enter these counts. Wall-clock (``phase_seconds``) is
the only uniform proxy covering both, which is why it is recorded here
alongside the token numbers.
"""
from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional

from common.logger import get_logger

log = get_logger("main")


# ---------------------------------------------------------------------------
# Phase dimension
# ---------------------------------------------------------------------------

BUILD = "build"
RETRIEVE = "retrieve"
ANSWER = "answer"
JUDGE = "judge"
OTHER = "other"

#: Canonical report order. `other` last — it is the "unattributed" bucket.
PHASES = (BUILD, RETRIEVE, ANSWER, JUDGE, OTHER)

#: Phases that constitute the cost of the memory system itself. `judge` is
#: evaluation overhead and `answer` is the cost of answering, not of having
#: the memory (see the issue's scoping rule).
MEMORY_PHASES = (BUILD, RETRIEVE)

_CURRENT_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "memevol_llm_phase", default=OTHER
)


# ---------------------------------------------------------------------------
# Per-user tally — the cost of ONE user's run
#
# The tracker above is process-global and cumulative, which is right for
# `token_usage.json` but wrong for "what did THIS user cost": users overlap on
# worker threads, and the same user_dir is re-run by every progressive stage.
# `tally_calls()` therefore hands out a FRESH tally per scope and binds it to
# the caller's context, so every LLM call made inside — on this thread, in a
# `to_thread` worker, or in a vendored ThreadPoolExecutor (see
# `install_thread_context_propagation`) — lands in that scope's counters and
# nowhere else. A scope's numbers are read once and dropped; nothing
# accumulates across users, stages or runs.
#
# Embedding calls are deliberately NOT tallied: per-token they cost ~1% of an
# LLM token, so adding the two together would misrepresent a system that
# embeds a lot as an expensive one. They stay in the global tracker
# (`token_usage.json`) where they can be read separately.
# ---------------------------------------------------------------------------


def is_embedding_model(model_name: Any) -> bool:
    """True for OpenAI embedding model names (`text-embedding-*`)."""
    return "text-embedding" in str(model_name or "").lower()


class CallTally:
    """Per-phase call/token counters for ONE bounded scope (one user's run)."""

    def __init__(self) -> None:
        self.by_phase: Dict[str, Dict[str, int]] = defaultdict(_new_entry)
        self._lock = threading.Lock()

    def add(self, phase_name: str, prompt: int, completion: int,
            total: int, reasoning: int, calls: int = 1) -> None:
        with self._lock:
            entry = self.by_phase[phase_name]
            entry["calls"] += int(calls)
            entry["prompt_tokens"] += int(prompt)
            entry["completion_tokens"] += int(completion)
            entry["total_tokens"] += int(total)
            entry["reasoning_tokens"] += int(reasoning)

    def phase(self, phase_name: str) -> Dict[str, int]:
        """This scope's counters for one phase (zeros when it never ran)."""
        with self._lock:
            return dict(self.by_phase.get(phase_name, _new_entry()))

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {ph: dict(c) for ph, c in self.by_phase.items()}


_CURRENT_TALLY: contextvars.ContextVar[Optional["CallTally"]] = contextvars.ContextVar(
    "memevol_call_tally", default=None
)


@contextlib.contextmanager
def tally_calls() -> Iterator["CallTally"]:
    """Tally every non-embedding LLM call made inside this block.

    Yields the tally itself; read it after the block. Nesting is honoured (the
    enclosing tally is restored on exit), and an inner scope does NOT feed its
    calls to an outer one — a scope means "exactly this span".
    """
    tally = CallTally()
    token = _CURRENT_TALLY.set(tally)
    try:
        yield tally
    finally:
        _CURRENT_TALLY.reset(token)


def current_tally() -> Optional["CallTally"]:
    """The tally LLM calls made right now are counted into (None outside a scope)."""
    return _CURRENT_TALLY.get()


def current_phase() -> str:
    """The phase LLM calls made right now are attributed to."""
    return _CURRENT_PHASE.get()


@contextlib.contextmanager
def phase(name: str, tracker: Optional["TokenTracker"] = None) -> Iterator[None]:
    """Attribute every LLM call made inside this block to `name`.

    Also accumulates wall-clock into the tracker (the global one unless
    `tracker` is given) — local model compute never shows up in the token
    counts, so elapsed time is the only cost signal that covers it.

    Nesting is honoured: the previous phase is restored on exit, so a
    `retrieve` block inside an `answer` block behaves sanely. Wall-clock is
    charged to the innermost phase only for the time actually spent there
    (an enclosing block's timer keeps running, so nested phase seconds may
    sum to more than the stage's elapsed time — read them per phase, not as
    a partition).
    """
    if name not in PHASES:
        raise ValueError(f"unknown phase {name!r}; expected one of {PHASES}")
    token = _CURRENT_PHASE.set(name)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        _CURRENT_PHASE.reset(token)
        tr = tracker if tracker is not None else GLOBAL_TOKEN_TRACKER
        if tr is not None:
            tr.add_phase_seconds(name, elapsed)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

_COUNTERS = ("calls", "total_tokens", "prompt_tokens", "completion_tokens",
             "reasoning_tokens")


def _new_entry() -> Dict[str, int]:
    return {k: 0 for k in _COUNTERS}


def _get(v: Any, key: str) -> Any:
    """Read `key` off a usage object that may be a dict or an SDK model."""
    if isinstance(v, dict):
        return v.get(key, 0) or 0
    return getattr(v, key, 0) or 0


class TokenTracker:
    """Per-(model, phase) call and token counts, plus per-phase wall-clock."""

    def __init__(self):
        # {model: {phase: {calls, total_tokens, prompt_tokens,
        #                  completion_tokens, reasoning_tokens}}}
        self.usage: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(_new_entry)
        )
        self.phase_seconds: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    # ---- recording ----

    def update(self, model_name: str, usage: Any, phase_name: Optional[str] = None,
               calls: int = 1) -> None:
        """Record ONE completed API call. Sync and thread-safe.

        `usage` is an OpenAI-style usage object or dict (the Anthropic path in
        common.llm normalizes to the same keys before calling here). A falsy
        `usage` still counts as a call — a request was made and paid for even
        when the server returned no usage block.

        `phase_name` overrides the ambient ContextVar phase; pass it only when
        recording a call made in a context you cannot enter — a judge forcing
        `judge` regardless of caller, or a replay of a subprocess's usage
        (alma's memo_manager), where this process's ContextVar says nothing
        about what the subprocess was doing.

        `calls` folds N already-counted requests into one update; only replays
        should pass anything but 1.
        """
        ph = phase_name or current_phase()
        # Two usage shapes reach this function. Chat Completions says
        # prompt/completion; the Responses API says input/output for the same
        # two numbers, and nests reasoning under output_tokens_details. Both
        # carry total_tokens, which is why a Responses-only baseline used to
        # report the right total with a 0/0 split — and why its per-query cost,
        # which sums prompt+completion, came out ZERO. zep did exactly that:
        # 3,059,840 build tokens, cost_build_per_query 0.0, first place on the
        # efficiency axis.
        prompt = _get(usage, "prompt_tokens") or _get(usage, "input_tokens")
        completion = _get(usage, "completion_tokens") or _get(usage, "output_tokens")
        total = _get(usage, "total_tokens")
        # Some SDK paths omit total_tokens; derive it so the headline number is
        # never silently short.
        if not total:
            total = prompt + completion
        # Live OpenAI usage nests reasoning under completion_tokens_details
        # (Responses: output_tokens_details); a replayed summary entry carries
        # it flat.
        details = (_get(usage, "completion_tokens_details")
                   or _get(usage, "output_tokens_details"))
        reasoning = _get(details, "reasoning_tokens") if details else _get(usage, "reasoning_tokens")

        with self._lock:
            entry = self.usage[model_name][ph]
            entry["calls"] += int(calls)
            entry["prompt_tokens"] += prompt
            entry["completion_tokens"] += completion
            entry["total_tokens"] += total
            entry["reasoning_tokens"] += reasoning

        # Per-user cost accounting (see CallTally): the ambient scope, if any.
        # Embeddings are excluded by design; a replay (`calls` > 1, or an
        # explicit `phase_name`) still counts — it is real usage by whoever is
        # in scope.
        tally = _CURRENT_TALLY.get()
        if tally is not None and not is_embedding_model(model_name):
            tally.add(ph, prompt, completion, total, reasoning, calls)

    async def aupdate(self, model_name: str, usage: Any,
                      phase_name: Optional[str] = None, calls: int = 1) -> None:
        """Async spelling of `update` (the counters are cheap; no awaiting)."""
        self.update(model_name, usage, phase_name, calls)

    def add_phase_seconds(self, phase_name: str, seconds: float) -> None:
        with self._lock:
            self.phase_seconds[phase_name] += seconds

    # ---- reporting ----

    def summary(self) -> Dict[str, Any]:
        """The `token_usage.json` payload: nested {model: {phase: counters}}
        plus derived rollups.

        Schema (changed 2026-08, see the LLM-accounting issue — the old shape
        was a flat {model: counters} with no phase dimension and no call
        counts; `_read_legacy_summary` still parses artifacts written then):

            {
              "by_model_phase": {model: {phase: {calls, total_tokens, ...}}},
              "by_phase":       {phase: {calls, total_tokens, ...}},
              "by_model":       {model: {calls, total_tokens, ...}},
              "totals":         {calls, total_tokens, ...},
              "memory_tokens_by_phase": ...  # build + retrieve only
              "phase_seconds":  {phase: float},
            }
        """
        with self._lock:
            by_model_phase = {
                model: {ph: dict(counters) for ph, counters in phases.items()}
                for model, phases in self.usage.items()
            }
            phase_seconds = dict(self.phase_seconds)

        by_phase: Dict[str, Dict[str, int]] = {}
        by_model: Dict[str, Dict[str, int]] = {}
        totals = _new_entry()
        for model, phases in by_model_phase.items():
            model_roll = _new_entry()
            for ph, counters in phases.items():
                phase_roll = by_phase.setdefault(ph, _new_entry())
                for k in _COUNTERS:
                    phase_roll[k] += counters[k]
                    model_roll[k] += counters[k]
                    totals[k] += counters[k]
            by_model[model] = model_roll

        memory = _new_entry()
        for ph in MEMORY_PHASES:
            for k in _COUNTERS:
                memory[k] += by_phase.get(ph, {}).get(k, 0)

        return {
            "by_model_phase": by_model_phase,
            "by_phase": by_phase,
            "by_model": by_model,
            "totals": totals,
            "memory_phase_totals": memory,
            "phase_seconds": phase_seconds,
        }

    def print_summary(self) -> None:
        s = self.summary()
        log.info("[blue]━━━━━━━━━━━━━━━ Token Usage Summary ━━━━━━━━━━━━━━━[/blue]")
        for ph in PHASES:
            c = s["by_phase"].get(ph)
            if not c:
                continue
            secs = s["phase_seconds"].get(ph, 0.0)
            log.info(
                f"[purple]{ph:<9} | calls {c['calls']:>5} | in {c['prompt_tokens']:>9} "
                f"| out {c['completion_tokens']:>8} (reasoning {c['reasoning_tokens']}) "
                f"| {secs:.1f}s[/purple]"
            )
        t = s["totals"]
        log.info(
            f"[purple]{'TOTAL':<9} | calls {t['calls']:>5} | "
            f"{t['total_tokens']} tokens[/purple]"
        )


# ---------------------------------------------------------------------------
# Summary arithmetic — per-stage deltas
#
# `TokenTracker` is process-global and cumulative, but each stage's
# `token_usage.json` must describe THAT stage only. Callers snapshot the
# summary before a stage and subtract it afterwards (common/evaluate.py).
# ---------------------------------------------------------------------------

def total_tokens(summary: Dict[str, Any]) -> int:
    """Total tokens across every model and phase in a summary."""
    return int(summary.get("totals", {}).get("total_tokens", 0))


def phase_tokens(summary: Dict[str, Any], *phase_names: str) -> int:
    """Total tokens for the named phases (e.g. the memory-building cost)."""
    by_phase = summary.get("by_phase", {})
    return sum(int(by_phase.get(p, {}).get("total_tokens", 0)) for p in phase_names)


def diff_summary(after: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
    """`after` minus `before`, structurally — the usage of one stage.

    Both arguments are `TokenTracker.summary()` outputs. Rollups are
    recomputed from the subtracted per-(model, phase) leaves rather than
    subtracted themselves, so they stay internally consistent.
    """
    a = after.get("by_model_phase", {})
    b = before.get("by_model_phase", {})

    delta_leaves: Dict[str, Dict[str, Dict[str, int]]] = {}
    for model, phases in a.items():
        for ph, counters in phases.items():
            prev = b.get(model, {}).get(ph, {})
            d = {k: int(counters.get(k, 0)) - int(prev.get(k, 0)) for k in _COUNTERS}
            # Drop entries that saw no activity during the stage.
            if any(v for v in d.values()):
                delta_leaves.setdefault(model, {})[ph] = d

    by_phase: Dict[str, Dict[str, int]] = {}
    by_model: Dict[str, Dict[str, int]] = {}
    totals = _new_entry()
    for model, phases in delta_leaves.items():
        model_roll = _new_entry()
        for ph, counters in phases.items():
            phase_roll = by_phase.setdefault(ph, _new_entry())
            for k in _COUNTERS:
                phase_roll[k] += counters[k]
                model_roll[k] += counters[k]
                totals[k] += counters[k]
        by_model[model] = model_roll

    memory = _new_entry()
    for ph in MEMORY_PHASES:
        for k in _COUNTERS:
            memory[k] += by_phase.get(ph, {}).get(k, 0)

    a_secs = after.get("phase_seconds", {})
    b_secs = before.get("phase_seconds", {})
    phase_seconds = {
        ph: round(float(a_secs.get(ph, 0.0)) - float(b_secs.get(ph, 0.0)), 3)
        for ph in a_secs
    }

    return {
        "by_model_phase": delta_leaves,
        "by_phase": by_phase,
        "by_model": by_model,
        "totals": totals,
        "memory_phase_totals": memory,
        "phase_seconds": {ph: s for ph, s in phase_seconds.items() if s},
    }


def read_total_tokens(data: Dict[str, Any]) -> int:
    """Total tokens from a `token_usage.json` payload of EITHER schema.

    The current schema carries a `totals` block; artifacts written before the
    phase dimension landed are a flat {model: {total_tokens, ...}} mapping.
    Consumers that read historical run directories (forge's orchestrator)
    must go through here so old runs don't silently read as 0.
    """
    if not isinstance(data, dict):
        return 0
    if "totals" in data or "by_model_phase" in data:
        return total_tokens(data)
    return sum(
        int(v.get("total_tokens", 0))
        for v in data.values()
        if isinstance(v, dict)
    )


# ---------------------------------------------------------------------------
# Phase propagation into worker threads
# ---------------------------------------------------------------------------

_thread_propagation_installed = False


def install_thread_context_propagation() -> bool:
    """Make `ThreadPoolExecutor.submit` carry the caller's context, so work
    handed to a worker thread keeps its phase. Idempotent.

    Several vendored systems fan their LLM calls out across their own thread
    pools — simplemem compresses windows at `max_parallel_workers` and
    searches at `max_retrieval_workers`, both `concurrent.futures`. `submit`
    does NOT copy contextvars (unlike `asyncio.to_thread`), so every call made
    inside those pools would be attributed to `other` instead of to the phase
    that started them, and the numbers would understate build for exactly the
    baselines that parallelize hardest.

    Editing those pools is not an option — they live under `src/`, which is
    byte-identical to upstream. So propagate at the stdlib boundary, the same
    pattern as `common/openai_usage.py`. This only widens contextvar
    visibility inside workers; it changes no scheduling or execution
    semantics, and copies (never shares) the context, so a worker's own
    contextvar writes stay local to it.
    """
    global _thread_propagation_installed
    if _thread_propagation_installed:
        return False
    import concurrent.futures

    real_submit = concurrent.futures.ThreadPoolExecutor.submit
    if getattr(real_submit, "__wrapped_by_memevol__", False):
        _thread_propagation_installed = True
        return False

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return real_submit(self, lambda: ctx.run(fn, *args, **kwargs))

    submit.__wrapped_by_memevol__ = True    # type: ignore[attr-defined]
    concurrent.futures.ThreadPoolExecutor.submit = submit
    _thread_propagation_installed = True
    return True


# ---------------------------------------------------------------------------
# Offline text token counting (memory-token measurement)
#
# Not usage reporting — this counts tokens in text we assembled ourselves,
# before any API call. Used by common/workflow.py to measure how much the
# retrieved memory added to the QA prompt.
# ---------------------------------------------------------------------------

_ENCODER_CACHE: Dict[str, Any] = {}
_ENCODER_LOCK = threading.Lock()


def _encoder_for(model: str):
    """tiktoken encoder for `model`, cached per process.

    Unknown / non-OpenAI model names (claude-*, local models) fall back to
    cl100k_base: the resulting count is then an approximation, but memory
    tokens are used as a RELATIVE cost signal, and the same encoding is
    applied to both sides of the differential.
    """
    with _ENCODER_LOCK:
        enc = _ENCODER_CACHE.get(model)
        if enc is None:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            _ENCODER_CACHE[model] = enc
        return enc


def count_text_tokens(text: str, model: str = "gpt-4o") -> int:
    """Token count of `text` under `model`'s encoding."""
    if not text:
        return 0
    return len(_encoder_for(model).encode(text))


GLOBAL_TOKEN_TRACKER: Optional[TokenTracker] = None


def init_global_tracker() -> TokenTracker:
    """Only called after all modules are imported."""
    global GLOBAL_TOKEN_TRACKER
    if GLOBAL_TOKEN_TRACKER is None:
        GLOBAL_TOKEN_TRACKER = TokenTracker()
    return GLOBAL_TOKEN_TRACKER
