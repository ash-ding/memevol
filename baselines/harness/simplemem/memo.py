"""SimpleMem (arXiv 2601.02553, https://github.com/aiming-lab/SimpleMem) as a
retrieval MemoClass.

BUILD: every ingestion unit becomes one SimpleMem ``Dialogue``; the vendored
pipeline runs untouched — ``MemoryBuilder`` windows the dialogues
(``WINDOW_SIZE=40``, overlap 2) and an LLM compresses each window into
self-contained ``MemoryEntry`` units (coref-resolved "lossless_restatement" +
keywords + timestamp/persons/entities/topic), embedded into a per-user LanceDB
multi-view index. ``add_dialogues`` + ``finalize`` per BUILD call → additive
across DynamicMem checkpoints (each call windows only the newly-visible delta;
``finalize`` flushes its remainder).

RETRIEVE: the paper's contribution — ``HybridRetriever.retrieve(query)`` runs
intent-aware planning + semantic/keyword/structured multi-view search (+ optional
reflection). Read-only (DynamicMem query non-pollution holds). Its retrieved
``MemoryEntry`` units are returned as ``{"passages": [...]}`` and the SHARED QA
agent answers, as it does for every memo (answering is not part of the contract),
which keeps the comparison about memory, not about SimpleMem's own answerer.

Ingestion units (recorder.init dispatch, cf. hipporag2's _init_to_passages):
  locomo ("conversation"): one Dialogue per turn (speaker, text), time = session
    date_time.  longmemeval ("sessions"): one Dialogue per message (role→speaker,
    content), time = session date.  dynamicmem ("app_logs"): one Dialogue per log
    entry, content = the shared app_log_to_passage text (identical content across
    baselines), speaker = app_name, time = log timestamp.
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.memo_class import MemoClass

from common.openai_usage import install as _install_openai_usage
from baselines.harness.concurrency import model_load_lock, quiet_stdout
from baselines.harness.passages import app_log_to_passage
from baselines.harness.model_config import (
    install_embedder_factory, install_openai_param_normalisation,
)

# SimpleMem's absolute imports (`from simplemem.core...`) must resolve to the
# byte-identical vendored copy under src/, not any pip-installed simplemem.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The shared embedder factory (baselines/harness/model_config.py) memoizes the
# heavy weights across per-user systems AND dispatches an API embedding model to
# an APIEmbedder. SimpleMem has no injection point — ``EmbeddingModel.__init__``
# dispatches on ``model_name.startswith("qwen3")`` and constructs the
# SentenceTransformer itself — so patching that constructor is the only lever
# that leaves ``src/`` byte-identical. Nothing here needs to know WHICH embedder
# was configured: the ``embedding_model`` key reaches SimpleMem through its
# ``EMBEDDING_MODEL`` env setting (see _ENV_FROM_CFG below), so the name the
# vendored code requests IS the configured one, and the factory dispatches on it.
#
# The param-normalisation patch is needed because SimpleMem's ``LLMClient``
# sends temperature=0.1..0.3 on every call, which the gpt-5 family rejects.
#
# Both must precede the vendored import (they bind their names at import time).
install_embedder_factory()
install_openai_param_normalisation()

from simplemem.text.system import SimpleMemSystem  # noqa: E402  (vendored, byte-identical)
from simplemem.core.models.memory_entry import Dialogue, MemoryEntry  # noqa: E402
from simplemem.core.settings import settings as _simplemem_settings  # noqa: E402

# SimpleMem's llm_client.py builds its own `openai` client; the SDK-boundary
# patch captures its calls without editing the vendored file.
_install_openai_usage()

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# SimpleMem reads part of its configuration ONLY from its global `settings`
# singleton (simplemem/core/settings.py), whose attribute lookup falls through
# to a top-level `config.py`, then the environment, then built-in defaults.
# `_pin_settings` assigns every setting the vendored code reads directly on that
# object — an instance attribute is found before any of those fallbacks — so
# the values come from this memo's config and nothing ambient can change them.
# OPENAI_API_KEY is the one setting left to the environment (a credential).
#
# No embedding-dimension key is needed: SimpleMem sizes its LanceDB table from
# `embedding_model.dimension` (vector_store.py:39), which the APIEmbedder
# answers for itself — so switching to a 1536-dim API embedder needs no second
# knob. It does invalidate any index built at the old width; the per-user
# LanceDB store is created with `clear_db=True`, so that resolves itself.
_SETTINGS_FROM_CONFIG = {
    "LLM_MODEL": "simplemem_llm_model",
    "OPENAI_BASE_URL": "base_url",
    "EMBEDDING_MODEL": "embedding_model",
    "WINDOW_SIZE": "window_size",
    "OVERLAP_SIZE": "overlap_size",
    "SEMANTIC_TOP_K": "semantic_top_k",
    "KEYWORD_TOP_K": "keyword_top_k",
    "STRUCTURED_TOP_K": "structured_top_k",
    "USE_JSON_FORMAT": "use_json_format",
    "ENABLE_PLANNING": "enable_planning",
    "ENABLE_REFLECTION": "enable_reflection",
    "MAX_REFLECTION_ROUNDS": "max_reflection_rounds",
    "ENABLE_PARALLEL_PROCESSING": "enable_parallel_processing",
    "MAX_PARALLEL_WORKERS": "max_parallel_workers",
    "ENABLE_PARALLEL_RETRIEVAL": "enable_parallel_retrieval",
    "MAX_RETRIEVAL_WORKERS": "max_retrieval_workers",
}
# Integration constants: non-streaming, no thinking mode (identical output, no
# log flood), and SimpleMem's own table name. LANCEDB_PATH is never used for
# real — every store gets its per-user `db_path` explicitly — but is pinned too.
_SETTINGS_FIXED = {
    "USE_STREAMING": False,
    "ENABLE_THINKING": False,
    "MEMORY_TABLE_NAME": "memory_entries",
    "LANCEDB_PATH": str(OUTPUTS_DIR),
}


def _pin_settings(cfg: Dict) -> None:
    """Assign every SimpleMem setting explicitly. Process-global, but identical
    for every user of a run, so safe under concurrent per-user instances."""
    for setting, key in _SETTINGS_FROM_CONFIG.items():
        setattr(_simplemem_settings, setting, cfg[key])
    for setting, value in _SETTINGS_FIXED.items():
        setattr(_simplemem_settings, setting, value)


def _init_to_dialogues(init: Dict, start_id: int) -> Tuple[List[Dialogue], int]:
    """recorder.init → (ordered [Dialogue, ...], next_dialogue_id). Ids are
    per-instance sequential and continue across BUILD calls (DynamicMem deltas)."""
    dialogues: List[Dialogue] = []
    did = start_id

    def _add(speaker: str, content: str, timestamp: Optional[str]):
        nonlocal did
        dialogues.append(Dialogue(dialogue_id=did, speaker=speaker or "",
                                  content=content or "", timestamp=timestamp or None))
        did += 1

    if "app_logs" in init:
        for e in init["app_logs"]:
            _add(e.get("app_name", "app"), app_log_to_passage(e), e.get("timestamp", ""))
    elif "conversation" in init:
        from benchmarks.locomo.env import extract_sessions   # memevol datasets — NOT the HF library
        # extract_sessions yields (session_idx, date_time, turns) — date_time is the
        # per-session timestamp, applied to every turn in that session.
        for _idx, date_time, turns in extract_sessions(init["conversation"]):
            for t in turns:
                _add(t.get("speaker", ""), t.get("text", ""), date_time)
    elif "sessions" in init:
        for s in init["sessions"]:
            for m in s.get("messages", []):
                _add(m.get("role", ""), m.get("content", ""), s.get("date", ""))
    else:
        raise KeyError(f"unrecognized recorder.init keys: {list(init)}")
    return dialogues, did


def _entry_to_passage(entry: MemoryEntry) -> str:
    """Render one retrieved SimpleMem memory unit into a passage string, mirroring
    SimpleMem's own AnswerGenerator._format_contexts field layout."""
    parts = [f"Content: {entry.lossless_restatement}"]
    if entry.timestamp:
        parts.append(f"Time: {entry.timestamp}")
    if entry.location:
        parts.append(f"Location: {entry.location}")
    if entry.persons:
        parts.append(f"Persons: {', '.join(entry.persons)}")
    if entry.entities:
        parts.append(f"Related Entities: {', '.join(entry.entities)}")
    if entry.topic:
        parts.append(f"Topic: {entry.topic}")
    return "\n".join(parts)


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
# SimpleMem's own defaults @ db80b6a, except where the paper's value wins
# (window_size) — see each comment.
CONFIG_DEFAULTS = {
    # PAPER §3.1/§3.3: GPT-4.1-mini is the backbone the headline LoCoMo and
    # LongMemEval-S tables and the full ablation are run on (the paper also
    # reports GPT-4o, Qwen-Plus, Qwen2.5-1.5B/3B, Qwen3-1.7B/8B). Its
    # LLMClient sends temperature=0.1..0.3, which the gpt-5 family rejects
    # — model_config normalises that away at the OpenAI-SDK boundary, so a
    # gpt-5 model IS runnable here.
    "simplemem_llm_model": "gpt-4.1-mini",
    # PAPER §3.1: "Qwen3-embedding-0.6b (1024 dimensions) for dense semantic
    # embeddings". Local sentence-transformer; all-MiniLM-L6-v2 is a light
    # fallback. A `text-embedding-*` name switches to the OpenAI API
    # embedder. No dimension knob: SimpleMem sizes its LanceDB table from
    # the embedder itself.
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "base_url": None,          # OpenAI-compatible base URL for the internal LLM (None = OpenAI default)
    # PAPER §3.1: "we use a sliding window of size W = 20". The VENDORED
    # code ships WINDOW_SIZE=40; the paper's value wins here (as memoryos
    # prefers its paper's values over the shipped defaults). Numbers
    # collected at 40 are NOT comparable to numbers collected at 20.
    "window_size": 20,
    # Window overlap for context continuity (SimpleMem OVERLAP_SIZE; the
    # paper states the window size but not the overlap → the code's value).
    "overlap_size": 2,
    # The next three are per-view CAPS in the vendored code. The paper does
    # not fix a top-k: §3.1 says the retrieval depth is "dynamically
    # adjusted based on estimated query complexity, ranging from k_min = 3
    # to k_max = 20 for complex reasoning queries" — that adaptive planner
    # runs inside the vendored code, under these caps. Left at the code's values.
    "semantic_top_k": 25,      # max vector-similarity hits per query (SEMANTIC_TOP_K)
    "keyword_top_k": 5,        # max keyword/BM25 hits per query (KEYWORD_TOP_K)
    "structured_top_k": 5,     # max structured-metadata hits per query (STRUCTURED_TOP_K)
    # SimpleMem's USE_JSON_FORMAT (default False): whether its LLM calls request
    # a JSON response format. Left at the code's value; the paper does not say.
    "use_json_format": False,
    "enable_planning": True,   # intent-aware multi-query retrieval planning
    "enable_reflection": True, # reflection-based additional retrieval rounds
    "max_reflection_rounds": 2,
    # SimpleMem default — parallel window compression. FAITHFUL path (serial
    # is NOT equivalent: it feeds each window the previous window's entries
    # as dedup context). Also the main build speed lever.
    "enable_parallel_processing": True,
    "max_parallel_workers": 16,    # threads for parallel build (MAX_PARALLEL_WORKERS)
    "enable_parallel_retrieval": True,   # SimpleMem default — parallel multi-view retrieval
    "max_retrieval_workers": 8,    # threads for parallel retrieval (MAX_RETRIEVAL_WORKERS)
}
# `arm: unified` writes unified_models.llm / .embedding into these keys.
# SimpleMem publishes on gpt-4.1-mini with a local Qwen3-Embedding-0.6B
# (1024-dim) index; both change. Do not quote the unified arm as SimpleMem's
# published result.
UNIFIED_MODEL_KEYS = {"llm": ("simplemem_llm_model",), "embedding": ("embedding_model",)}


class SimpleMemMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._system: Optional[SimpleMemSystem] = None   # lazy — built on first hook call
        self._instance_id = uuid.uuid4().hex[:12]        # per-user LanceDB scoping
        self._next_id = 1                                # sequential Dialogue ids across BUILD calls

    def _ensure_system(self):
        if self._system is not None:
            return
        # One user at a time: construction loads local models (concurrency.model_load_lock).
        with model_load_lock:
            cfg = self.config
            _pin_settings(cfg)

            save_dir = str(OUTPUTS_DIR / self._instance_id)
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            with quiet_stdout():
                # EmbeddingModel is constructed here → patched SentenceTransformer
                self._system = SimpleMemSystem(
                    api_key=None,                  # credential: resolved from the environment by SimpleMem
                    model=cfg["simplemem_llm_model"],
                    base_url=cfg["base_url"] or None,
                    db_path=save_dir,
                    clear_db=True,
                    enable_thinking=False,
                    use_streaming=False,
                    enable_planning=cfg["enable_planning"],
                    enable_reflection=cfg["enable_reflection"],
                    max_reflection_rounds=cfg["max_reflection_rounds"],
                    # SimpleMem's internal thread pools are KEPT ON (its shipped
                    # default, and the path the paper's numbers use): the serial and
                    # parallel build paths are NOT equivalent — the serial path feeds
                    # each window the previous window's entries as dedup context,
                    # while the parallel path processes windows independently, so the
                    # faithful output is the parallel one. Workers only run the LLM
                    # extraction concurrently; embedding into LanceDB is batched once
                    # at the end. NOTE the multiplication: users now overlap (the
                    # hooks run on worker threads), so up to max_sample_concurrent ×
                    # max_parallel_workers compression calls can be in flight at once
                    # — keep both within your OpenAI rate limits. The shared embedder
                    # is serialized per model (concurrency.serialize_calls).
                    enable_parallel_processing=cfg["enable_parallel_processing"],
                    max_parallel_workers=cfg["max_parallel_workers"],
                    enable_parallel_retrieval=cfg["enable_parallel_retrieval"],
                    max_retrieval_workers=cfg["max_retrieval_workers"],
                )

    # SimpleMem is synchronous (window compression, planning/reflection LLM calls,
    # LanceDB): each hook runs its body on a worker thread so other users keep
    # going meanwhile (see baselines/harness/concurrency.py).

    async def build_memory_from_data(self, recorder) -> None:
        await asyncio.to_thread(self._build, recorder)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return await asyncio.to_thread(self._retrieve, recorder)

    def _build(self, recorder) -> None:
        self._ensure_system()
        dialogues, self._next_id = _init_to_dialogues(recorder.init, self._next_id)
        # add_dialogues + finalize is ADDITIVE across checkpoints; SimpleMem's LLM
        # compression + LanceDB indexing run here.
        with quiet_stdout():
            self._system.add_dialogues(dialogues)
            self._system.finalize()

    def _retrieve(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        with quiet_stdout():
            entries = self._system.hybrid_retriever.retrieve(query)   # intent-aware multi-view search
        if not entries:
            return {}
        return {"passages": [_entry_to_passage(e) for e in entries]}
