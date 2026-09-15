"""A-mem (Agentic Memory, arXiv:2502.12110) as a retrieval MemoClass.

BUILD: every ingestion unit becomes one A-mem note via
`AgenticMemorySystem.add_note(content, time=...)` — the method's own pipeline
(LLM content analysis → LLM memory evolution → MiniLM embedding index →
consolidation every evo_threshold=100 evolutions) runs untouched, exactly as
in the official LoCoMo eval driver (test_advanced.py @ 0c8039f).

RETRIEVE: the official eval pipeline verbatim — an LLM rewrites the question
into keywords (prompt + JSON schema copied from
test_advanced.py::generate_query_llm; a separate LLMController mirroring
advancedMemAgent.retriever_llm), then `find_related_memories_raw(keywords, k)`
returns one formatted string (top-k notes + linked neighbors). Read-only
(DynamicMem query non-pollution holds). The shared QA agent answers —
`use_memory_to_answer` is NOT overridden (hipporag2 pattern).

Ingestion units (recorder.init dispatch, cf. hipporag2's _init_to_passages):
  locomo ("conversation"): one note per turn, VERBATIM official unit
    "Speaker {speaker}says : {text}" (missing space intended), time = session
    date_time.  longmemeval ("sessions"): one note per message,
    "{role}: {content}", time = session date (A-mem never defined these
    benchmarks; analogous mapping).  dynamicmem ("app_logs"): one note per
    log entry, content = hipporag2's app_log_to_passage text (identical
    content across baselines), time = log timestamp.
"""
from __future__ import annotations

import asyncio
import json
from typing import Dict, List, Tuple

from common.memo_class import MemoClass
from common.openai_usage import install as _install_openai_usage
from baselines.harness.concurrency import model_load_lock, quiet_stdout
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.model_config import (
    install_embedder_factory, install_openai_param_normalisation,
)

# Both patches MUST precede the vendored import (see model_config's docstring):
# memory_layer binds `SentenceTransformer` at ITS import time, and its
# OpenAIController hardcodes temperature=0.7 + max_tokens=1000 on every call,
# which the gpt-5 family rejects.
install_embedder_factory()
install_openai_param_normalisation()


# A-mem's LLMController builds its own `openai` client (a lazy `from openai
# import OpenAI` inside OpenAIController.__init__), so its 2 calls per
# ingested note never reached common.tokens. Patching the SDK boundary here —
# integration code — captures them with ZERO edits under src/, preserving the
# byte-identity the README's `diff -r` asserts.
_install_openai_usage()
# NOTE: sentence-transformers eagerly imports HuggingFace `datasets`. That used
# to collide with memevol's own top-level `datasets/` package and needed a
# sys.modules shim (_st_shim.py); the package was renamed to `benchmarks/`
# (2026-08-07), so plain imports are correct now.
from baselines.harness.amem.src.memory_layer import AgenticMemorySystem, LLMController  # noqa: E402

# Copied VERBATIM from A-mem test_advanced.py::generate_query_llm (@ 0c8039f).
# (.format on this template renders byte-identically to upstream's f-string.)
_KEYWORDS_PROMPT = """Given the following question, generate several keywords, using 'cosmos' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the selected text. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""

_KEYWORDS_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "response",
    "schema": {
        "type": "object",
        "properties": {"keywords": {"type": "string"}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
    "strict": True,
}}


def _init_to_note_units(init: Dict) -> List[Tuple[str, str]]:
    """recorder.init → ordered [(content, time), ...] A-mem note units."""
    if "app_logs" in init:
        return [(app_log_to_passage(e), e.get("timestamp", "")) for e in init["app_logs"]]
    if "conversation" in init:
        from benchmarks.locomo.env import extract_sessions
        units: List[Tuple[str, str]] = []
        for _idx, date_time, turns in extract_sessions(init["conversation"]):
            for t in turns:
                # VERBATIM official A-mem LoCoMo unit (incl. missing space):
                #   test_advanced.py: "Speaker "+ turn.speaker + "says : " + turn.text
                units.append(("Speaker " + t.get("speaker", "") + "says : " + t.get("text", ""), date_time))
        return units
    if "sessions" in init:
        units = []
        for s in init["sessions"]:
            for m in s.get("messages", []):
                units.append((f"{m.get('role', '')}: {m.get('content', '')}", s.get("date", "")))
        return units
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
CONFIG_DEFAULTS = {
    # PAPER (arXiv 2502.12110) Table 1: GPT-4o-mini is the primary GPT
    # backbone (the paper also reports GPT-4o, Qwen2.5-1.5B/3B, Llama3.2-1B/3B).
    # A-mem's OpenAIController hardcodes temperature+max_tokens, which the
    # gpt-5 family rejects — model_config normalises those away at the
    # OpenAI-SDK boundary, so a gpt-5 model IS runnable (unified arm).
    "amem_llm_model": "gpt-4o-mini",
    # PAPER §4.2: "For text embedding, we implement the all-minilm-l6-v2
    # model across all experiments." 384-dim, local. A `text-embedding-*`
    # name switches to the OpenAI API embedder instead.
    "amem_embedding_model": "all-MiniLM-L6-v2",
    "retrieve_k": 10,          # PAPER §4.2: "we primarily employ k=10 for top-k memory selection"
}
# `arm: unified` writes unified_models.llm / .embedding into these keys. A-mem
# publishes on gpt-4o-mini with a local all-MiniLM-L6-v2 index; both change.
# Do not quote the unified arm as A-mem's published result.
UNIFIED_MODEL_KEYS = {"llm": ("amem_llm_model",), "embedding": ("amem_embedding_model",)}


class AMemMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._system = None          # AgenticMemorySystem (lazy — built on first hook call)
        self._retriever_llm = None   # LLMController for the query→keywords rewrite

    def _ensure_system(self):
        if self._system is not None:
            return
        # One user at a time: construction loads local models (concurrency.model_load_lock).
        with model_load_lock:
            model = self.config["amem_llm_model"]
            # A-mem's embedder IS a constructor parameter (AgenticMemorySystem's
            # `model_name`), so the config key needs no special plumbing: the name
            # flows to SimpleEmbeddingRetriever, which calls the patched
            # SentenceTransformer factory installed above. That factory both shares
            # ONE embedder across users (a fresh MemoClass is built per user, so
            # otherwise the weights reload per conversation) and returns an
            # APIEmbedder when the name is a `text-embedding-*` model.
            embedder = self.config["amem_embedding_model"]
            # Mirrors test_advanced.py::advancedMemAgent.__init__ (openai backend):
            # one AgenticMemorySystem + a separate retriever_llm, same model.
            self._system = AgenticMemorySystem(
                model_name=embedder, llm_backend="openai", llm_model=model,
            )
            self._retriever_llm = LLMController(backend="openai", model=model, api_key=None)

    # A-mem is synchronous end to end (LLM calls, embedding, evolution): each hook
    # runs its body on a worker thread so other users keep going meanwhile
    # (see baselines/harness/concurrency.py).

    async def build_memory_from_data(self, recorder) -> None:
        await asyncio.to_thread(self._build, recorder)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return await asyncio.to_thread(self._retrieve, recorder)

    def _build(self, recorder) -> None:
        self._ensure_system()
        units = _init_to_note_units(recorder.init)
        # A-mem prints every analysis + evolution prompt; silence the flood
        # (console-output-only adaptation — the algorithm is untouched).
        with quiet_stdout():
            for content, t in units:
                self._system.add_note(content, time=t)

    def _rewrite_query(self, question: str) -> str:
        # VERBATIM logic of test_advanced.py::generate_query_llm (@ 0c8039f).
        response = self._retriever_llm.llm.get_completion(
            _KEYWORDS_PROMPT.format(question=question), response_format=_KEYWORDS_SCHEMA,
        )
        try:
            return json.loads(response)["keywords"]
        except Exception:
            return response.strip()

    def _retrieve(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        k = int(self.config["retrieve_k"])
        with quiet_stdout():
            keywords = self._rewrite_query(query)   # LLM call
            memory_str = self._system.find_related_memories_raw(keywords, k=k)
        if not memory_str:   # upstream returns [] when the store is empty
            return {}
        return {"memories": memory_str}
