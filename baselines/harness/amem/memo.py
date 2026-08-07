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

import contextlib
import json
import os
from typing import Dict, List, Tuple

from common.memo_class import MemoClass
from baselines.harness.hipporag2.memo import app_log_to_passage
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


class AMemMemo(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        self._system = None          # AgenticMemorySystem (lazy — built on first hook call)
        self._retriever_llm = None   # LLMController for the query→keywords rewrite

    def _ensure_system(self):
        if self._system is not None:
            return
        model = self.config.get("amem_llm_model", "gpt-4o-mini")
        # Mirrors test_advanced.py::advancedMemAgent.__init__ (openai backend):
        # one AgenticMemorySystem + a separate retriever_llm, same model.
        self._system = AgenticMemorySystem(
            model_name="all-MiniLM-L6-v2", llm_backend="openai", llm_model=model,
        )
        self._retriever_llm = LLMController(backend="openai", model=model, api_key=None)

    async def build_memory_from_data(self, recorder) -> None:
        self._ensure_system()
        units = _init_to_note_units(recorder.init)
        # A-mem prints every analysis + evolution prompt; silence the flood
        # (console-output-only adaptation — the algorithm is untouched). The
        # block contains no awaits, so redirect_stdout can't leak across tasks.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
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

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        k = int(self.config.get("retrieve_k", 10))
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            keywords = self._rewrite_query(query)   # LLM call
            memory_str = self._system.find_related_memories_raw(keywords, k=k)
        if not memory_str:   # upstream returns [] when the store is empty
            return {}
        return {"memories": memory_str}
