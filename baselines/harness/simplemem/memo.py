"""SimpleMem (arXiv 2510.xxxxx, https://github.com/aiming-lab/SimpleMem) as a
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
agent answers — ``use_memory_to_answer`` is NOT overridden (hipporag2/amem
pattern; keeps the comparison about memory, not about SimpleMem's own answerer).

Ingestion units (recorder.init dispatch, cf. hipporag2's _init_to_passages):
  locomo ("conversation"): one Dialogue per turn (speaker, text), time = session
    date_time.  longmemeval ("sessions"): one Dialogue per message (role→speaker,
    content), time = session date.  dynamicmem ("app_logs"): one Dialogue per log
    entry, content = hipporag2's app_log_to_passage text (identical content across
    baselines), speaker = app_name, time = log timestamp.
"""
from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.memo_class import MemoClass
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.simplemem._st_shim import (
    ensure_sentence_transformers, install_embedding_cache, import_simplemem_system,
)

ensure_sentence_transformers()     # import ST past memevol's datasets/ shadow (embedding.py needs it)
install_embedding_cache()          # share the ~0.6B Qwen3 embedder across per-user systems
# Import the vendored simplemem chain (which pulls in lancedb) with HF `datasets`
# active, so lancedb's import-time `from datasets import Dataset` resolves to the
# HF library, not memevol's `datasets/` package.
SimpleMemSystem, Dialogue, MemoryEntry = import_simplemem_system()   # vendored, byte-identical

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# SimpleMem settings that are read ONLY from its global `settings` singleton
# (env / config.py), never from the SimpleMemSystem constructor. We stamp them
# into os.environ before the first system is built so per-run config takes hold.
# Keyed by _cfg name → SimpleMem env var name.
_ENV_FROM_CFG = {
    "embedding_model": "EMBEDDING_MODEL",
    "window_size": "WINDOW_SIZE",
    "overlap_size": "OVERLAP_SIZE",
    "semantic_top_k": "SEMANTIC_TOP_K",
    "keyword_top_k": "KEYWORD_TOP_K",
    "structured_top_k": "STRUCTURED_TOP_K",
}


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
        from datasets.locomo.env import extract_sessions   # memevol datasets — NOT the HF library
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


class SimpleMemMemo(MemoClass):
    _cfg: Dict = {}   # overridden per-run by eval_common.make_memo_class

    def __init__(self):
        super().__init__()
        self._system: Optional[SimpleMemSystem] = None   # lazy — built on first hook call
        self._instance_id = uuid.uuid4().hex[:12]        # per-user LanceDB scoping
        self._next_id = 1                                # sequential Dialogue ids across BUILD calls

    def _ensure_system(self):
        if self._system is not None:
            return
        cfg = self._cfg
        # Stamp settings that SimpleMem reads only from env (idempotent; identical
        # for every user, so process-global env is safe).
        for key, env in _ENV_FROM_CFG.items():
            val = cfg.get(key)
            if val is not None:
                os.environ[env] = str(val)
        os.environ["USE_STREAMING"] = "false"   # integration: non-streaming (identical output, no log flood)

        save_dir = str(OUTPUTS_DIR / self._instance_id)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            # EmbeddingModel is constructed here → patched SentenceTransformer
            # (shared cache + hf_datasets_active) handles the datasets shadow.
            self._system = SimpleMemSystem(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model=cfg.get("simplemem_llm_model"),
                base_url=cfg.get("base_url") or None,
                db_path=save_dir,
                clear_db=True,
                enable_thinking=False,
                use_streaming=False,
                enable_planning=cfg.get("enable_planning"),
                enable_reflection=cfg.get("enable_reflection"),
                max_reflection_rounds=cfg.get("max_reflection_rounds"),
                # SimpleMem's internal thread pools are KEPT ON (its shipped
                # default, and the path the paper's numbers use): the serial and
                # parallel build paths are NOT equivalent — the serial path feeds
                # each window the previous window's entries as dedup context,
                # while the parallel path processes windows independently, so the
                # faithful output is the parallel one. It is also the only real
                # build parallelism: the async build hook body is synchronous +
                # blocking, so it blocks the event loop and users don't actually
                # overlap under max_sample_concurrent — meaning there is no
                # thread-count multiplication to fear. Workers only run the LLM
                # extraction concurrently; embedding into LanceDB is batched once
                # at the end (single-threaded), so the shared embedder is never
                # hit from multiple threads.
                enable_parallel_processing=cfg.get("enable_parallel_processing"),
                max_parallel_workers=cfg.get("max_parallel_workers"),
                enable_parallel_retrieval=cfg.get("enable_parallel_retrieval"),
                max_retrieval_workers=cfg.get("max_retrieval_workers"),
            )

    async def build_memory_from_data(self, recorder) -> None:
        self._ensure_system()
        # _init_to_dialogues touches memevol's `datasets` (locomo branch) — compute
        # BEFORE the stdout redirect and OUTSIDE any hf_datasets_active window.
        dialogues, self._next_id = _init_to_dialogues(recorder.init, self._next_id)
        # add_dialogues + finalize is ADDITIVE across checkpoints; SimpleMem's LLM
        # compression + LanceDB indexing run here (synchronous, no await → the
        # redirect can't leak across concurrently-scheduled samples).
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            self._system.add_dialogues(dialogues)
            self._system.finalize()

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            entries = self._system.hybrid_retriever.retrieve(query)   # intent-aware multi-view search
        if not entries:
            return {}
        return {"passages": [_entry_to_passage(e) for e in entries]}
