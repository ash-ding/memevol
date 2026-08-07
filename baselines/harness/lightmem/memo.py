"""LightMem (https://github.com/zjunlp/LightMem) as a retrieval MemoClass.

BUILD: every ingestion unit becomes a LightMem turn (a ``[user, assistant]``
message pair carrying a session-level ``time_stamp``); the vendored pipeline runs
untouched, exactly as in LightMem's own experiment drivers — per turn
``add_memory(messages, force_segment=is_last, force_extract=is_last)`` runs
optional LLMlingua-2 pre-compression → topic segmentation → LLM metadata/summary
extraction → HuggingFace embedding → insertion into a per-user Qdrant index
(``update="offline"``). After the last turn of a build call, the offline-update
refinement phase (``construct_update_queue_all_entries`` +
``offline_update_all_entries``) is run when enabled — the full LoCoMo-paper
pipeline (config knob ``offline_update``, default on).

RETRIEVE: ``LightMemory.retrieve(query, limit)`` — embed the query, search the
Qdrant index, return the top-k memories as formatted strings (this is exactly how
LightMem's own LongMemEval driver retrieves). Read-only (DynamicMem query
non-pollution holds). Returned as ``{"passages": [...]}`` and the SHARED QA agent
answers — ``use_memory_to_answer`` is NOT overridden (hipporag2/amem/simplemem
pattern; keeps the comparison about memory, not about a bespoke answerer).

Ingestion units (recorder.init dispatch, cf. hipporag2's _init_to_passages):
  locomo ("conversation"): one turn per utterance — user content = turn text,
    time = ``parse_locomo_timestamp`` of the session date_time (verbatim from
    LightMem's add_locomo.py), speaker_id/name from the conversation.
  longmemeval ("sessions"): one turn per (user, assistant) message pair, time =
    session date (LightMem's native LongMemEval mapping).
  dynamicmem ("app_logs"): one turn per log entry, user content = hipporag2's
    app_log_to_passage text (identical content across baselines), time = log
    timestamp (LightMem never defined this benchmark; analogous mapping).
"""
from __future__ import annotations

import contextlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from common.memo_class import MemoClass
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.lightmem._st_shim import (
    ensure_sentence_transformers, install_embedding_cache, install_eager_attention,
    hf_datasets_active, import_lightmemory,
)

ensure_sentence_transformers()     # import ST past memevol's datasets/ shadow
install_embedding_cache()          # share the embedder across per-user systems
install_eager_attention()          # LLMlingua-2 segmenter needs eager output_attentions (transformers>=5)
LightMemory = import_lightmemory()  # vendored, byte-identical

# LightMem logs verbosely at INFO per add_memory/retrieve call; pin its logger to
# WARNING + a NullHandler (console-only integration adaptation; the algorithm is
# untouched). Stray print()s in the vendored code are silenced per-hook below.
_lm_logger = logging.getLogger("LightMemory")
_lm_logger.setLevel(logging.WARNING)
_lm_logger.addHandler(logging.NullHandler())
_lm_logger.propagate = False

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


@contextlib.contextmanager
def _quiet_stdout():
    """Silence LightMem's vendored debug ``print()``s during a hook body.

    The sink is opened as UTF-8 (``errors="replace"``) on purpose: LightMem's
    extractor ``print()``s the raw user prompt, and on Windows the default
    ``cp1252`` stdout can't encode non-Latin-1 characters (e.g. ``č`` in
    multilingual LongMemEval content) → ``UnicodeEncodeError``. That exception is
    swallowed deep in LightMem's parallel extractor (→ a result with
    ``usage=None`` → an ``AttributeError`` crash in its token accounting), so a
    plain devnull redirect (cp1252) would still trip it. Routing to a UTF-8 sink
    lets the print succeed and discards it. No await inside → the process-global
    stdout swap can't leak across concurrently-scheduled samples."""
    with open(os.devnull, "w", encoding="utf-8", errors="replace") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def parse_locomo_timestamp(timestamp_str: str) -> str:
    # verbatim from LightMem experiments/locomo/add_locomo.py
    timestamp_str = timestamp_str.strip("()")
    try:
        dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def _turn_pair(role_content_user: str, ts: str, speaker_id: str, speaker_name: str) -> List[Dict]:
    """One LightMem turn: a [user, assistant] pair (assistant empty), mirroring
    LightMem's own drivers (user_only messages_use). Both carry the session-level
    time_stamp; speaker fields are stored in the payload."""
    return [
        {"role": "user", "content": role_content_user, "time_stamp": ts,
         "speaker_id": speaker_id, "speaker_name": speaker_name},
        {"role": "assistant", "content": "", "time_stamp": ts,
         "speaker_id": speaker_id, "speaker_name": speaker_name},
    ]


def _init_to_turns(init: Dict) -> List[List[Dict]]:
    """recorder.init → ordered [turn, ...], each turn a [user, assistant] pair."""
    if "app_logs" in init:
        turns: List[List[Dict]] = []
        for e in init["app_logs"]:
            app = e.get("app_name", "app")
            turns.append(_turn_pair(app_log_to_passage(e), e.get("timestamp", ""), app, app))
        return turns
    if "conversation" in init:
        from datasets.locomo.env import extract_sessions   # memevol datasets — NOT the HF library
        conv = init["conversation"]
        speaker_a = conv.get("speaker_a")
        speaker_b = conv.get("speaker_b")
        turns = []
        # extract_sessions yields (session_idx, date_time, turns) — date_time is the
        # per-session timestamp, applied to every utterance in that session.
        for _idx, date_time, session_turns in extract_sessions(conv):
            ts = parse_locomo_timestamp(date_time)
            for t in session_turns:
                name = t.get("speaker", "")
                sid = "speaker_a" if name == speaker_a else ("speaker_b" if name == speaker_b else name)
                content = t.get("text", "")
                # verbatim from add_locomo.py: fold a blip_caption into the content
                if t.get("blip_caption"):
                    content = f"{content} (image description: {t['blip_caption']})"
                turns.append(_turn_pair(content, ts, sid, name))
        return turns
    if "sessions" in init:
        turns = []
        for s in init["sessions"]:
            date = s.get("date", "")
            msgs = [m for m in s.get("messages", [])]
            # drop leading non-user messages, then pair up user+assistant (verbatim
            # loop shape from LightMem's run_lightmem_gpt.py).
            while msgs and msgs[0].get("role") != "user":
                msgs.pop(0)
            for turn_idx in range(len(msgs) // 2):
                pair = msgs[turn_idx * 2: turn_idx * 2 + 2]
                if len(pair) < 2 or pair[0].get("role") != "user" or pair[1].get("role") != "assistant":
                    continue
                turns.append([
                    {"role": "user", "content": pair[0].get("content", ""), "time_stamp": date},
                    {"role": "assistant", "content": pair[1].get("content", ""), "time_stamp": date},
                ])
        return turns
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


class LightMemMemo(MemoClass):
    _cfg: Dict = {}   # overridden per-run by eval_common.make_memo_class

    def __init__(self):
        super().__init__()
        self._system = None                        # LightMemory (lazy — built on first hook call)
        self._instance_id = uuid.uuid4().hex[:12]  # per-user Qdrant scoping

    def _build_config(self) -> Dict:
        cfg = self._cfg
        if cfg["topic_segment"] and not cfg["pre_compress"]:
            # LightMem shares one LLMlingua-2 model between the pre-compressor and
            # the topic segmenter (precomp_topic_shared); the segmenter reads
            # self.compressor, which only exists when pre_compress is on.
            raise ValueError("lightmem: topic_segment requires pre_compress "
                             "(they share the LLMlingua-2 model).")
        save_dir = str(OUTPUTS_DIR / self._instance_id)
        config: Dict = {
            "pre_compress": cfg["pre_compress"],
            "pre_compressor": ({
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": cfg["llmlingua_model"],
                        "device_map": cfg["llmlingua_device"],
                        "use_llmlingua2": True,
                    },
                    "compress_config": {
                        "instruction": "",
                        "rate": cfg["compress_rate"],
                        "target_token": -1,
                    },
                },
            } if cfg["pre_compress"] else None),
            "topic_segment": cfg["topic_segment"],
            "precomp_topic_shared": True,
            "topic_segmenter": ({"model_name": "llmlingua-2"} if cfg["topic_segment"] else None),
            "messages_use": cfg["messages_use"],
            "metadata_generate": True,
            "text_summary": True,
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": cfg["lightmem_llm_model"],
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "max_tokens": cfg["manager_max_tokens"],
                    "openai_base_url": cfg["base_url"] or "",
                },
            },
            "extract_threshold": cfg["extract_threshold"],
            "index_strategy": "embedding",
            "text_embedder": {
                "model_name": "huggingface",
                "configs": {
                    "model": cfg["embedding_model"],
                    "embedding_dims": cfg["embedding_dims"],
                    "model_kwargs": {"device": cfg["embedding_device"]},
                },
            },
            "retrieve_strategy": "embedding",
            "embedding_retriever": {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": self._instance_id,
                    "embedding_model_dims": cfg["embedding_dims"],
                    "path": save_dir,
                    "on_disk": True,
                },
            },
            "update": "offline",
            "extraction_mode": cfg["extraction_mode"],
        }
        return config

    def _ensure_system(self):
        if self._system is not None:
            return
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        config = self._build_config()
        # Construction builds the LLMlingua-2 compressor + SentenceTransformer
        # embedder + Qdrant client; the patched SentenceTransformer (shared cache
        # + hf_datasets_active) handles the datasets shadow. No await inside, so
        # the stdout redirect + sys.modules swap can't leak across samples.
        with _quiet_stdout():
            with hf_datasets_active():
                self._system = LightMemory.from_config(config)

    async def build_memory_from_data(self, recorder) -> None:
        self._ensure_system()
        # _init_to_turns touches memevol's `datasets` (locomo branch) — compute
        # BEFORE the stdout redirect and OUTSIDE any hf_datasets_active window.
        turns = _init_to_turns(recorder.init)
        if not turns:
            return
        n = len(turns)
        # add_memory is ADDITIVE across checkpoints; its LLM extraction + Qdrant
        # indexing run here (synchronous, no await → the redirect can't leak
        # across concurrently-scheduled samples). force_segment/force_extract on
        # the last turn flush the buffers so this call's data is fully committed
        # before any retrieval (matches the drivers' is_last_turn flush).
        with _quiet_stdout():
            for i, turn_msgs in enumerate(turns):
                is_last = i == n - 1
                self._system.add_memory(
                    messages=turn_msgs, force_segment=is_last, force_extract=is_last,
                )
            if self._cfg["offline_update"]:
                # Full LoCoMo-paper offline refinement over the whole current
                # index (per-entry LLM dedup/merge/delete). For DynamicMem this
                # runs at each checkpoint's build call (integration adaptation:
                # there is no single "final" build call in the interleaved
                # protocol), so each checkpoint's queries see a refined memory.
                self._system.construct_update_queue_all_entries()
                self._system.offline_update_all_entries(
                    score_threshold=self._cfg["update_sim_threshold"],
                )

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        k = int(self._cfg["retrieve_limit"])
        with _quiet_stdout():
            passages = self._system.retrieve(query, limit=k)   # embed + Qdrant search (read-only)
        if not passages:
            return {}
        return {"passages": list(passages)}
