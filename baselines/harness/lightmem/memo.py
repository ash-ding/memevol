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

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from common.memo_class import MemoClass

from common.openai_usage import install as _install_openai_usage
from baselines.harness.concurrency import quiet_stdout
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.model_config import (
    install_embedder_factory, install_openai_param_normalisation,
    is_api_embedding_model, resolve_device,
)

# LightMem's absolute imports (`from lightmem.memory...`) must resolve to the
# byte-identical vendored copy under src/, not any pip-installed lightmem.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The shared embedder factory (baselines/harness/model_config.py) memoizes the
# heavy weights across per-user systems: a fresh MemoClass is built per user, and
# ``TextEmbedderHuggingface.__init__`` does
# ``self.model = SentenceTransformer(config.model, **config.model_kwargs)`` with
# no way to pass a pre-built model through, so patching that constructor is the
# only lever that leaves ``src/`` byte-identical.
#
# LightMem is the one baseline that does NOT need the factory's API-embedder
# dispatch: its vendored factory already ships a real OpenAI arm
# (``src/lightmem/factory/text_embedder/openai.py::TextEmbedderOpenAI``), so the
# API arm is a `text_embedder.model_name` flip in _build_config below — a genuine
# vendored code path rather than an adapter. The patch stays for the HF arm.
#
# The param-normalisation patch is needed because LightMem's memory manager sends
# temperature + max_tokens on every call, which the gpt-5 family rejects.
#
# Both must precede the vendored import (they bind their names at import time).
install_embedder_factory()
install_openai_param_normalisation()

from lightmem.memory.lightmem import LightMemory  # noqa: E402  (vendored, byte-identical)

# LightMem's memory managers build their own `openai` clients (5 files under
# src/); the SDK-boundary patch captures them without editing any of them.
_install_openai_usage()

# LightMem logs verbosely at INFO per add_memory/retrieve call; pin its logger to
# WARNING + a NullHandler (console-only integration adaptation; the algorithm is
# untouched). Stray print()s in the vendored code are silenced per-hook below.
_lm_logger = logging.getLogger("LightMemory")
_lm_logger.setLevel(logging.WARNING)
_lm_logger.addHandler(logging.NullHandler())
_lm_logger.propagate = False

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


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
        from benchmarks.locomo.env import extract_sessions   # memevol datasets — NOT the HF library
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


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
# LightMem's own experiment defaults @ 34410f4.
CONFIG_DEFAULTS = {
    "pre_compress": True,      # LLMlingua-2 token pre-compression (a core LightMem stage; needs the llmlingua model + a GPU)
    "topic_segment": True,     # attention-based topic segmentation. REQUIRES pre_compress (shares the LLMlingua-2 model)
    # PAPER Table 5: LLMlingua-2 is the token-compression AND
    # topic-segmentation model (both, shared). HF hub id or a local path.
    "llmlingua_model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    # device_map for the LLMlingua-2 model. None/"auto" => cuda if a GPU is
    # visible, else cpu (slow). Pin with "cpu" / "cuda" / "cuda:1" if needed.
    "llmlingua_device": None,
    "compress_rate": 0.6,      # LLMlingua-2 target compression rate (LoCoMo experiment value; library default is 0.8)
    "messages_use": "user_only",   # which turns feed extraction: user_only | assistant_only | hybrid
    "extract_threshold": 0.1,  # segmentation/extraction trigger threshold (LoCoMo experiment value)
    "extraction_mode": "flat", # flat (factual entries only) | event (factual + relational, temporally bound)
    # PAPER Table 5: GPT-4o-mini is the system backbone for BOTH
    # f_sum/extract() and f_update() (the paper also reports
    # Qwen3-30B-A3B-Instruct-2507 and GLM-4.6). It sends temperature +
    # max_tokens, which the gpt-5 family rejects — model_config normalises
    # those away at the OpenAI-SDK boundary, so a gpt-5 model IS runnable.
    "lightmem_llm_model": "gpt-4o-mini",
    "manager_max_tokens": 16000,   # max_tokens for the internal LLM (LoCoMo experiment value)
    # OpenAI-compatible base URL for LightMem's internal LLM AND, on the API
    # embedder arm, for its embedding calls (None = OpenAI default)
    "base_url": None,
    # PAPER Table 5: f_index() embedding model = all-MiniLM-L6-v2 (a
    # black-font entry there, i.e. shared by LightMem AND its baselines).
    # 384-dim, local HF sentence-transformer. A `text-embedding-*` name
    # switches to LightMem's OWN vendored TextEmbedderOpenAI arm — a real
    # code path, not an adapter.
    "embedding_model": "all-MiniLM-L6-v2",
    # MUST match the embedder: it sizes the Qdrant collection AND is sent as
    # the API `dimensions` parameter, so a mismatch fails hard.
    # text-embedding-3-small => 1536. Changing this invalidates any existing
    # index built at the old width.
    "embedding_dims": 384,
    "embedding_device": None,  # device for the embedder (HF arm only). None/"auto" => cuda if visible, else cpu
    # Run the offline-update refinement phase after build
    # (construct_update_queue + offline_update_all_entries) — the full
    # LoCoMo-paper pipeline. Adds per-entry LLM cost.
    "offline_update": True,
    "update_sim_threshold": 0.9,   # score_threshold for offline_update_all_entries (LoCoMo experiment value)
    "retrieve_limit": 20,      # top-k memories LightMemory.retrieve returns per query (LongMemEval driver value)
}
# `arm: unified` writes unified_models.llm / .embedding into these keys, and the
# embedder's width into `embedding_dims` (it sizes the Qdrant collection AND is
# sent as the API `dimensions` parameter, so it must move with the embedder).
# LightMem publishes on gpt-4o-mini with a local all-MiniLM-L6-v2 (384-dim)
# index. WHAT STAYS LOCAL: the LLMlingua-2 pre-compressor — a BERT token
# classifier with NO API equivalent, so it remains a real, untracked local
# compute cost in BOTH arms.
UNIFIED_MODEL_KEYS = {
    "llm": ("lightmem_llm_model",),
    "embedding": ("embedding_model",),
    "embedding_dims": ("embedding_dims",),
}


class LightMemMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._system = None                        # LightMemory (lazy — built on first hook call)
        self._instance_id = uuid.uuid4().hex[:12]  # per-user Qdrant scoping

    def _build_config(self) -> Dict:
        cfg = self.config
        if cfg["topic_segment"] and not cfg["pre_compress"]:
            # LightMem shares one LLMlingua-2 model between the pre-compressor and
            # the topic segmenter (precomp_topic_shared); the segmenter reads
            # self.compressor, which only exists when pre_compress is on.
            raise ValueError("lightmem: topic_segment requires pre_compress "
                             "(they share the LLMlingua-2 model).")
        save_dir = str(OUTPUTS_DIR / self._instance_id)
        # Both device knobs used to default to a hardcoded "cuda" and crashed
        # outright on a CPU-only box. They now default to null → auto-detect.
        llmlingua_device = resolve_device(cfg["llmlingua_device"])
        embedding_device = resolve_device(cfg["embedding_device"])
        config: Dict = {
            "pre_compress": cfg["pre_compress"],
            "pre_compressor": ({
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": cfg["llmlingua_model"],
                        "device_map": llmlingua_device,
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
                    "api_key": None,   # credential: resolved from the environment by LightMem / the SDK
                    "max_tokens": cfg["manager_max_tokens"],
                    "openai_base_url": cfg["base_url"] or "",
                },
            },
            "extract_threshold": cfg["extract_threshold"],
            "index_strategy": "embedding",
            # Embedder arm, chosen by the SHAPE of `embedding_model`:
            #   text-embedding-* → LightMem's own vendored TextEmbedderOpenAI
            #   anything else    → its TextEmbedderHuggingface (paper-faithful)
            # Both are real vendored code paths (TextEmbedderFactory dispatches
            # on `model_name`), so the API arm needs no adapter here — unlike
            # amem/memoryos/simplemem, which have no OpenAI embedder at all.
            #
            # `embedding_dims` MUST move with the embedder: it sizes the Qdrant
            # collection below AND is sent as the API `dimensions` parameter, so
            # a mismatch is a hard failure rather than a silent degradation.
            "text_embedder": ({
                "model_name": "openai",
                "configs": {
                    "model": cfg["embedding_model"],
                    "embedding_dims": cfg["embedding_dims"],
                    "api_key": None,   # credential: the OpenAI SDK reads it from the environment
                    "openai_base_url": cfg["base_url"] or None,
                },
            } if is_api_embedding_model(cfg["embedding_model"]) else {
                "model_name": "huggingface",
                "configs": {
                    "model": cfg["embedding_model"],
                    "embedding_dims": cfg["embedding_dims"],
                    "model_kwargs": {"device": embedding_device},
                },
            }),
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
        # Construction builds the LLMlingua-2 compressor + the embedder (shared
        # across users by model_config's factory) + the Qdrant client; stdout is
        # silenced for the vendored debug prints.
        with quiet_stdout():
            self._system = LightMemory.from_config(config)

    # LightMem is synchronous (LLMlingua-2, LLM extraction, embedding, Qdrant):
    # each hook runs its body on a worker thread so other users keep going
    # meanwhile (see baselines/harness/concurrency.py).

    async def build_memory_from_data(self, recorder) -> None:
        await asyncio.to_thread(self._build, recorder)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return await asyncio.to_thread(self._retrieve, recorder)

    def _build(self, recorder) -> None:
        self._ensure_system()
        turns = _init_to_turns(recorder.init)
        if not turns:
            return
        n = len(turns)
        # add_memory is ADDITIVE across checkpoints; its LLM extraction + Qdrant
        # indexing run here. force_segment/force_extract on
        # the last turn flush the buffers so this call's data is fully committed
        # before any retrieval (matches the drivers' is_last_turn flush).
        with quiet_stdout():
            for i, turn_msgs in enumerate(turns):
                is_last = i == n - 1
                self._system.add_memory(
                    messages=turn_msgs, force_segment=is_last, force_extract=is_last,
                )
            if self.config["offline_update"]:
                # Full LoCoMo-paper offline refinement over the whole current
                # index (per-entry LLM dedup/merge/delete). For DynamicMem this
                # runs at each checkpoint's build call (integration adaptation:
                # there is no single "final" build call in the interleaved
                # protocol), so each checkpoint's queries see a refined memory.
                self._system.construct_update_queue_all_entries()
                self._system.offline_update_all_entries(
                    score_threshold=self.config["update_sim_threshold"],
                )

    def _retrieve(self, recorder) -> Dict:
        self._ensure_system()
        query = recorder.init.get("query", "")
        k = int(self.config["retrieve_limit"])
        with quiet_stdout():
            passages = self._system.retrieve(query, limit=k)   # embed + Qdrant search (read-only)
        if not passages:
            return {}
        return {"passages": list(passages)}
