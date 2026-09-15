"""MemoryOS (arXiv 2506.06326, EMNLP 2025 Oral — github.com/BAI-LAB/MemoryOS)
as a retrieval MemoClass.

BUILD: every ingestion unit becomes one MemoryOS *dialogue page* via
``Memoryos.add_memory(user_input, agent_response)``, and the vendored pipeline
runs untouched: pages land in the fixed-length STM queue (capacity 7 in the
paper), FIFO-evict into the segmented-paging MTM where a page joins a segment
when ``F_score = cos(e_s, e_p) + Jaccard(K_s, K_p) > theta``, and segments whose
``Heat = alpha*N_visit + beta*L_interaction + gamma*R_recency`` crosses the
threshold are distilled into the LPM (user profile / user KB / assistant
traits). Additive across DynamicMem checkpoints — each BUILD call feeds only the
newly-visible delta.

RETRIEVE: ``Retriever.retrieve_context(query)`` — the paper's three-tier read:
all of STM, a two-stage MTM search (top-m segments by the same F_score, then
top-k pages by semantic similarity, updating N_visit/R_recency as a side effect),
and the LPM's top-10 user-knowledge and assistant-knowledge entries. Returned as
``{"passages": [...]}`` for the SHARED QA agent — ``use_memory_to_answer`` is NOT
overridden, so MemoryOS's own ``get_response`` generator is deliberately unused
(hipporag2/amem/simplemem pattern; keeps the comparison about memory rather than
about each method's answerer).

PAGE MODEL. MemoryOS stores a page as a (user_input, agent_response) PAIR, not a
single utterance — its updater, heat accounting and prompts all assume that
shape. Ingestion therefore pairs consecutive turns rather than emitting one page
per turn, which is also what the upstream LoCoMo scripts do:
  locomo ("conversation"): consecutive turns of the two speakers pair up;
    a trailing odd turn becomes a page with an empty agent_response.
  longmemeval ("sessions"): role user -> user_input, role assistant ->
    agent_response, paired in order within a session.
  dynamicmem ("app_logs"): no dialogue exists, so each log becomes a page with
    hipporag2's app_log text as user_input and an empty agent_response
    (identical passage content across baselines).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.memo_class import MemoClass
from baselines.harness.concurrency import model_load_lock, quiet_stdout, serialize_calls
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.model_config import (
    get_embedder, install_openai_param_normalisation,
)

from common.openai_usage import install as _install_openai_usage

# The vendored utils.py builds its own `openai` client; patching the SDK
# boundary captures its calls with ZERO edits under src/.
_install_openai_usage()
# NOTE: the vendored utils.py imports sentence-transformers, which eagerly
# imports HuggingFace `datasets`. That used to collide with memevol's own
# top-level `datasets/` package and needed a sys.modules shim; the package was
# renamed to `benchmarks/` (2026-08-07), so plain imports are correct now.

# MUST precede the vendored import: utils.chat_completion hardcodes temperature
# + max_tokens on every call, which the gpt-5 family rejects.
# (No embedder-constructor patch here — see _ensure_system.)
install_openai_param_normalisation()

_SRC = Path(__file__).resolve().parent / "src"
# The vendored modules import each other ABSOLUTELY (`from long_term import ...`,
# `import prompts`), so the package directory itself has to be importable — not
# just its parent. Both paths go on sys.path, innermost first.
for _p in (str(_SRC / "memoryos"), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memoryos import Memoryos            # noqa: E402  (vendored, byte-identical)
from memoryos import utils as _mos_utils  # noqa: E402  (for the embedder seed below)
from memoryos import long_term as _mos_long_term, mid_term as _mos_mid_term  # noqa: E402

# The hooks run on worker threads, so several users call `get_embedding` at
# once. Its process-global `_embedding_cache` is not thread-safe: a hit is
# `if key in cache: return cache[key]`, and past 10000 entries it evicts by
# snapshotting the first 1000 keys and `del`-ing them one by one — two threads
# evicting together delete the same key twice (KeyError), and an eviction
# between another thread's `in` and `[]` fails that lookup. One lock around the
# whole function makes it safe without editing utils.py. long_term / mid_term
# imported the function by name, so their references are swapped too. The
# cost is small: local encodes are serialized anyway and the LLM calls around
# them still overlap.
serialize_calls(_mos_utils, "get_embedding")
_mos_long_term.get_embedding = _mos_mid_term.get_embedding = _mos_utils.get_embedding

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"

# The embedder name MemoryOS's vendored code always asks for. `get_embedding`
# carries it as a DEFAULT ARGUMENT (`utils.py::get_embedding`) and every call
# site takes the default, so it is the cache key regardless of what we configure.
_VENDORED_EMBEDDER_KEY = "all-MiniLM-L6-v2"


def _seed_embedder(model_name: str) -> None:
    """Give MemoryOS the configured embedder by pre-filling its own model cache.

    MemoryOS exposes no embedder parameter — `get_embedding(text, model_name=
    "all-MiniLM-L6-v2")` builds one behind a process-global dict:

        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)

    Seeding that dict under the name the vendored code asks for means the
    `if` never fires and every call site (long_term, mid_term, updater) uses
    what we put there — including an :class:`APIEmbedder` for a
    `text-embedding-*` name. Editing `utils.py` would break the ``src/``
    byte-identity, and patching the global ``SentenceTransformer`` constructor
    (what amem/lightmem/simplemem must do, since they pass the configured name
    through) would be a process-wide swap for what is really one dict entry.

    Idempotent and safe to call per user: `get_embedder` memoizes, so every
    instance shares one embedder rather than reloading the weights per
    conversation. `get_embedding`'s own embedding cache keys on the default name
    too, which stays correct because one process only ever holds one embedder.
    """
    if _VENDORED_EMBEDDER_KEY not in _mos_utils._model_cache:
        _mos_utils._model_cache[_VENDORED_EMBEDDER_KEY] = get_embedder(model_name)


def _pairs_from_init(init: Dict) -> List[Tuple[str, str, str]]:
    """recorder.init -> [(user_input, agent_response, timestamp)] dialogue pages."""
    out: List[Tuple[str, str, str]] = []

    if init.get("app_logs") is not None:
        for entry in init["app_logs"]:
            if isinstance(entry, dict):
                out.append((app_log_to_passage(entry), "", str(entry.get("timestamp", ""))))
        return out

    if init.get("conversation") is not None:
        conv = init["conversation"]

        def _session_no(key: str) -> int:
            try:
                return int(key.split("_")[1])
            except (IndexError, ValueError):
                return 0

        for key in sorted((k for k in conv
                           if k.startswith("session_") and not k.endswith("_date_time")),
                          key=_session_no):
            turns = conv.get(key) or []
            date = str(conv.get(f"{key}_date_time", ""))
            buf: List[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text", ""))
                caption = str(turn.get("blip_caption") or "").strip()
                if caption:
                    text = f"{text} [shared image: {caption}]" if text else f"[shared image: {caption}]"
                buf.append(f"{turn.get('speaker', '?')}: {text}")
                if len(buf) == 2:
                    out.append((buf[0], buf[1], date))
                    buf = []
            if buf:
                out.append((buf[0], "", date))
        return out

    if init.get("sessions") is not None:
        for sess in init["sessions"]:
            if not isinstance(sess, dict):
                continue
            date = str(sess.get("date", ""))
            pending: Optional[str] = None
            for msg in (sess.get("messages") or []):
                if not isinstance(msg, dict):
                    continue
                content = str(msg.get("content", ""))
                if str(msg.get("role", "")).lower() == "assistant" and pending is not None:
                    out.append((pending, content, date))
                    pending = None
                else:
                    if pending is not None:
                        out.append((pending, "", date))
                    pending = content
            if pending is not None:
                out.append((pending, "", date))
        return out

    return out


def _page_to_passage(page: Dict[str, Any]) -> str:
    """One retrieved MTM page rendered for the shared QA agent."""
    when = page.get("timestamp") or ""
    user = page.get("user_input") or ""
    agent = page.get("agent_response") or ""
    head = f"[{when}] " if when else ""
    body = f"{user}\n{agent}".strip()
    return f"{head}{body}"


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
# Vendored @ memoryos-pro 0.1.0. Paper = arXiv 2506.06326 §4.1
# "Implementation Details". Where the shipped code and the paper disagree,
# the paper's value is the default here and the code's is named in the
# comment (this repo's chosen values — see README).
CONFIG_DEFAULTS = {
    # PAPER Tables 1-2: GPT-4o-mini is the headline backbone on both GVD and
    # LoCoMo (the paper also reports Qwen2.5-7B/3B). Drives page/segment
    # summarisation, keyword extraction, persona + knowledge distillation.
    # Its vendored chat_completion hardcodes temperature+max_tokens, which
    # the gpt-5 family rejects — model_config normalises those away, so a
    # gpt-5 model IS runnable here.
    "memoryos_llm_model": "gpt-4o-mini",
    # THE PAPER NAMES NO EMBEDDER — §4.1 covers hardware and the
    # STM/MTM/LPM capacities but never states an embedding model. This is
    # therefore the VENDORED CODE's value (utils.get_embedding's default
    # argument), the only evidence available. 384-dim, local. Because that
    # default argument means the requested name is never the configured
    # one, the key is applied by seeding MemoryOS's own model cache under
    # the requested name (_seed_embedder). A `text-embedding-*` name
    # switches to the OpenAI API embedder; no dimension knob is needed
    # (MemoryOS sizes its FAISS indexes from the embedding array itself).
    "memoryos_embedding_model": "all-MiniLM-L6-v2",
    "base_url": None,              # OpenAI-compatible base URL for the internal LLM (None = OpenAI default)
    "short_term_capacity": 7,      # STM dialogue-page queue length. Paper: 7 (vendored default is 10)
    # Max MTM segments before LFU eviction. PAPER: 200. The vendored default
    # is 2000, which never binds — one LoCoMo conversation produces ~176
    # segments, so 200 evicts and 2000 does not.
    "mid_term_capacity": 200,
    # tau: Heat = a*N_visit + b*L_interaction + c*R_recency above which a
    # segment is distilled into the LPM. Paper: 5 (a=b=c=1)
    "mid_term_heat_threshold": 5.0,
    # theta in `F_score = cos(e_s,e_p) + Jaccard(K_s,K_p) > theta` for
    # merging a page into a segment. Paper: 0.6
    "mid_term_similarity_threshold": 0.6,
    # MTM pages returned per query (the paper's top-k). PAPER: 10 on LoCoMo
    # (5 on GVD). The vendored default is 7.
    "retrieval_queue_capacity": 10,
    "long_term_knowledge_capacity": 100,   # FIFO capacity of User KB / Assistant Traits. Paper: 100
}
# `arm: unified` writes unified_models.llm / .embedding into these keys.
# MemoryOS publishes its headline LoCoMo numbers on gpt-4o-mini with a local
# all-MiniLM-L6-v2 index; both change. Do not quote the unified arm as
# MemoryOS's published result.
UNIFIED_MODEL_KEYS = {"llm": ("memoryos_llm_model",), "embedding": ("memoryos_embedding_model",)}


class MemoryOSMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._memo: Optional[Memoryos] = None
        self._instance_id = uuid.uuid4().hex[:12]   # per-user on-disk store

    def _ensure_system(self) -> None:
        if self._memo is not None:
            return
        # One user at a time: construction loads local models (concurrency.model_load_lock).
        with model_load_lock:
            cfg = self.config
            _seed_embedder(cfg["memoryos_embedding_model"])
            save_dir = OUTPUTS_DIR / self._instance_id
            if save_dir.exists():
                shutil.rmtree(save_dir, ignore_errors=True)
            save_dir.mkdir(parents=True, exist_ok=True)
            with quiet_stdout():
                self._memo = Memoryos(
                    user_id=f"u_{self._instance_id}",
                    assistant_id=f"a_{self._instance_id}",
                    openai_api_key=None,   # credential: the OpenAI SDK reads it from the environment
                    openai_base_url=cfg["base_url"] or "",
                    data_storage_path=str(save_dir),
                    llm_model=cfg["memoryos_llm_model"],
                    short_term_capacity=cfg["short_term_capacity"],
                    mid_term_capacity=cfg["mid_term_capacity"],
                    mid_term_heat_threshold=cfg["mid_term_heat_threshold"],
                    mid_term_similarity_threshold=cfg["mid_term_similarity_threshold"],
                    retrieval_queue_capacity=cfg["retrieval_queue_capacity"],
                    long_term_knowledge_capacity=cfg["long_term_knowledge_capacity"],
                )
        # No dimension knob is needed: MemoryOS sizes its FAISS indexes from the
        # embedding array itself (`dim = embeddings_np.shape[1]`, long_term.py
        # and mid_term.py), so a 1536-dim API embedder drops straight in.

    # MemoryOS is synchronous (LLM summarisation, embedding, FAISS, JSON
    # persistence): each hook runs its body on a worker thread so other users keep
    # going meanwhile (see baselines/harness/concurrency.py).

    async def build_memory_from_data(self, recorder) -> None:
        await asyncio.to_thread(self._build, recorder)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        return await asyncio.to_thread(self._retrieve, recorder)

    def _build(self, recorder) -> None:
        self._ensure_system()
        pages = _pairs_from_init(recorder.init)
        # add_memory drives the whole update chain (STM append -> FIFO ->
        # MTM segmentation -> heat -> LPM distillation), including its LLM calls.
        with quiet_stdout():
            for user_input, agent_response, when in pages:
                self._memo.add_memory(user_input=user_input,
                                      agent_response=agent_response,
                                      timestamp=when or None)

    def _retrieve(self, recorder) -> Dict:
        self._ensure_system()
        query = str(recorder.init.get("query", ""))
        with quiet_stdout():
            res = self._memo.retriever.retrieve_context(user_query=query,
                                                        user_id=self._memo.user_id)
            stm = self._memo.short_term_memory.get_all()

        passages: List[str] = []
        # STM: the paper returns ALL of it (it is the recent-context tier).
        for qa in stm or []:
            passages.append(_page_to_passage(qa))
        # MTM: top-m segments -> top-k pages.
        for page in res.get("retrieved_pages") or []:
            passages.append(_page_to_passage(page))
        # LPM: top-10 user-knowledge + assistant-knowledge entries.
        for entry in res.get("retrieved_user_knowledge") or []:
            passages.append(f"[user knowledge] {entry.get('knowledge', '')}")
        for entry in res.get("retrieved_assistant_knowledge") or []:
            passages.append(f"[assistant knowledge] {entry.get('knowledge', '')}")

        return {"passages": passages} if passages else {}
