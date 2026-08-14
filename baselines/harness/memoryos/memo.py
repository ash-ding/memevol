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

import contextlib
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.memo_class import MemoClass
from common.openai_usage import install as _install_openai_usage
from baselines.harness.hipporag2.memo import app_log_to_passage

# The vendored utils.py builds its own `openai` client; patching the SDK
# boundary captures its calls with ZERO edits under src/.
_install_openai_usage()

# NOTE: the vendored utils.py imports sentence-transformers, which eagerly
# imports HuggingFace `datasets`. That used to collide with memevol's own
# top-level `datasets/` package and needed a sys.modules shim; the package was
# renamed to `benchmarks/` (2026-08-07), so plain imports are correct now.

_SRC = Path(__file__).resolve().parent / "src"
# The vendored modules import each other ABSOLUTELY (`from long_term import ...`,
# `import prompts`), so the package directory itself has to be importable — not
# just its parent. Both paths go on sys.path, innermost first.
for _p in (str(_SRC / "memoryos"), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from memoryos import Memoryos            # noqa: E402  (vendored, byte-identical)

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


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


class MemoryOSMemo(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        self._memo: Optional[Memoryos] = None
        self._instance_id = uuid.uuid4().hex[:12]   # per-user on-disk store

    def _ensure_system(self) -> None:
        if self._memo is not None:
            return
        cfg = self.config
        save_dir = OUTPUTS_DIR / self._instance_id
        if save_dir.exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull):
            self._memo = Memoryos(
                user_id=f"u_{self._instance_id}",
                assistant_id=f"a_{self._instance_id}",
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                openai_base_url=cfg.get("base_url") or "",
                data_storage_path=str(save_dir),
                llm_model=cfg.get("memoryos_llm_model"),
                short_term_capacity=cfg.get("short_term_capacity"),
                mid_term_capacity=cfg.get("mid_term_capacity"),
                mid_term_heat_threshold=cfg.get("mid_term_heat_threshold"),
                mid_term_similarity_threshold=cfg.get("mid_term_similarity_threshold"),
                retrieval_queue_capacity=cfg.get("retrieval_queue_capacity"),
                long_term_knowledge_capacity=cfg.get("long_term_knowledge_capacity"),
            )
        # The embedder is NOT a constructor argument in this version: utils.py's
        # get_embedding() hardcodes `all-MiniLM-L6-v2` behind a process-global
        # cache. Recorded here so the config cannot imply a knob that does not
        # exist (the README's `embedding_model_name` belongs to a later build).

    async def build_memory_from_data(self, recorder) -> None:
        self._ensure_system()
        pages = _pairs_from_init(recorder.init)
        # add_memory drives the whole update chain (STM append -> FIFO ->
        # MTM segmentation -> heat -> LPM distillation), including its LLM calls.
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull):
            for user_input, agent_response, when in pages:
                self._memo.add_memory(user_input=user_input,
                                      agent_response=agent_response,
                                      timestamp=when or None)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_system()
        query = str(recorder.init.get("query", ""))
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull):
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
