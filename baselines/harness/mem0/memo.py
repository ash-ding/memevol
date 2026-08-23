"""Mem0 (https://github.com/mem0ai/mem0) as a retrieval MemoClass.

BUILD: ingestion units are batched into ``Memory.add(messages, user_id=...)``,
which is Mem0's actual contribution — an LLM reads each batch, EXTRACTS
standalone facts from it, and then decides per fact whether to ADD it, UPDATE an
existing one, or DELETE a contradicted one against what is already stored. The
memory therefore holds distilled facts ("User adopted a rescue dog named Pico, a
beagle mix"), not raw turns. Additive across DynamicMem checkpoints: each BUILD
call passes only the newly-visible delta.

RETRIEVE: ``Memory.search(query, filters={"user_id": ...}, top_k=...)`` — vector
search over those facts. Returned as ``{"passages": [...]}`` for the SHARED QA
agent; ``use_memory_to_answer`` is NOT overridden (hipporag2/amem/simplemem
pattern), so the comparison stays about the memory rather than each method's own
answerer.

ISOLATION: one Mem0 instance per user, each with its own on-disk Qdrant
collection and its own history DB, so nothing leaks between conversations. The
``user_id`` filter is redundant given that, and is passed anyway because Mem0's
update/delete bookkeeping keys on it.

Ingestion units (recorder.init dispatch, cf. hipporag2's _init_to_passages):
  locomo ("conversation"): one message per turn, speaker folded into the text
    (Mem0 has no speaker field), time = session date_time.
  longmemeval ("sessions"): one message per message, role preserved.
  dynamicmem ("app_logs"): one message per log entry, content = hipporag2's
    app_log_to_passage text (identical passage content across baselines).
"""
from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.memo_class import MemoClass
from common.store_cache import DiskStoreCache
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.model_config import install_openai_param_normalisation

# Mem0's OpenAI provider sends temperature + max_tokens on every extractor call,
# which the gpt-5 family rejects. Installed before the mem0 import so the patch
# is in place regardless of when mem0 builds its clients. No embedder factory:
# mem0 is already on an API embedder (`embedder.provider: openai`).
install_openai_param_normalisation()

# `import mem0` must resolve to the byte-identical vendored copy under src/, not
# to any pip-installed mem0ai (there is none in this baseline's env — mem0ai was
# dropped from pyproject.toml when the source was vendored). Prepended, so the
# vendored copy wins even if one is installed in an ambient env.
# (zep/memoryos pattern; no _st_shim.py needed — nothing here imports HF datasets.)
_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Must be set BEFORE importing mem0: the flag is read at module import time
# (mem0/memory/telemetry.py), and with it on, every Memory() additionally opens a
# telemetry vector store at the process-global path ~/.mem0/migrations_qdrant.
# Local-mode Qdrant takes an exclusive lock on a storage folder, so the second
# concurrent user in the same process dies with "already accessed by another
# instance of Qdrant client" — which is what happens here, since the workflow
# builds several users' memories concurrently. Off is also simply correct: a
# benchmark run should not phone the evaluation home.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from mem0 import Memory  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def _init_to_messages(init: Dict) -> List[Dict[str, str]]:
    """recorder.init -> Mem0 chat messages."""
    out: List[Dict[str, str]] = []

    if init.get("app_logs") is not None:
        for entry in init["app_logs"]:
            if isinstance(entry, dict):
                out.append({"role": "user", "content": app_log_to_passage(entry)})
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
            date = str(conv.get(f"{key}_date_time", ""))
            for turn in (conv.get(key) or []):
                if not isinstance(turn, dict):
                    continue
                text = str(turn.get("text", ""))
                caption = str(turn.get("blip_caption") or "").strip()
                if caption:
                    text = f"{text} [shared image: {caption}]" if text else f"[shared image: {caption}]"
                # Speaker and date go INTO the content: Mem0's extractor only
                # sees role+content, so dropping them would make every
                # "who said it / when" question unanswerable from memory.
                out.append({"role": "user",
                            "content": f"[{date}] {turn.get('speaker', '?')}: {text}"})
        return out

    if init.get("sessions") is not None:
        for sess in init["sessions"]:
            if not isinstance(sess, dict):
                continue
            date = str(sess.get("date", ""))
            for msg in (sess.get("messages") or []):
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user")).lower()
                role = role if role in ("user", "assistant", "system") else "user"
                out.append({"role": role, "content": f"[{date}] {msg.get('content', '')}"})
        return out

    return out


class Mem0Memo(DiskStoreCache, MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        self._memory: Optional[Memory] = None
        self._instance_id = uuid.uuid4().hex[:12]
        self._user_id = f"u_{self._instance_id}"

    # -- memory-cache hooks (common/store_cache.py) --
    _store_handle = "_memory"

    def _store_path(self):
        """Per-user store: the Qdrant collection AND the history DB live here."""
        return OUTPUTS_DIR / self._instance_id

    def _ensure_system(self) -> None:
        if self._memory is not None:
            return
        cfg = self.config
        store = OUTPUTS_DIR / self._instance_id
        # Never wipe a store restored from the memory cache (DiskStoreCache).
        if store.exists() and not self.restored_from_cache:
            shutil.rmtree(store, ignore_errors=True)
        store.mkdir(parents=True, exist_ok=True)
        llm_conf: Dict[str, Any] = {"model": cfg.get("mem0_llm_model")}
        if cfg.get("base_url"):
            llm_conf["openai_base_url"] = cfg["base_url"]
        self._memory = Memory.from_config({
            # Embedded Qdrant on disk — per-user collection, no server process.
            "vector_store": {"provider": "qdrant", "config": {
                "collection_name": f"c_{self._instance_id}",
                "path": str(store / "qdrant"),
                "on_disk": True,
            }},
            "llm": {"provider": "openai", "config": llm_conf},
            "embedder": {"provider": "openai",
                         "config": {"model": cfg.get("embedding_model")}},
            "history_db_path": str(store / "history.db"),
        })

    async def build_memory_from_data(self, recorder) -> None:
        self._ensure_system()
        messages = _init_to_messages(recorder.init)
        if not messages:
            return
        # Batched rather than one add() per turn: Mem0's extractor reads a whole
        # message list at once, so batching is both its intended usage and the
        # difference between a few dozen LLM calls and several hundred. The
        # batch size is a knob because the extraction prompt has to fit.
        size = max(1, int(self.config.get("add_batch_size") or 20))
        for i in range(0, len(messages), size):
            self._memory.add(messages[i:i + size], user_id=self._user_id,
                             infer=bool(self.config.get("infer", True)))

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_system()
        query = str(recorder.init.get("query", ""))
        res = self._memory.search(
            query,
            filters={"user_id": self._user_id},
            top_k=int(self.config.get("top_k") or 20),
            threshold=float(self.config.get("threshold") or 0.0),
        )
        results = res.get("results") if isinstance(res, dict) else res
        passages = [str(r.get("memory", "")) for r in (results or []) if r.get("memory")]
        return {"passages": passages} if passages else {}
