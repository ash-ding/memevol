"""HippoRAG2 as a retrieval MemoClass: Phase 1 indexes the ingested unit's
passages into a per-user HippoRAG graph; Phase 2 retrieves top-k passages. The
shared QA agent answers from those passages (fair 'HippoRAG-as-memory'
comparison), and the per-dataset workflow judges/scores identically to the main
method.

HippoRAG API notes (verified against the installed `hipporag==2.0.0a4` package
at /export/scratch_large/ding/code/HippoRAG/src/hipporag/HippoRAG.py):
  - `HippoRAG.index(docs=...)` IS additive across calls: it delegates to
    `EmbeddingStore.insert_strings`, which dedups by content hash and upserts
    only the genuinely-new strings (see embedding_store.py:63-90); OpenIE is
    likewise only re-run for chunk ids not already indexed. So each
    `build_memory_from_data` call only needs to pass the NEW segment (already true for
    DynamicMem's per-checkpoint slices — `datasets/dynamicmem/workflow.py`
    passes `app_logs[prev_end:len(visible)]`, a non-overlapping suffix) rather
    than re-indexing the full accumulated `self._passages`.
  - `HippoRAG.retrieve(queries=, num_to_retrieve=, gold_docs=None)` exists and
    (with the default `gold_docs=None`) returns `List[QuerySolution]`, each
    with a `.docs` attribute (top-k passage strings) — no native answer
    involved. Used directly; the `rag_qa(...).docs` path is kept as a fallback
    for older/alternate HippoRAG builds that lack `retrieve`, but is not
    exercised by the installed version.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, List

from common.memo_class import MemoClass

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


def app_log_to_passage(log_entry: dict) -> str:
    # verbatim from the old eval_hipporag2.py (lines 99-113)
    ts = log_entry.get("timestamp", ""); app = log_entry.get("app_name", "")
    api = log_entry.get("api_name", "")
    req = json.dumps(log_entry.get("request", {}), ensure_ascii=False)
    resp = json.dumps(log_entry.get("response", {}), ensure_ascii=False)
    domain = log_entry.get("metadata", {}).get("domain", "")
    return (f"[{ts}] App: {app}, Action: {api}\nDomain: {domain}\n"
            f"Request: {req}\nResponse: {resp}")


def _init_to_passages(init: Dict) -> List[str]:
    if "app_logs" in init:
        return [app_log_to_passage(e) for e in init["app_logs"]]
    if "conversation" in init:
        from datasets.locomo.env import extract_sessions
        conv = init["conversation"]
        out = []
        # extract_sessions returns (session_idx, date_time, turns) tuples —
        # NOT bare turn lists — so date_time must be unpacked alongside turns.
        for _idx, date_time, turns in extract_sessions(conv):
            for t in turns:
                out.append(f"[{date_time}] {t.get('speaker', '')}: {t.get('text', '')}")
        return out
    if "sessions" in init:
        out = []
        for s in init["sessions"]:
            for m in s.get("messages", []):
                out.append(f"[{s.get('date', '')}] {m.get('role', '')}: {m.get('content', '')}")
        return out
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


class HippoRAGMemo(MemoClass):
    def __init__(self, config=None):
        super().__init__(config)
        self._hippo = None
        self._passages: List[str] = []
        # NOTE: the recorders actually handed to build_memory_from_data/retrieve_memory_for_query
        # by common/workflow.py and datasets/*/workflow.py are throwaway
        # `self.recorder_class()` instances that are NEVER given `.user_id`
        # (only a separate bookkeeping recorder used for trace/step logging
        # gets `user_id = user_tag` — verified across common/workflow.py:490-508,
        # datasets/dynamicmem/workflow.py:107-208, datasets/locomo/workflow.py:95).
        # So `recorder.user_id` is always the dataclass default "" in practice —
        # keying save_dir on it would collapse every user's HippoRAG graph onto
        # the same path (silent cross-user contamination under concurrent
        # eval). Instead rely on the documented invariant "a fresh MemoClass
        # instance is created per user — no cross-user state" and key save_dir
        # on an instance-scoped id generated once here.
        self._instance_id = uuid.uuid4().hex[:12]

    def _ensure_hippo(self):
        if self._hippo is not None:
            return
        cfg = self.config
        factory = cfg.get("_hippo_factory")
        embedding = cfg["embedding"]
        save_dir = str(OUTPUTS_DIR / f"{self._instance_id}_{embedding.replace('/', '_')}")
        if factory is not None:
            self._hippo = factory(save_dir=save_dir)
            return
        from hipporag import HippoRAG
        from hipporag.utils.config_utils import BaseConfig
        is_local = "text-embedding" not in embedding
        conf = BaseConfig(
            llm_name=cfg["llm_model"], embedding_model_name=embedding,
            save_dir=save_dir, response_format=None, temperature=1, seed=None,
            embedding_batch_size=cfg.get("embedding_batch_size") or (4 if is_local else 16),
            embedding_model_dtype=cfg.get("embedding_dtype") or ("float16" if is_local else "auto"),
        )
        self._hippo = HippoRAG(global_config=conf)

    async def build_memory_from_data(self, recorder) -> None:
        # Called once (all_at_once) for locomo/longmemeval; per checkpoint for
        # DynamicMem TCE — accumulate + index each new segment.
        self._ensure_hippo()
        new = _init_to_passages(recorder.init)
        self._passages.extend(new)
        if new:
            self._hippo.index(docs=new)   # HippoRAG.index is additive (verified: dedup-by-hash upsert)

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        self._ensure_hippo()   # defensive no-op if build_memory_from_data already ran
        query = recorder.init.get("query", "")
        k = int(self.config.get("top_k", 5))
        # Prefer retrieve-only; fall back to rag_qa(...).docs if absent.
        if hasattr(self._hippo, "retrieve"):
            sols = self._hippo.retrieve(queries=[query], num_to_retrieve=k)
            docs = list(getattr(sols[0], "docs", []))[:k] if sols else []
        else:
            sols, _msgs, _meta = self._hippo.rag_qa(queries=[query])
            docs = list(getattr(sols[0], "docs", []))[:k] if sols else []
        return {"passages": docs}
