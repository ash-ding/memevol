"""Zep (Graphiti temporal knowledge graph, arXiv:2501.13956) as a retrieval
MemoClass.

BUILD: every ingestion unit becomes one Graphiti **episode** via
`add_episode(episode_body, reference_time, source, ...)` — Graphiti's own pipeline
runs untouched (LLM entity extraction + reflection, entity resolution/dedup, fact
extraction, bi-temporal edge extraction + invalidation, BGE-m3 embedding). This is
the paper's LongMemEval protocol (§4): "integrate the conversation history into a
Zep knowledge graph through Zep's APIs" — one episode per message. Additive across
calls (Graphiti persists in the per-user on-disk FalkorDB Lite store), so
DynamicMem's per-checkpoint delta builds accumulate correctly.

RETRIEVE: `search_(query, COMBINED_HYBRID_SEARCH_CROSS_ENCODER, limit=k)` — the
paper's retrieval (§3): hybrid BM25 + cosine + breadth-first search over edges and
nodes, reranked by the BGE cross-encoder. The paper retrieves the top-20 edges
(facts) and entity nodes (summaries) and reformats them into a FACTS/ENTITIES
context string (§3, template replicated in `_format_context` since the
compose-context step lives in Zep's hosted service, not OSS Graphiti). Read-only.
The shared QA agent answers — `use_memory_to_answer` is NOT overridden (the paper
uses a separate chat agent over the retrieved context; hipporag2/amem pattern).

Backend: **FalkorDB Lite** (embedded, in-process, on-disk, no server), scoped per
instance (`uuid` dbfilename + group_id) — see _st_shim / README faithfulness
boundary for the Neo4j-Lucene vs RediSearch retrieval-backend deviation.

Ingestion units (recorder.init dispatch, cf. hipporag2/amem):
  locomo ("conversation"): one episode per turn, "{speaker}: {text}" (message
    type — Graphiti auto-extracts the speaker as an entity), reference_time =
    session date.  longmemeval ("sessions"): one per message, "{role}: {content}",
    reference_time = session date.  dynamicmem ("app_logs"): one per log entry,
    content = hipporag2's app_log_to_passage text (identical across baselines),
    text type, reference_time = log timestamp.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.memo_class import MemoClass
from common.openai_usage import install as _install_openai_usage
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.zep import _st_shim

# Graphiti calls the OpenAI SDK directly from 7 files under src/. Patching the
# SDK boundary captures its graph-construction traffic with ZERO edits under
# src/ (byte-identity per the README's `diff -r`).
_install_openai_usage()

_st_shim.ensure_src_on_path()          # `import graphiti_core` -> vendored copy
_st_shim.ensure_sentence_transformers()   # before any graphiti cross_encoder import

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.llm_client import LLMConfig, OpenAIClient  # noqa: E402
from graphiti_core.search.search_config_recipes import (  # noqa: E402
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)

from baselines.harness.zep._bge_embedder import BGEM3Embedder, CachedBGEReranker  # noqa: E402

# The embedded FalkorDB Lite (redislite) store MUST live on a native POSIX
# filesystem: redislite starts a redis-server bound to a UNIX SOCKET next to the
# db file, and unix sockets are unsupported on WSL's DrvFs (/mnt/c, ...) and other
# network/9p mounts → "redis-server process failed to start". Default to the
# system temp dir (ext4 on WSL2, tmpfs/local on Linux); override via cfg["db_root"].
# NOT the repo's outputs/ (that sits on /mnt/c under WSL).
def _db_root(cfg: Dict) -> Path:
    return Path(cfg.get("db_root") or tempfile.gettempdir()) / "zep_falkordb"

# Paper §3 context template (structure verbatim); the compose step is a Zep
# service feature, replicated here because OSS Graphiti stops at search results.
_CONTEXT_TEMPLATE = (
    "FACTS and ENTITIES represent relevant context to the current conversation.\n"
    "These are the most relevant facts and their valid date ranges. If the fact "
    "is about an event, the event takes place during this time.\n"
    "format: FACT (Date range: from - to)\n"
    "<FACTS>\n{facts}\n</FACTS>\n"
    "These are the most relevant entities\n"
    "ENTITY_NAME: entity summary\n"
    "<ENTITIES>\n{entities}\n</ENTITIES>"
)


def _parse_dt(value: Any) -> datetime:
    """Best-effort parse of a dataset timestamp into a tz-aware datetime
    (add_episode requires reference_time; Graphiti resolves relative dates against
    it). Falls back to now(UTC) when a value is missing/unparseable — a faithful
    default (the paper anchors episodes on the message's own send time)."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value or "").strip()
    if s:
        try:
            from dateutil import parser as _p
            # fuzzy=True so extra tokens don't abort the parse — notably
            # LongMemEval's "2023/05/20 (Sat) 02:21" weekday-in-parens (without
            # fuzzy, dateutil raises on "(Sat)" → every episode would fall back to
            # now(), losing the real session date Graphiti anchors temporal facts
            # on). LoCoMo ("7:00 pm on 20 May, 2023") + dynamicmem (ISO) parse
            # either way; fuzzy is a safe superset for those.
            dt = _p.parse(s, fuzzy=True)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            try:
                dt = datetime.fromisoformat(s)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _init_to_episodes(init: Dict) -> List[Dict[str, Any]]:
    """recorder.init → ordered list of episode kwargs for add_episode."""
    if "app_logs" in init:
        return [
            {"name": str(e.get("app_log_id", f"app_log_{i}")),
             "body": app_log_to_passage(e),
             "source": EpisodeType.text,
             "source_description": "app log",
             "reference_time": _parse_dt(e.get("timestamp", ""))}
            for i, e in enumerate(init["app_logs"])
        ]
    if "conversation" in init:
        from benchmarks.locomo.env import extract_sessions
        eps: List[Dict[str, Any]] = []
        for idx, date_time, turns in extract_sessions(init["conversation"]):
            ref = _parse_dt(date_time)
            for t in turns:
                eps.append({
                    "name": str(t.get("dia_id", f"session_{idx}")),
                    "body": f"{t.get('speaker', '')}: {t.get('text', '')}",
                    "source": EpisodeType.message,
                    "source_description": "conversation message",
                    "reference_time": ref,
                })
        return eps
    if "sessions" in init:
        eps = []
        for s in init["sessions"]:
            ref = _parse_dt(s.get("date", ""))
            sid = s.get("session_id", "session")
            for i, m in enumerate(s.get("messages", [])):
                eps.append({
                    "name": f"{sid}_{i}",
                    "body": f"{m.get('role', '')}: {m.get('content', '')}",
                    "source": EpisodeType.message,
                    "source_description": "chat session message",
                    "reference_time": ref,
                })
        return eps
    raise KeyError(f"unrecognized recorder.init keys: {list(init)}")


def _fmt_dt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.date().isoformat()
    except Exception:
        return str(dt)


def _format_context(edges: List[Any], nodes: List[Any]) -> str:
    """Render Graphiti search results into the paper's FACTS/ENTITIES template."""
    facts: List[str] = []
    for e in edges:
        va, ia = _fmt_dt(getattr(e, "valid_at", None)), _fmt_dt(getattr(e, "invalid_at", None))
        rng = f" (Date range: {va or '...'} - {ia or 'present'})" if (va or ia) else ""
        facts.append(f"  - {getattr(e, 'fact', '')}{rng}")
    entities: List[str] = []
    for n in nodes:
        summary = (getattr(n, "summary", "") or "").strip()
        name = getattr(n, "name", "")
        entities.append(f"  {name}: {summary}" if summary else f"  {name}")
    if not facts and not entities:
        return ""
    return _CONTEXT_TEMPLATE.format(facts="\n".join(facts), entities="\n".join(entities))


class ZepMemo(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)
        # Per-user isolation: a fresh instance == one user (no cross-user state —
        # recorder.user_id is always "" at memo call sites). Scope the embedded
        # on-disk FalkorDB Lite store AND the Graphiti group_id on this uuid.
        self._instance_id = uuid.uuid4().hex[:12]
        self._gid = self._instance_id
        self._graphiti: Optional[Graphiti] = None
        self._falkor_db = None
        self._db_path: Optional[str] = None

    async def _ensure(self) -> None:
        if self._graphiti is not None:
            return
        from redislite.async_falkordb_client import AsyncFalkorDB   # embedded, async

        base = _db_root(self.config)   # native POSIX FS — see _db_root (NOT /mnt/c under WSL)
        base.mkdir(parents=True, exist_ok=True)
        self._db_path = str(base / f"{self._instance_id}.db")
        self._falkor_db = AsyncFalkorDB(dbfilename=self._db_path)
        driver = FalkorDriver(falkor_db=self._falkor_db)

        device = self.config.get("device")
        # Embedder: paper-faithful BGE-m3 (default) or OpenAI (fallback).
        if self.config.get("embedder", "bge-m3") == "openai":
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            embedder = OpenAIEmbedder(OpenAIEmbedderConfig(
                embedding_model=self.config.get("embedder_model", "text-embedding-3-small")))
        else:
            embedder = BGEM3Embedder(self.config.get("embedder_model", "BAAI/bge-m3"), device=device)

        graph_model = self.config.get("graph_llm_model", "gpt-4o-mini")
        # Reranker: paper-faithful BGE cross-encoder (default) or OpenAI (fallback).
        if self.config.get("reranker", "bge") == "openai":
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            cross_encoder = OpenAIRerankerClient(config=LLMConfig(model=graph_model))
        else:
            cross_encoder = CachedBGEReranker(
                self.config.get("reranker_model", "BAAI/bge-reranker-v2-m3"), device=device)

        # Internal graph-construction LLM. Paper: gpt-4o-mini-2024-07-18 (4-series).
        # NOTE: Graphiti uses the openai SDK directly, so these calls do NOT flow
        # through common.tokens (same caveat as amem/hipporag2/simplemem/lightmem).
        llm_client = OpenAIClient(config=LLMConfig(
            model=graph_model,
            small_model=self.config.get("graph_llm_small_model") or graph_model,
        ))

        self._graphiti = Graphiti(
            graph_driver=driver, llm_client=llm_client,
            embedder=embedder, cross_encoder=cross_encoder,
        )
        await self._graphiti.build_indices_and_constraints()

    async def build_memory_from_data(self, recorder) -> None:
        await self._ensure()
        # Sequential ingestion: Graphiti dedups/resolves each episode against the
        # accumulated graph, so episodes must be added one at a time (concurrent
        # add_episode on one graph would corrupt entity/edge resolution). Faithful
        # to the paper's sequential ingestion.
        for ep in _init_to_episodes(recorder.init):
            await self._graphiti.add_episode(
                name=ep["name"], episode_body=ep["body"],
                source_description=ep["source_description"],
                reference_time=ep["reference_time"], source=ep["source"],
                group_id=self._gid,
            )

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        await self._ensure()
        query = recorder.init.get("query", "")
        if not query:
            return {}
        k = int(self.config.get("retrieve_k", 20))   # paper: top-20 edges + nodes
        config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
        config.limit = k
        results = await self._graphiti.search_(query, config=config, group_ids=[self._gid])
        ctx = _format_context(results.edges[:k], results.nodes[:k])
        if not ctx:
            return {}
        return {"inline_memory_blocks": [ctx]}

    def __del__(self):
        # Best-effort teardown of the per-instance embedded store (one instance ==
        # one user, so this never touches another user's data). The FalkorDB Lite
        # subprocess is managed by redislite's own atexit cleanup; we additionally
        # try a sync close and remove the on-disk db file. Guarded against
        # interpreter-shutdown teardown ordering.
        db = getattr(self, "_falkor_db", None)
        if db is not None:
            for name in ("close", "shutdown"):
                fn = getattr(db, name, None)
                if callable(fn):
                    try:
                        res = fn()
                        if hasattr(res, "__await__"):   # coroutine — can't await in __del__
                            res.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    break
        path = getattr(self, "_db_path", None)
        if path:
            for p in (path, path + ".dir"):
                try:
                    if Path(p).is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
