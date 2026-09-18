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
The shared QA agent answers, as it does for every memo (answering is not part of
the contract; the paper likewise uses a separate chat agent over the retrieved
context).

Backend: **FalkorDB Lite** (embedded, in-process, on-disk, no server), scoped per
instance (`uuid` dbfilename + group_id) — see the README faithfulness boundary
for the Neo4j-Lucene vs RediSearch retrieval-backend deviation.

Ingestion units (recorder.init dispatch, cf. hipporag2/amem):
  locomo ("conversation"): one episode per turn, "{speaker}: {text}" (message
    type — Graphiti auto-extracts the speaker as an entity), reference_time =
    session date.  longmemeval ("sessions"): one per message, "{role}: {content}",
    reference_time = session date.  dynamicmem ("app_logs"): one per log entry,
    content = hipporag2's app_log_to_passage text (identical across baselines),
    text type, reference_time = log timestamp.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.memo_class import MemoClass
from common.openai_usage import install as _install_openai_usage
from baselines.harness.hipporag2.memo import app_log_to_passage
from baselines.harness.concurrency import serialize_calls
from baselines.harness.model_config import (
    api_embedding_dims, install_openai_param_normalisation, is_api_embedding_model,
    resolve_device,
)

# `import graphiti_core` must resolve to the byte-identical vendored copy under
# src/, not any pip-installed graphiti-core. Idempotent.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Graphiti's OpenAIClient ALREADY drops `temperature` for the gpt-5 family and
# routes structured output through `responses.parse` — but its plain-JSON path
# still sends `max_tokens` (openai_client.py:126), which those models reject in
# favour of `max_completion_tokens`. The shim renames it. No embedder factory
# here: Graphiti accepts an injected EmbedderClient, so zep configures its
# embedder directly (see _ensure) rather than patching a constructor.
install_openai_param_normalisation()

# Graphiti calls the OpenAI SDK directly from 7 files under src/. Patching the
# SDK boundary captures its graph-construction traffic with ZERO edits under
# src/ (byte-identity per the README's `diff -r`).
_install_openai_usage()

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.llm_client import LLMConfig, OpenAIClient  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig  # noqa: E402
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient  # noqa: E402
from graphiti_core.search.search_config_recipes import (  # noqa: E402
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
import graphiti_core.telemetry.telemetry as _graphiti_telemetry  # noqa: E402

# Graphiti sends an anonymous PostHog event on every Graphiti() construction —
# once per user here — unless GRAPHITI_TELEMETRY_ENABLED says otherwise. A
# benchmark run should not phone home (mem0's is off for the same reason), and
# the switch is set here explicitly rather than through the environment:
# `capture_event` looks this function up at call time.
_graphiti_telemetry.is_telemetry_enabled = lambda: False

# --- Paper-faithful BGE-m3 embedder + cached BGE reranker --------------------
#
# The Zep paper (§4.1) uses BAAI's BGE-m3 for BOTH embedding and reranking.
# graphiti_core ships a BGE *reranker* (cross_encoder/bge_reranker_client.py) but
# NO local/sentence-transformers *embedder* (only openai/azure/gemini/voyage), so
# the embedder is supplied through Graphiti's public `EmbedderClient` extension
# point — integration code, NOT an edit to vendored graphiti_core. This mirrors
# what the paper's authors must have done (BGE-m3 is not in graphiti's OSS list).
#
# A fresh ZepMemo is built per user, so both classes read their model from the
# process-wide caches below: the ~2GB BGE-m3 weights and the BGE reranker load
# ONCE per process, not once per user. Because Graphiti accepts injected clients,
# this is a plain cached factory — nothing is monkeypatched, unlike the
# lightmem/simplemem baselines whose vendored code builds its embedder internally
# with no injection point.

DEFAULT_EMBEDDER_MODEL = "BAAI/bge-m3"          # paper: BGE-m3, 1024-dim
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # graphiti's BGERerankerClient default

# Process-wide model caches, keyed by (model_name, device).
_embedder_cache: dict = {}
_reranker_cache: dict = {}


def get_bge_embedder(model_name: str, device: str | None = None):
    """Cached SentenceTransformer for the BGE-m3 embedder. Constructed once per
    (model_name, device)."""
    key = (model_name, device)
    if key not in _embedder_cache:
        from sentence_transformers import SentenceTransformer
        # Users encode through the default executor concurrently (BGEM3Embedder):
        # one encode at a time on the shared model — see concurrency.serialize_calls.
        _embedder_cache[key] = serialize_calls(SentenceTransformer(model_name, device=device), "encode")
    return _embedder_cache[key]


def get_bge_reranker(model_name: str, device: str | None = None):
    """Cached CrossEncoder for the BGE reranker. Constructed once per
    (model_name, device)."""
    key = (model_name, device)
    if key not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        # graphiti's BGERerankerClient.rank runs `predict` in the default executor,
        # so concurrent users would share it: one predict at a time.
        _reranker_cache[key] = serialize_calls(CrossEncoder(model_name, device=device), "predict")
    return _reranker_cache[key]


class BGEM3Embedder(EmbedderClient):
    """SentenceTransformer('BAAI/bge-m3') as a graphiti EmbedderClient. Encodes on
    the ST model (cached, shared) in an executor so the async graph pipeline never
    blocks. Embeddings are L2-normalized (graphiti retrieval uses cosine
    similarity) and kept at the model's native width — set explicitly, because
    EmbedderConfig's default width comes from the EMBEDDING_DIM env var."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER_MODEL, device: str | None = None):
        self.model = get_bge_embedder(model_name, device=device)
        self.config = EmbedderConfig(embedding_dim=self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        dim = self.config.embedding_dim
        return [list(map(float, v))[:dim] for v in vecs]

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        texts = [input_data] if isinstance(input_data, str) else list(input_data)  # type: ignore[list-item]
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(None, self._encode, texts)
        return vecs[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode, input_data_list)


class CachedBGEReranker(BGERerankerClient):
    """graphiti's BGERerankerClient, but the CrossEncoder comes from the shared
    cache above instead of being reconstructed per user. Behavior (the async
    `rank`) is inherited byte-for-byte from the vendored client."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.model = get_bge_reranker(model_name, device=device)

# The embedded FalkorDB Lite (redislite) store MUST live on a native POSIX
# filesystem: redislite starts a redis-server bound to a UNIX SOCKET next to the
# db file, and unix sockets are unsupported on WSL's DrvFs (/mnt/c, ...) and other
# network/9p mounts → "redis-server process failed to start". Default to the
# system temp dir (ext4 on WSL2, tmpfs/local on Linux); override via cfg["db_root"].
# NOT the repo's outputs/ (that sits on /mnt/c under WSL).
def _db_root(cfg: Dict) -> Path:
    return Path(cfg["db_root"] or tempfile.gettempdir()) / "zep_falkordb"

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


# Method config, faithful arm — module-level DATA: eval_harness.py resolves it
# (with `arm` / `unified_models` / `memo:`) and hands the result to the memo's
# constructor; the class itself only reads self.config.
CONFIG_DEFAULTS = {
    "retrieve_k": 20,          # PAPER §4: "retrieve the 20 most relevant edges (facts) and entity nodes"
    # PAPER §4.1: "the BGE-m3 models from BAAI for both reranking and
    # embedding tasks". 1024-dim, local sentence-transformers. A
    # `text-embedding-*` name builds Graphiti's OpenAIEmbedder instead.
    "embedder_model": "BAAI/bge-m3",
    "reranker": "bge",         # bge (paper-faithful cross-encoder rerank) | openai
    # PAPER §4.1 names the BGE-m3 family for reranking without pinning a
    # checkpoint; this is that family's cross-encoder and graphiti's own
    # BGERerankerClient default.
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    # sentence-transformers device for BGE-m3 + reranker. None/"auto" =>
    # cuda if a GPU is visible, else cpu. Pin with "cpu" / "cuda:1" if needed.
    "device": None,
    # Dir for the embedded FalkorDB Lite store; None => system temp. MUST be
    # a native POSIX FS (redislite binds a unix socket) — do NOT point at
    # /mnt/c or another DrvFs/9p mount under WSL.
    "db_root": None,
    # PAPER §4.1: "we utilize gpt-4o-mini-2024-07-18 for graph construction".
    # The DATED snapshot is the paper's, and is what this pins — the undated
    # `gpt-4o-mini` alias now resolves to a later snapshot, so it would
    # silently stop reproducing the paper. Graphiti already drops
    # temperature for the gpt-5 family but still sends max_tokens, which
    # they reject; model_config renames it, so a gpt-5 model is runnable too.
    "graph_llm_model": "gpt-4o-mini-2024-07-18",
    # Max concurrent graph operations — Graphiti's own default (its
    # SEMAPHORE_LIMIT), passed as `max_coroutines` so the environment can't
    # change it. The paper states none.
    "max_coroutines": 20,
}
# `arm: unified` writes unified_models.llm / .embedding into these keys; an API
# embedder is then built through Graphiti's own OpenAIEmbedder (an injected
# EmbedderClient, no adapter). WHAT STAYS LOCAL: the reranker.
# bge-reranker-v2-m3 is a CROSS-ENCODER scoring (query, doc) pairs, so it has
# no API equivalent; Graphiti's OpenAIRerankerClient is an LLM-scoring reranker
# — a materially different retrieval algorithm — so the paper's cross-encoder is
# kept in BOTH arms (still the heaviest local cost in the fleet: ~570M params,
# k passes/query).
UNIFIED_MODEL_KEYS = {"llm": ("graph_llm_model",), "embedding": ("embedder_model",)}


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

        device = resolve_device(self.config["device"])   # None → cuda if visible, else cpu
        # Embedder: paper-faithful BGE-m3 (default) or an OpenAI API model
        # (unified arm). The width is passed explicitly: OpenAIEmbedder truncates
        # every vector to it, and its default comes from the EMBEDDING_DIM env var
        # (1024 when unset) — which silently cut text-embedding-3-small to 1024.
        model = self.config["embedder_model"]
        if is_api_embedding_model(model):
            from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
            embedder = OpenAIEmbedder(OpenAIEmbedderConfig(
                embedding_model=model, embedding_dim=api_embedding_dims(model)))
        else:
            embedder = BGEM3Embedder(model, device=device)

        graph_model = self.config["graph_llm_model"]
        # Reranker: paper-faithful BGE cross-encoder (default) or OpenAI (fallback).
        if self.config["reranker"] == "openai":
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
            cross_encoder = OpenAIRerankerClient(config=LLMConfig(model=graph_model))
        else:
            cross_encoder = CachedBGEReranker(
                self.config["reranker_model"], device=device)

        # Internal graph-construction LLM (paper: gpt-4o-mini-2024-07-18) — also
        # Graphiti's "small" model: the paper names no separate one. Its calls
        # are captured by common.openai_usage at the SDK boundary.
        llm_client = OpenAIClient(config=LLMConfig(model=graph_model, small_model=graph_model))

        self._graphiti = Graphiti(
            graph_driver=driver, llm_client=llm_client,
            embedder=embedder, cross_encoder=cross_encoder,
            max_coroutines=self.config["max_coroutines"],
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
        k = int(self.config["retrieve_k"])   # paper: top-20 edges + nodes
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
