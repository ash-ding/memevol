"""EvolveMem memory system — a fixed architecture whose retrieval
configuration is the evolvable action space (see action_space.py).

Implements the paper's three layers on this repo's 3-hook contract:
  §3.1 Structured Memory Store  → build_memory_from_data
       typed units + consolidation (Jaccard dedup, importance decay,
       entity reinforcement)
  §3.2 Retrieval Layer          → retrieve_memory_for_query
       three views (BM25 lexical / dense semantic / structured metadata),
       evolvable fusion (sum | weighted_sum | rrf), query augmentation
       (entity swap, decomposition), context budget B_ctx, answer style α
  §3.3 Self-Evolution Engine    → lives OUTSIDE this class (run_main.py);
       this class only consumes the current θ.

θ is loaded from the JSON file named by $EVOLVEMEM_CONFIG at instance
construction (the eval subprocess sets the env var); absent → defaults,
so the file also runs standalone as a plain multi-view retrieval memo.

Contract deviations from the paper, both forced by the repo protocol and
documented in README.md:
  - retrieve_memory_for_query MUST be read-only w.r.t. memory (DynamicMem
    checkpoint isolation), so entity reinforcement ρ is accumulated at
    BUILD time from entity co-occurrence across ingested items, not from
    query hits.
  - answer generation stays with the benchmark's own QA agent (score
    comparability); α is realized as a guidance sentence in the retrieved
    dict.

Input dispatch follows the standard init-key shapes:
  dynamicmem   recorder.init["app_logs"]      list[app_log dict]
  locomo       recorder.init["conversation"]  session dict
  longmemeval  recorder.init["sessions"]      list[session dict]
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common.harness_base import Basic_Recorder, MemoStructure
from common.logger import get_logger

from baselines.evolve.evolvemem.action_space import (
    ANSWER_STYLE_TEXT,
    load_config,
)

log = get_logger("main")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "was", "are", "were", "be", "been", "with", "by", "from", "as",
    "that", "this", "it", "its", "his", "her", "their", "your", "my",
    "i", "you", "he", "she", "they", "we", "me", "him", "them", "us",
    "do", "did", "does", "have", "has", "had", "not", "no", "so", "but",
    "what", "when", "where", "which", "who", "how", "why", "about",
}

MEMORY_TYPES = ("episodic", "semantic", "preference", "project_state",
                "working_summary", "procedural")


def _tokenize(text: str, min_len: int = 2, extra_stop: Optional[set] = None) -> List[str]:
    stop = _STOPWORDS if not extra_stop else (_STOPWORDS | extra_stop)
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= min_len and t not in stop]


def _parse_ts(raw: Any) -> Optional[float]:
    """Best-effort timestamp → epoch seconds. Handles the formats the three
    benchmarks actually emit; returns None when unparseable."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Heuristic: ms vs s epochs.
        return float(raw) / 1000.0 if raw > 1e11 else float(raw)
    s = str(raw).strip()
    fmts = (
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
        "%Y/%m/%d %H:%M", "%Y/%m/%d",
        "%I:%M %p on %d %B, %Y",   # LoCoMo session_date_time
        "%d %B, %Y",
    )
    # Strip trailing weekday/annotation like "2023/05/20 (Sat) 02:21".
    m = re.match(r"(\d{4}/\d{2}/\d{2})(?:\s*\(\w+\))?\s*(\d{2}:\d{2})?", s)
    if m:
        s2 = m.group(1) + (f" {m.group(2)}" if m.group(2) else "")
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                return datetime.strptime(s2, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class _BM25:
    """Minimal Okapi BM25 over pre-tokenized docs (no external dep)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens: List[List[str]] = []
        self.doc_freq: Counter = Counter()
        self.doc_len: List[int] = []
        self._avg_len = 0.0

    def add(self, tokens: List[str]) -> None:
        self.doc_tokens.append(tokens)
        self.doc_len.append(len(tokens))
        for t in set(tokens):
            self.doc_freq[t] += 1
        self._avg_len = sum(self.doc_len) / max(1, len(self.doc_len))

    def scores(self, query_tokens: List[str]) -> List[float]:
        n = len(self.doc_tokens)
        out = [0.0] * n
        if n == 0:
            return out
        for term in set(query_tokens):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, toks in enumerate(self.doc_tokens):
                tf = toks.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-9, self._avg_len))
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out


class EvolveMemMemo(MemoStructure):
    """Typed memory store + multi-view retrieval, parameterized by θ."""

    def __init__(self):
        super().__init__()
        cfg_path = os.environ.get("EVOLVEMEM_CONFIG")
        self.cfg = load_config(Path(cfg_path) if cfg_path else None)
        unknown_extras = [k for k in self.cfg.get("extras", {})
                          if k not in ("min_token_len", "stopword_extra", "dedup_scope",
                                       "session_join", "query_expand_keywords")]
        if unknown_extras:
            log.info(f"evolvemem: inert extras dims (stored, not implemented): {unknown_extras}")

        # Memory units, parallel arrays for speed.
        self.contents: List[str] = []
        self.types: List[str] = []
        self.timestamps: List[Optional[float]] = []
        self.importances: List[float] = []
        self.entities: List[set] = []
        self.tokens: List[List[str]] = []
        self.embeddings: List[List[float]] = []
        self.source_ids: List[str] = []

        self.bm25 = _BM25()
        self.entity_reinforcement: Counter = Counter()  # entity → ρ (build-time co-occurrence)
        self._ingested_sources: set = set()             # cross-checkpoint dedup by source id
        self._embedder = None
        self._latest_ts: Optional[float] = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _extra(self, key: str, default: Any) -> Any:
        return self.cfg.get("extras", {}).get(key, default)

    def _min_tok(self) -> int:
        try:
            return int(self._extra("min_token_len", 2))
        except (TypeError, ValueError):
            return 2

    def _extra_stop(self) -> Optional[set]:
        raw = self._extra("stopword_extra", None)
        return {str(w).lower() for w in raw} if isinstance(raw, list) else None

    def _get_embedder(self):
        if self._embedder is None:
            from common.llm import Embedding
            self._embedder = Embedding(model="text-embedding-3-small")
        return self._embedder

    # ------------------------------------------------------------------
    # BUILD — §3.1 structured memory store
    # ------------------------------------------------------------------

    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        items = self._extract_items(recorder.init)
        # Cross-call dedup by source id: DynamicMem re-presents the full
        # visible prefix at each checkpoint; only genuinely new items are
        # ingested so consolidation state accumulates instead of resetting.
        fresh = [it for it in items if it[2] not in self._ingested_sources]
        if not fresh:
            return
        for _, _, sid in fresh:
            self._ingested_sources.add(sid)

        if self.cfg["extraction_mode"] == "llm":
            units = await self._extract_units_llm(fresh)
        else:
            units = [self._extract_unit_raw(content, ts, sid) for content, ts, sid in fresh]

        units = self._consolidate(units)
        if not units:
            return

        texts = [u["content"] for u in units]
        embs = await self._get_embedder().get_batch_embeddings(texts)

        for u, emb in zip(units, embs):
            self.contents.append(u["content"])
            self.types.append(u["type"])
            self.timestamps.append(u["ts"])
            self.importances.append(u["importance"])
            self.entities.append(u["entities"])
            self.tokens.append(u["tokens"])
            self.embeddings.append(emb)
            self.source_ids.append(u["source_id"])
            self.bm25.add(u["tokens"])
            # Entity reinforcement ρ accumulates at build time (see module
            # docstring on the read-only retrieve constraint).
            for ent in u["entities"]:
                self.entity_reinforcement[ent] += 1
            if u["ts"] is not None:
                self._latest_ts = max(self._latest_ts or u["ts"], u["ts"])

        self._apply_importance_decay()

    def _extract_items(self, init: Dict) -> List[Tuple[str, Optional[float], str]]:
        """Normalize any benchmark's init into (content, epoch_ts, source_id)."""
        if not isinstance(init, dict):
            return []
        if "app_logs" in init and init.get("app_logs") is not None:
            return self._items_from_app_logs(init["app_logs"])
        if "conversation" in init and init.get("conversation") is not None:
            return self._items_from_conversation(init["conversation"])
        if "sessions" in init and init.get("sessions") is not None:
            return self._items_from_sessions(init["sessions"])
        return []

    def _items_from_app_logs(self, app_logs: List[Dict]) -> List[Tuple[str, Optional[float], str]]:
        out = []
        for entry in app_logs or []:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("app_log_id", f"log{len(out)}"))
            ts = _parse_ts(entry.get("timestamp"))
            parts = [f"[{entry.get('timestamp', '')}]",
                     str(entry.get("app_name", "")), str(entry.get("api_name", ""))]
            for key in ("request", "response"):
                val = entry.get(key)
                if val:
                    txt = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
                    parts.append(f"{key}: {txt}")
            out.append((" | ".join(p for p in parts if p), ts, sid))
        return out

    def _items_from_conversation(self, conv: Dict) -> List[Tuple[str, Optional[float], str]]:
        out = []
        for key in sorted(conv):
            if not re.fullmatch(r"session_\d+", key):
                continue
            date_str = conv.get(f"{key}_date_time", "")
            ts = _parse_ts(date_str)
            for turn in conv.get(key) or []:
                if not isinstance(turn, dict):
                    continue
                sid = str(turn.get("dia_id", f"{key}:{len(out)}"))
                content = f"[{date_str}] {turn.get('speaker', '?')}: {turn.get('text', '')}"
                out.append((content, ts, sid))
        return out

    def _items_from_sessions(self, sessions: List[Dict]) -> List[Tuple[str, Optional[float], str]]:
        gran = self.cfg["unit_granularity"]
        join_n = self._extra("session_join", 4)
        try:
            join_n = max(1, int(join_n))
        except (TypeError, ValueError):
            join_n = 4
        out = []
        for sess in sessions or []:
            if not isinstance(sess, dict):
                continue
            sess_id = str(sess.get("session_id", f"sess{len(out)}"))
            date_str = sess.get("date", "")
            ts = _parse_ts(date_str)
            msgs = sess.get("messages") or []
            if gran == "session":
                body = " ".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in msgs)
                out.append((f"[{date_str}] {body}", ts, sess_id))
                continue
            # item/auto granularity: join_n consecutive messages per unit —
            # keeps user/assistant adjacency while bounding unit count on
            # large haystacks.
            for i in range(0, len(msgs), join_n):
                chunk = msgs[i:i + join_n]
                body = " ".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in chunk)
                out.append((f"[{date_str}] {body}", ts, f"{sess_id}:{i}"))
        return out

    def _extract_unit_raw(self, content: str, ts: Optional[float], sid: str) -> Dict:
        toks = _tokenize(content, self._min_tok(), self._extra_stop())
        ents = {e.strip() for e in _ENTITY_RE.findall(content) if len(e.strip()) > 2}
        return {
            "content": content, "type": self._heuristic_type(content),
            "ts": ts, "importance": 0.5, "entities": ents,
            "tokens": toks, "source_id": sid,
        }

    @staticmethod
    def _heuristic_type(content: str) -> str:
        low = content.lower()
        if any(w in low for w in ("prefer", "favorite", "favourite", "like", "love", "hate")):
            return "preference"
        if any(w in low for w in ("how to", "steps", "procedure", "instructions")):
            return "procedural"
        return "episodic"

    async def _extract_units_llm(self, fresh: List[Tuple[str, Optional[float], str]]) -> List[Dict]:
        """LLM typed extraction over sliding windows (§3.1). Falls back to raw
        units for a window whose LLM call ultimately fails (Agent has its own
        retry ladder), preserving partial results as the paper prescribes."""
        from common.llm import Agent
        schema = {
            "type": "object",
            "properties": {"memories": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "type": {"type": "string", "enum": list(MEMORY_TYPES)},
                    "importance": {"type": "number"},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "source_index": {"type": "integer"},
                },
                "required": ["content", "type", "importance", "source_index"],
            }}},
            "required": ["memories"],
        }
        window = int(self.cfg["extraction_window"])
        units: List[Dict] = []
        prev_tail = ""
        for w_start in range(0, len(fresh), window):
            batch = fresh[w_start:w_start + window]
            listing = "\n".join(f"{i}: {content[:600]}" for i, (content, _, _) in enumerate(batch))
            agent = Agent(
                system_prompt=(
                    "You extract typed memory units from a user-data stream for a "
                    "long-term memory system. Emit self-contained factual memories "
                    "(who/what/when preserved verbatim), typed as one of "
                    f"{list(MEMORY_TYPES)}. importance ∈ [0,1]. source_index refers "
                    "to the numbered input item a memory came from. Avoid "
                    "duplicating memories already extracted from the previous window."
                ),
                output_schema=schema,
                model=self.cfg["extraction_model"],
                timeout=300,
            )
            prompt = (f"Previously extracted (tail, for dedup):\n{prev_tail}\n\n"
                      f"Input items:\n{listing}")
            try:
                result = await agent.ask(prompt)
                extracted = result.get("memories", [])
            except Exception as exc:
                log.warning(f"evolvemem: LLM extraction window failed ({exc}); raw fallback")
                units.extend(self._extract_unit_raw(c, t, s) for c, t, s in batch)
                continue
            for mem in extracted:
                idx = mem.get("source_index", 0)
                idx = idx if isinstance(idx, int) and 0 <= idx < len(batch) else 0
                _, ts, sid = batch[idx]
                content = str(mem.get("content", "")).strip()
                if not content:
                    continue
                units.append({
                    "content": content,
                    "type": mem.get("type", "episodic"),
                    "ts": ts,
                    "importance": min(1.0, max(0.0, float(mem.get("importance", 0.5)))),
                    "entities": {str(e) for e in (mem.get("entities") or [])},
                    "tokens": _tokenize(content, self._min_tok(), self._extra_stop()),
                    "source_id": sid,
                })
            prev_tail = "; ".join(u["content"][:80] for u in units[-5:])
        return units

    def _consolidate(self, units: List[Dict]) -> List[Dict]:
        """Jaccard dedup of incoming units against each other AND recent
        stored units, keeping the higher-importance duplicate's importance."""
        tau = float(self.cfg["dedup_tau"])
        scope_global = self._extra("dedup_scope", "window") == "global"
        recent_sets = [set(t) for t in (self.tokens if scope_global else self.tokens[-500:])]
        kept: List[Dict] = []
        kept_sets: List[set] = []
        for u in units:
            uset = set(u["tokens"])
            dup = False
            if uset:
                for other in kept_sets + recent_sets:
                    if not other:
                        continue
                    j = len(uset & other) / len(uset | other)
                    if j >= tau:
                        dup = True
                        break
            if not dup:
                kept.append(u)
                kept_sets.append(uset)
        return kept

    def _apply_importance_decay(self) -> None:
        """Linear ι decay by age (α_d per day) with floor ι_min, computed
        against the newest ingested timestamp (no wall-clock dependence)."""
        rate = float(self.cfg["importance_decay"])
        floor = float(self.cfg["importance_floor"])
        if rate <= 0 or self._latest_ts is None:
            return
        for i, ts in enumerate(self.timestamps):
            if ts is None:
                continue
            age_days = max(0.0, (self._latest_ts - ts) / 86400.0)
            self.importances[i] = max(floor, self.importances[i] - rate * age_days)

    # ------------------------------------------------------------------
    # RETRIEVE — §3.2 multi-view fusion (read-only)
    # ------------------------------------------------------------------

    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        query = str(recorder.init.get("query", ""))
        if not self.contents:
            return {"memories": []}
        cfg = self._config_for_query(query)

        sub_queries = [query]
        if cfg["query_decomposition"]:
            sub_queries = await self._decompose(query, cfg)
        if cfg["entity_swap"]:
            stripped = _ENTITY_RE.sub(" ", query).strip()
            if stripped and stripped != query:
                sub_queries.append(stripped)

        # Rank per sub-query, merge via RRF across sub-queries (paper §3.2).
        merged: Dict[int, float] = defaultdict(float)
        for sq in sub_queries:
            ranking = await self._rank_single(sq, cfg)
            for rank, (idx, _) in enumerate(ranking):
                merged[idx] += 1.0 / (int(cfg["rrf_k"]) + rank + 1)
        final = sorted(merged.items(), key=lambda kv: -kv[1])[:int(cfg["b_ctx"])]

        memories = []
        for idx, _ in final:
            ts = self.timestamps[idx]
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else ""
            memories.append({"content": self.contents[idx],
                             "type": self.types[idx], "date": date})
        out: Dict[str, Any] = {"memories": memories}
        style = ANSWER_STYLE_TEXT.get(cfg["answer_style"])
        if style:
            out["answer_guidance"] = style
        return out

    def _config_for_query(self, query: str) -> Dict[str, Any]:
        """θ_c — first matching per-category override wins."""
        for cat in self.cfg.get("per_category", []):
            try:
                if re.search(cat["pattern"], query):
                    merged = dict(self.cfg)
                    merged.update(cat["overrides"])
                    return merged
            except re.error:
                continue
        return self.cfg

    async def _rank_single(self, query: str, cfg: Dict) -> List[Tuple[int, float]]:
        """One fused ranking pass: three views → fusion → intrinsic bonuses."""
        n = len(self.contents)
        q_tokens = _tokenize(query, self._min_tok(), self._extra_stop())

        sem_rank = await self._semantic_view(query, int(cfg["k_sem"]))

        expand_n = self._extra("query_expand_keywords", 0)
        try:
            expand_n = max(0, int(expand_n))
        except (TypeError, ValueError):
            expand_n = 0
        if expand_n and sem_rank:
            expansion = Counter()
            for idx, _ in sem_rank[:3]:
                expansion.update(self.tokens[idx])
            q_tokens = q_tokens + [t for t, _ in expansion.most_common(expand_n)]

        kw_rank = self._lexical_view(q_tokens, int(cfg["k_kw"]))
        str_rank = self._metadata_view(query, q_tokens, int(cfg["k_str"]))

        views = {"sem": sem_rank, "kw": kw_rank, "str": str_rank}
        weights = {"sem": float(cfg["w_sem"]), "kw": float(cfg["w_kw"]), "str": float(cfg["w_str"])}
        mode = cfg["fusion_mode"]

        fused: Dict[int, float] = defaultdict(float)
        if mode == "rrf":
            k = int(cfg["rrf_k"])
            for name, ranking in views.items():
                for rank, (idx, _) in enumerate(ranking):
                    fused[idx] += 1.0 / (k + rank + 1)
        else:
            # Normalize per-view scores to [0,1] before (weighted) sum.
            for name, ranking in views.items():
                if not ranking:
                    continue
                top = max(s for _, s in ranking) or 1.0
                w = weights[name] if mode == "weighted_sum" else 1.0
                for idx, s in ranking:
                    fused[idx] += w * (s / top)

        # s(q,m;θ) = s_fuse + λ_ι·ι + λ_r·rec + ρ  (Eq. 1; ρ normalized).
        lam_i = float(cfg["lambda_importance"])
        lam_r = float(cfg["lambda_recency"])
        halflife = float(cfg["recency_halflife_days"])
        rho_max = max(self.entity_reinforcement.values()) if self.entity_reinforcement else 1
        scored: List[Tuple[int, float]] = []
        for idx, base in fused.items():
            score = base + lam_i * self.importances[idx]
            ts = self.timestamps[idx]
            if ts is not None and self._latest_ts is not None:
                age_days = max(0.0, (self._latest_ts - ts) / 86400.0)
                score += lam_r * (0.5 ** (age_days / max(1e-9, halflife)))
            ents = self.entities[idx]
            if ents:
                rho = max(self.entity_reinforcement.get(e, 0) for e in ents) / rho_max
                score += 0.1 * rho
            scored.append((idx, score))
        scored.sort(key=lambda kv: -kv[1])
        _ = n  # (kept for clarity: rankings index into the full store)
        return scored

    async def _semantic_view(self, query: str, k: int) -> List[Tuple[int, float]]:
        if k <= 0 or not self.embeddings:
            return []
        q_emb = await self._get_embedder().get_embedding(query)
        q_norm = math.sqrt(sum(x * x for x in q_emb)) or 1.0
        sims = []
        for i, emb in enumerate(self.embeddings):
            dot = sum(a * b for a, b in zip(q_emb, emb))
            e_norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            sims.append((i, dot / (q_norm * e_norm)))
        sims.sort(key=lambda kv: -kv[1])
        return sims[:k]

    def _lexical_view(self, q_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        if k <= 0 or not q_tokens:
            return []
        scores = self.bm25.scores(q_tokens)
        ranked = sorted(((i, s) for i, s in enumerate(scores) if s > 0), key=lambda kv: -kv[1])
        return ranked[:k]

    def _metadata_view(self, query: str, q_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        """Structured view: entity + keyword overlap between query and unit
        metadata (entities, type keywords)."""
        if k <= 0:
            return []
        q_ents = {e.strip() for e in _ENTITY_RE.findall(query)}
        q_set = set(q_tokens)
        scored = []
        for i in range(len(self.contents)):
            s = 2.0 * len(q_ents & self.entities[i])
            s += len(q_set & set(self.tokens[i])) / max(1, len(q_set))
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]

    async def _decompose(self, query: str, cfg: Dict) -> List[str]:
        from common.llm import Agent
        schema = {"type": "object",
                  "properties": {"sub_queries": {"type": "array", "items": {"type": "string"}}},
                  "required": ["sub_queries"]}
        agent = Agent(
            system_prompt=("Split a possibly multi-hop question into at most 3 "
                           "single-hop sub-queries for memory retrieval. If the "
                           "question is already single-hop, return it unchanged."),
            output_schema=schema, model=cfg["decomposition_model"], timeout=120,
        )
        try:
            result = await agent.ask(query)
            subs = [str(s) for s in result.get("sub_queries", []) if str(s).strip()]
            return ([query] + subs[:3]) if subs else [query]
        except Exception as exc:
            log.warning(f"evolvemem: query decomposition failed ({exc}); using raw query")
            return [query]
