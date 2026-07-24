"""Iteration-0 seed genotypes — two deliberately simple, contrasting
(E,U,R,G) implementations so the first Pareto front has diversity along
both the performance and the cost axis (paper: the initial candidate set;
EvolveLab's simplest members play this role there).

Operator sources are written against the skeleton toolkit in
design_space.SKELETON_HEADER (_tokenize/_parse_ts/_BM25 are module-level
names inside the assembled file).
"""
from __future__ import annotations

from typing import Dict

# --------------------------------------------------------------------------
# Seed 1 — flat lexical store: append-only units + BM25 retrieval. Cheapest
# possible architecture (no LLM, no embeddings); the cost/latency anchor.
# --------------------------------------------------------------------------

_FLAT_BM25: Dict[str, str] = {
    "encode": '''\
async def encode(items, state):
    units = []
    for it in items:
        units.append({
            "content": it["content"],
            "ts": it["ts"],
            "source_id": it["source_id"],
            "tokens": _tokenize(it["content"]),
        })
    return units
''',
    "store": '''\
async def store(units, state):
    if "units" not in state:
        state["units"] = []
        state["bm25"] = _BM25()
    for u in units:
        state["units"].append(u)
        state["bm25"].add(u["tokens"])
''',
    "retrieve": '''\
async def retrieve(query, state):
    units = state.get("units", [])
    if not units:
        return {"memories": []}
    scores = state["bm25"].scores(_tokenize(query))
    ranked = sorted(range(len(units)), key=lambda i: -scores[i])[:12]
    return {"memories": [units[i]["content"] for i in ranked if scores[i] > 0]}
''',
    "manage": '''\
async def manage(state):
    return None
''',
}

# --------------------------------------------------------------------------
# Seed 2 — dense semantic store: embedded units, cosine retrieval with a
# recency bonus, dedup-on-manage. The quality-leaning anchor (embedding
# cost, better conceptual matching).
# --------------------------------------------------------------------------

_EMBED_RECENCY: Dict[str, str] = {
    "encode": '''\
async def encode(items, state):
    units = []
    for it in items:
        units.append({
            "content": it["content"],
            "ts": it["ts"],
            "source_id": it["source_id"],
            "tokens": _tokenize(it["content"]),
        })
    return units
''',
    "store": '''\
async def store(units, state):
    from common.llm import Embedding
    if "units" not in state:
        state["units"] = []
        state["embeddings"] = []
        state["latest_ts"] = None
    if not units:
        return
    embs = await Embedding(model="text-embedding-3-small").get_batch_embeddings(
        [u["content"] for u in units])
    for u, e in zip(units, embs):
        state["units"].append(u)
        state["embeddings"].append(e)
        if u["ts"] is not None:
            prev = state["latest_ts"]
            state["latest_ts"] = u["ts"] if prev is None else max(prev, u["ts"])
''',
    "retrieve": '''\
async def retrieve(query, state):
    from common.llm import Embedding
    units = state.get("units", [])
    if not units:
        return {"memories": []}
    q = await Embedding(model="text-embedding-3-small").get_embedding(query)
    qn = math.sqrt(sum(x * x for x in q)) or 1.0
    latest = state.get("latest_ts")
    scored = []
    for i, e in enumerate(state["embeddings"]):
        en = math.sqrt(sum(x * x for x in e)) or 1.0
        sim = sum(a * b for a, b in zip(q, e)) / (qn * en)
        ts = units[i]["ts"]
        if ts is not None and latest is not None:
            age_days = max(0.0, (latest - ts) / 86400.0)
            sim += 0.1 * (0.5 ** (age_days / 90.0))
        scored.append((i, sim))
    scored.sort(key=lambda kv: -kv[1])
    return {"memories": [
        {"content": units[i]["content"],
         "date": datetime.fromtimestamp(units[i]["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
                 if units[i]["ts"] else ""}
        for i, _ in scored[:12]
    ]}
''',
    "manage": '''\
async def manage(state):
    # Jaccard dedup over the most recent window; drop the later duplicate.
    units = state.get("units", [])
    embs = state.get("embeddings", [])
    if len(units) < 2:
        return
    keep = [True] * len(units)
    start = max(0, len(units) - 300)
    sets = [set(u["tokens"]) for u in units]
    for i in range(start, len(units)):
        if not keep[i] or not sets[i]:
            continue
        for j in range(max(start, i - 50), i):
            if not keep[j] or not sets[j]:
                continue
            jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            if jac >= 0.92:
                keep[i] = False
                break
    state["units"] = [u for u, k in zip(units, keep) if k]
    state["embeddings"] = [e for e, k in zip(embs, keep) if k]
''',
}

SEED_GENOTYPES: Dict[str, Dict[str, str]] = {
    "flat_bm25": _FLAT_BM25,
    "embed_recency": _EMBED_RECENCY,
}
