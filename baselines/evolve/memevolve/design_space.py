"""MemEvolve's modular memory design space (paper §3 / EvolveLab).

A memory architecture ("genotype") is four operator implementations:

    Ω = (E, U, R, G)  =  (encode, store, retrieve, manage)

  async def encode(items, state)  -> list[dict]   E: raw items → memory units
  async def store(units, state)   -> None         U: integrate units into state
  async def retrieve(query, state)-> dict         R: query → retrieved context
  async def manage(state)         -> None         G: periodic maintenance
                                                     (consolidate/forget/reorganize)

`state` is a plain dict the operators own entirely; `items` are
benchmark-normalized dicts {"content", "ts", "source_id"} produced by the
fixed adapter (so operators are dataset-agnostic). `retrieve` must be
READ-ONLY w.r.t. state (repo contract: DynamicMem checkpoint isolation).

This module owns:
  - SKELETON_HEADER: the fixed adapter + helper toolkit every assembled
    genotype file starts with (item normalization, tokenizer, BM25,
    timestamp parsing — so designed operators have a stable stdlib);
  - assemble_genotype(): operator sources → complete runnable .py that
    defines a `MemoStructure` subclass loadable by launch.py;
  - validate_operators(): compile + AST check that the four operator
    functions exist with the right arity before any subprocess is spent.

The meta-evolution operator F (meta_evolver.py) redesigns ONLY the four
operator bodies — the adapter/skeleton is immutable, which is what keeps
architectural changes "structurally constrained within the unified design
space" (paper §4.2).
"""
from __future__ import annotations

import ast
import hashlib
from typing import Dict, List

OPERATORS = ("encode", "store", "retrieve", "manage")

OPERATOR_SIGNATURES = {
    "encode": "async def encode(items, state):",
    "store": "async def store(units, state):",
    "retrieve": "async def retrieve(query, state):",
    "manage": "async def manage(state):",
}

# ---------------------------------------------------------------------------
# Fixed skeleton — helpers + adapter. Everything here is available to the
# designed operators by module-level name.
# ---------------------------------------------------------------------------

SKELETON_HEADER = '''\
"""Auto-assembled MemEvolve genotype. DO NOT EDIT — regenerate via
baselines/evolve/memevolve (the four operator sections are the genotype;
everything else is the fixed skeleton from design_space.py)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from common.harness_base import Basic_Recorder, MemoStructure

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "was", "are", "were", "be", "been", "with", "by", "from", "as",
    "that", "this", "it", "its", "his", "her", "their", "your", "my",
    "i", "you", "he", "she", "they", "we", "me", "him", "them", "us",
    "do", "did", "does", "have", "has", "had", "not", "no", "so", "but",
    "what", "when", "where", "which", "who", "how", "why", "about",
}


def _tokenize(text: str, min_len: int = 2) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= min_len and t not in _STOPWORDS]


def _parse_ts(raw: Any) -> Optional[float]:
    """Best-effort timestamp → epoch seconds for the three benchmarks."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) / 1000.0 if raw > 1e11 else float(raw)
    s = str(raw).strip()
    m = re.match(r"(\\d{4}/\\d{2}/\\d{2})(?:\\s*\\(\\w+\\))?\\s*(\\d{2}:\\d{2})?", s)
    if m:
        s2 = m.group(1) + (f" {m.group(2)}" if m.group(2) else "")
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
            try:
                return datetime.strptime(s2, fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%I:%M %p on %d %B, %Y", "%d %B, %Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


class _BM25:
    """Minimal Okapi BM25 over pre-tokenized docs."""

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


def _extract_items(init: Dict) -> List[Dict[str, Any]]:
    """Normalize any benchmark's recorder.init into
    [{"content", "ts", "source_id"}] so operators stay dataset-agnostic."""
    out: List[Dict[str, Any]] = []
    if not isinstance(init, dict):
        return out
    if init.get("app_logs") is not None:
        for entry in init["app_logs"]:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("app_log_id", f"log{len(out)}"))
            parts = [f"[{entry.get('timestamp', '')}]",
                     str(entry.get("app_name", "")), str(entry.get("api_name", ""))]
            for key in ("request", "response"):
                val = entry.get(key)
                if val:
                    txt = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
                    parts.append(f"{key}: {txt}")
            out.append({"content": " | ".join(p for p in parts if p),
                        "ts": _parse_ts(entry.get("timestamp")), "source_id": sid})
        return out
    if init.get("conversation") is not None:
        conv = init["conversation"]
        for key in sorted(conv):
            if not re.fullmatch(r"session_\\d+", key):
                continue
            date_str = conv.get(f"{key}_date_time", "")
            ts = _parse_ts(date_str)
            for turn in conv.get(key) or []:
                if not isinstance(turn, dict):
                    continue
                sid = str(turn.get("dia_id", f"{key}:{len(out)}"))
                out.append({"content": f"[{date_str}] {turn.get('speaker', '?')}: {turn.get('text', '')}",
                            "ts": ts, "source_id": sid})
        return out
    if init.get("sessions") is not None:
        for sess in init["sessions"]:
            if not isinstance(sess, dict):
                continue
            sess_id = str(sess.get("session_id", f"sess{len(out)}"))
            date_str = sess.get("date", "")
            ts = _parse_ts(date_str)
            msgs = sess.get("messages") or []
            for i in range(0, len(msgs), 4):
                chunk = msgs[i:i + 4]
                body = " ".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in chunk)
                out.append({"content": f"[{date_str}] {body}", "ts": ts,
                            "source_id": f"{sess_id}:{i}"})
        return out
    return out
'''

SKELETON_ADAPTER = '''

# ===========================================================================
# Fixed adapter — routes the repo's 3-hook contract through the operators.
# ===========================================================================

class MemEvolveMemo(MemoStructure):
    """Genotype-driven memory: build → encode+store+manage, retrieve → retrieve."""

    def __init__(self):
        super().__init__()
        self.state: Dict[str, Any] = {}
        self._ingested: set = set()

    async def build_memory_from_data(self, recorder: Basic_Recorder) -> None:
        items = _extract_items(recorder.init)
        # DynamicMem re-presents the visible prefix each checkpoint — only
        # genuinely new items flow into the operators.
        fresh = [it for it in items if it["source_id"] not in self._ingested]
        if not fresh:
            return
        for it in fresh:
            self._ingested.add(it["source_id"])
        units = await encode(fresh, self.state)
        await store(units or [], self.state)
        await manage(self.state)

    async def retrieve_memory_for_query(self, recorder: Basic_Recorder) -> Dict:
        query = str(recorder.init.get("query", ""))
        result = await retrieve(query, self.state)
        return result if isinstance(result, dict) else {"memories": result}
'''


def assemble_genotype(operators: Dict[str, str]) -> str:
    """Operator sources → complete genotype .py (validated)."""
    validate_operators(operators)
    return assemble_from_validated(operators)


def validate_operators(operators: Dict[str, str]) -> None:
    """Compile each operator + check the expected async def exists with the
    right arity. Raises ValueError with an LLM-consumable message."""
    missing = [op for op in OPERATORS if not (operators.get(op) or "").strip()]
    if missing:
        raise ValueError(f"missing operator implementations: {missing}")
    arity = {"encode": 2, "store": 2, "retrieve": 2, "manage": 1}
    for name in OPERATORS:
        src = operators[name]
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            raise ValueError(f"operator `{name}` has a syntax error: {exc}") from exc
        fns = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name]
        if not fns:
            raise ValueError(
                f"operator `{name}` must define `{OPERATOR_SIGNATURES[name]}` "
                f"at module top level (async def, exact name)")
        n_args = len(fns[0].args.args)
        if n_args != arity[name]:
            raise ValueError(
                f"operator `{name}` must take exactly {arity[name]} positional "
                f"args ({OPERATOR_SIGNATURES[name]}), got {n_args}")
    # Whole-file compile catches cross-operator issues (e.g. duplicate defs).
    compile(assemble_from_validated(operators), "<genotype>", "exec")


def assemble_from_validated(operators: Dict[str, str]) -> str:
    """assemble without re-validating (internal, avoids recursion)."""
    sections = []
    for name in OPERATORS:
        sections.append(f"\n\n# ===== OPERATOR: {name} =====\n{operators[name].strip()}\n")
    return SKELETON_HEADER + "".join(sections) + SKELETON_ADAPTER


def genotype_sha(operators: Dict[str, str]) -> str:
    """Content fingerprint over the four operator sources."""
    blob = "\n\x00\n".join(operators[name].strip() for name in OPERATORS)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]
