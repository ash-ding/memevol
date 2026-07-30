"""Embedding utilities — OpenAI-embedding backend.

PATCH #2 (of 3; see the harness README): upstream's EmbeddingModel loads a
local Qwen3-Embedding-0.6B (or MiniLM) via sentence-transformers + torch —
a multi-GB dependency this host doesn't carry, and whose cost would be
invisible to the framework's token accounting. This drop-in replacement
keeps the exact interface the rest of the vendored code consumes
(`.dimension`, `.encode(texts, is_query)`, `.encode_single`,
`.encode_documents`, `.encode_query`) but computes embeddings through the
OpenAI embeddings API (model via $SIMPLEMEM_EMBED_MODEL, default
text-embedding-3-small/1536 — the same embedding every other baseline in
this repo uses, which keeps cross-baseline retrieval comparisons apples-to-
apples). Fidelity note: the paper's numbers used Qwen3 embeddings; scores
are therefore comparable WITHIN this repo, not with the paper's table.
"""
from typing import List
import os
import threading

import numpy as np

_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbeddingModel:
    """OpenAI-backed embedding model (interface-compatible with upstream)."""

    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or os.getenv(
            "SIMPLEMEM_EMBED_MODEL", "text-embedding-3-small")
        self.dimension = _DIMENSIONS.get(self.model_name, 1536)
        self.model_type = "openai"
        self.supports_query_prompt = False
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        # Lazy + thread-safe: upstream calls encode from parallel workers.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    # ---- upstream interface --------------------------------------------

    def encode(self, texts, is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        client = self._get_client()
        out: List[List[float]] = []
        BATCH = 512  # stay under OpenAI's items-per-request cap
        for i in range(0, len(texts), BATCH):
            batch = [t if isinstance(t, str) and t.strip() else " "
                     for t in texts[i:i + BATCH]]
            resp = client.embeddings.create(model=self.model_name, input=batch)
            out.extend(d.embedding for d in resp.data)
            _tally_embedding_usage(self.model_name, resp)
        arr = np.asarray(out, dtype=np.float32)
        # Upstream normalizes embeddings (normalize_embeddings=True); match it
        # so downstream cosine/L2 assumptions hold identically.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)


# ---------------------------------------------------------------------------
# Usage tally — shared with llm_client (PATCH #3); the harness adapter drains
# this into the framework's GLOBAL_TOKEN_TRACKER after each hook call.
# ---------------------------------------------------------------------------

USAGE_TALLY = {}
_TALLY_LOCK = threading.Lock()


def _tally_embedding_usage(model: str, resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    with _TALLY_LOCK:
        slot = USAGE_TALLY.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
        slot["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0


def tally_llm_usage(model: str, usage) -> None:
    """Called by llm_client (PATCH #3) after each non-streaming completion."""
    if usage is None:
        return
    with _TALLY_LOCK:
        slot = USAGE_TALLY.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
        slot["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        slot["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0


def drain_usage_tally() -> dict:
    """Return-and-reset the accumulated usage (adapter-side accounting)."""
    global USAGE_TALLY
    with _TALLY_LOCK:
        out, USAGE_TALLY = USAGE_TALLY, {}
    return out
