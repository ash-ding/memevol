"""Paper-faithful BGE-m3 embedder + cached BGE reranker for the zep baseline.

The Zep paper (§4.1) uses BAAI's BGE-m3 for BOTH embedding and reranking.
graphiti_core ships a BGE **reranker** (cross_encoder/bge_reranker_client.py) but
NO local/sentence-transformers **embedder** (only openai/azure/gemini/voyage), so
the embedder is supplied here through Graphiti's public `EmbedderClient` extension
point — integration code, NOT an edit to vendored graphiti_core. This mirrors what
the paper's authors must have done (BGE-m3 is not in graphiti's OSS embedder list).

Both classes reuse the process-wide model cache in `_st_shim` so the ~2GB BGE-m3
weights and the BGE reranker load ONCE per process, not once per user.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterable

from baselines.harness.zep import _st_shim

_st_shim.ensure_src_on_path()
_st_shim.ensure_sentence_transformers()   # before importing the graphiti reranker (imports ST)

from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig  # noqa: E402
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient  # noqa: E402

DEFAULT_EMBEDDER_MODEL = "BAAI/bge-m3"          # paper: BGE-m3, 1024-dim
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # graphiti's BGERerankerClient default


class BGEM3Embedder(EmbedderClient):
    """SentenceTransformer('BAAI/bge-m3') as a graphiti EmbedderClient. Encodes on
    the ST model (cached, shared) in an executor so the async graph pipeline never
    blocks. Embeddings are L2-normalized (graphiti retrieval uses cosine
    similarity) and truncated to config.embedding_dim (default 1024 = BGE-m3's
    native dim, so a no-op)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDER_MODEL,
                 device: str | None = None, config: EmbedderConfig | None = None):
        self.config = config or EmbedderConfig()
        self.model = _st_shim.get_bge_embedder(model_name, device=device)

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
    _st_shim cache instead of being reconstructed per user. Behavior (the async
    `rank`) is inherited byte-for-byte from the vendored client."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.model = _st_shim.get_bge_reranker(model_name, device=device)
