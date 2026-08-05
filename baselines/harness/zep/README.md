# Zep baseline

[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](zep.pdf)
(arXiv:2501.13956) as a ready-made memory system on the 3-hook `MemoStructure`
contract. Zep's memory engine is **Graphiti**; the baseline vendors and drives
`graphiti_core` directly.

**Provenance**: `vendor/graphiti_core/` is vendored VERBATIM (byte-identical) from
<https://github.com/getzep/graphiti> @
`4f62cfe7a2d519e55bfdf2dc4a2fd06649dc00b3`, excluding the top-level `server/` and
`mcp_server/` service dirs (unused). No file under `vendor/graphiti_core/` is
edited — provenance lives here, not in per-file headers, to preserve byte-identity:

    diff -r <(git -C <graphiti-clone> show 4f62cfe:graphiti_core) vendor/graphiti_core

## How it works

Graphiti builds a **temporal knowledge graph** from a stream of *episodes*
(messages). Each `add_episode` call runs the paper's pipeline untouched: LLM entity
extraction (+ reflection), entity resolution/dedup, fact (edge) extraction,
bi-temporal edge extraction with contradiction-driven **edge invalidation**, and
BGE-m3 embedding. Retrieval (`search_`) runs a hybrid **BM25 + cosine + breadth-first**
search over edges and nodes, reranked by the **BGE cross-encoder** (the paper's
`COMBINED_HYBRID_SEARCH_CROSS_ENCODER` recipe), returning the top-k facts and entity
summaries. These are reformatted into the paper's FACTS/ENTITIES context string and
returned as `{"inline_memory_blocks": [...]}`. The shared QA agent answers —
`use_memory_to_answer` is not overridden (the paper uses a separate chat agent over
the retrieved context; hipporag2/amem pattern).

**Backend: embedded FalkorDB Lite** (`falkordblite`, in-process, on-disk, no
server) — each user gets its own `outputs/<uuid>.db` store (+ Graphiti `group_id`),
so there is no cross-user state. This is the operational model of amem/simplemem/
lightmem (pip-only, no daemon), not Graphiti's default Neo4j server.

## Setup

zep has its own venv, built from its own self-contained `requirements.txt`
(`-r ../../core-requirements.txt` + neo4j, tenacity, posthog, falkordblite,
redis, sentence-transformers):

    baselines/setup_venv.sh zep

This creates `baselines/harness/zep/venv/`. The shared `baselines/venv/` is
dev/test only and cannot run zep.

## Usage

    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py \
        --config baselines/harness/zep/config.example.yaml
    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py \
        --config my_zep.yaml --dataset dynamicmem --split search

**Requires Python 3.12+** (falkordblite constraint; zep's own venv qualifies —
`baselines/setup_venv.sh` defaults to `python3.12`).
Key flags: `--config` (YAML; CLI overrides it); `--retrieve_k` (default 20, the
paper's top-k); `--embedder` (`bge-m3` paper-faithful local | `openai`);
`--reranker` (`bge` paper-faithful cross-encoder | `openai`); `--device`
(`cuda`|`cpu`, sentence-transformers device for BGE models); `--graph_llm_model`
(default `gpt-4o-mini`, the paper's graph-construction LLM — keep a 4-series model);
`--llm_model` / `--judge_model` (default `gpt-5-mini` — shared QA agent + judge,
baseline convention); `--split`.

Shared progressive-sampling flags (same as cc/hipporag2/amem): `--progressive` /
`--no-progressive` (default off); `--sampling-seed` (default 42); `--no-memory-cache`.
**Sizing is config-file only** — `single_stage` (progressive: false, REQUIRED) or
`stages` (progressive: true). See `config.example.yaml`.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `graphiti_core` (@ 4f62cfe); Graphiti's construction pipeline (entity/fact/temporal/community extraction, resolution, edge invalidation); BGE reranker (`BAAI/bge-reranker-v2-m3`); `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` recipe; retrieve_k=20; internal graph LLM gpt-4o-mini; paper's FACTS/ENTITIES context template (§3) |
| Integration adaptations (not algorithm) | **FalkorDB Lite** backend instead of the paper's Neo4j — full-text search is RediSearch, not Neo4j Lucene BM25 (a retrieval-backend difference; graph construction is backend-agnostic and identical); **BGE-m3 embedder** supplied via Graphiti's public `EmbedderClient` extension point (`_bge_embedder.py`) since graphiti_core ships no local embedder — the paper used BGE-m3, which is not in the OSS embedder list; longmemeval (per message) / dynamicmem (per app-log entry, hipporag2's `app_log_to_passage` text) episode mappings — the paper only ran LoCoMo/LongMemEval conversations; answering via the shared QA agent; `_st_shim.py` (memevol's `datasets/` shadows HF `datasets`, a sentence-transformers import-time dep) + shared model cache; context compose replicated here (a Zep-service feature, not in OSS Graphiti) |
| Upstream quirks preserved | Graphiti's last-n-message context window (paper n=4) and all prompts/thresholds untouched; episode `source=message` auto-extracts the speaker as an entity |

## Caveats

- **Internal LLM cost is not tracked**: Graphiti calls the OpenAI SDK directly, so
  its gpt-4o-mini graph-construction calls do NOT flow through `common.tokens`
  (same caveat as amem/hipporag2/simplemem/lightmem). `token_usage.json` reflects
  only the shared QA + judge (gpt-5-mini).
- **Build is expensive**: `add_episode` runs several LLM calls per episode
  (extraction, resolution, fact, temporal, dedup) and is sequential per user (the
  graph dedups against accumulated state). Cost/latency scale like amem's per-note
  model. BGE-m3 embedding + BGE reranking are local (GPU-friendly).
- **Native filesystem required**: the embedded FalkorDB Lite (redislite) store
  starts a redis-server bound to a **unix socket**, which is unsupported on WSL's
  DrvFs (`/mnt/c`, ...) and other 9p/network mounts → `RedisLiteServerStartError:
  redis-server process failed to start`. The store therefore defaults to the
  system temp dir (`/tmp`, ext4 on WSL2), NOT the repo's `outputs/`. Override with
  `db_root` / `--db_root` (must be a native POSIX FS). Only the redislite store is
  affected; `results/` traces still write under the repo.
- **falkordblite maturity**: the embedded backend is newer than the Neo4j path.
  Each concurrent user spins its own embedded FalkorDB Lite store; teardown of the
  embedded process + on-disk file is best-effort (`ZepMemo.__del__`).

## Smoke verification (per code path)

`_init_to_episodes` has three ingestion branches (app_logs / conversation /
sessions); smoke on `--split search`, 1 sample each, confirming build → retrieve →
QA runs, `invalid_users` is empty, and retrieved context is non-empty:

    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py --config baselines/harness/zep/smoke_locomo.yaml
    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py --config baselines/harness/zep/smoke_longmemeval.yaml
    baselines/harness/zep/venv/bin/python baselines/harness/zep/run.py --config baselines/harness/zep/smoke_dynamicmem.yaml

Smoke scores are single-sample sanity signals, NOT benchmark numbers. Real numbers
belong on the `--split test` runs (touch the test split once per reported number).
