# Zep baseline

[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](zep.pdf)
(arXiv:2501.13956) as a ready-made memory system on the 3-hook `MemoClass`
contract. Zep's memory engine is **Graphiti**; the baseline vendors and drives
`graphiti_core` directly.

**Provenance**: `src/graphiti_core/` is vendored VERBATIM (byte-identical) from
<https://github.com/getzep/graphiti> @
`4f62cfe7a2d519e55bfdf2dc4a2fd06649dc00b3`, excluding the top-level `server/` and
`mcp_server/` service dirs (unused). No file under `src/graphiti_core/` is
edited — provenance lives here, not in per-file headers, to preserve byte-identity:

    diff -r <(git -C <graphiti-clone> show 4f62cfe:graphiti_core) src/graphiti_core

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

zep has its own uv-managed environment, defined by its own self-contained
`pyproject.toml` + committed `uv.lock` (neo4j, tenacity, posthog, falkordblite,
redis, sentence-transformers, plus the shared core deps):

    cd baselines/harness/zep && uv sync

This creates `baselines/harness/zep/.venv/`. The repo-root `.venv/` is
dev/test only and cannot run zep.

## Usage

    cd baselines/harness/zep && uv run python run.py --config config.example.yaml

**Requires Python 3.12+** (falkordblite constraint; pinned via zep's own
`.python-version`).

`run.py` takes exactly one flag, `--config <yaml>` (required) — there is no
other CLI surface. Every parameter lives in the config file:
`config.example.yaml` documents each key inline — copy it, edit the values
(e.g. `dataset: dynamicmem`, `split: search`), and point `--config` at your
copy. The YAML must list EXACTLY the keys `run.py`'s `REQUIRED_KEYS`
expects — a missing key OR an unknown key aborts the run before anything
executes; a `null` value counts as listed.

Keys worth calling out: `retrieve_k` (default 20, the paper's top-k);
`embedder` (`bge-m3` paper-faithful local | `openai`); `reranker` (`bge`
paper-faithful cross-encoder | `openai`); `device` (`cuda`|`cpu`,
sentence-transformers device for BGE models); `graph_llm_model` (default
`gpt-4o-mini`, the paper's graph-construction LLM — keep a 4-series model).
The rest (`embedder_model`, `reranker_model`, `db_root`,
`graph_llm_small_model`, `llm_model`, `judge_model`, `progressive`,
`sampling_seed`, `memory_cache`) are documented inline in
`config.example.yaml`.

**Sizing is config-file only** (there is no sizing CLI surface either) —
`single_stage` (progressive: false, REQUIRED) or `stages` (progressive:
true). See `config.example.yaml`.

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`config.example.yaml`) | unified arm (`config.unified.yaml`) |
|---|---|---|
| graph LLM (`graph_llm_model`) | `gpt-4o-mini-2024-07-18` — the paper's exact pin (§4.1) | `gpt-5-mini` |
| embedder (`embedder` / `embedder_model`) | `BAAI/bge-m3`, local, 1024-dim — the paper's (§4.1) | `text-embedding-3-small`, API, 1536-dim |
| reranker (`reranker` / `reranker_model`) | `BAAI/bge-reranker-v2-m3` cross-encoder — the paper's family (§4.1) | **unchanged** — no API equivalent |

The graph LLM pins the **dated** snapshot, quoting §4.1: *"we utilize
gpt-4o-mini-2024-07-18 for graph construction"*. The undated `gpt-4o-mini` alias
now resolves to a later snapshot, so leaving it undated would have silently
stopped reproducing the paper. §4.1 names "the BGE-m3 models from BAAI for both
reranking and embedding tasks" without pinning a reranker checkpoint;
`bge-reranker-v2-m3` is that family's cross-encoder and graphiti's own default.

**The faithful arm is the default**, and it is what the faithfulness table
below and every number in this README describe. The unified arm puts all seven
baselines on one LLM and one embedder so the comparison against the main method
is like-for-like — it is a deliberate deviation from the paper, and its numbers
must not be quoted as Zep's published result.

Both arms leave `src/` **byte-identical** — the `diff -r` above still passes.

- **the embedder** needs no patching at all: Graphiti accepts an injected
  `EmbedderClient`, so `memo.py` simply constructs `BGEM3Embedder` or Graphiti's
  own `OpenAIEmbedder`. Zep is the baseline the other three are measured
  against — it is the only one with a real injection point.
- **the LLM.** Graphiti already drops `temperature` for the gpt-5 family and
  routes structured output through `responses.parse`, but its plain-JSON path
  still sends `max_tokens` (`openai_client.py:126`), which those models reject.
  The shim in [`../model_config.py`](../model_config.py) renames it to
  `max_completion_tokens` at the OpenAI-SDK boundary.
- **the reranker stays local.** `bge-reranker-v2-m3` is a CROSS-ENCODER: it
  scores (query, doc) pairs, so it has no API equivalent. Graphiti does ship an
  `OpenAIRerankerClient`, but that is an LLM-scoring reranker — a materially
  different retrieval algorithm — so the unified arm keeps the paper's
  cross-encoder and confines the change to the LLM and the embedder. It remains
  the heaviest local cost in the fleet (~570M params, k forward passes per
  query) in BOTH arms, which is why `device` still matters on the unified arm.

`device` now defaults to `null` = **auto-detect** (cuda if a GPU is visible,
else cpu); it used to default to a hardcoded `cuda` and crashed outright on a
CPU-only box. Switching the embedder changes the vector width (1024 → 1536), so
any FalkorDB store or `memory_cache: true` snapshot built on the other arm is
invalid.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `graphiti_core` (@ 4f62cfe); Graphiti's construction pipeline (entity/fact/temporal/community extraction, resolution, edge invalidation); BGE reranker (`BAAI/bge-reranker-v2-m3`); `COMBINED_HYBRID_SEARCH_CROSS_ENCODER` recipe; retrieve_k=20 (§4); internal graph LLM `gpt-4o-mini-2024-07-18`, the paper's dated pin (faithful arm); paper's FACTS/ENTITIES context template (§3) |
| Integration adaptations (not algorithm) | **FalkorDB Lite** backend instead of the paper's Neo4j — full-text search is RediSearch, not Neo4j Lucene BM25 (a retrieval-backend difference; graph construction is backend-agnostic and identical); **BGE-m3 embedder** supplied via Graphiti's public `EmbedderClient` extension point (`BGEM3Embedder` in `memo.py`) since graphiti_core ships no local embedder — the paper used BGE-m3, which is not in the OSS embedder list; longmemeval (per message) / dynamicmem (per app-log entry, hipporag2's `app_log_to_passage` text) episode mappings — the paper only ran LoCoMo/LongMemEval conversations; answering via the shared QA agent; a process-wide model cache in `memo.py` so the BGE-m3 and reranker weights load once per process rather than once per user (a plain cached factory — Graphiti accepts injected clients, so nothing is monkeypatched); context compose replicated here (a Zep-service feature, not in OSS Graphiti) |
| Upstream quirks preserved | Graphiti's last-n-message context window (paper n=4) and all prompts/thresholds untouched; episode `source=message` auto-extracts the speaker as an entity |

## Caveats

- **Internal LLM cost IS tracked; local compute is not.** Graphiti calls the
  OpenAI SDK directly, and `memo.py` installs `common.openai_usage` before
  importing it, so its gpt-4o-mini graph-construction calls are captured at the
  SDK boundary and land in `token_usage.json` under the `build` phase — with no
  edit under `src/` (byte-identity preserved). What can NEVER be counted is the
  local compute: BGE-m3 embedding and the **bge-reranker-v2-m3 cross-encoder**,
  which scores (query, doc) PAIRS and so runs k forward passes per query — the
  heaviest per-query cost in the fleet. Neither is an API call, so neither
  produces a usage object. `run_record.json` names them (with device) and
  `phase_seconds` is the only figure covering both. **Any cost comparison
  against another baseline must state that it covers API calls only** — zep is
  where that understatement is largest.
- **Build is expensive**: `add_episode` runs several LLM calls per episode
  (extraction, resolution, fact, temporal, dedup) and is sequential per user (the
  graph dedups against accumulated state). Cost/latency scale like amem's per-note
  model. BGE-m3 embedding + BGE reranking are local (GPU-friendly).
- **Native filesystem required**: the embedded FalkorDB Lite (redislite) store
  starts a redis-server bound to a **unix socket**, which is unsupported on WSL's
  DrvFs (`/mnt/c`, ...) and other 9p/network mounts → `RedisLiteServerStartError:
  redis-server process failed to start`. The store therefore defaults to the
  system temp dir (`/tmp`, ext4 on WSL2), NOT the repo's `outputs/`. Override with
  the `db_root` config key (must be a native POSIX FS). Only the redislite store is
  affected; `results/` traces still write under the repo.
- **falkordblite maturity**: the embedded backend is newer than the Neo4j path.
  Each concurrent user spins its own embedded FalkorDB Lite store; teardown of the
  embedded process + on-disk file is best-effort (`ZepMemo.__del__`).

## Smoke verification (per code path)

`_init_to_episodes` has three ingestion branches (app_logs / conversation /
sessions); smoke with `split: search` in the config, 1 sample each, confirming
build → retrieve → QA runs, `invalid_users` is empty, and retrieved context
is non-empty:

    cd baselines/harness/zep && uv run python run.py --config smoke_locomo.yaml
    uv run python run.py --config smoke_longmemeval.yaml
    uv run python run.py --config smoke_dynamicmem.yaml

Smoke scores are single-sample sanity signals, NOT benchmark numbers. Real numbers
belong on `split: test` runs (touch the test split once per reported number).
