# Zep baseline

[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)
(arXiv:2501.13956) as a ready-made memory system on the 2-hook `MemoClass`
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
the shared QA agent answers, as for every memo (the paper likewise uses a separate chat agent over
the retrieved context; hipporag2/amem pattern).

**Backend: embedded FalkorDB Lite** (`falkordblite`, in-process, on-disk, no
server) — each user gets its own `<db_root>/zep_falkordb/<uuid>.db` store (+ Graphiti
`group_id`; `db_root` defaults to the system temp dir, see Caveats), so there is no
cross-user state. This is the operational model of amem/simplemem/
lightmem (pip-only, no daemon), not Graphiti's default Neo4j server.

## Setup

zep has its own uv-managed environment, defined by its own self-contained
`pyproject.toml` + committed `uv.lock` (neo4j, tenacity, posthog, falkordblite,
redis, sentence-transformers, plus the shared core deps):

    cd baselines/harness/zep && uv sync

This creates `baselines/harness/zep/.venv/`. The repo-root `.venv/` is
dev/test only and cannot run zep.

## Usage

    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness \
        --config baselines/harness/config.example.yaml      # harness: zep

**Requires Python 3.12+** (falkordblite constraint; pinned via zep's own
`.python-version`).

All seven harness baselines share ONE entrypoint
([`../eval_harness.py`](../eval_harness.py)) and ONE frame config
([`../config.example.yaml`](../config.example.yaml)): set `harness: zep`,
choose `arm`, dataset/split/sizing and the shared QA + judge models, and point
`--config` at your copy. The YAML must list EXACTLY the frame keys — a missing
key OR an unknown key aborts the run before anything executes; a `null` value
counts as listed. `--project` is not optional: this baseline's deps live only
in its own venv.

**Method knobs are not in the config file.** Every zep-specific parameter is
declared once, with its justification, in `CONFIG_DEFAULTS` at the top of
[`memo.py`](memo.py) (the faithful arm); `UNIFIED_MODEL_KEYS` names the keys
`arm: unified` writes `unified_models` into. Print them:

    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness --describe zep

Every run keeps its config as `runs/<run_id>/config.yaml` (copy it to re-run) and
the fully resolved values in `runs/<run_id>/memo_config.resolved.yaml`.
To override one for an ablation, add a `memo:` block to your config
(`memo: {retrieve_k: 5}`) — validated against `CONFIG_DEFAULTS`, so a typo aborts;
model keys can't be set there (use `arm` / `unified_models`).

Keys worth calling out: `retrieve_k` (default 20, the paper's top-k);
`embedder_model` (`BAAI/bge-m3` paper-faithful local; a `text-embedding-*` name
builds Graphiti's OpenAIEmbedder, told its width explicitly); `reranker` (`bge`
paper-faithful cross-encoder | `openai`); `device` (`null` = auto-detect, else
`cpu` / `cuda:N`, sentence-transformers device for the BGE models);
`graph_llm_model` (default `gpt-4o-mini-2024-07-18`, the paper's
graph-construction LLM — also used as Graphiti's "small" model);
`max_coroutines` (default 20, Graphiti's own concurrency limit, passed
explicitly). The frame keys (`llm_model`, `judge_model`, `progressive`,
`sampling_seed`, ...) are documented inline in `../config.example.yaml`.

Nothing is read from the environment: Graphiti's own env-driven defaults
(`EMBEDDING_DIM`, `SEMAPHORE_LIMIT`) are overridden by explicit arguments, the
ones it reads with no override (`CHUNK_*`, index names, ...) make a run refuse
to start if set, and Graphiti's PostHog telemetry is switched off in `memo.py`.

**Sizing is config-file only** (there is no sizing CLI surface either) —
`single_stage` (progressive: false, REQUIRED) or `stages` (progressive:
true). See `../config.example.yaml`.

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`arm: faithful` — `CONFIG_DEFAULTS`) | unified arm (`arm: unified` — example `unified_models`) |
|---|---|---|
| graph LLM (`graph_llm_model`) | `gpt-4o-mini-2024-07-18` — the paper's exact pin (§4.1) | `gpt-5-mini` |
| embedder (`embedder_model`) | `BAAI/bge-m3`, local, 1024-dim — the paper's (§4.1) | `text-embedding-3-small`, API, 1536-dim |
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
any FalkorDB store built on the other arm is invalid.

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
  affected; `runs/` traces still write under the repo.
- **falkordblite maturity**: the embedded backend is newer than the Neo4j path.
  Each concurrent user spins its own embedded FalkorDB Lite store; teardown of the
  embedded process + on-disk file is best-effort (`ZepMemo.__del__`).

## Smoke verification (per code path)

`_init_to_episodes` has three ingestion branches (app_logs / conversation /
sessions); smoke with `split: search` in the config, 1 sample each, confirming
build → retrieve → QA runs, `invalid_users` is empty, and retrieved context
is non-empty:

    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness --config smoke_locomo.yaml
    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness --config smoke_longmemeval.yaml
    uv run --project baselines/harness/zep python -m baselines.harness.eval_harness --config smoke_dynamicmem.yaml

Smoke scores are single-sample sanity signals, NOT benchmark numbers. Real numbers
belong on `split: test` runs (touch the test split once per reported number).
