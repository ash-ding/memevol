# LightMem baseline

[LightMem](https://github.com/zjunlp/LightMem) — a lightweight, compression-first
lifelong memory (LLMlingua-2 pre-compression → attention topic segmentation → LLM
metadata/summary extraction → embedding index → offline update) — as a ready-made
memory system on the 3-hook `MemoClass` contract. The paper PDF is in this
directory ([lightmem.pdf](lightmem.pdf)).

**Provenance**: the `vendor/lightmem/` subtree is vendored from
<https://github.com/zjunlp/LightMem> @
`34410f41173f9107fbe8daf273092f398bfaf85b`. Only the **core text** package is
vendored — `src/lightmem/{__init__,configs,factory,memory}`, **byte-identical** to
upstream. The `src/lightmem/memory_toolkits/` subtree (LightMem's OWN vendored
copies of mem0 / A-mem / LangMem for its internal comparisons) and the multimodal
`src/em2mem` / `VLM2Vec` trees are NOT vendored — this baseline evaluates
LightMem's text memory only. Verify the vendored files are unmodified:

    UP=$(mktemp -d) && git clone https://github.com/zjunlp/LightMem "$UP" \
      && git -C "$UP" checkout 34410f4
    for sub in __init__.py configs factory memory; do \
      diff -r "$UP/src/lightmem/$sub" vendor/lightmem/$sub && echo "$sub: identical"; done

Nothing under `vendor/` is edited (unlike the simplemem baseline, LightMem's
top-level `__init__.py` is empty, so no trimming is needed). All integration code
lives in `memo.py` / `run.py` / `_st_shim.py`, never in `vendor/`. The vendored
`memory/graph.py` is a broken one-line upstream stub (`class GraphMem:` with no
body); it is imported only when `graph_mem=True`, which this baseline never sets,
so it is kept byte-identical and never loaded.

## How it works

LightMem's text pipeline runs untouched, driven exactly as in its own experiment
scripts (`experiments/{locomo,longmemeval}`):

1. **BUILD** (`build_memory_from_data`) — each ingestion unit becomes a LightMem
   turn (a `[user, assistant]` message pair with a session-level `time_stamp`),
   fed one turn at a time to `add_memory(messages, force_segment=is_last,
   force_extract=is_last)`. Internally: optional **LLMlingua-2 pre-compression**
   → **topic segmentation** (attention-based, sharing the LLMlingua-2 model) →
   **LLM metadata + summary extraction** (internal `gpt-4o-mini`) → **HF
   embedding** → insertion into a per-user **Qdrant** index (`update="offline"`).
   `force_*` on the last turn flush the buffers so the call's data is fully
   committed before any retrieval (mirrors the drivers' `is_last_turn`). Additive
   across DynamicMem checkpoints.
2. **Offline update** (config `offline_update`, default **on**) — after the last
   turn of a build call, `construct_update_queue_all_entries()` +
   `offline_update_all_entries(0.9)` run the full LoCoMo-paper refinement
   (per-entry LLM dedup / merge / delete over the current index).
3. **RETRIEVE** (`retrieve_memory_for_query`) — `LightMemory.retrieve(query,
   limit)` embeds the query, searches Qdrant, and returns the top-k memories as
   formatted `"timestamp weekday memory"` strings (exactly how LightMem's own
   LongMemEval driver retrieves). Read-only.

Retrieved memories are returned as `{"passages": [...]}` and the **shared QA agent
answers** — `use_memory_to_answer` is not overridden (hipporag2/amem/simplemem
pattern). This keeps the comparison about *memory* (LightMem's compression +
offline-refined retrieval), not about LightMem's own answer generator.

## Setup

lightmem is its own standalone uv project (`pyproject.toml` + `.python-version`
+ committed `uv.lock`, pulling in sentence-transformers, llmlingua,
qdrant-client, transformers, torch, pydantic):

    cd baselines/harness/lightmem && uv sync

This creates `baselines/harness/lightmem/.venv/`. The repo-root
`.venv/` is dev/test only and cannot run lightmem.

## Usage

    cd baselines/harness/lightmem && uv run python run.py \
        --config config.example.yaml
    uv run python run.py --config my_lightmem.yaml --dataset dynamicmem --split search

Flags: `--config` (YAML path; CLI flags override it). LightMem knobs:
`--pre_compress` / `--topic_segment` (default on; `topic_segment` requires
`pre_compress` — shared LLMlingua-2 model); `--llmlingua_model` (HF hub id or
local path) / `--llmlingua_device` (`cuda`|`cpu`); `--compress_rate` (0.6);
`--messages_use`; `--extract_threshold` (0.1); `--extraction_mode` (`flat`|`event`);
`--lightmem_llm_model` (default `gpt-4o-mini`; **4-series only** — LightMem sends
`temperature=0.1`, which the gpt-5 family rejects); `--manager_max_tokens`;
`--embedding_model` (default `all-MiniLM-L6-v2`) / `--embedding_dims` (384) /
`--embedding_device`; `--offline_update` (default on) / `--update_sim_threshold`
(0.9); `--retrieve_limit` (20). Shared: `--llm_model` / `--judge_model` (default
`gpt-5-mini` — shared QA agent + judge, baseline convention); `--split`.

Shared progressive-sampling flags (same as cc/hipporag2/amem/simplemem):
`--progressive` / `--no-progressive` (default off — staged stage1→2→3 gauntlet vs
one single-stage pass); `--sampling-seed` (default `42`); `--no-memory-cache`.

**Sizing is config-file only** (no sizing CLI flags). `progressive: false`
(default) REQUIRES a `single_stage` block; `progressive: true` sizes from a
`stages` block. See `config.example.yaml`.

## Ingestion mapping (`recorder.init` → LightMem turns)

| Benchmark | init key | one turn per… | user content / time_stamp |
|---|---|---|---|
| locomo | `conversation` | conversation utterance | turn text / `parse_locomo_timestamp(session date_time)` (verbatim from `add_locomo.py`) |
| longmemeval_s/m | `sessions` | (user, assistant) message pair | user message content / session `date` (LightMem's native LongMemEval mapping) |
| dynamicmem | `app_logs` | app-log entry | hipporag2's `app_log_to_passage` text / log `timestamp` |

Turns are ingested in order; the per-user Qdrant index is instance-scoped
(`uuid`), so a fresh `MemoClass` per user means no cross-user state.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `vendor/lightmem/{configs,factory,memory}` (pre-compression, topic segmentation, extraction, offline update, Qdrant backend, `retrieve`); `parse_locomo_timestamp`; per-turn `add_memory(force_segment/force_extract=is_last)` loop; LLMlingua-2 model + `compress_rate=0.6`; `extract_threshold=0.1`; internal `gpt-4o-mini`; `retrieve(limit=20)`; `offline_update_all_entries(0.9)` |
| Design choices (recorded) | **retrieval** = `LightMemory.retrieve()` (LightMem's own LongMemEval-driver path, uniform across datasets) — NOT the LoCoMo experiment's separate `VectorRetriever`/per-speaker glue; **offline update** = config knob, default on (the LoCoMo-paper full pipeline); **answering** via the shared QA agent |
| Integration adaptations (not algorithm) | longmemeval (per message pair) / dynamicmem (per app-log entry, hipporag2's text) ingestion mapping — LightMem only defined LoCoMo/LongMemEval; default extraction prompt (as the LongMemEval driver uses — the LoCoMo-specific `experiments/` prompt is not vendored); `_st_shim.py` (sentence-transformers vs memevol's `datasets/` shadow — same class of issue as amem/simplemem, imported once; + a process-wide embedder cache so the weights load once, not per user); LightMem's per-call `print`/INFO logging silenced (stdout redirect + logger pinned to WARNING); Qdrant path scoped per-uuid instance |
| Cross-user note | `memory/lightmem.py` has module globals `GLOBAL_TOPIC_IDX` / `GLOBAL_LAST_SUMMARY_TIME` that never reset across instances. `GLOBAL_TOPIC_IDX` monotonically grows (topic ids don't restart at 0 per user), but retrieval never filters on `topic_id`, so this is inert for scoring; `GLOBAL_LAST_SUMMARY_TIME` is only touched by `summarize()`, which this baseline never calls |
| Known consequences | LightMem compresses source turns into memory units that carry NO `app_log_id`, so DynamicMem evidence-citation scoring is disadvantaged (inherent to compression-first memory); for DynamicMem the offline-update phase runs at EACH checkpoint's build call (no single "final" build in the interleaved protocol), multiplying its cost |

## Cost / performance profile

- **Build** dominates: per turn, optional LLMlingua-2 compression (local) +
  segmentation (local) + 1 `gpt-4o-mini` extraction call (when a segment
  triggers) + a local embedding pass. **Offline update** adds a per-entry
  `gpt-4o-mini` dedup/update call over the whole index. **Retrieve** is a local
  embedding + Qdrant search, then the shared `gpt-5-mini` QA + `gpt-5-mini` judge.
- LightMem's internal `gpt-4o-mini` calls do **not** flow through `common.tokens`
  (same caveat as amem / simplemem / HippoRAG), so only the shared QA/judge side
  appears in `token_usage.json`.
- The LLMlingua-2 pre-compressor (a BERT model) and the sentence-transformer
  embedder are local and benefit from a GPU (`llmlingua_device` / `embedding_device`,
  default `cuda`; set `cpu` to run without a GPU, slowly). The embedder is loaded
  once per process and shared across users (`_st_shim`).
- Build/retrieve are synchronous + blocking, so users don't overlap under
  `--max_sample_concurrent` (a blocking hook body stalls the event loop) — the
  same profile as amem/simplemem.

## Validation status

Written against the vendored code and the 3-hook contract; **not yet run
end-to-end** here (this baseline's own uv env, incl. `llmlingua`/`qdrant-client`,
+ a GPU + an OpenAI key are only available on the eval server). All adapter + vendored files
pass `py_compile`. To smoke each ingestion branch cheaply on the search split:

    cd baselines/harness/lightmem
    uv run python run.py --config smoke_locomo.yaml
    uv run python run.py --config smoke_dynamicmem.yaml
    uv run python run.py --config smoke_longmemeval.yaml

Read `results/<dataset>/search/traces/<user>.json` to confirm build → retrieve →
QA runs, `invalid_users` is empty, and the retrieved `passages` are non-empty
memory strings in the expected `timestamp weekday memory` format.

## Output

    baselines/harness/lightmem/
    ├── outputs/<instance_id>/     # per-user Qdrant index (on-disk) — gitignored
    └── results/<dataset>/<split>/
        ├── score.json             # {"benchmark_eval_score": {...}, "per_user": {...}, ...}
        ├── token_usage.json        # shared QA/judge token totals (common.tokens)
        └── traces/<user_id>.json   # full per-user QA trajectory
