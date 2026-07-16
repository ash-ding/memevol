# hipporag2 baseline — graph-based RAG pipeline as a retrieval MemoStructure

`HippoRAGMemo` (`memo.py`) wraps [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s
pipeline (OpenIE → NER + triples → knowledge graph + entity embeddings →
personalized PageRank retrieval) as a `MemoStructure` — a fixed,
non-evolved memory architecture:

- **Phase 1 (`build_memory_from_data`)**: converts the ingested unit's data into
  text passages (dynamicmem: app_logs; locomo: conversation turns;
  longmemeval: session messages — dispatch on `recorder.init` keys) and
  indexes them into a per-user HippoRAG graph. `HippoRAG.index()` is
  additive/dedup-by-hash, so DynamicMem's per-checkpoint segments accumulate
  correctly across calls instead of re-indexing from scratch each time.
- **Phase 2 (`retrieve_memory_for_query`)**: fact retrieval → reranking →
  personalized PageRank → top-k passages, returned as `{"passages": [...]}`.
  The **shared QA agent** (not HippoRAG's own `rag_qa` reader) answers from
  those passages, and the per-dataset workflow judges/scores identically to
  the main method — this is a fair "HippoRAG-as-memory" comparison, not an
  end-to-end HippoRAG pipeline comparison.

hipporag2 runs through the **same shared runner** as cc
(`baselines.eval_common.run_baseline`), which resolves the SAME production
per-dataset workflow the main method uses (`baselines.registry.resolve`) —
so DynamicMem gets the official TCE v2 checkpoint protocol + holistic judge,
not a compat-shim script. The old single-dataset `eval_hipporag2.py` was
replaced by `run.py` (2026-07); numbers from that script are NOT comparable
to `run.py`'s DynamicMem output (different protocol).

Per the standardized `MemoStructure` contract, `build_memory_from_data` is called
ONCE per build call with the whole newly-visible data already in
`recorder.init` — the memo ingests the handed data however it chooses;
`HippoRAGMemo` indexes the whole call's passages in one shot and relies on
`HippoRAG.index()`'s additive/dedup-by-hash behavior for correctness across
calls.

## Usage

```bash
baselines/venv/bin/python baselines/hipporag2/run.py \
    --dataset {dynamicmem,locomo,longmemeval_s,longmemeval_m} \
    [--split test|search] \
    [--stage-spec '<json>'] \
    [--embedding text-embedding-3-small|nvidia/NV-Embed-v2|...] \
    [--llm_model gpt-5-mini] \
    [--judge_model gpt-5-mini] \
    [--embedding_batch_size N] [--embedding_dtype float16|auto] \
    [--max_sample_concurrent 3]
```

- `--dataset` (defaults to `dynamicmem`) — one of the four registered
  benchmarks (`baselines.registry.DATASETS`).
- `--split` (default `test`) — `test` = held-out split, `search` = the
  split the main method's search loop sees.
- `--stage-spec` (default `None` = **whole test split, full coverage** —
  byte-identical to the main method's `forge.heldout` default; see "Stage
  spec fields" below).
- `--embedding` — any OpenAI embedding model name (API, no GPU needed) or a
  local HF/NV embedding model (GPU, loaded in-process by HippoRAG).
- `--llm_model` — used BOTH as HippoRAG's internal OpenIE/triple-extraction
  LLM and as the shared QA agent's model.
- `--judge_model` — the judge model (default `gpt-5-mini`).
- `--embedding_batch_size` / `--embedding_dtype` — left `None`: for API
  embeddings (the default `text-embedding-3-small`) → batch 16, dtype auto; for
  local embeddings → batch 4, dtype float16 — computed automatically in
  `_ensure_hippo`.
- `--max_sample_concurrent` — per-eval user/sample concurrency (default 3).

Examples:

```bash
# OpenAI API embedding (no GPU needed), full held-out eval
baselines/venv/bin/python baselines/hipporag2/run.py \
    --dataset locomo --embedding text-embedding-3-small

# Local GPU embedding (NVIDIA)
baselines/venv/bin/python baselines/hipporag2/run.py \
    --dataset longmemeval_s --embedding nvidia/NV-Embed-v2 \
    --embedding_batch_size 2 --embedding_dtype float16

# Quick check, capped to 2 units
baselines/venv/bin/python baselines/hipporag2/run.py \
    --dataset dynamicmem --stage-spec '{"n_samples": 2}'
```

## Stage-spec fields

`--stage-spec` is a raw JSON object of USER OVERRIDES merged over the
family's full-coverage base (`baselines.eval_common.family_full_spec` /
`effective_stage_spec` — mirrors `forge.orchestrator.full_wire_spec`
exactly, so an omitted field stays `null` = uncapped, not zero). All units
are PER-RUN counts (no gauntlet/staging here — this is a single pass, same
as `forge.heldout`'s `coverage=full`):

| Field | Applies to | Meaning |
|---|---|---|
| `n_samples` | all datasets | Generic wire field for the unit count: **users** for dynamicmem, **conversations** for locomo, **questions** for longmemeval. `null`/omitted = whole split. |
| `n_checkpoints` | dynamicmem only | DynamicMem TCE checkpoints per user (of 5 quarterly checkpoints). `null` = all 5. |
| `n_task_a` | dynamicmem only | Task-A (`state_completion`) items sampled per checkpoint. `null` = full bucket. |
| `n_task_c` | dynamicmem only | Task-C (`apply_service`) items sampled per checkpoint. `null` = full bucket. |
| `n_qa` | locomo only | QA pairs sampled per conversation. `null` = all (categories 1-4 only; cat-5 excluded). |
| — | longmemeval | No extra field — one question is one unit, so `n_samples` alone controls both. |

## Output

Each eval run creates a fresh per-instance `outputs/<uuid>_<embedding>/` HippoRAG graph directory (embeddings + knowledge graph) per user, and these directories accumulate across runs (gitignored but growing on disk) — periodically clean `baselines/hipporag2/outputs/` if disk is a concern.

```
baselines/hipporag2/
├── outputs/<instance_id>_<embedding>/   # HippoRAG's own per-instance graph
│                                       # cache (OpenIE, embeddings, KG) — gitignored
└── results/<dataset>/<split>/
    ├── score.json          # {"benchmark_eval_score": {...}, "per_user": {...}, "invalid_users": [...]}
    ├── token_usage.json     # per-model token totals (common.tokens.TokenTracker)
    └── traces/<user_id>.json   # full per-user QA trajectory (no sampling)
```

## Notes

- `HippoRAGMemo` runs on all four datasets via dispatch on `recorder.init`
  keys (`app_logs` / `conversation` / `sessions` — see `_init_to_passages`
  in `memo.py`), mirroring how `forge`-evolved harnesses dispatch.
- Useful as a hand-designed comparison point: how does a published RAG
  pipeline (used purely as a memory/retrieval layer, answered by the SAME
  shared QA agent the main method uses) perform on these benchmarks versus
  forge-evolved harnesses?
- The old `eval_hipporag2.py` (DynamicMem-only, compat-shim last-checkpoint
  data) was fully replaced by `run.py` + `memo.py` — do not resurrect it;
  its results are not comparable (different protocol, no full TCE
  checkpoint interleaving).
