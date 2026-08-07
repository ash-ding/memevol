# hipporag2 baseline — graph-based RAG pipeline as a retrieval MemoClass

`HippoRAGMemo` (`memo.py`) wraps [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s
pipeline (OpenIE → NER + triples → knowledge graph + entity embeddings →
personalized PageRank retrieval) as a `MemoClass` — a fixed,
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
(`baselines.harness.eval_utility.run_baseline`), which resolves the SAME production
per-dataset workflow the main method uses (`baselines.registry.resolve`) —
so DynamicMem gets the official TCE v2 checkpoint protocol + holistic judge,
not a compat-shim script. The old single-dataset `eval_hipporag2.py` was
replaced by `run.py` (2026-07); numbers from that script are NOT comparable
to `run.py`'s DynamicMem output (different protocol).

Per the standardized `MemoClass` contract, `build_memory_from_data` is called
ONCE per build call with the whole newly-visible data already in
`recorder.init` — the memo ingests the handed data however it chooses;
`HippoRAGMemo` indexes the whole call's passages in one shot and relies on
`HippoRAG.index()`'s additive/dedup-by-hash behavior for correctness across
calls.

## Setup

hipporag2 is its own uv project (`pyproject.toml` + `.python-version` +
committed `uv.lock`, plus HippoRAG's runtime deps — chromadb,
langchain-chroma, finch-clust, networkx, nltk, rank_bm25,
sentence-transformers):

```bash
cd baselines/harness/hipporag2
uv sync
# hipporag2 ALSO needs the external HippoRAG package as an EDITABLE install
# (memo.py: `from hipporag import HippoRAG`) — `hipporag` is NOT vendored and
# NOT a pyproject.toml dependency, so `uv sync` alone does not install it:
uv pip install -e /export/scratch_large/ding/code/HippoRAG   # point at your HippoRAG checkout
```

This is the ONLY baseline whose setup needs a second step beyond `uv sync`.

This creates `baselines/harness/hipporag2/.venv/`. The repo-root
`.venv/` is dev/test only and cannot run hipporag2.

## Usage

```bash
cd baselines/harness/hipporag2 && uv run python run.py --config config.example.yaml
```

`run.py` takes exactly one flag, `--config <yaml>` (required) — there is no
other CLI surface (dataset, split, progressive, embedding, etc. are no
longer flags). Every parameter lives in the config file: `config.example.yaml` documents
each key inline — copy it, edit the values, and point `--config` at your
copy. The YAML must list EXACTLY the keys `run.py`'s `REQUIRED_KEYS`
expects — a missing key OR an unknown key aborts the run before anything
executes; a `null` value counts as listed. Sizing fields (`single_stage` /
`stages`) are checked down to the leaf, and `null` there means "whole
split" for that dimension (see "Sizing" below).

A couple of keys are worth calling out beyond what the YAML comments say:

- `embedding` — any OpenAI embedding model name (API, no GPU needed) or a
  local HF/NV embedding model (GPU, loaded in-process by HippoRAG).
- `llm_model` — used BOTH as HippoRAG's internal OpenIE/triple-extraction
  LLM and as the shared QA agent's model.
- `embedding_batch_size` / `embedding_dtype` — leave `null`: for API
  embeddings (the default `text-embedding-3-small`) this resolves to
  batch 16 / dtype auto; for local embeddings, batch 4 / dtype float16 —
  computed automatically in `_ensure_hippo`.

To switch embedders, datasets, etc., edit those keys in your config yaml —
e.g.:

```yaml
# my_hr.yaml — local GPU embedding, LongMemEval-s
dataset: longmemeval_s
embedding: nvidia/NV-Embed-v2
embedding_batch_size: 2
embedding_dtype: float16
```

then `uv run python run.py --config my_hr.yaml`.

## Sizing (config file only)

There are **no CLI flags at all** — evaluation sizes are config-file keys
resolved through the shared `common.evaluate` layer (the same one forge
uses):

- **`progressive: false` (default)** REQUIRES a `single_stage` block — ONE pass
  sized by its fields (`common.evaluate.single_stage_wire_spec`; a `null`
  or omitted field = the WHOLE split for that dimension, byte-identical to the
  main method's `forge.heldout` `progressive=false` when all-null). Omitting
  `single_stage` raises a clear `ValueError` (no silent whole-split).
- **`progressive: true`** runs the staged stage1→2→3 gauntlet; a `stages` block
  overrides the family `DEFAULT_STAGES`.

Both blocks use the family's NATIVE size fields (PER-UNIT counts):

| Field | Applies to | Meaning |
|---|---|---|
| `n_users` | dynamicmem | Users sampled from the split. `null`/omitted = whole split. |
| `n_conversations` | locomo | Conversations sampled from the split. `null`/omitted = whole split. |
| `n_questions` | longmemeval | Questions sampled from the split. `null`/omitted = whole split. |
| `n_checkpoints` | dynamicmem only | DynamicMem TCE checkpoints per user (of 5 quarterly checkpoints). `null` = all 5. |
| `n_task_a` | dynamicmem only | Task-A (`state_completion`) items sampled per checkpoint. `null` = full bucket. |
| `n_task_c` | dynamicmem only | Task-C (`apply_service`) items sampled per checkpoint. `null` = full bucket. |
| `n_qa` | locomo only | QA pairs sampled per conversation. `null` = all (categories 1-4 only; cat-5 excluded). |

```yaml
# my_hr.yaml — 2 DynamicMem users, whole checkpoint/item buckets
dataset: dynamicmem
progressive: false
single_stage: {n_users: 2}
```

## Output

Each eval run creates a fresh per-instance `outputs/<uuid>_<embedding>/` HippoRAG graph directory (embeddings + knowledge graph) per user, and these directories accumulate across runs (gitignored but growing on disk) — periodically clean `baselines/harness/hipporag2/outputs/` if disk is a concern.

```
baselines/harness/hipporag2/
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
