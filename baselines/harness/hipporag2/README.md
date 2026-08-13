# hipporag2 baseline — graph-based RAG pipeline as a retrieval MemoClass

`HippoRAGMemo` (`memo.py`) wraps [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s
pipeline (OpenIE → NER + triples → knowledge graph + entity embeddings →
personalized PageRank retrieval) as a `MemoClass` — a fixed,
non-evolved memory architecture. The baseline vendors and drives `hipporag`
directly.

**Provenance**: `src/hipporag/` is vendored VERBATIM (byte-identical, all 61
files, no exclusions inside the package) from
<https://github.com/OSU-NLP-Group/HippoRAG> @
`c617143f01477243992a63b2e2151cc003dd3b21` (`main`, version `2.0.0-alpha.4` —
the version the integration was written against), excluding only the upstream
repo's non-package content (`examples/`, `reproduce/`, `tests/`, `main.py`,
`images/`, `outputs/`, `setup.py`, `requirements.txt`). No file under
`src/hipporag/` is edited — provenance lives here, not in per-file headers, to
preserve byte-identity:

    git -C <hipporag-clone> archive c617143 src/hipporag | tar -x -C /tmp/hr
    diff -r /tmp/hr/src/hipporag src/hipporag        # 0 diffs

Nothing inside the package is pruned because upstream's
`embedding_model/__init__.py` and `llm/__init__.py` eagerly import every backend,
so pruning would mean editing vendored files; the whole package is 524K.

**This replaces an editable install of an external checkout.** Until 2026-08 the
`hipporag` package came from `uv pip install -e <path-to-a-HippoRAG-clone>` — not
in `pyproject.toml`, not in `uv.lock`, so `uv sync` produced a broken env, no
commit was pinned to any recorded number, and the checkout could be edited
underneath a run. (By the time this was vendored that path no longer existed on
the machine, which is the failure mode itself.) Whoever needs the old behaviour
should note that the vendored commit is `main` at vendoring time, not a recovered
copy of that checkout — an unrecorded local edit to it, if there was one, is not
reproduced here.

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
committed `uv.lock`), and `uv sync` alone is the whole setup — no editable
install, no external checkout:

```bash
cd baselines/harness/hipporag2 && uv sync
```

This creates `baselines/harness/hipporag2/.venv/`. The repo-root
`.venv/` is dev/test only and cannot run hipporag2.

`pyproject.toml` carries the deps the vendored package actually imports, pinned
as upstream's own `setup.py` pins them (torch 2.5.1, transformers 4.45.2, litellm
1.73.1, gritlm 1.0.2, python_igraph 0.11.8, pydantic 2.10.4, tenacity 8.5.0,
tiktoken 0.7.0) plus sentence-transformers, boto3, pandas/pyarrow, filelock,
packaging, requests, tqdm. Two deliberate departures from upstream's list, both
noted inline in `pyproject.toml`:

- **`vllm` / `outlines` are not installed.** Only `llm/vllm_offline.py`,
  `llm/transformers_offline.py` and `information_extraction/openie_vllm_offline.py`
  import them, and none is on the import path of the OpenAI-backed config this
  baseline runs. Those offline backends are therefore vendored but unavailable —
  selecting one raises `ModuleNotFoundError`. `vllm` also hard-pins torch and is a
  multi-GB install.
- **`networkx`, `scipy`, `einops`, `nest_asyncio` are dropped**: declared by
  upstream's `setup.py` but imported nowhere in `src/hipporag` (they serve
  upstream's `reproduce/` scripts). scipy/numpy/openai/httpx are already in the
  shared-core block.

`gritlm`, `boto3` and `litellm` are needed only because upstream's
`embedding_model/__init__.py` / `llm/__init__.py` import every backend eagerly —
not because the exercised path uses GritLM, Bedrock or LiteLLM. The env is ~6GB,
dominated by torch + gritlm's own transitive deps (mteb, wandb).

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

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | the whole `hipporag` package (@ c617143, 61 files, 0 diffs): OpenIE (NER + triple extraction), knowledge-graph construction, synonym/entity linking, personalized-PageRank retrieval, fact reranking, all prompts and thresholds, the parquet embedding store's dedup-by-hash upsert |
| Integration adaptations (not algorithm) | `src/` prepended to `sys.path` in `memo.py` so `import hipporag` resolves to the vendored copy; retrieval via `HippoRAG.retrieve()` only — HippoRAG's own `rag_qa` reader is NOT used, the shared QA agent answers from the returned passages (fair "HippoRAG-as-memory" comparison); per-dataset passage mappings (`_init_to_passages`: app_logs / conversation turns / session messages) — upstream ran QA corpora, not conversational memory benchmarks; one graph per `MemoClass` instance under `outputs/<instance_id>_<embedding>/`, keyed on an instance-scoped id rather than `recorder.user_id` (which is always `""` in practice — see the note in `memo.py`); incremental `index()` per build call, relying on upstream's additive dedup-by-hash behaviour |
| Not installed, so unavailable | the vLLM / offline-transformers LLM + OpenIE backends (vendored but `vllm`/`outlines` are not deps — see Setup); the chroma / milvus / qdrant vector-store backends (lazy imports, un-installed; the default parquet store is what runs) |
| Upstream quirks preserved | `embedding_model/__init__.py` and `llm/__init__.py` import every backend eagerly (hence gritlm/boto3/litellm in the deps); internal OpenIE LLM cost is not tracked by `common.tokens` (same caveat as amem/zep/simplemem/lightmem) |

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
