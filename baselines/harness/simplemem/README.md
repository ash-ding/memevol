# SimpleMem baseline

[SimpleMem](https://github.com/aiming-lab/SimpleMem) — semantic-compression
lifelong memory — as a ready-made memory system on the 3-hook `MemoClass`
contract. The paper PDF is in this directory ([simplemem.pdf](simplemem.pdf)).

**Provenance**: the `src/simplemem/` subtree is vendored from
<https://github.com/aiming-lab/SimpleMem> @
`db80b6a7c591e0ea730a058e9f5fc4eb06572299`. Only the **text** pipeline is
vendored — `simplemem/text/` + `simplemem/core/` — which is byte-identical to
upstream; the heavy `multimodal/`, `evolver/`, and `integrations/` subtrees are
NOT vendored (this baseline evaluates SimpleMem's text memory only). Verify the
core+text files are unmodified:

    UP=$(mktemp -d) && git clone https://github.com/aiming-lab/SimpleMem "$UP" \
      && git -C "$UP" checkout db80b6a
    diff -r "$UP/simplemem/core" src/simplemem/core && echo "core: identical"
    diff -r "$UP/simplemem/text" src/simplemem/text && echo "text: identical"

The one vendored file that is **not** byte-identical is
`src/simplemem/__init__.py` — upstream's package initializer eagerly imports
the AutoMemory router (→ multimodal / evolver), so it is replaced with a minimal
initializer that imports nothing (the baseline imports
`simplemem.text.system.SimpleMemSystem` directly). All integration code lives in
`memo.py` / `run.py`, never in `src/`.

## How it works

SimpleMem's three-stage text pipeline runs untouched:

1. **Semantic structured compression** — BUILD windows the ingested `Dialogue`s
   (`WINDOW_SIZE=40`, overlap 2) and an LLM distills each window into
   self-contained `MemoryEntry` units: a coreference-resolved "lossless
   restatement" with absolute timestamps, plus keywords and
   persons/entities/location/topic metadata.
2. **Online semantic synthesis** — intra-session consolidation during the write
   path (redundancy removed at ingestion, not retrieval).
3. **Intent-aware retrieval planning** — RETRIEVE runs
   `HybridRetriever.retrieve(query)`: multi-query planning + a three-view search
   (semantic vector / keyword / structured-metadata) over a per-user LanceDB
   index, with optional reflection rounds.

Retrieved memory units are returned as `{"passages": [...]}` and the **shared QA
agent answers** — `use_memory_to_answer` is not overridden (hipporag2/amem
pattern). This keeps the comparison about *memory* (SimpleMem's compression +
retrieval), not about SimpleMem's own `answer_generator`.

## Setup

simplemem has its own uv-managed venv, built from its own self-contained
`pyproject.toml` (lancedb, pyarrow, tantivy, dateparser,
sentence-transformers, transformers, torch):

    cd baselines/harness/simplemem && uv sync

This creates `baselines/harness/simplemem/.venv/`. The repo-root
`.venv/` is dev/test only and cannot run simplemem.

## Usage

    cd baselines/harness/simplemem && uv run python run.py --config config.example.yaml

`run.py` takes exactly one flag, `--config <yaml>` (required) — there is no
other CLI surface. Every parameter lives in the config file:
`config.example.yaml` documents each key inline — copy it, edit the values
(e.g. `dataset: dynamicmem`, `split: search`), and point `--config` at your
copy. The YAML must list EXACTLY the keys `run.py`'s `REQUIRED_KEYS`
expects — a missing key OR an unknown key aborts the run before anything
executes; a `null` value counts as listed.

SimpleMem-specific keys worth calling out: `simplemem_llm_model` (default
`gpt-4.1-mini` — SimpleMem's own default; **4-series only**, since its
`LLMClient` sends `temperature=0.1..0.3`, which the gpt-5 family rejects);
`embedding_model` (default `Qwen/Qwen3-Embedding-0.6B`, the faithful local
sentence-transformer; set `all-MiniLM-L6-v2` for the light fallback). The
rest (`base_url`, `window_size`, `overlap_size`, `semantic_top_k`,
`keyword_top_k`, `structured_top_k`, `enable_planning`, `enable_reflection`,
`max_reflection_rounds`, `enable_parallel_processing`,
`max_parallel_workers`, `enable_parallel_retrieval`,
`max_retrieval_workers`, `llm_model`, `judge_model`, `progressive`,
`sampling_seed`, `memory_cache`) are documented inline in
`config.example.yaml`.

**Sizing is config-file only** (there is no sizing CLI surface either).
`progressive: false` (default) REQUIRES a `single_stage` block; `progressive:
true` sizes from a `stages` block. See `config.example.yaml`.

## Ingestion mapping (`recorder.init` → `Dialogue`)

| Benchmark | init key | one `Dialogue` per… | speaker / content / time |
|---|---|---|---|
| locomo | `conversation` | conversation turn | `speaker` / `text` / session `date_time` (native SimpleMem format) |
| longmemeval_s/m | `sessions` | message | `role` / `content` / session `date` |
| dynamicmem | `app_logs` | app-log entry | `app_name` / hipporag2's `app_log_to_passage` text / log `timestamp` |

`Dialogue` ids are per-instance sequential and continue across BUILD calls, so
DynamicMem's per-checkpoint deltas accumulate correctly (each BUILD windows only
the new segment; `finalize` flushes its remainder).

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`config.example.yaml`) | unified arm (`config.unified.yaml`) |
|---|---|---|
| internal LLM (`simplemem_llm_model`) | `gpt-4.1-mini` — the paper's backbone (§3.1/§3.3) | `gpt-5-mini` |
| embedder (`embedding_model`) | `Qwen/Qwen3-Embedding-0.6B`, local, 1024-dim — the paper's (§3.1) | `text-embedding-3-small`, API, 1536-dim |

**The faithful arm is the default**, and it is what the faithfulness table
below and every number in this README describe. The unified arm puts all seven
baselines on one LLM and one embedder so the comparison against the main method
is like-for-like — it is a deliberate deviation from the paper, and its numbers
must not be quoted as SimpleMem's published result.

Both arms leave `src/` **byte-identical** — the `diff -r` above still passes.
Two boundary levers in [`../model_config.py`](../model_config.py) make that
possible:

- **the embedder.** `EmbeddingModel.__init__` dispatches on
  `model_name.startswith("qwen3")` and builds its own SentenceTransformer, with
  no injection point, so the shared factory patches that constructor. It
  memoizes local weights across users (a fresh MemoClass is built per user, and
  the Qwen3 weights are ~0.6B) and returns an API-backed `.encode()`-compatible
  adapter for a `text-embedding-*` name. The configured value reaches SimpleMem
  through its `EMBEDDING_MODEL` env setting, so the name the vendored code
  requests IS the configured one and the factory dispatches on it directly.
- **the LLM.** SimpleMem's `LLMClient` sends `temperature=0.1..0.3`, which the
  gpt-5 family rejects — so before this, SimpleMem could not run a gpt-5 model
  at all. The shim drops the rejected params and renames `max_tokens` →
  `max_completion_tokens` at the OpenAI-SDK boundary.

No dimension knob has to move with the embedder: SimpleMem sizes its LanceDB
table from `embedding_model.dimension`. The per-user store is rebuilt with
`clear_db=True`, but a `memory_cache: true` gauntlet snapshot taken at 1024-dim
is invalid under the 1536-dim arm.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `src/simplemem/{core,text}` (compression, hybrid retrieval, planning, reflection, answer prompts, LanceDB backend); `OVERLAP_SIZE=2`; `SEMANTIC/KEYWORD/STRUCTURED_TOP_K=25/5/5`; internal LLM `gpt-4.1-mini` (paper §3.1/§3.3); `Qwen/Qwen3-Embedding-0.6B` embedder (paper §3.1, 1024-dim) |
| Paper over code | `window_size: 20` — the paper (§3.1) states "a sliding window of size W = 20" while the vendored code ships `WINDOW_SIZE=40`. The paper's value is the default, matching how memoryos resolves the same kind of divergence. **Numbers collected at 40 are not comparable to numbers collected at 20.** |
| Integration adaptations (not algorithm) | longmemeval (per message) / dynamicmem (per app-log entry, hipporag2's `app_log_to_passage` text) ingestion mapping — SimpleMem only defined LoCoMo; answering via the shared QA agent; the shared embedder factory from [`../model_config.py`](../model_config.py) (see **Model configuration** above — a process-wide cache so the 0.6B weights load once, not per user, and the seam the API-embedder arm is injected through); the vendored chain imported exactly once at `memo.py` module scope, avoiding a pyarrow re-registration crash; `src/simplemem/__init__.py` trimmed to keep multimodal/evolver off the import path; `use_streaming=false` (identical output, no console flood) |
| Kept at SimpleMem defaults (faithful path) | `enable_parallel_processing`/`enable_parallel_retrieval=true` (SimpleMem's shipped defaults, and the path its LoCoMo eval uses) — NOTE the serial and parallel build paths are **not** equivalent: the serial path feeds each window the previous window's entries as dedup context, the parallel path processes windows independently, so the parallel output is the faithful one. It is also the only real build parallelism (the async build hook body is synchronous + blocking, so users don't overlap under `max_sample_concurrent` and there is no thread-count multiplication). Tune via `max_parallel_workers` (16) / `max_retrieval_workers` (8) |
| Known consequences | SimpleMem compresses source turns into `MemoryEntry` units that carry NO `app_log_id`, so DynamicMem evidence-citation scoring is disadvantaged (inherent to compression-first memory); LoCoMo is SimpleMem's home benchmark and its tuned `WINDOW_SIZE` — other datasets use the same size unless overridden |

## Cost / performance profile

- **Build** dominates: 1 `gpt-4.1-mini` compression call per ~`WINDOW_SIZE`
  dialogues (with overlap), plus a local embedding pass. **Retrieve** adds
  planning + (optional) reflection `gpt-4.1-mini` calls per query, then the
  shared `gpt-5-mini` QA + `gpt-5-mini` judge.
- SimpleMem's internal `gpt-4.1-mini` calls do **not** flow through
  `common.tokens` (same caveat as amem / HippoRAG), so only the shared QA/judge
  side appears in `token_usage.json`.
- The faithful embedder (`Qwen/Qwen3-Embedding-0.6B`) is a ~0.6B local model:
  it benefits from a GPU and downloads once from HuggingFace.
  The shared embedder factory loads it once per process and shares it
  across users.
- Build/retrieve are synchronous + blocking, so users don't overlap under
  `max_sample_concurrent` (a blocking hook body stalls the event loop). The
  real build speedup is SimpleMem's own window parallelism
  (`enable_parallel_processing`, `max_parallel_workers`, default 16), which
  runs the per-window compression LLM calls concurrently; retrieval parallelism
  is `max_retrieval_workers` (default 8). Keep the worker counts within your
  OpenAI rate limits.

## Validation status

Written against the vendored code and the 3-hook contract; **not yet run
end-to-end** here (this baseline's own venv + a GPU for the Qwen3 embedder + an
OpenAI key are only available on the eval server). To smoke each ingestion branch
cheaply on the search split (mirrors amem's per-branch check):

Smoke configs are LOCAL scratch files — `.gitignore` keeps
`baselines/harness/simplemem/smoke_*.yaml` out of the repo (same as lightmem and
zep), so create them yourself. Each is a full copy of `config.example.yaml` with
only `dataset` / `split` / `single_stage` changed; exact config is
unconditional, so a partial override file will abort before running.

    # locomo (conversation branch) — 1 conv, 3 QAs
    #   smoke_locomo.yaml: dataset: locomo, split: search,
    #   single_stage: {n_conversations: 1, n_qa: 3}
    cd baselines/harness/simplemem && uv run python run.py --config smoke_locomo.yaml

    # dynamicmem (app_logs branch) — 1 user, checkpoint interleaving
    #   smoke_dm.yaml: dataset: dynamicmem, split: search,
    #   single_stage: {n_users: 1, n_checkpoints: 1, n_task_a: 1, n_task_c: 1}
    uv run python run.py --config smoke_dm.yaml

    # longmemeval_s (sessions branch) — 1 question
    #   smoke_lme.yaml: dataset: longmemeval_s, split: search,
    #   single_stage: {n_questions: 1}
    uv run python run.py --config smoke_lme.yaml

Read `results/<dataset>/search/traces/<user>.json` to confirm build → retrieve
→ QA runs, `invalid_users` is empty, and the retrieved `passages` are non-empty
compressed memory units in the expected format.
