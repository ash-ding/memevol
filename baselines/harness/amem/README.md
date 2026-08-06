# A-mem baseline

[A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110) as a
ready-made memory system on the 3-hook `MemoStructure` contract.

**Provenance**: `memory_layer.py` is vendored VERBATIM from
<https://github.com/WujiangXu/A-mem> @
`0c8039f28fdcc08189a23c07a3437d9d2482f9c2` — the paper-reproduction core
(NOT `memory_layer_robust.py`, NOT the A-mem-sys package). Below its 8-line
provenance header the file is byte-identical to upstream:

    tail -n +9 memory_layer.py | diff - <(git -C /export/scratch_large/ding/code/A-mem show 0c8039f:memory_layer.py)

## How it works

Every ingestion unit becomes an A-mem note (`AgenticMemorySystem.add_note`):
an LLM analyzes content into keywords/context/tags, a second LLM call decides
memory evolution (strengthen links / update neighbors), notes are embedded
with `all-MiniLM-L6-v2` (sentence-transformers, local), and the store
consolidates every 100 evolutions. Retrieval rewrites the question into
keywords (LLM; prompt verbatim from the official eval driver
`test_advanced.py::generate_query_llm`) and returns the top-k notes + linked
neighbors as one formatted string in `{"memories": ...}`. The shared QA agent
answers — `use_memory_to_answer` is not overridden (hipporag2 pattern; note
memevol's locomo QA prompts are themselves ported from the IREM A-mem
baseline, so the answer side is already A-mem-shaped there).

## Setup

amem has its own `.venv/`, built from its own self-contained `pyproject.toml`
(sentence-transformers, transformers, torch, nltk, rank_bm25, scikit-learn,
litellm on top of the shared core deps) with a committed `uv.lock`:

    baselines/setup_venv.sh amem   # thin wrapper over `uv sync`
    # or directly:
    cd baselines/harness/amem && uv sync

This creates `baselines/harness/amem/.venv/`. The repo-root `.venv/` is
dev/test only and cannot run amem.

## Usage

    cd baselines/harness/amem && uv run python run.py \
        --config config.example.yaml
    uv run python run.py \
        --config my_amem.yaml --dataset dynamicmem --split search

Flags: `--config` (YAML path; CLI flags override it); `--amem_llm_model`
(default `gpt-4o-mini`, A-mem's own default — its `OpenAIController` hardcodes
`temperature=0.7` + `max_tokens=1000`, which the gpt-5 family rejects, so keep
a 4-series model); `--retrieve_k` (default 10, upstream default); `--llm_model`
/ `--judge_model` (default `gpt-5-mini` — shared QA agent + judge, baseline
convention); `--split`.

Shared progressive-sampling flags (same as cc/hipporag2): `--progressive` /
`--no-progressive` (default off — run the staged stage1→2→3 gauntlet with
threshold elimination instead of one single-stage pass); `--sampling-seed`
(default `42`, base seed for the fixed step-0 sample; a no-op at whole-split);
`--no-memory-cache` (disable cross-stage Phase-1 memory reuse, on by default).

**Sizing is config-file only** (there are no sizing CLI flags).
`progressive: false` (default) REQUIRES a `single_stage` block — ONE
pass sized by its native fields (`n_conversations` / `n_qa` for locomo,
`n_users` / `n_checkpoints` / `n_task_a` / `n_task_c` for dynamicmem,
`n_questions` for longmemeval; a `null`/omitted field = the WHOLE split for
that dimension). Omitting `single_stage` raises a clear `ValueError` (no silent
whole-split). `progressive: true` sizes from a `stages` block (overrides the
family `DEFAULT_STAGES`). See `config.example.yaml`.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `memory_layer.py`; locomo note unit `"Speaker {X}says : {text}"` + session date (missing-space quirk preserved); keywords-rewrite prompt + JSON schema; `retrieve_k=10`; `evo_threshold=100`; internal gpt-4o-mini |
| Integration adaptations (not algorithm) | longmemeval (per message) / dynamicmem (per app-log entry, hipporag2's `app_log_to_passage` text) ingestion mapping — A-mem only defined LoCoMo; answering via the shared QA agent; `_st_shim.py` (memevol's `datasets/` shadows HF `datasets`, an ST 5.x import-time dep); per-note `print` flood redirected to devnull |
| Upstream quirks preserved | `find_related_memories_raw` neighbor-cap loop behavior; `"says :"` spacing |

## Smoke verification (per code path)

`_init_to_note_units` has three ingestion branches; each was verified
end-to-end on a small slice of real data (`--split search`), confirming
build → retrieve → QA runs, `invalid_users` is empty, and retrieved memories
are non-empty in the expected note format:

| Dataset | Branch | Spec | Result | Retrieved-memory format |
|---|---|---|---|---|
| dynamicmem | `app_logs` | `{"n_samples":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1}` | overall 0.75, invalid=[] | `talk start time:…memory content:` (app-log passage) |
| locomo | `conversation` | `{"n_samples":1,"n_qa":3}` | overall 0.667, invalid=[] | verbatim `Speaker Xsays :` (missing-space quirk) |
| longmemeval_s | `sessions` | `{"n_samples":1}` | overall 1.0, invalid=[] | `role: content` (`assistant: …` / `user: …`) |
| longmemeval_m | `sessions` (same as _s) | — | not run | — |

Scores are single-sample sanity signals, NOT benchmark numbers.
`longmemeval_m` shares the exact `sessions` branch as `longmemeval_s`, so `_s`
already exercises its code path; it was skipped because one question ≈ 5000
messages (~10× `_s`) — see the estimate below.

## Cost profile

Build dominates: **2 gpt-4o-mini calls per ingested note** (analyze_content +
process_memory evolution), plus per query 1 gpt-4o-mini keyword rewrite + 1
gpt-5-mini QA + 1 gpt-5-mini judge. A-mem's internal gpt-4o-mini calls do NOT
flow through `common.tokens` (same caveat as HippoRAG), so the build-side
figures below are STRUCTURAL estimates (~1.5k in / ~0.33k out per note, ±50%);
the gpt-5-mini QA/judge side is calibrated from the smoke runs' tracked usage.

Build is **serial + blocking** (A-mem is synchronous), and
`consolidate_memories` re-embeds the whole accumulated corpus every 100
evolutions (≈ O(n²) local MiniLM work), so per-note wall-clock GROWS with
corpus size (~3 s/note at ~500 notes, more beyond). Samples run at
`--max_sample_concurrent` (default 3).

### Full held-out test-set estimate (per dataset)

Assumed pricing: gpt-4o-mini $0.15 / $0.60 per 1M in/out; gpt-5-mini ~$0.25 /
$2.00 per 1M in/out (plug in real rates — the total is gpt-4o-mini-build-dominated).

| Dataset (test split) | build notes | queries | est. API $ | est. wall-clock |
|---|---|---|---|---|
| dynamicmem (4 users) | ~7 k | ~1.6 k | ~$14 | ~4–5 h |
| locomo (4 convs) | ~2.4 k | 655 | ~$2 | ~1 h |
| longmemeval_s (200 q) | ~101 k | 200 | ~$40 | ~2.5 days |
| longmemeval_m (200 q) | ~977 k | 200 | ~$390 | ~5–6 weeks |

- **Excluding longmemeval_m: ≈ $55 total, ≈ 3 days wall-clock** (dominated by
  longmemeval_s's serial build).
- **Including longmemeval_m: ≈ $450 total, many weeks** — the per-message note
  model + O(n²) consolidate makes it impractical at full scale without
  engineering changes (coarser ingestion / lower consolidate frequency), which
  would depart from the faithful method.

Time, not money, is the binding constraint. The build is API-bound, so raising
`--max_sample_concurrent` (within OpenAI rate limits) is the main lever.

Tests: `cd baselines/harness/amem && uv run python tests/test_amem_baseline.py`
(amem's own project — heavy imports; the repo-root `.venv/` is dev/test
core only and doesn't have them).
