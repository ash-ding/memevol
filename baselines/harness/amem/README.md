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

## Usage

    baselines/venv/bin/python baselines/harness/amem/run.py --dataset locomo
    baselines/venv/bin/python baselines/harness/amem/run.py --dataset dynamicmem \
        --split search --stage-spec '{"n_samples": 1, "n_checkpoints": 1, "n_task_a": 1, "n_task_c": 1}'

Flags: `--amem_llm_model` (default `gpt-4o-mini`, A-mem's own default — its
`OpenAIController` hardcodes `temperature=0.7` + `max_tokens=1000`, which the
gpt-5 family rejects, so keep a 4-series model); `--retrieve_k` (default 10,
upstream default); `--llm_model` / `--judge_model` (default `gpt-5-mini` —
shared QA agent + judge, baseline convention); `--split`; `--stage-spec`.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | whole `memory_layer.py`; locomo note unit `"Speaker {X}says : {text}"` + session date (missing-space quirk preserved); keywords-rewrite prompt + JSON schema; `retrieve_k=10`; `evo_threshold=100`; internal gpt-4o-mini |
| Integration adaptations (not algorithm) | longmemeval (per message) / dynamicmem (per app-log entry, hipporag2's `app_log_to_passage` text) ingestion mapping — A-mem only defined LoCoMo; answering via the shared QA agent; `_st_shim.py` (memevol's `datasets/` shadows HF `datasets`, an ST 5.x import-time dep); per-note `print` flood redirected to devnull |
| Upstream quirks preserved | `find_related_memories_raw` neighbor-cap loop behavior; `"says :"` spacing |

## Cost profile

2 gpt-4o-mini calls per ingested note (analysis + evolution) + 1 per query
(keywords rewrite). Internal calls use A-mem's own OpenAI client and are NOT
tracked by `common.tokens` (same caveat as HippoRAG's internal calls). LoCoMo
≈ 300–600 turns/conversation → expect ~20–60 min and ~$1 per conversation at
build time (calls are sync + serial).

Tests: `baselines/venv/bin/python tests/test_amem_baseline.py` (baselines
venv only — heavy imports).
