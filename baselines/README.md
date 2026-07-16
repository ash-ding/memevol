# Baselines

Comparison methods for the memory-architecture search. Each baseline runs
against the same benchmark set as the main method ([forge/](../forge/)),
producing comparable metrics: per-user reward, judge-scored accuracy, and
(for some) token / latency telemetry.

## Layout — two kinds of baseline

`baselines/` is split by WHAT is being compared:

```
baselines/
├── registry.py          # shared dataset registry (both sides import it)
├── requirements.txt     # shared full ML install
├── venv/                # shared Python 3.12 venv (gitignored)
├── evolve/              # SEARCH-METHOD baselines — compared against forge ITSELF
│   └── alma/            #   LLM-meta-agent search loop (memevol's original method)
└── harness/             # READY-MADE MEMORY SYSTEMS — compared against forge's
    ├── eval_common.py   #   EVOLVED HARNESSES. Shared runner: run_baseline()
    ├── cc/              #   Claude Code as direct QA agent (native answer)
    └── hipporag2/       #   HippoRAG2 graph-RAG pipeline as retrieval memory
```

- **`evolve/`** — methods that SEARCH over memory-structure code, like forge
  does. Their unit of comparison is the search loop itself (proposer quality,
  sample efficiency, final evolved-harness score vs forge's).
- **`harness/`** — fixed, hand-written memory systems implementing the same
  standardized 3-hook `MemoStructure` contract
  (`build_memory_from_data` / `retrieve_memory_for_query` /
  `use_memory_to_answer`) that forge-evolved harnesses implement. Their unit
  of comparison is the harness artifact: they run through the SAME
  per-dataset workflows via `baselines.harness.eval_common.run_baseline`, so
  their scores sit on the same axis as any evolved harness's.

**DynamicMem protocol status (since the 2026-07 TCE upgrade)**: alma runs
the shared `DynamicMemWorkflow`, so it automatically follows the official
TCE v2 checkpoint protocol (checkpoint-interleaved ingestion, two task
families, official holistic 0–1 judge) — its numbers ARE comparable with
forge. `hipporag2` and `cc` both drive their `MemoStructure`
(`HippoRAGMemo` / `CCMemo`) through `baselines.harness.eval_common.run_baseline`,
which resolves the SAME production per-dataset workflow the main method
uses (`baselines.registry.resolve`), so their DynamicMem numbers get the
official TCE protocol too, and both run on all four datasets
(dynamicmem/locomo/longmemeval_s/longmemeval_m) via one `run.py --dataset
...` entrypoint. `cc` bypasses the shared QA agent (`CCMemo.use_memory_to_answer`,
the standardized answer hook every `MemoStructure` may implement) — its own
tool-using answer is judged verbatim instead of being relayed through a
second LLM call.

All baselines share **`baselines/venv/`** (full ML install:
`pip install -r baselines/requirements.txt`) and write artifacts under
each baseline's own `logs/` and `results/` directories (gitignored).

| Baseline | Kind | Approach | Optimization | Best for |
|---|---|---|---|---|
| **[evolve/alma](evolve/alma/)** | search method | LLM-meta-agent search loop | Yes — propose / select / evolve over harness code | Established baseline; the framework's "v1" memory-architecture search |
| **[harness/cc](harness/cc/)** | ready-made harness | Claude Code as direct QA agent (native answer, multi-dataset) | None (zero-shot) | What-if: just give CC the raw user data + tools and let it answer |
| **[harness/hipporag2](harness/hipporag2/)** | ready-made harness | Graph-based RAG pipeline as retrieval memory (OpenIE → KG → PPR retrieval → passages; shared QA agent answers) | None (fixed pipeline) | Hand-designed memory architecture comparison point, multi-dataset |

---

## evolve/alma — meta-learning loop

The original method memevol shipped with: an LLM meta-agent reads sampled
QA trajectories, identifies failure patterns, and proposes new memory-
structure code. Sanity-checks each candidate, then evaluates on the search
split. Softmax-weighted parent selection over reward.

```bash
# Smoke — 2 users, 2 steps
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status search --eval_n_samples 2 --eval_n_qa 10 --steps 2

# Full training
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status search --eval_n_samples 6 --eval_n_qa 20 --steps 10

# Held-out evaluation of a saved memo
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status test --memo_SHA <SHA>
```

Artifacts: `baselines/evolve/alma/{logs/, memo_archive/, results/}`. See
[evolve/alma/README.md](evolve/alma/README.md) for layout details.

**Key difference from forge**: alma's proposer is a single LLM call with
compressed feedback (sampled trajectories + meta-prompt), whereas forge's
proposer is an agentic CC SDK call with full filesystem access to all
prior code, traces, and scores.

---

## harness/cc — Claude Code as direct QA agent

Skips memory-architecture design entirely. `CCMemo`
(`baselines/harness/cc/memo.py`) is a `MemoStructure` run through the same
`baselines.harness.eval_common.run_baseline` shared runner as hipporag2, on
any of the four datasets:

- **Phase 1 (`build_memory_from_data`)**: stashes the currently-visible data
  (dynamicmem: app_logs; locomo: conversation; longmemeval: sessions —
  dispatch on `recorder.init` keys) into a per-user temp directory as a
  single JSON file.
- **Phase 2 (`retrieve_memory_for_query`)**: runs Claude Code with tool access
  (Read, Grep, Glob) to that temp directory and asks it to answer the
  question directly — no separate retrieval step.
- **Answer (`use_memory_to_answer`)**: runs Claude Code on the workflow's exact
  formatted prompt so the workflow judges cc's own answer verbatim,
  bypassing the shared QA agent entirely.

```bash
# Full eval on one dataset
baselines/venv/bin/python baselines/harness/cc/run.py \
    --dataset locomo --model claude-sonnet-4-20250514

baselines/venv/bin/python baselines/harness/cc/run.py \
    --dataset dynamicmem --stage-spec '{"n_samples": 2}'
```

Useful as an upper-bound reference: how well does a strong agent do **with
no learned memory structure at all**, just raw access + tools?

Artifacts: `baselines/harness/cc/results/<dataset>/<split>/`.

---

## harness/hipporag2 — graph-based RAG pipeline as a retrieval MemoStructure

`HippoRAGMemo` (`baselines/harness/hipporag2/memo.py`) wraps
[HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s pipeline as a
`MemoStructure` — a fixed (non-evolved) memory architecture run through the
same `baselines.harness.eval_common.run_baseline` shared runner as the rest
of the baselines, on any of the four datasets:

- **Phase 1 (`build_memory_from_data`)**: converts the ingested unit's data into text
  passages (dynamicmem: app_logs; locomo: conversation turns; longmemeval:
  session messages — dispatch on `recorder.init` keys) and indexes them into a
  per-user HippoRAG graph (OpenIE → NER + triples → knowledge graph + entity
  embeddings). Indexing is additive across calls, so DynamicMem's per-
  checkpoint segments accumulate correctly.
- **Phase 2 (`retrieve_memory_for_query`)**: fact retrieval → reranking → personalized
  PageRank → top-k passages, returned as `{"passages": [...]}`. The **shared
  QA agent** (not HippoRAG's own `rag_qa` reader) answers from those passages,
  and the per-dataset workflow judges/scores identically to the main method —
  a fair "HippoRAG-as-memory" comparison point rather than an end-to-end
  HippoRAG pipeline comparison.

```bash
# OpenAI API embedding (no GPU needed)
baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset locomo --embedding text-embedding-3-small

baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset dynamicmem --stage-spec '{"n_samples": 2}'

# Local GPU embedding (NVIDIA)
baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset longmemeval_s --embedding nvidia/NV-Embed-v2 \
    --embedding_batch_size 2 --embedding_dtype float16
```

Useful as a hand-designed comparison: how does a published RAG pipeline
perform on these benchmarks vs forge-evolved harnesses?

Artifacts: `baselines/harness/hipporag2/{outputs/, results/<dataset>/<split>/}`.

---

## Adding a new harness baseline (mem0 / letta / zep / ...)

A new ready-made memory system needs exactly two files under
`baselines/harness/<name>/`:

1. **`memo.py`** — a `common.harness_base.MemoStructure` subclass
   implementing the 3-hook contract:
   - `async build_memory_from_data(recorder)` — ingest the data visible in
     `recorder.init` (dispatch on its keys: `app_logs` / `conversation` /
     `sessions`). Called once per build call; accumulate across calls
     (DynamicMem delivers per-checkpoint deltas).
   - `async retrieve_memory_for_query(recorder)` — return the dict fed to
     the shared QA agent (e.g. `{"passages": [...]}`).
   - `async use_memory_to_answer(recorder, retrieved, prompt)` — OPTIONAL;
     return an answer string to bypass the shared QA agent (like cc), or
     `None`/omit to use it (like hipporag2).
   Per-run config travels as the `_cfg` class attribute
   (`eval_common.make_memo_class`) — the workflow instantiates the class
   with no args.
2. **`run.py`** — an argparse CLI that builds the configured memo class and
   calls `baselines.harness.eval_common.run_baseline(dataset=..., split=...,
   user_stage_spec=parse_stage_spec(...), memo_class=..., qa_model=...,
   judge_model=..., out_dir=...)`. Copy `harness/hipporag2/run.py` as the
   template — it is ~40 lines.

That's it: split resolution, per-dataset judge, scoring, and the full
DynamicMem TCE checkpoint protocol all come from the shared runner, so the
numbers are directly comparable to forge-evolved harnesses (same code, not
"comparable" code). No `__init__.py` needed (namespace packages).

---

## Shared foundation

All baselines (and forge) build on the same dataset adapters and judge:

- **[`baselines/registry.py`](registry.py)** — dataset name → (workflow,
  env module, recorder) resolution, shared by BOTH `evolve/` and `harness/`
  (mirrors `forge/launch.py::WORKFLOWS`; baselines never import forge).
- **[`datasets/<bench>/env.py`](../datasets/)** — `load_user_data`,
  `get_task_list`, per-benchmark Recorder.
- **[`common/judge.py`](../common/judge.py)** — LLM-as-judge with
  configurable prompt template and score range. Scores are directly
  comparable when methods share the same judge config (see the DynamicMem
  protocol-status note above for the current exception).
- **[`common/llm.py`](../common/llm.py)** — `Agent` / `Embedding`
  wrappers with automatic token tracking; baselines use these so their
  cost numbers are comparable to forge's.

## License

MIT (same as the main project).
