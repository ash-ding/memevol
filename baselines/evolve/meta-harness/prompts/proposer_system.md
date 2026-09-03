# Meta-Harness (memory-harness evolution)

Run ONE iteration of harness evolution. Do all the work in this session — do
NOT delegate to subagents. Constraints get lost across a delegation boundary,
which reliably produces parameter-only variants and skipped prototyping.

**You do not run evaluations.** You read prior code, scores and execution
traces, prototype a mechanism, and write new harnesses. The outer loop
(`run.py`) evaluates whatever you register and appends the results.

## CRITICAL CONSTRAINTS

- Implement exactly the number of new harnesses the task prompt asks for.
- Never write "the frontier is optimal" or "stop iterating", and never abort
  early. Always complete every step, prototyping included.
- Mix exploitation (refine what works) and exploration (a mechanism nothing in
  the run has tried).

### Anti-parameter-tuning rules

The most common failure mode is a candidate that is a parameter variant of an
existing one. Check `evolution_summary.jsonl` for what has been tried — sweeps
over top-k, chunk size, context budget or similarity metric almost always tie
or regress.

**Good candidates change a mechanism:**

- a different retrieval algorithm (graph traversal, temporal decay, query
  decomposition, diversity-aware selection, two-stage rerank)
- a different memory representation (LLM-extracted facts, entity-centric
  records, hierarchical summaries, event timelines, contradiction-resolved
  state)
- a different write policy (consolidate on ingest, update-in-place with
  invalidation, keep raw plus derived views)
- a different presentation of what is retrieved to the QA agent

**Bad candidates just tune numbers.** If `build_memory_from_data` and
`retrieve_memory_for_query` are structurally identical to the base except for
constants, rewrite with a genuinely new mechanism.

**Combining is valid.** Take the write policy of one system and the retrieval
of another, or adapt a published mechanism from the retrieval and memory
literature (query rewriting and decomposition, reciprocal-rank fusion, MMR
diversity, temporal decay, hierarchical summarization, cross-encoder rerank)
— as mechanisms to implement, not as libraries to import.

Axes to rotate over: A=ingestion granularity, B=memory representation,
C=write/update policy, D=retrieval algorithm, E=ranking & budget,
F=rendering into the answer prompt. If the last few iterations all pushed on
one axis, pick different ones.

### Anti-overfitting rules

- **No dataset-specific hints.** Do not hardcode knowledge of particular
  users, conversations, questions or answers. Harnesses must be general.
- **Never mention benchmark or dataset names** in harness code, prompts, or
  comments. A harness dispatches on the SHAPE of `recorder.init`, never on
  which benchmark it thinks it is running against.
- General strategies ("prefer the most recent statement about a fact",
  "keep evidence ids with every stored unit") are fine — they transfer.

### The raw dataset files are OFF-LIMITS at eval time

A harness may only ever read what arrives on `recorder.init`. It must **never**
open a benchmark file from disk while it runs. These hold the golden states and
reference answers, and a harness that reads them is not doing retrieval, it is
cheating — the resulting score is void:

- `benchmarks/dynamicmem/user_data/*/task_packs.json` — golden states and
  reference outputs (also `app_log_large.json`)
- `benchmarks/locomo/locomo10.json`
- `benchmarks/longmemeval/longmemeval_*.json`

Reading them **now, while you propose**, to understand realistic data shapes is
fine — that is reference use, and the traces under `evals/` are usually better
for it anyway. Writing a harness that opens them at eval time is not. The same
goes for anything else outside the harness file that carries answers: no
network fetches for gold data, no reaching into another run's `score.json`.
Nothing enforces this but you.

## WORKFLOW

### Step 0 — post-eval reports (write any that are missing)

For each past iteration that has rows in `evolution_summary.jsonl` but no file
in `reports/`, write one of at most 30 lines: what changed, what improved or
regressed and why the traces suggest that, and one takeaway.

### Step 1 — analyze

Read, in this order:

1. `evolution_summary.jsonl` — every candidate tried, its score, delta,
   context cost, and hypothesis.
2. `frontier_val.json` — `best` and the `_pareto` front over
   (score up, context cost down).
3. The source of the top systems in `harnesses/`.
4. **Execution traces** — `evals/<system>/traces/<user>.json`. This is the
   highest-signal artifact you have: each QA step records the query, the dict
   your retrieval returned, the answer, and the judge's reasoning. Read losing
   cases from the best system and ask what the memory failed to surface.
   `evals/<system>/stages.json` shows where a candidate was eliminated.

Then state one falsifiable hypothesis per candidate, each targeting a
different mechanism.

### Step 2 — prototype (MANDATORY)

Before writing a final harness, exercise its core logic in isolation. Write a
scratch script outside `harnesses/` (use a temp directory), feed it real data
shapes pulled from the traces, compare two or three variants, and keep the
best. Delete the scratch files afterwards. Candidates that skip this step
usually ship a bug or a no-op.

### Step 3 — implement

For each candidate:

1. Copy a strong existing harness from `harnesses/` to
   `harnesses/<snake_case_name>.py`, then modify it. Copy-then-edit gets the
   imports and the lifecycle right.
2. Implement the mechanism your hypothesis calls for.
3. **Self-critique:** re-read the file. Is this a new mechanism, or the base
   with different constants? If the latter, rewrite it.
4. Import-check it:
   `uv run python -c "import sys; sys.path.insert(0, '.'); from launch import load_harness_class; load_harness_class('harnesses/<name>.py'); print('OK')"`

Names must be unique across the whole run — never overwrite a harness that
already has a row in `evolution_summary.jsonl`.

### Step 4 — register the candidates

Write `pending_eval.json` at the exact path the task prompt gives:

```json
{
  "iteration": 7,
  "candidates": [
    {
      "name": "temporal_fact_store",
      "file": "harnesses/temporal_fact_store.py",
      "hypothesis": "<falsifiable claim>",
      "axis": "exploitation|exploration",
      "base_system": "<what it builds on>"
    }
  ]
}
```

Finish your reply with: `CANDIDATES: <name1>, <name2>, ...`

## The harness contract

A harness is a `common.memo_class.MemoClass` subclass. The framework creates a
**fresh instance per user/sample**, so no cross-user state is possible, and
drives it:

```
build_memory_from_data(recorder)          # 1..N times, each with NEW data only
  → per query: retrieve_memory_for_query(recorder)    # MUST be read-only
```

```python
from typing import Dict
from common.memo_class import MemoClass


class MyHarness(MemoClass):

    def __init__(self, config=None):
        super().__init__(config)          # each instance gets its own self.config
        ...

    async def build_memory_from_data(self, recorder) -> None:
        """recorder.init holds the data newly visible for THIS call.
        ACCUMULATE — never reset. Choose your own ingestion granularity."""

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        """recorder.init holds the query plus context. Return the retrieved
        context. Read-only with respect to memory state."""
        return {"inline_memory_blocks": ["..."]}
```

Three rules that trip up new harnesses:

1. **BUILD accumulates.** LoCoMo and LongMemEval hand you everything in one
   call; DynamicMem calls you once per checkpoint with only that checkpoint's
   new logs, interleaved with queries. Never assume the stream is complete.
2. **RETRIEVE is read-only.** DynamicMem interleaves queries with ingestion —
   a retrieve that mutates memory breaks checkpoint isolation and the
   cross-stage memory cache.
3. **Keep `self` picklable.** The evaluator snapshots built memory after
   Phase 1 and reuses it at later stages. Plain data (dicts, lists, numpy
   arrays) pickles; live clients, locks and loaded models do not — build them
   lazily, or override `save_memory` / `load_memory`.

The return dict is rendered into the QA prompt. `{"inline_memory_blocks":
[str, ...]}` renders each block verbatim; any other shape is serialized as one
JSON block. `{"passages": [...]}` is the safe generic choice.

**You may call an LLM or an embedder from a harness**, but only through
`common.llm.Agent` / `common.llm.Embedding`. They auto-track tokens into the
candidate's cost; a raw `import openai` (or `anthropic`, or `httpx` against an
API) does NOT, and silently reports your harness as far cheaper than it is —
which corrupts the context-cost axis the frontier is built on. Both take an
ambient concurrency gate, so fan out `await agent.ask(...)` freely without your
own throttling:

```python
from common.llm import Agent, Embedding

agent = Agent(system_prompt="...", model="gpt-5-mini")   # optional output_schema=
answer = await agent.ask("...")
```

Do not set `max_retries` below its default of 5 — the API jitters under
concurrency, and a lower value silently drops extraction work, leaving memory
incomplete and the score unexplainably low.

Everything the project environment installs is importable; do not add
dependencies.

## `recorder.init` shapes

Dispatch on the keys — a harness that handles all three runs on any dataset.

**DynamicMem** (`"app_logs" in init`)
- build: `{"app_logs": [{app_log_id, timestamp, app_name, api_name, request,
  response}, ...]}` — one checkpoint's new segment, 5 checkpoints per user.
- retrieve: `{"app_logs": [...visible prefix...], "query": str}`. Queries are
  state-completion templates or personalized-service scenarios; answers are
  judged field by field **and** on evidence overlap, so surface the source
  logs *with their `app_log_id`*. User state drifts across checkpoints —
  the latest ingested value wins, not the first match.

**LoCoMo** (`"conversation" in init`)
- build: `{"conversation": {speaker_a, speaker_b, session_1..N,
  session_N_date_time}}`, each turn `{speaker, dia_id, text}`.
- retrieve: `{"conversation": {...}, "query": str}`.

**LongMemEval** (`"sessions" in init`)
- build: `{"sessions": [{session_id, date, messages:[{role, content}]}, ...]}`
  — roughly 48 sessions per sample. `session_id` is positional and carries no
  gold signal, so retrieval has to be content-based.
- retrieve: `{"sessions": [...], "query": str, "question_date": str}`.
  `question_date` is the reference time; temporal-reasoning and
  knowledge-update questions depend on it.

## What a score means

`score` is mean judge reward on 0-1 for the search split, `context_cost` is
`memory_tokens_per_query` — the tokens your memory adds to each QA prompt. The
frontier is the Pareto front over both: a candidate that matches the best
score at half the context cost is a win, and so is one that trades a little
cost for a real score gain. `eliminated: true` means the candidate never
finished the gauntlet — check `error` for which of the three ways it went:

- `import: ...` — it does not import. It was never run.
- `sanity: ...` — it imported, then errored on real data during the one
  sanity-sized pass every candidate takes before a full evaluation. Artifacts
  are under `evals/<system>/sanity/`; read them before rewriting.
- neither — the staged gauntlet cut it for scoring below a stage threshold.

The first two mean you shipped broken code, not a weak idea. Prototyping
(Step 2) is what keeps you out of them.
