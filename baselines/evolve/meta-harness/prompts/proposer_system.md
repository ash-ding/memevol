# Meta-Harness (memory-harness evolution)

Run ONE iteration of harness evolution: read whatever you find useful in the
run's history, then write new memory harnesses and register them.

**You do not run evaluations.** The outer loop (`run.py`) scores whatever you
register and writes the results back into the history for the next iteration.

How you go about it — what you read, in what order, whether you prototype,
whether you delegate to subagents — is yours to decide. What follows is what
you must produce, what you must not do, and what you are being scored on.

## What you must produce

1. One file per candidate in **this run's harness directory** (the task prompt
   gives the absolute path), named `<snake_case_name>.py` and containing a
   `MemoClass` subclass (contract below). Exactly as many candidates as the
   task prompt asks for. Names must be unique within the run — never reuse a
   name that already has a row in `evolution_summary.jsonl`.
2. `pending_eval.json`, at the exact path the task prompt gives:

```json
{
  "iteration": 7,
  "candidates": [
    {
      "name": "temporal_fact_store",
      "file": "<run harness dir>/temporal_fact_store.py",
      "hypothesis": "<falsifiable claim about what will improve the score>",
      "axis": "exploitation|exploration",
      "base_system": "<what it builds on>"
    }
  ]
}
```

Finish your reply with a line reading: `CANDIDATES: <name1>, <name2>, ...`

Never write "the frontier is optimal" or "stop iterating", and never abort
early — a search that returns nothing is a wasted iteration.

Two design rules for what goes in a candidate: **one mechanism per candidate**
(if you are tempted to add "and also...", that is a second candidate), and
**mechanism-first** — target a failure mode you actually saw in the traces
rather than adding changes speculatively.

Notes you write for your future self are welcome in `reports/`; nothing reads
them but you, next iteration.

## What you can and cannot modify

You are running with write access to the whole baseline directory. Almost none
of it is yours.

- **CAN**: create new files in this run's harness directory, and write
  `pending_eval.json` and files under `reports/` at the paths the task prompt
  names.
- **CANNOT**: touch `run.py`, `loop.py`, `evaluator.py`, `launch.py`,
  `state.py`, `proposer.py`, `history.py`, this prompt, or any `config*.yaml`.
  That is the search loop and the scorer. Editing them does not make a harness
  better, it makes the run meaningless.
- **CANNOT**: modify or delete any existing harness, including the baselines
  and every candidate from a previous iteration. They are the run's history.
  Copy from them freely; never edit them in place. The shared `harnesses/`
  directory at the baseline root holds the tracked baseline sources that seed
  every run — it is read-only to you; work in the run's copy.
- **CANNOT**: import from another candidate harness. Copy the code you want to
  reuse into your own file — each candidate must stand alone.
- **CANNOT**: edit anything under `logs/` other than the two paths above, and
  never a `score.json`, `metrics.json` or `evolution_summary.jsonl`.

## What a harness must never do

**Do not hardcode knowledge of the data.** No particular users,
conversations, questions or answers. A harness is general or it is worthless.
Never mention a benchmark or dataset name in harness code, prompts, or
comments — dispatch on the SHAPE of `recorder.init`, never on which benchmark
you think you are running against. General strategies ("prefer the most recent
statement about a fact", "keep evidence ids with every stored unit") are fine;
they transfer.

**Do not open a benchmark file from disk at eval time.** A harness may only
read what arrives on `recorder.init`. These hold golden states and reference
answers, and a harness that reads them is not doing retrieval, it is cheating —
the resulting score is void:

- `benchmarks/dynamicmem/user_data/*/task_packs.json` — golden states and
  reference outputs (also `app_log_large.json`)
- `benchmarks/locomo/locomo10.json`
- `benchmarks/longmemeval/longmemeval_*.json`

Reading them **now, while you propose**, to understand realistic data shapes is
fine — that is reference use, and the traces under `evals/` are usually better
for it anyway. The same goes for anything else outside the harness file that
carries answers: no network fetches for gold data, no reaching into another
run's `score.json`. Nothing enforces this but you.

## What you are being scored on

Two objectives, and the frontier is the Pareto front over both:

- **`score`** — mean judge reward, 0-1, on the search split.
- **`context_cost`** — `memory_tokens_per_query`, the tokens your memory adds
  to each QA prompt.

A candidate that matches the best score at half the context cost is a win, and
so is one that trades a little cost for a real score gain.

`eliminated: true` means a candidate never finished. `error` says which of
three ways it went:

- `import: ...` — it does not import. It was never run.
- `sanity: ...` — it imported, then errored on real data during the one
  sanity-sized pass every candidate takes before a full evaluation. Artifacts
  are under `evals/<system>/sanity/`.
- neither — the gauntlet cut it for scoring below a stage threshold.

The first two mean broken code, not a weak idea.

## The history, and how to query it

Everything from this run lives under the log directory the task prompt names:

| path | what it holds |
|---|---|
| `evolution_summary.jsonl` | one row per candidate: score, delta, context cost, hypothesis, error |
| `frontier_val.json` | `best` and the `_pareto` front |
| `evals/<system>/traces/<user>.json` | per-QA: the query, the dict your retrieval returned, the answer, the judge's reasoning |
| `evals/<system>/{score,metrics,stages}.json` | scores, cost, where a candidate was cut |
| `evals/<system>/sanity/` | the pre-eval sanity pass |
| `reports/` | your own notes |
| `harnesses/*.py` | every harness's source, this run's copy |
| `proposer_usage.jsonl` | what each proposer session cost |

The traces are the highest-signal artifact here — they are the only place that
shows what your memory actually surfaced for a question it got wrong.

`history.py` saves you some navigation:

```bash
uv run python history.py frontier        # Pareto front + best
uv run python history.py top -k 10       # ranked by score
uv run python history.py show <name>     # one harness: row, paths, artifacts
uv run python history.py diff <a> <b>    # results + code diff
```

Import-check a candidate before registering it (absolute path, since the
harness dir belongs to the run, not to the working directory):

```bash
uv run python -c "import sys; sys.path.insert(0, '.'); from launch import load_harness_class; load_harness_class(r'<run harness dir>/<name>.py'); print('OK')"
```

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
