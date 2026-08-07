# cc baseline — Claude Code as a direct QA agent

`CCMemo` (`memo.py`) is a `MemoClass` that skips memory-architecture
design entirely: Phase 1 stashes the currently-visible user data to a
per-user temp directory; Phase 2 runs Claude Code with file tools
(Read/Grep/Glob) against that directory and answers the question directly.
`CCMemo.use_memory_to_answer` (the shared workflow's standardized answer hook)
judges cc's own tool-using answer verbatim on the EXACT formatted prompt the
main method would give its own QA agent — this is what makes cc emit each
benchmark's required output format (e.g. DynamicMem TCE's "Return JSON
only" skeleton) instead of free prose the judge can't parse.

cc runs through the **same shared runner** as hipporag2
(`baselines.harness.eval_utility.run_baseline`), which resolves the SAME production
per-dataset workflow the main method uses (`baselines.registry.resolve`) —
so DynamicMem gets the official TCE v2 checkpoint protocol + holistic judge,
not a compat-shim script. The old single-dataset `eval_cc.py` was replaced
by `run.py` (2026-07); numbers from that script are NOT comparable to
`run.py`'s DynamicMem output (different protocol).

Per the standardized `MemoClass` contract, `build_memory_from_data` is called
ONCE per build call with the whole newly-visible data already in
`recorder.init` — the memo ingests the handed data however it chooses;
`CCMemo` doesn't split it at all, it just stashes the whole payload to disk
each call.

## Setup

cc has its own uv-managed environment, defined by its own self-contained
`pyproject.toml` + committed `uv.lock` (`.python-version` pins 3.12):

```bash
cd baselines/harness/cc && uv sync
```

This creates `baselines/harness/cc/.venv/`. The repo-root `.venv/` is
dev/test only and cannot run cc.

## Usage

```bash
cd baselines/harness/cc && uv run python run.py --config config.example.yaml
```

`run.py` takes exactly one flag, `--config <yaml>` (required) — there is no
other CLI surface (dataset, split, progressive, model, etc. are no longer
flags). Every parameter lives in the config file: `config.example.yaml` documents
each key inline — copy it, edit the values, and point `--config` at your
copy. The YAML must list EXACTLY the keys `run.py`'s `REQUIRED_KEYS`
expects — a missing key OR an unknown key (typo, stale setting) aborts the
run before anything executes; a `null` value counts as listed. Sizing
fields (`single_stage` / `stages`) are checked down to the leaf, and `null`
there means "whole split" for that dimension (see "Sizing" below).

A couple of keys are worth calling out beyond what the YAML comments say:

- `model` — the Claude Code model cc itself answers with (aliases
  `sonnet` → `claude-sonnet-4-20250514`, `opus` → `claude-opus-4-20250514`;
  any explicit model id also works). Does NOT affect scoring model choice —
  cc bypasses the shared QA agent (see the next bullet).
- `judge_model` — also passed as `qa_model` to `run_baseline` for
  API-compatibility (`BaseWorkflow.__init__` requires one), but it is never
  actually invoked to answer: `CCMemo.use_memory_to_answer` answers before
  the shared QA agent would be reached.

To change dataset, model, split, sizing, etc., edit those keys in your
config yaml — e.g.:

```yaml
# my_cc.yaml — Opus, DynamicMem, 2 users
dataset: dynamicmem
model: opus
single_stage: {n_users: 2, n_checkpoints: 5, n_task_a: 5, n_task_c: 5}
```

then `uv run python run.py --config my_cc.yaml`.

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
# my_cc.yaml — 2 DynamicMem users, all 5 checkpoints, 3 Task-A + 3 Task-C per checkpoint
dataset: dynamicmem
progressive: false
single_stage: {n_users: 2, n_checkpoints: 5, n_task_a: 3, n_task_c: 3}
```

```yaml
# my_cc.yaml — 3 LoCoMo conversations, 10 QA each
dataset: locomo
progressive: false
single_stage: {n_conversations: 3, n_qa: 10}
```

## Output

```
baselines/harness/cc/results/<dataset>/<split>/
├── score.json          # {"benchmark_eval_score": {...}, "per_user": {...}, "invalid_users": [...]}
├── token_usage.json     # per-model token totals (common.tokens.TokenTracker)
└── traces/<user_id>.json   # full per-user QA trajectory (no sampling)
```

## Notes

- `CCMemo` runs on all four datasets via dispatch on `recorder.init` keys
  (`app_logs` / `conversation` / `sessions` — see `_CONTEXT` in `memo.py`),
  mirroring how `forge`-evolved harnesses dispatch.
- Useful as an upper-bound reference: how well does a strong general-purpose
  agent do with raw data + tools and **no learned memory structure at all**?
- The old `eval_cc.py` (DynamicMem-only, compat-shim last-checkpoint data)
  was fully replaced by `run.py` + `memo.py` — do not resurrect it; its
  results are not comparable (different protocol, no full TCE checkpoint
  interleaving).
