# cc baseline — Claude Code as a direct QA agent

`CCMemo` (`memo.py`) is a `MemoStructure` that skips memory-architecture
design entirely: Phase 1 stashes the currently-visible user data to a
per-user temp directory; Phase 2 runs Claude Code with file tools
(Read/Grep/Glob) against that directory and answers the question directly.
`CCMemo.use_memory_to_answer` (the shared workflow's standardized answer hook)
judges cc's own tool-using answer verbatim on the EXACT formatted prompt the
main method would give its own QA agent — this is what makes cc emit each
benchmark's required output format (e.g. DynamicMem TCE's "Return JSON
only" skeleton) instead of free prose the judge can't parse.

cc runs through the **same shared runner** as hipporag2
(`baselines.harness.eval_common.run_baseline`), which resolves the SAME production
per-dataset workflow the main method uses (`baselines.registry.resolve`) —
so DynamicMem gets the official TCE v2 checkpoint protocol + holistic judge,
not a compat-shim script. The old single-dataset `eval_cc.py` was replaced
by `run.py` (2026-07); numbers from that script are NOT comparable to
`run.py`'s DynamicMem output (different protocol).

Per the standardized `MemoStructure` contract, `build_memory_from_data` is called
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
cd baselines/harness/cc && uv run python run.py \
    --config config.example.yaml \
    [--dataset {dynamicmem,locomo,longmemeval_s,longmemeval_m}] \
    [--split test|search] \
    [--progressive|--no-progressive] \
    [--sampling-seed 42] \
    [--model sonnet|opus|claude-sonnet-4-20250514|...] \
    [--max_turns 30] \
    [--judge_model gpt-5-mini] \
    [--max_sample_concurrent 3] \
    [--no-memory-cache]
```

- `--config` — YAML config path (CLI flags override it). **Evaluation SIZES
  live in the config file only** (`single_stage` / `stages` — see "Sizing"
  below); there are no sizing CLI flags.
- `--dataset` (required by convention; defaults to `dynamicmem`) — one of
  the four registered benchmarks (`baselines.registry.DATASETS`).
- `--split` (default `test`) — `test` = held-out split, `search` = the
  split the main method's search loop sees. Cross-run leakage direction:
  never run `search` and treat it as a comparable held-out number.
- `--model` — the Claude Code model cc itself answers with (aliases
  `sonnet` → `claude-sonnet-4-20250514`, `opus` → `claude-opus-4-20250514`;
  any explicit model id also works). Does NOT affect scoring model choice —
  cc bypasses the shared QA agent (see "Why `--judge_model` doubles as
  `qa_model`" below).
- `--max_turns` — max Claude Code tool-use turns per QA question (default
  30).
- `--judge_model` — the judge model. Also passed as `qa_model` to
  `run_baseline` for API-compatibility (`BaseWorkflow.__init__` requires
  one), but it is never actually invoked to answer: `CCMemo.use_memory_to_answer`
  answers before the shared QA agent would be reached.
- `--max_sample_concurrent` — per-eval user/sample concurrency (default 3).
- `--progressive` / `--no-progressive` (default off, matching this baseline's
  historical single-pass behavior) — run the staged stage1→2→3 gauntlet
  (with threshold elimination) instead of one single-stage pass. Sizes come
  from the config `stages` block (or family `DEFAULT_STAGES`).
- `--sampling-seed` (default `42`) — base seed for the (fixed step-0) sample
  this baseline evaluates. A no-op at whole-split (`null` sizes); it only
  selects a subset when a size field caps.
- `--no-memory-cache` — disable cross-stage Phase-1 memory reuse (on by
  default).

Examples:

```bash
# Config-driven run (sizes from the config's single_stage / stages)
uv run python run.py \
    --config config.example.yaml

# Opus, one dataset (size via a config with single_stage: {n_users: 2})
uv run python run.py \
    --config my_cc.yaml --dataset dynamicmem --model opus

# LongMemEval-m, search split (comparable to what the proposer itself sees)
uv run python run.py \
    --config my_cc.yaml --dataset longmemeval_m --split search
```

## Sizing (config file only)

There are **no sizing CLI flags** — evaluation sizes are config-file keys
resolved through the shared `common.evaluate` layer (the same one forge
uses):

- **`progressive: false` (default)** REQUIRES a `single_stage` block — ONE pass
  sized by its fields (`common.evaluate.single_stage_wire_spec`; a `null`
  or omitted field = the WHOLE split for that dimension, byte-identical to the
  main method's `forge.heldout` `coverage=full` when all-null). Omitting
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
