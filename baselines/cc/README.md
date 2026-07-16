# cc baseline — Claude Code as a direct QA agent

`CCMemo` (`memo.py`) is a `MemoStructure` that skips memory-architecture
design entirely: Phase 1 stashes the currently-visible user data to a
per-user temp directory; Phase 2 runs Claude Code with file tools
(Read/Grep/Glob) against that directory and answers the question directly.
`CCMemo.general_answer` (the shared workflow's standardized answer hook)
judges cc's own tool-using answer verbatim on the EXACT formatted prompt the
main method would give its own QA agent — this is what makes cc emit each
benchmark's required output format (e.g. DynamicMem TCE's "Return JSON
only" skeleton) instead of free prose the judge can't parse.

cc runs through the **same shared runner** as hipporag2
(`baselines.eval_common.run_baseline`), which resolves the SAME production
per-dataset workflow the main method uses (`baselines.registry.resolve`) —
so DynamicMem gets the official TCE v2 checkpoint protocol + holistic judge,
not a compat-shim script. The old single-dataset `eval_cc.py` was replaced
by `run.py` (2026-07); numbers from that script are NOT comparable to
`run.py`'s DynamicMem output (different protocol).

## Usage

```bash
baselines/venv/bin/python baselines/cc/run.py \
    --dataset {dynamicmem,locomo,longmemeval_s,longmemeval_m} \
    [--split test|search] \
    [--stage-spec '<json>'] \
    [--model sonnet|opus|claude-sonnet-4-20250514|...] \
    [--max_turns 30] \
    [--judge_model gpt-5-mini] \
    [--max_sample_concurrent 3]
```

- `--dataset` (required by convention; defaults to `dynamicmem`) — one of
  the four registered benchmarks (`baselines.registry.DATASETS`).
- `--split` (default `test`) — `test` = held-out split, `search` = the
  split the main method's search loop sees. Cross-run leakage direction:
  never run `search` and treat it as a comparable held-out number.
- `--stage-spec` (default `None` = **whole test split, full coverage** —
  byte-identical to the main method's `forge.heldout` default; see "Stage
  spec fields" below).
- `--model` — the Claude Code model cc itself answers with (aliases
  `sonnet` → `claude-sonnet-4-20250514`, `opus` → `claude-opus-4-20250514`;
  any explicit model id also works). Does NOT affect scoring model choice —
  cc bypasses the shared QA agent (see "Why `--judge_model` doubles as
  `qa_model`" below).
- `--max_turns` — max Claude Code tool-use turns per QA question (default
  30).
- `--judge_model` — the judge model. Also passed as `qa_model` to
  `run_baseline` for API-compatibility (`BaseWorkflow.__init__` requires
  one), but it is never actually invoked to answer: `CCMemo.general_answer`
  answers before the shared QA agent would be reached.
- `--max_sample_concurrent` — per-eval user/sample concurrency (default 3).

Examples:

```bash
# Full held-out eval, one dataset, default model (sonnet)
baselines/venv/bin/python baselines/cc/run.py --dataset locomo

# Opus, capped to 2 units for a quick check
baselines/venv/bin/python baselines/cc/run.py \
    --dataset dynamicmem --model opus --stage-spec '{"n_samples": 2}'

# LongMemEval-m, search split (comparable to what the proposer itself sees)
baselines/venv/bin/python baselines/cc/run.py \
    --dataset longmemeval_m --split search
```

## Stage-spec fields

`--stage-spec` is a raw JSON object of USER OVERRIDES merged over the
family's full-coverage base (`baselines.eval_common.family_full_spec` /
`effective_stage_spec` — mirrors `forge.orchestrator.full_wire_spec`
exactly, so an omitted field stays `null` = uncapped, not zero). All units
are PER-RUN counts (no gauntlet/staging here — this is a single pass, same
as `forge.heldout`'s `coverage=full`):

| Field | Applies to | Meaning |
|---|---|---|
| `n_samples` | all datasets | Generic wire field for the unit count: **users** for dynamicmem, **conversations** for locomo, **questions** for longmemeval. `null`/omitted = whole split. |
| `n_checkpoints` | dynamicmem only | DynamicMem TCE checkpoints per user (of 5 quarterly checkpoints). `null` = all 5. |
| `n_task_a` | dynamicmem only | Task-A (`state_completion`) items sampled per checkpoint. `null` = full bucket. |
| `n_task_c` | dynamicmem only | Task-C (`apply_service`) items sampled per checkpoint. `null` = full bucket. |
| `n_qa` | locomo only | QA pairs sampled per conversation. `null` = all (categories 1-4 only; cat-5 excluded). |
| — | longmemeval | No extra field — one question is one unit, so `n_samples` alone controls both. |

```bash
# 2 DynamicMem users, all 5 checkpoints, 3 Task-A + 3 Task-C items per checkpoint
baselines/cc/run.py --dataset dynamicmem \
    --stage-spec '{"n_samples": 2, "n_checkpoints": 5, "n_task_a": 3, "n_task_c": 3}'

# 3 LoCoMo conversations, 10 QA each
baselines/cc/run.py --dataset locomo --stage-spec '{"n_samples": 3, "n_qa": 10}'
```

## Output

```
baselines/cc/results/<dataset>/<split>/
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
