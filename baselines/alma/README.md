# alma baseline

Meta-learning loop that evolves memory structures through an LLM-driven
propose → sanity-check → evaluate → select pipeline. This was the original
method in memevol's root; it now lives here as a baseline so a new method
(Meta-Harness) can be developed at the project root.

## Quick start

Run from the **project root**:

```bash
# Smoke test — 2 users, 2 steps, 10 QA per user
python baselines/alma/run_main.py \
    --status search \
    --eval_n_samples 2 --eval_n_qa 10 \
    --steps 2

# Full training — 6 users, 10 steps, 20 QA per user
python baselines/alma/run_main.py \
    --status search \
    --eval_n_samples 6 --eval_n_qa 20 \
    --steps 10

# Evaluate a saved memo on held-out users (007–010)
python baselines/alma/run_main.py \
    --status test \
    --memo_SHA <SHA>
```

See `search.sh` for search-loop examples and `test.sh` for held-out evaluation examples.

## Layout

```
baselines/alma/
├── run_main.py         # CLI entry
├── meta_agent.py       # MetaAgent: analyze → generate → examine → evaluate
├── memo_manager.py     # memo lifecycle, reward, softmax selection
├── meta_agent_prompt.py  # meta-LLM prompts (shows common/harness_base.py as the contract)
├── launch.py           # subprocess entry (score.json + full traces, no sampling)
├── eval_runner.py      # subprocess manager (2h/8h wall-clock timeout)
├── sampling.py         # single-user bin sampling → analysis artifact (alma-only)
├── memo_archive/       # evolved memo code, one file per content SHA
│   ├── baseline/memo_structure_no_mem.py
│   └── dynamicmem/memo_structure_<SHA>.py
├── memo_test/          # staging: eval_runner copies each memo to memo_test_<SHA>.py
│                       # before running launch.py (per-SHA to survive concurrency)
├── logs/               # meta-learning checkpoints (check_*.json) + run logs
├── results/            # per-run eval outputs
│   └── dynamicmem/<SHA>_<status>_<mode>/
│       ├── score.json
│       ├── traces/<user_id>.json   # FULL trajectory (no sampling)
│       └── token_usage.json
├── search.sh           # meta-learning search invocations
├── supervisor.sh       # unattended long-run wrapper (auto-resume from checkpoint)
└── test.sh             # held-out evaluation invocations
```

Shared infrastructure (`harness_base.py`, `llm.py`, `tokens.py`,
`logger.py`, the `DynamicMemWorkflow`) was long ago extracted to
[`common/`](../../common/) and [`datasets/dynamicmem/`](../../datasets/dynamicmem/)
— alma imports it from there. Since the 2026-07 TCE upgrade, alma's
DynamicMem evals therefore follow the official checkpoint protocol
(0–1 holistic judge) automatically.

## Dataset layer

The dataset-specific code (`get_task_list`, `load_user_checkpoints`,
`DynamicMemRecorder`, the official TCE prompts + holistic judge in
`tce_prompts.py`) is shared at
[`datasets/dynamicmem/`](../../datasets/dynamicmem/). It has zero dependency
on alma so other methods can reuse it.
