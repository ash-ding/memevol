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
    --eval_n_users 2 --eval_n_qa 10 \
    --steps 2

# Full training — 6 users, 10 steps, 20 QA per user
python baselines/alma/run_main.py \
    --status search \
    --eval_n_users 6 --eval_n_qa 20 \
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
├── meta_agent_prompt.py
├── harness_base.py     # MemoStructure / Sub_memo_layer abstract classes (alma-specific contract)
├── llm.py              # Agent + Embedding wrappers
├── tokens.py           # TokenTracker
├── logger.py           # rich logger
├── workflow.py         # DynamicMem_Workflow + save_full_traces
├── launch.py           # subprocess entry (score.json + full traces, no sampling)
├── eval_runner.py      # subprocess manager (2h/8h wall-clock timeout)
├── sampling.py         # single-user bin sampling → analysis artifact (alma-only)
├── memo_archive/
│   ├── baseline/memo_structure_no_mem.py
│   └── dynamicmem/memo_structure_<SHA>.py
├── memo_test/          # staging: subprocess imports memo_test.py
├── logs/               # checkpoints + rotating log files
├── results/            # per-run eval outputs
│   └── dynamicmem/<SHA>_<status>_<mode>/
│       ├── score.json
│       ├── traces/<user_id>.json   # FULL trajectory (no sampling)
│       ├── memory_dumps/<user_id>.json
│       └── token_usage.json
├── search.sh           # meta-learning search invocations
└── test.sh             # held-out evaluation invocations
```

## Dataset layer

The dataset-specific code (`get_task_list`, `load_user_data`, `judge_answer`,
`DynamicMemRecorder`, prompts) is shared at
[`datasets/dynamicmem/`](../../datasets/dynamicmem/). It has zero dependency
on alma so other methods can reuse it.
