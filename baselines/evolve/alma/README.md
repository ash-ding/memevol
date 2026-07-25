# alma baseline

Meta-learning loop that evolves memory structures through an LLM-driven
propose → sanity-check → evaluate → select pipeline. This was the original
method in memevol's root; it now lives here as a baseline so a new method
(Meta-Harness) can be developed at the project root.

## Quick start

Run from the **project root**:

Evaluation sizes no longer live on flat `--eval_n_*/--check_n_*` flags (removed).
Each candidate is scored through the shared staged gauntlet (`--progressive`);
sizes come from the family `DEFAULT_STAGES` (override with `--stages '<json>'`).

alma is **config-first**: pass `--config <yaml>` for a reusable settings file and
override individual fields on the CLI (precedence: `DEFAULT_CONFIG` < `--config`
YAML < CLI flags). `config.example.yaml` is a documented, runnable example:

```bash
# Config-first: everything from the YAML, override --status / --steps on the CLI
python baselines/evolve/alma/run.py \
    --config baselines/evolve/alma/config.example.yaml \
    --status search --steps 10
```

```bash
# Smoke test — tiny 1-user/1-checkpoint gauntlet, 2 steps
# (--stages sizes must be non-decreasing across stage1..3)
SMOKE_STAGES='{"sanity_check":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1},"stage1":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage2":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1,"threshold":0.0},"stage3":{"n_users":1,"n_checkpoints":1,"n_task_a":1,"n_task_c":1}}'
python baselines/evolve/alma/run.py \
    --status search \
    --progressive --stages "$SMOKE_STAGES" \
    --steps 2

# Full training — stage1→2→3 gauntlet (default stages), 10 steps
python baselines/evolve/alma/run.py \
    --status search \
    --progressive \
    --steps 10

# Evaluate a saved memo on held-out users (007–010)
python baselines/evolve/alma/run.py \
    --status test \
    --memo_SHA <SHA> --progressive
```

See `search.sh` for search-loop examples and `test.sh` for held-out evaluation examples.

## Layout

```
baselines/evolve/alma/
├── run.py              # CLI entry (config-first: --config <yaml> + CLI overrides)
├── config.example.yaml # documented, runnable example config (DEFAULT_CONFIG < YAML < CLI)
├── meta_agent.py       # MetaAgent: analyze → generate → examine → evaluate
├── memo_manager.py     # memo lifecycle, reward, softmax selection
├── meta_agent_prompt.py  # meta-LLM prompts (shows common/harness_base.py as the contract)
├── launch.py           # subprocess entry (score.json + full traces, no sampling)
├── eval_runner.py      # subprocess manager (2h/8h wall-clock timeout)
├── sampling.py         # single-user bin sampling → analysis artifact (alma-only)
├── memo_archive/       # evolved memo code, one file per content SHA
│   ├── baseline/memo_structure_no_mem.py
│   └── <dataset>/memo_structure_<SHA>.py
├── memo_test/          # staging: eval_runner copies each memo to memo_test_<SHA>.py
│                       # before running launch.py (per-SHA to survive concurrency)
├── logs/               # meta-learning checkpoints (check_*.json) + run logs
├── results/            # per-run eval outputs
│   └── <dataset>/<SHA>_<status>_<mode>/
│       ├── score.json
│       ├── traces/<user_id>.json   # FULL trajectory (no sampling)
│       └── token_usage.json
├── search.sh           # meta-learning search invocations
├── supervisor.sh       # unattended long-run wrapper (auto-resume from checkpoint)
└── test.sh             # held-out evaluation invocations
```

Shared infrastructure (`harness_base.py`, `llm.py`, `tokens.py`,
`logger.py`, the `DynamicMemWorkflow`) was long ago extracted to
[`common/`](../../../common/) and [`datasets/dynamicmem/`](../../../datasets/dynamicmem/)
— alma imports it from there. Since the 2026-07 TCE upgrade, alma's
DynamicMem evals therefore follow the official checkpoint protocol
(0–1 holistic judge) automatically.

## Dataset layer

The dataset-specific code (`get_task_list`, `load_user_checkpoints`,
`DynamicMemRecorder`, the official TCE prompts + holistic judge in
`tce_prompts.py`) is shared at
[`datasets/dynamicmem/`](../../../datasets/dynamicmem/). It has zero dependency
on alma so other methods can reuse it.

## Datasets

ALMA targets exactly **one benchmark per run**, selected with
`--dataset {dynamicmem,locomo,longmemeval_s,longmemeval_m}` (default
`dynamicmem`). The dataset name resolves through
[`registry.py`](registry.py) to a workflow class, env module, and recorder
class (mirrors `forge/launch.py::WORKFLOWS`, but ALMA is standalone and does
not import forge), and through [`dataset_info.py`](dataset_info.py) to the
prompt fragments (`recorder.init` shape, evidence key, etc.) that the
meta-agent's analysis/generation/reflection prompts render for that dataset.

Per-dataset paths (templated by `<dataset>`):

```
memo_archive/<dataset>/memo_structure_<SHA>.py   # evolved memo code
results/<dataset>/<SHA>_<status>_<mode>/          # per-run eval output
    score.json
    traces/<user_id>.json
    token_usage.json
```

`memo_archive/baseline/` (the `no_mem` seed) is **shared across datasets** —
it is not templated, since the no-memory baseline is dataset-agnostic.

DynamicMem's behaviour and paths are unchanged from before multi-dataset
support was added (`--dataset` defaults to `dynamicmem`, and its prompt
fragments are a byte-identical extraction of the original hardcoded text —
see `tests/test_alma_multidataset.py::test_dynamicmem_prompts_byte_identical`).

Example commands (mirroring "Quick start" above, run from the project root):

```bash
# LoCoMo — smoke-size search loop, 1 step, per-step random subset
LC_STAGES='{"sanity_check":{"n_conversations":1,"n_qa":2},"stage1":{"n_conversations":1,"n_qa":3,"threshold":0.0},"stage2":{"n_conversations":1,"n_qa":3,"threshold":0.0},"stage3":{"n_conversations":1,"n_qa":3}}'
python baselines/evolve/alma/run.py \
    --dataset locomo --status search --steps 1 \
    --progressive --random_sample --sampling_seed 42 --stages "$LC_STAGES" \
    --meta_model gpt-5-mini --execution_model gpt-5-mini --judge_model gpt-5-mini

# LongMemEval (s or m variant) — evaluate a saved memo on the held-out split
python baselines/evolve/alma/run.py \
    --dataset longmemeval_s --status test --memo_SHA <SHA> --progressive
```
