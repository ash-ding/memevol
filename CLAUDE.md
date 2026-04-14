# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**memevol** evolves memory structures for AI agents on the DynamicMem benchmark.

The project is being reorganized:
- The original meta-learning loop (an LLM meta-agent that proposes / evaluates / reflects) now lives under `baselines/alma/` as a baseline.
- The project root is reserved for a new method inspired by [Meta-Harness](docs/meta%20hearness.pdf) (when implemented, a Claude Code SDK-driven search over harness code with full filesystem access). Not yet implemented.
- `baselines/cc/` and `baselines/hipporag2/` remain as alternative baselines.

## Virtual Environment

The alma baseline uses a venv under `baselines/venv/` (a future Meta-Harness implementation may use a separate environment, hence the per-baseline location).

```bash
source /export/scratch_large/ding/code/memevol/baselines/venv/bin/activate
```

Python binary: `/export/scratch_large/ding/code/memevol/baselines/venv/bin/python`

Requires `OPENAI_API_KEY` in `.env` at the project root.

## Directory Layout

```
memevol/
├── datasets/                 # Cross-method dataset utilities (zero dependency on any method package)
│   └── dynamicmem/
│       ├── env.py            # Basic_Recorder, DynamicMemRecorder, get_task_list, load_user_data, judge_answer
│       ├── prompts.py        # QA agent prompt
│       ├── user_data/        # DynamicMem app logs + profiles per user (gitignored)
│       └── user_qa/          # QA pairs per user (gitignored)
│
├── baselines/
│   ├── alma/                 # The original meta-learning method (run from project root)
│   │   ├── run_main.py       #   python baselines/alma/run_main.py --status search --steps 10
│   │   ├── meta_agent.py
│   │   ├── memo_manager.py
│   │   ├── meta_agent_prompt.py
│   │   ├── harness_base.py   # MemoStructure / Sub_memo_layer (alma's 2-phase contract)
│   │   ├── llm.py            # Agent + Embedding wrappers
│   │   ├── tokens.py
│   │   ├── logger.py
│   │   ├── workflow.py       # Phase1 / Phase2 per-user runner + save_full_traces
│   │   ├── launch.py         # subprocess entry: writes score.json + full traces
│   │   ├── eval_runner.py    # subprocess manager (2h check / 8h eval wall-clock cap)
│   │   ├── sampling.py       # single-user bin sampling → analysis artifact (alma-only)
│   │   ├── memo_archive/     # generated memo code (gitignored except .gitkeep)
│   │   ├── memo_test/        # runtime staging for subprocess
│   │   ├── logs/             # checkpoints + rotating run logs
│   │   ├── results/          # per-run eval outputs (gitignored)
│   │   │   └── dynamicmem/<SHA>_<status>_<mode>/
│   │   │       ├── score.json
│   │   │       ├── traces/<user_id>.json   # FULL trajectory per user
│   │   │       ├── memory_dumps/<user_id>.json
│   │   │       └── token_usage.json
│   │   ├── search.sh
│   │   ├── test.sh
│   │   └── README.md
│   ├── cc/                   # Claude Code SDK baseline (existing)
│   └── hipporag2/            # HippoRAG2 baseline (existing)
│
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Alma — run meta-learning search (full loop)
python baselines/alma/run_main.py --status search --steps 10

# Alma — evaluate a saved memo on held-out users
python baselines/alma/run_main.py --status test --memo_SHA <SHA>

# Alma — resume from checkpoint
python baselines/alma/run_main.py --status search --history_ckpt_path <filename.json>
```

Checkpoint JSONs and rotating log files live under `baselines/alma/logs/`.
Eval products (scores, traces, memory dumps) live under `baselines/alma/results/dynamicmem/<SHA>_<status>_<mode>/`.

See [`baselines/alma/README.md`](baselines/alma/README.md) for method-specific docs; [`baselines/alma/search.sh`](baselines/alma/search.sh) and [`baselines/alma/test.sh`](baselines/alma/test.sh) for more example invocations.

## Architecture (alma baseline)

### Two-Phase Protocol (per user)

1. **Phase 1 — `general_update(recorder)`**: Build a memory/profile from the user's app logs. Called N times depending on `update_type`.
2. **Phase 2 — `general_retrieve(recorder)`**: Given a question, retrieve relevant info from memory. Returns a `Dict` fed to the QA agent. An LLM judge scores the answer on a 0–10 scale.

Each user gets a fresh `MemoStructure` instance — no cross-user memory.

### Meta-Learning Loop

- **`MetaAgent`** (`baselines/alma/meta_agent.py`): orchestrates analyze → generate → sanity-check → evaluate.
- **`Memo_Manager`** (`baselines/alma/memo_manager.py`): persists memo code, tracks rewards, does softmax parent selection with exploration bonus.
- **`meta_agent_prompt.py`**: prompt templates for analysis / generation / error reflection. Includes Chroma and NetworkX cheatsheets.

Loop steps:
1. Run no-memory baseline to calibrate normalized reward.
2. Generate initial memory structure.
3. Sanity-check on `check_n_users` × `check_n_qa` (default 3×10, up to 3 retries with reflection).
4. Full eval → `benchmark_overall_eval_score`.
5. Repeat: select K structures via softmax(final_score) → analyze → generate → evaluate.

Selection: `final_score = sigmoid(reward - baseline) - α·log(1 + visit_count)`.

### Evaluation output (alma's contract)

The subprocess writes **only full traces + scores** — no sampling. Alma's main process reads the full traces and calls `baselines/alma/sampling.py::build_analysis_artifact` to assemble a compressed examples list for the meta-agent's analysis prompt. Other methods that reuse `datasets/dynamicmem/env.py` (cc, hipporag2, future Meta-Harness) do not go through alma's subprocess or sampling.

### Dataset layer boundary

`datasets/dynamicmem/` is the **only** cross-method shared code. It intentionally imports nothing from any method package. The judge (`judge_answer`) uses `openai.AsyncOpenAI` directly; method packages that want per-call token tracking can inject a tracker via `datasets.dynamicmem.env.set_token_tracker(tracker)`.

## Key Patterns (alma)

- **Subprocess isolation**: evaluation runs in a subprocess (`baselines/alma/eval_runner.py` → `baselines/alma/launch.py`). Memo code is copied to `baselines/alma/memo_test/memo_test.py` before execution.
- **Wall-clock subprocess timeout**: check=2h, eval=8h; on timeout the subprocess is force-killed and `RuntimeError` is raised.
- **Per-run output directory**: `baselines/alma/results/dynamicmem/<SHA>_<status>_<mode>/` contains `score.json`, `traces/<user_id>.json`, `memory_dumps/<user_id>.json`, `token_usage.json`.
- **Full traces always saved**: every QA from every user is persisted to `traces/` (not sampled). Sampling for the meta-agent analysis happens in the main process via `sampling.py`.
- **Error propagation**: runtime errors in `general_update` / `general_retrieve` are re-raised with `[Phase1_Update]` / `[Phase2_Retrieve]` tags so reflection can target the right phase.
- **Partial Phase 2 preservation**: if `general_retrieve` fails mid-way, already-answered QAs are kept on the recorder and `failure_info` is set; the user is not lost entirely.
- **Retry & robustness**: `Agent.ask()` retries transient API errors (timeout / connection / 5xx) up to 5 times (exponential backoff). `sem_task()` in the meta-loop retries failed memo evaluations up to 2 times with 30s→60s backoff. Judge retries 3 times. Embedding retries with exponential backoff.

## Terminology

| Term | Meaning |
|---|---|
| `status=search` | Meta-learning phase, using users 001–006 |
| `status=test` | Held-out evaluation, using users 007–010 |
| `mode=check` | Quick sanity check (subset of users/QA) |
| `mode=eval` | Full evaluation (all users in split) |
