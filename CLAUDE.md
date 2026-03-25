# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**memevol** is a meta-learning framework that evolves memory structures for AI agents on the DynamicMem benchmark. An LLM-driven loop generates, evaluates, and iteratively improves Python code that defines how an agent stores and retrieves information from user app logs to answer personalization questions.

## Virtual Environment

A pre-built venv (Python 3.12.3) exists at the project root. **Activate before running any commands:**

```bash
source /export/scratch_large/ding/code/memevol/venv/bin/activate
```

Python binary: `/export/scratch_large/ding/code/memevol/venv/bin/python`

## Environment

Requires `OPENAI_API_KEY` in `.env` at the project root. Uses `python-dotenv` to load it automatically.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run meta-learning training (full loop)
python run_main.py --status train --steps 10

# Evaluate a specific memo structure on held-out users
python run_main.py --status eval --memo_SHA <SHA>

# Resume training from checkpoint
python run_main.py --status train --history_ckpt_path logs/<checkpoint>.json
```

## `run_main.py` — Complete CLI Reference

`run_main.py` is the **sole entry point** for the entire framework. It routes to `meta_agent.forward()` when `--status train`, or `meta_agent.run_single_memo()` otherwise.

### Meta-Learning Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--meta_model` | str | `gpt-5` | LLM used by MetaAgent for analyzing QA trajectories, generating memory structure code, and reflecting on errors. Any OpenAI model name accepted. |
| `--execution_model` | str | `gpt-5-mini` | LLM used by the downstream QA agent that answers questions using retrieved memory. Supports `model/reasoning_effort` format (e.g. `gpt-5-mini/low`). |
| `--steps` | int | `10` | Number of meta-learning iterations in the evolution loop. Each step: select parent structures → analyze → generate new code → sanity check → evaluate. |
| `--max_container_concurrent` | int | `5` | Max parallel memo structure evaluations within the meta-learning loop. Controls how many `eval_runner` subprocesses run simultaneously. |
| `--result_dir` | str | `check` | Prefix for checkpoint JSON filenames saved in `logs/`. E.g. `check` produces `logs/check_step0.json`, `logs/check_step1.json`, etc. |

### Data Split & Evaluation Mode

| Argument | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--status` | str | `train` | `train`, `eval` | **`train`**: runs the full meta-learning evolution loop on users 001–006 (6 users). Calls `meta_agent.forward()`. **`eval`**: evaluates a single memo structure (specified by `--memo_SHA`) on held-out users 007–010 (4 users). Calls `meta_agent.run_single_memo()`. |
| `--train_size` | int | `6` | — | Number of train users per evaluation round. Max 6 (the total train pool). |
| `--memo_SHA` | str | `None` | — | SHA identifier of a specific memo structure to evaluate. **Required when `--status eval`**. Looks up code in `memo_archive/dynamicmem/memo_structure_<SHA>.py` (falls back to `memo_archive/baseline/`). |
| `--history_ckpt_path` | str | `None` | — | Path to a checkpoint JSON in `logs/` to resume training from. Restores the `Memo_Manager` state (memo_db, baseline reward, visit counts). |

### Update-Type Hyperparameters

Controls how app logs are batched and fed to the generated `general_update()` method during Phase 1.

| Argument | Type | Default | Choices | Description |
|---|---|---|---|---|
| `--update_type` | str | `all_at_once` | `all_at_once`, `chunked`, `sequential` | **`all_at_once`**: single `general_update()` call with all app logs (or the last `--max_logs` entries). **`chunked`**: splits logs into `--n_chunks` equal chunks, one `general_update()` call per chunk. **`sequential`**: one `general_update()` call per individual app log entry. |
| `--n_chunks` | int | `5` | — | Number of chunks to split app logs into. **Only effective when `--update_type chunked`**. |
| `--max_logs` | int | `None` | — | Maximum number of app log entries to keep (truncates oldest first). **Only effective when `--update_type all_at_once`**. `None` = use all logs. |

### QA Evaluation Settings

| Argument | Type | Default | Description |
|---|---|---|---|
| `--qa_sample_size` | int | `None` | Number of QA pairs to evaluate per user. `None` = use all available QA pairs. Set to e.g. `20` during training to reduce API cost. When set, samples deterministically (seed=42). |
| `--max_concurrent` | int | `5` | Max parallel users within a single evaluation subprocess. Controls the asyncio semaphore in `DynamicMem_Workflow.run_all_users()`. |
| `--n_score_bins` | int | `3` | Number of equal-width bins over score range 0–10 for trajectory sampling. After evaluation, QA results are binned by judge score, then `--samples_per_bin` trajectories are sampled from each bin to feed the meta-agent's analysis. |
| `--samples_per_bin` | int | `3` | Max QA trajectories sampled per score bin for meta-agent analysis. Total trajectories ≤ `n_score_bins × samples_per_bin`. |
| `--judge_model` | str | `gpt-5-mini` | LLM used by the judge to score QA answers on a 0–10 scale. Outputs JSON `{reason, score}`. |

## Architecture

### Two-Phase Protocol (per user)

1. **Phase 1 — `general_update(recorder)`**: Build a memory/profile from the user's app logs. Called N times depending on `update_type` strategy.
2. **Phase 2 — `general_retrieve(recorder)`**: Given a question, retrieve relevant info from memory. Returns a `Dict` fed to the QA agent which answers the question. An LLM judge scores the answer on a 0–10 scale.

Each user gets a fresh `MemoStructure` instance — no cross-user memory.

### Meta-Learning Loop (`core/`)

- **`MetaAgent`** (`core/meta_agent.py`): Orchestrates the evolution loop — selection, analysis, code generation, sanity check (up to 3 retries with LLM reflection on errors), evaluation.
- **`Memo_Manager`** (`core/memo_manager.py`): Manages memo lifecycle — saving to `memo_archive/`, loading, computing normalized rewards, softmax selection with exploration bonus.
- **`core/meta_agent_prompt.py`**: All prompt templates for LLM-driven analysis, generation, and error reflection. Includes cheatsheets for Chroma, NetworkX, and available tools (Agent, Embedding).

Loop steps:
1. Run no-memory baseline to establish reward baseline
2. Generate initial memory structure code
3. Sanity check on 3 users (retry up to 3x with LLM reflection on errors)
4. Evaluate on full split → `benchmark_overall_eval_score`
5. Repeat: select K structures via softmax(final_score), analyze QA trajectories, generate improved code, evaluate

Selection formula: `final_score = sigmoid(reward - baseline) - α·log(1 + visit_count)`

### Evaluation Framework (`evals/`)

- **`evals/launch.py`**: Subprocess entry point called by `eval_runner.py`. Dynamically imports the generated `MemoStructure` subclass, runs the workflow, bins QA scores, and samples trajectories for meta-agent analysis.
- **`evals/agents/memo_structure.py`**: Abstract base classes `MemoStructure` (with `general_update`/`general_retrieve`) and `Sub_memo_layer` (with `update`/`retrieve`).
- **`evals/agents/base.py`**: `TokenTracker` for per-model token usage tracking (prompt, completion, reasoning tokens). Global singleton via `init_global_tracker()`.
- **`evals/workflows/dynamicmem_workflow.py`**: Executes Phase 1 + Phase 2 for each user with semaphore-controlled concurrency. 300s timeout per `general_update`/`general_retrieve` call.
- **`evals/utils/hire_agent.py`**: `Agent` class — async OpenAI Chat wrapper with JSON schema validation. `Embedding` class — async embedding manager with batch operations, cosine similarity, and Chroma-compatible interfaces.

### Data & Environments (`dynamicmem/`, `envs/`)

- `dynamicmem/user_data/<user_id>/`: Contains `app_log_large.json` and `user_basic_profile.json` per user.
- `dynamicmem/user_qa/`: QA pairs with reference answers, indexed by user.
- Data splits in `envs/dynamicmem_env.py:get_task_list()`: **train** = users 001–006 (6 users), **eval** = users 007–010 (4 users).
- `envs/dynamicmem_env.py`: `DynamicMemRecorder` dataclass, data loading, and LLM judge (scoring 0–10 with reason).
- `envs/prompts/dynamicmem_prompt.py`: System/user prompts for the QA agent.

### Generated Code Storage (`memo_archive/`)

- `dynamicmem/memo_structure_<SHA>.py`: Evolved memory structures named by content SHA.
- `baseline/memo_structure_no_mem.py`: Zero-memory baseline (no-op update, empty retrieve).

## Key Patterns

- **Subprocess isolation**: Evaluation runs in a subprocess (`eval_runner.py` → `evals/launch.py`) to isolate generated code execution. Memo code is copied to `evals/memo_test/` before execution.
- **sys.path manipulation**: `run_main.py` adds both project root and `evals/` to `sys.path` for cross-module imports.
- **All async**: The entire pipeline is async (`asyncio`). Entry point uses `asyncio.run()`.
- **Environment variables for evals subprocess**: `OPENAI_API_KEY`, `EVALS_LOG_DIR`, `DYNAMICMEM_DATA` are passed via `eval_runner.py`.
- **Logs directory**: `logs/` stores checkpoint JSONs. Created automatically by `run_main.py`.
- **Eval result files**: Written to `evals/logs/dynamicmem/<SHA>_<mode>.json` containing `benchmark_eval_score`, sampled `examples`, and `token_usage`.
