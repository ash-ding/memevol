# memevol

A meta-learning framework that **evolves memory structures** for AI agents. An LLM-driven loop automatically generates, evaluates, and iteratively improves Python code that defines how an agent stores and retrieves information from user app logs to answer personalization questions.

Built on the [DynamicMem](https://github.com/ash-ding/DynamicMem) benchmark.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                   Meta-Learning Loop                    │
│                                                         │
│  1. Select parent structures (softmax + exploration)    │
│  2. Analyze QA trajectories with LLM                    │
│  3. Generate improved memory structure code              │
│  4. Sanity check on 3 users (retry up to 3×)            │
│  5. Evaluate on full training split                      │
│  6. Update rewards → repeat                             │
└─────────────────────────────────────────────────────────┘
```

Each generated memory structure implements a **two-phase protocol** that runs independently per user:

- **Phase 1 — `general_update(recorder)`**: Ingest app logs and build a memory representation (vector DB, knowledge graph, structured profile, etc.)
- **Phase 2 — `general_retrieve(recorder)`**: Given a question, retrieve relevant information from memory. The retrieved context is passed to a QA agent, whose answer is scored by an LLM judge (0–10).

The meta-agent observes which structures score well, analyzes failure patterns, and generates improved code — an evolutionary search over program space.

## Setup

**Requirements**: Python 3.12+, an OpenAI API key.

```bash
# Clone
git clone https://github.com/ash-ding/memevol.git
cd memevol

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Data**: Place the DynamicMem dataset under `dynamicmem/` with the following structure:
```
dynamicmem/
├── user_data/
│   ├── 001_user_001/
│   │   ├── app_log_large.json
│   │   └── user_basic_profile.json
│   ├── 002_user_002/
│   │   └── ...
│   └── ...  (users 001–010)
└── user_qa/
    ├── 001.json
    ├── 002.json
    └── ...
```

## Quick Start

```bash
# Smoke test — 2 users, 2 steps, 10 QA per user
python run_main.py \
    --status train \
    --train_size 2 \
    --qa_sample_size 10 \
    --steps 2

# Full training — 6 users, 10 steps, 20 QA per user
python run_main.py \
    --status train \
    --train_size 6 \
    --qa_sample_size 20 \
    --steps 10

# Evaluate a learned structure on held-out users (007–010)
python run_main.py \
    --status eval \
    --memo_SHA <SHA>

# Evaluate the no-memory baseline
python run_main.py \
    --status eval \
    --memo_SHA no_mem
```

See `training.sh` and `evaluation.sh` for more examples.

## CLI Reference

`run_main.py` is the sole entry point. All arguments:

### Mode & Data Split

| Argument | Default | Description |
|---|---|---|
| `--status` | `train` | `train` — run evolution loop on users 001–006; `eval` — evaluate a single structure on held-out users 007–010 |
| `--train_size` | `6` | Number of train users per evaluation round (max 6) |
| `--memo_SHA` | — | SHA of a memo structure to evaluate (required for `--status eval`) |
| `--history_ckpt_path` | — | Checkpoint JSON in `logs/` to resume training from |

### Models

| Argument | Default | Description |
|---|---|---|
| `--meta_model` | `gpt-5` | LLM for meta-agent (analysis, code generation, error reflection) |
| `--execution_model` | `gpt-5-mini` | LLM for the downstream QA agent; supports `model/reasoning_effort` format |
| `--judge_model` | `gpt-5-mini` | LLM judge for scoring answers (0–10) |

### Update Strategy

Controls how app logs are batched for `general_update()` in Phase 1.

| Argument | Default | Description |
|---|---|---|
| `--update_type` | `all_at_once` | `all_at_once` — one call with all logs; `chunked` — split into N chunks; `sequential` — one call per log entry |
| `--n_chunks` | `5` | Number of chunks (only for `chunked` mode) |
| `--max_logs` | all | Max log entries to keep, truncates oldest (only for `all_at_once` mode) |

### Evaluation & Sampling

| Argument | Default | Description |
|---|---|---|
| `--qa_sample_size` | all | QA pairs per user; set to e.g. `20` during training to control cost |
| `--steps` | `10` | Number of meta-learning iterations |
| `--max_container_concurrent` | `5` | Parallel memo evaluations in the meta-loop |
| `--max_concurrent` | `5` | Parallel users within one evaluation subprocess |
| `--n_score_bins` | `3` | Equal-width bins over 0–10 score range for trajectory sampling |
| `--samples_per_bin` | `3` | Max sampled trajectories per bin fed to meta-agent analysis |
| `--result_dir` | `check` | Prefix for checkpoint filenames in `logs/` |

## Project Structure

```
memevol/
├── run_main.py                 # Sole CLI entry point
├── eval_runner.py              # Spawns evaluation subprocesses
├── core/
│   ├── meta_agent.py           # MetaAgent: evolution loop orchestrator
│   ├── memo_manager.py         # Memo lifecycle, reward tracking, selection
│   └── meta_agent_prompt.py    # LLM prompts for analysis/generation/reflection
├── evals/
│   ├── launch.py               # Subprocess entry point for evaluation
│   ├── agents/
│   │   ├── memo_structure.py   # Abstract base: MemoStructure, Sub_memo_layer
│   │   └── base.py             # TokenTracker for API usage monitoring
│   ├── workflows/
│   │   └── dynamicmem_workflow.py  # Two-phase per-user execution
│   └── utils/
│       └── hire_agent.py       # Agent (Chat API) & Embedding wrappers
├── envs/
│   ├── dynamicmem_env.py       # Recorder, data loaders, LLM judge
│   └── prompts/
│       └── dynamicmem_prompt.py
├── memo_archive/
│   ├── baseline/               # No-memory baseline
│   └── dynamicmem/             # Generated structures (memo_structure_<SHA>.py)
├── dynamicmem/                 # Dataset (not in git)
│   ├── user_data/              # App logs & profiles per user
│   └── user_qa/                # QA pairs per user
├── logs/                       # Checkpoints & results (auto-created)
├── training.sh                 # Training command examples
├── evaluation.sh               # Evaluation command examples
├── requirements.txt
└── .env.example
```

## How Memory Structures Are Evolved

Each generated memory structure is a Python file containing:

1. One or more **`Sub_memo_layer`** subclasses — each owns a database (Chroma vector store, NetworkX graph, dict, etc.) and implements `update()` / `retrieve()`.
2. A **`MemoStructure`** subclass that orchestrates the layers in `general_update()` and `general_retrieve()`.

Available tools for generated code:
- **Chroma** — vector similarity search
- **NetworkX** — knowledge graphs
- **Agent** — async LLM calls with JSON schema validation
- **Embedding** — text embeddings with cosine similarity

The meta-agent selects parent structures via softmax over:

```
final_score = sigmoid(reward - baseline) - α · log(1 + visit_count)
```

This balances exploitation (high-reward structures) with exploration (less-visited ones).

## License

MIT
