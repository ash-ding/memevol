# memevol

A research framework for **evolving memory structures** for AI agents on the [DynamicMem](https://github.com/ash-ding/DynamicMem) benchmark.

Each method generates Python code that defines how an agent stores and retrieves information from a user's app logs to answer personalization questions. An LLM judge scores each answer on a 0–10 scale; the framework iterates on the memory code to maximize judge score.

## Status

This repository is being reorganized:

- **`baselines/alma/`** — the original meta-learning loop (structured-prompt LLM proposer + softmax parent selection + sampled QA traces). Fully functional.
- **Project root** — reserved for a new method inspired by [Meta-Harness](docs/meta%20hearness.pdf): a Claude Code SDK proposer with full filesystem access to all prior code, scores, and execution traces. Not yet implemented.
- **`baselines/cc/`**, **`baselines/hipporag2/`** — alternative baselines (Claude Code as direct QA agent; HippoRAG2 RAG pipeline).

The shared `datasets/dynamicmem/` layer (data loaders, recorder, LLM judge) is used by all of the above and has zero dependency on any specific method.

## How alma works

```
┌─────────────────────────────────────────────────────────┐
│                   Meta-Learning Loop                    │
│                                                         │
│  1. Select parent structures (softmax + exploration)    │
│  2. Analyze QA trajectories with LLM                    │
│  3. Generate improved memory-structure code             │
│  4. Sanity check on 3 users (retry up to 3×)            │
│  5. Evaluate on full training split                     │
│  6. Update rewards → repeat                             │
└─────────────────────────────────────────────────────────┘
```

Each generated memory structure implements a **two-phase protocol** that runs independently per user:

- **Phase 1 — `general_update(recorder)`**: ingest app logs and build a memory representation (vector DB, knowledge graph, structured profile, etc.).
- **Phase 2 — `general_retrieve(recorder)`**: given a question, retrieve relevant information. The retrieved context is passed to a QA agent whose answer is scored by the LLM judge.

The meta-agent observes which structures score well, analyzes failure patterns, and generates improved code — an evolutionary search over program space.

## Setup

**Requirements**: Python 3.12+, an OpenAI API key.

```bash
git clone https://github.com/ash-ding/memevol.git
cd memevol

# The alma baseline uses its own venv under baselines/venv/.
# (A future Meta-Harness method may use a separate environment — hence per-baseline venvs.)
python -m venv baselines/venv
source baselines/venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**Data**: place the DynamicMem dataset under `datasets/dynamicmem/`:
```
datasets/dynamicmem/
├── user_data/
│   ├── 001_user_001/
│   │   ├── app_log_large.json
│   │   └── user_basic_profile.json
│   └── ...
└── user_qa/
    ├── 001.json
    └── ...
```

(Override the default location with the `DYNAMICMEM_DATA` environment variable
if you want to keep the data elsewhere.)

## Quick start (alma baseline)

All commands run from the **project root**.

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

# Evaluate a learned structure on held-out users (007–010)
python baselines/alma/run_main.py --status test --memo_SHA <SHA>

# Evaluate the no-memory baseline
python baselines/alma/run_main.py --status test --memo_SHA no_mem
```

See [`baselines/alma/search.sh`](baselines/alma/search.sh) for meta-learning
examples, [`baselines/alma/test.sh`](baselines/alma/test.sh) for held-out
evaluation examples, and [`baselines/alma/README.md`](baselines/alma/README.md)
for details on alma's directory layout and output locations.

## Output locations (alma)

- **Checkpoints** (meta-learning state): `baselines/alma/logs/*.json`
- **Eval products** per run (scores + full traces + memory dumps):
  `baselines/alma/results/dynamicmem/<SHA>_<status>_<mode>/`
- **Generated memo code**: `baselines/alma/memo_archive/dynamicmem/memo_structure_<SHA>.py` (gitignored)

## Shared utilities

- **`datasets/dynamicmem/env.py`** — dataset loaders, `DynamicMemRecorder`, LLM judge. No method-layer dependencies.
- **`datasets/dynamicmem/prompts.py`** — QA agent prompt.

Every method and baseline imports from `datasets.dynamicmem.env`; nothing imports from any method package into this layer.

## License

MIT
