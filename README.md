# memevol

Evolutionary search over **memory architectures for AI agents**.

A Claude-Code-SDK-driven proposer iteratively writes Python harnesses
implementing memory systems and evaluates them across multiple long-context
QA benchmarks. The main method targets the open question:

> Can a coding agent invent memory architectures that approach human-brain-level
> capabilities — multi-tier organization, principled forgetting and
> consolidation, temporal awareness, dynamic re-organization — going beyond
> what hand-designed retrieval databases can achieve?

The main method lives at **[`forge/`](forge/)**. Comparison baselines
(alma meta-learning loop, Claude Code as direct QA agent, HippoRAG2) live
under [`baselines/`](baselines/) — see [baselines/README.md](baselines/README.md).

## How forge works

```
┌─────────────────────────────────────────────────────────────┐
│                  Outer loop (forge orchestrator)            │
│                                                             │
│  for step in range(steps):                                  │
│    for j in range(k_per_step):                              │
│      ┌─────────────────┐                                    │
│      │ Proposer (CC)   │  reads workspace history,          │
│      │ Singularity-    │  writes new harness/<id>/          │
│      │ sandboxed       │  decides which priors to use       │
│      └────────┬────────┘                                    │
│               │                                             │
│      ┌────────▼────────┐                                    │
│      │ Sanity check    │  small eval (1×3 QA per benchmark) │
│      │ (optional)      │  — propose_with_fix on failure     │
│      └────────┬────────┘                                    │
│               │ pass                                        │
│      ┌────────▼────────┐                                    │
│      │ Evaluator       │  full eval on each benchmark       │
│      │ (Singularity)   │  → score.json + traces             │
│      └────────┬────────┘                                    │
│               │                                             │
│      ┌────────▼────────┐                                    │
│      │ Frontier update │  objectives = {accuracy,           │
│      └─────────────────┘    accuracy_<dataset>,             │
│                              robustness, code_length,       │
│                              tokens_total, ...}             │
└─────────────────────────────────────────────────────────────┘
```

Each generated **harness** is a Python class inheriting
`common.harness_base.MemoStructure`, implementing the two-phase contract:

- **Phase 1** — `general_update(recorder)`: ingest a stream of user data
  (app logs, conversation turns, chat sessions) into any internal structure
  (dict, vector store, graph, hierarchy, ...). Called multiple times per
  user when `update_type=chunked|sequential`.
- **Phase 2** — `general_retrieve(recorder)`: given the current QA query
  in `recorder.init["query"]`, return a dict that's fed to the QA agent
  as context.

Per-user isolation: a fresh instance is created for every user — no
cross-user state. The harness must handle all benchmark `recorder.init`
shapes (typically by dispatching on init keys).

The proposer is **not** told which prior to use. Per the
[Meta-Harness paper](docs/meta%20hearness.pdf), CC browses
`workspace/<run_id>/harnesses/` and `frontier.json` itself, decides which
candidates to read (via Read/Grep/Glob + Bash + jq + WebSearch), and
records its chosen priors in the new harness's `meta.json::parent_ids`.

## Benchmarks

Three benchmarks are wired in (each runs as its own Singularity exec
against the same harness, scored to `[0, 1]` then averaged):

| Dataset | Source | What it tests |
|---|---|---|
| **[DynamicMem](datasets/dynamicmem/)** | App-activity logs (Fitbit, banking, calendar, ...) over months | Personalized profile construction + longitudinal pattern extraction + cross-domain synthesis |
| **[LoCoMo](datasets/locomo/)** | Multi-session two-person conversations (10 samples) | Long-conversation memory + temporal questions + adversarial "Not mentioned" handling |
| **[LongMemEval](datasets/longmemeval/)** | Haystack of chat sessions (`s` ~48/sample, `m` ~475/sample) | Single/multi-session reasoning, temporal reasoning, knowledge updates |

Data files are **not** in the repo (DynamicMem 3.7 GB user_data; LongMemEval
m_cleaned.json 2.6 GB). Acquire separately and place under
`datasets/<bench>/`; each `env.py` honors a `MEMEVOL_DATA_DIR` override.

## Setup

**Requirements**: Python 3.12, an OpenAI API key, the host's `claude` CLI
(subscription login), and Singularity.

```bash
git clone https://github.com/ash-ding/memevol.git
cd memevol

# Forge's host venv is intentionally minimal — only YAML parsing,
# the SDK proposer driver, and subprocess management. All ML
# dependencies (chromadb, sentence-transformers, ...) live in the
# Singularity images.
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...

claude login   # one-time CC SDK auth (if not already logged in)
```

**Build the Singularity images** (one-time, ~15 minutes total):

```bash
# Eval container (~3.3 GB) — full ML stack; harnesses run inside this
PATH=$HOME/.local/bin:$PATH singularity build \
  /export/scratch_large/ding/forge_images/eval-base.sif \
  containers/eval-base.def

# Proposer container (~3.4 GB) — inherits eval-base + jq + tree
PATH=$HOME/.local/bin:$PATH singularity build \
  /export/scratch_large/ding/forge_images/proposer-base.sif \
  containers/proposer-base.def
```

(Image storage path is set in [`forge/paths.py`](forge/paths.py); the
default points to `/export/scratch_large/ding/forge_images/` which is
host-specific. Override there if needed.)

## Quick start

```bash
# 1-min smoke: evaluate the seed harness on dynamicmem (1 user × 2 QA)
venv/bin/python -m forge.orchestrator --config configs/smoke.yaml

# Smoke + 1 propose iteration (exercises the full propose → eval pipeline)
venv/bin/python -m forge.orchestrator --config configs/smoke.yaml --steps 1

# Small multi-benchmark search (10 propose iterations × 2 candidates)
venv/bin/python -m forge.orchestrator --config configs/search_mini.yaml

# Full multi-benchmark search (production)
venv/bin/python -m forge.orchestrator --config configs/search.yaml

# CLI overrides (any YAML field has a matching CLI flag)
venv/bin/python -m forge.orchestrator \
  --config configs/search.yaml \
  --steps 3 --datasets dynamicmem,locomo --gpu
```

See **[`configs/example.yaml`](configs/example.yaml)** for the full
documented schema (every field, default, and effect).

## Status modes

`status` is a single user-facing knob that determines the search direction:

| `status` | Data split | Eval size | Sanity layer |
|---|---|---|---|
| `search` (default) | search split | full `eval_n_*` | respected (`sanity.enabled`) |
| `test` | held-out test split | full `eval_n_*` | respected |
| `devtest` | search split | small `check_n_*` | always skipped |

`devtest` is the "did my harness even import / run" smoke. `search` is
training/exploration. `test` is final held-out evaluation.

## Architecture

```
memevol/
├── forge/                  Main method (Singularity-sandboxed CC proposer
│                           + selective-bind evaluator + frontier)
├── common/                 Cross-method utilities used by forge AND baselines
│   ├── harness_base.py     MemoStructure ABC, Recorder
│   ├── workflow.py         BaseWorkflow scheduler (per-user concurrency,
│   │                       Phase 1 chunking, Phase 2 QA loop, persistence)
│   ├── llm.py              Agent / Embedding (OpenAI wrappers; auto-tracked)
│   ├── judge.py            LLM-as-judge with prompt template + score range
│   ├── tokens.py           TokenTracker (used by Agent/Embedding/Judge)
│   └── logger.py           rich-based logger
├── datasets/<bench>/       Per-benchmark adapter
│   ├── env.py              Recorder subclass + load_user_data + task list
│   ├── workflow.py         BaseWorkflow subclass (10 hooks)
│   └── prompts.py          QA agent prompt template
├── containers/             Singularity .def files for both images
├── configs/                YAML configs (smoke / search_mini / search / example)
├── seeds/                  Project-level seed harness library (git-tracked)
├── baselines/              Comparison methods — see baselines/README.md
└── workspace/<run_id>/     Per-run runtime state (gitignored)
    ├── harnesses/<int>_<hash8>/
    ├── frontier.json
    ├── runs/                  transient evaluator output
    └── orchestrator.log       host-side per-run log
```

## Logging

| Where | Scope |
|---|---|
| `forge/logs/orchestrator.log` | Global tape — every forge invocation appends; rotates at 5 MB × 3 |
| `workspace/<run_id>/orchestrator.log` | Per-run host-side log (one file per `--run-name`) |
| `workspace/<run_id>/runs/<id>_<ts>_<ds>/subprocess.log` | Per-eval-execution log inside the container |

Proposer tool calls are forwarded to orchestrator.log at INFO level
(`proposer·tool: #N <name> <input>`), with a per-call `proposer·info:
finished turns=X tools=Y duration=Zms cost=$W` summary.

## Method-design references

- [`docs/meta hearness.pdf`](docs/) — Meta-Harness paper (the design we follow:
  agent-driven parent selection, full filesystem feedback, no compressed
  per-candidate summaries).
- The mission framing inside the active template under
  [`forge/prompts/templates/`](forge/prompts/templates/) (the stem listed in
  `forge/prompts/templates/_default`) decomposes "biological memory" into a
  12-axis taxonomy across three families (Functional Core / Performance &
  Runtime / Learning & Adaptation) — these are the search dimensions the
  proposer is asked to advance.

## License

MIT
