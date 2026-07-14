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
│      │ Sanity check    │  tiny real-data run per benchmark  │
│      │ (optional)      │  — propose_with_fix on failure     │
│      └────────┬────────┘                                    │
│               │ pass                                        │
│      ┌────────▼────────┐  staged gauntlet, per benchmark:   │
│      │ Evaluator       │  stage1 →[≥thr]→ stage2 →[≥thr]→   │
│      │ (Singularity)   │  stage3 (below threshold ⇒ out);   │
│      └────────┬────────┘  Phase-1 memory cached across      │
│               │           stages → score.json + traces      │
│      ┌────────▼────────┐  per-benchmark axes only —         │
│      │ Frontier update │  objectives = {accuracy_<ds>,      │
│      └─────────────────┘    stage_<ds>, robustness_<ds>,    │
│                              code_length, tokens_total, ...} │
│                              (no cross-benchmark mean;       │
│                               proposer self-selects priors)  │
└─────────────────────────────────────────────────────────────┘
```

Two evaluation-efficiency mechanisms (2026-07):

- **Staged evaluation** — each benchmark runs a stage1→2→3 promotion
  gauntlet with config thresholds (per-benchmark independent); bad
  candidates die on a ~20-item stage1 instead of consuming a full eval.
  Sampling is deterministic and *nested* (a smaller stage's task set is a
  strict subset of a larger one).
- **Cross-stage memory cache** — the memory a harness builds in Phase 1 is
  snapshotted (pickle; per checkpoint for DynamicMem) and reused at deeper
  stages instead of being rebuilt. See `common/memory_cache.py`.

Each generated **harness** is a Python class inheriting
`forge.harness_base.MemoStructure` (a documented subclass of the common
ABC), implementing the two-phase contract:

- **Phase 1** — `general_update(recorder)`: ingest a stream of user data
  (app logs, conversation turns, chat sessions) into any internal structure
  (dict, vector store, graph, hierarchy, ...). Called multiple times per
  user when `update_type=chunked|sequential`.
- **Phase 2** — `general_retrieve(recorder)`: given the current query in
  `recorder.init["query"]`, return a dict that's fed to the QA agent as
  context. Must be read-only w.r.t. memory state (DynamicMem interleaves
  queries with ingestion at checkpoints — query pollution breaks
  checkpoint isolation).

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

| Dataset | Source | Protocol | Split |
|---|---|---|---|
| **[DynamicMem](datasets/dynamicmem/)** | App-activity logs (~1500/user over 15 months) | Official **TCE v2 checkpoint protocol**: ingestion interleaved with tasks at 5 quarterly checkpoints; two task families (state completion + personalized service); official holistic Core+Detail judge, scores 0–1 | 6 users search / 4 test |
| **[LoCoMo](datasets/locomo/)** | Multi-session two-person conversations (~154 QA each after filtering) | Two-phase; binary CORRECT/WRONG judge (community-standard); QA **categories 1–4 only** (cat-5 adversarial excluded 2026-07-08 — the data carries no gold answers for them) | 6 conv search / 4 test |
| **[LongMemEval](datasets/longmemeval/)** | 500 questions, each with its own haystack of chat sessions (`s` ~48, `m` ~476) | Two-phase, 1 QA per question; binary yes/no judge (paper) | 300 search / 200 test (stratified by question type) |

Data files are **not** in the repo (DynamicMem `user_data/<user>/{app_log_large,task_packs}.json`,
~77 MB total; LongMemEval `m_cleaned.json` 2.6 GB). Acquire separately and
place under `datasets/<bench>/`; DynamicMem honors a `DYNAMICMEM_DATA`
env-var override.

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
# Quick smoke: --smoke-test turns any config into a sanity-size single pass
# (no gauntlet, no sanity gate). With --steps 1 it exercises the whole
# propose → eval → score pipeline cheaply (~1-2 min, a handful of LLM calls).
venv/bin/python -m forge.orchestrator --config configs/search_mini.yaml --smoke-test --steps 1

# Small multi-benchmark search (10 propose iterations × 2 candidates)
venv/bin/python -m forge.orchestrator --config configs/search_mini.yaml

# Full multi-benchmark search (production)
venv/bin/python -m forge.orchestrator --config configs/search.yaml

# CLI overrides (any YAML field has a matching CLI flag)
venv/bin/python -m forge.orchestrator \
  --config configs/search.yaml \
  --steps 3 --datasets dynamicmem,locomo --gpu
```

See **[`configs/search_example.yaml`](configs/search_example.yaml)** for the full
documented schema (every field, default, and effect).

## Run modes

The orchestrator always runs the **search loop** on the search (training)
split — the old `mode:` switch was removed 2026-07-14. Two entry points:

| Entry | Data split | Eval size | Sanity layer |
|---|---|---|---|
| `forge.orchestrator` (default) | search split | full staged gauntlet (`stages.stage1..3` with thresholds) | respected (`sanity.enabled`) |
| `forge.orchestrator --smoke-test` | search split | one `stages.sanity_check`-sized run, no gauntlet | always skipped |
| `python -m forge.heldout` | held-out **test** split | `coverage: full` (whole split) by default, or the sampled gauntlet | n/a (harnesses already passed sanity in their own run) |

`--smoke-test` is the "did my harness even import / run" check. Held-out
evaluation is deliberately a separate entry (frozen harnesses only — running
the search loop on test data would optimize against the held-out split).
Per-benchmark stage sizes/thresholds live in each `datasets.<ds>.stages`
config block (see `configs/search_example.yaml`).

## Architecture

```
memevol/
├── forge/                  Main method (Singularity-sandboxed CC proposer
│                           + selective-bind evaluator + frontier)
├── common/                 Cross-method utilities used by forge AND baselines
│   ├── harness_base.py     MemoStructure ABC, Recorder (baseline contract;
│   │                       forge harnesses inherit forge/harness_base.py)
│   ├── workflow.py         BaseWorkflow scheduler (per-user concurrency,
│   │                       Phase 1 chunking, Phase 2 QA loop, persistence)
│   ├── memory_cache.py     cross-stage Phase-1 memory snapshots
│   ├── llm.py              Agent / Embedding (shared client, unified retry
│   │                       kernel, global concurrency gate, token tracking)
│   ├── judge.py            LLM-as-judge with prompt template + score range
│   ├── tokens.py           TokenTracker (used by Agent/Embedding/Judge)
│   └── logger.py           rich-based logger
├── datasets/<bench>/       Per-benchmark adapter
│   ├── env.py              Recorder subclass + loaders + task list
│   ├── workflow.py         BaseWorkflow subclass (dynamicmem overrides
│   │                       run_single_user: checkpoint-interleaved TCE)
│   └── prompts.py          QA agent prompt template
│                           (dynamicmem: tce_prompts.py — official TCE
│                           prompts + holistic judge, ported verbatim)
├── containers/             Singularity .def files for both images
├── configs/                YAML configs (smoke / search_mini / search / example)
├── seeds/                  Project-level seed harness library (git-tracked)
├── baselines/              Comparison methods — see baselines/README.md
└── workspace/<run_id>/     Per-run runtime state (gitignored)
    ├── harnesses/<int>_<hash8>/
    │   └── <dataset>/         score.json + stages.json + <stage>/traces
    │                          + memory_cache/ (cross-stage snapshots)
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
