# memevol

**memevol sits at the intersection of self-evolving agents and agent memory: a
coding agent evolves memory-system code for AI agents, evaluated across
long-context memory benchmarks.**

## Contents

1. [Codebase structure](#1-codebase-structure)
2. [How forge works](#2-how-forge-works)
3. [The `MemoStructure` contract](#3-the-memostructure-contract)
4. [Setup & quickstart](#4-setup--quickstart)

Comparison baselines live under [`baselines/`](baselines/) — see
**[baselines/README.md](baselines/README.md)** for what they are and for the
step-by-step guide to adapting an existing memory system into this repo's
evaluation protocol.

---

## 1. Codebase structure

| Directory | Role |
|---|---|
| [`forge/`](forge/) | **The main method.** Claude-Code-SDK proposer + Singularity-sandboxed evaluator + frontier record store; searches over harness code |
| [`common/`](common/) | The shared evaluation platform: the [`MemoStructure`](common/harness_base.py) contract, the [`Basic_Recorder`](common/recorder.py) data envelope, the [`BaseWorkflow`](common/workflow.py) scheduler, LLM/judge/embedding kernel, token tracking, memory cache, logging |
| [`datasets/`](datasets/) | One adapter per benchmark: `env.py` (data loading + recorder + split), `workflow.py` (evaluation protocol), `prompts.py` (QA-agent prompt) |
| [`baselines/`](baselines/) | Comparison methods, split into `evolve/` (search-method baselines, compared against forge itself) and `harness/` (ready-made memory systems, compared against forge-evolved harnesses) — [README](baselines/README.md) |
| [`seeds/`](seeds/) | Opt-in seed harness library. A seed is copied into a run as candidate #0 (e.g. `no_memory` — the calibration floor any real memory design must beat) |
| [`configs/`](configs/) | [`search_example.yaml`](configs/search_example.yaml) (documented AND runnable search config) + [`test_example.yaml`](configs/test_example.yaml) (held-out test flow) |
| [`containers/`](containers/) | Singularity image definitions (eval base + proposer base) |
| [`tools/`](tools/) | Operator scripts (prompt-version bookkeeping, run watchdog) |
| [`tests/`](tests/) | Zero-dependency test suites, run under both venvs |
| `workspace/` | Per-run state (gitignored): one directory per search run — harnesses, scores, traces, frontier |

---

## 2. How forge works

### The outer loop

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

The proposer is **not** told which prior to use. Following the
[Meta-Harness paper](docs/meta%20hearness.pdf), CC browses the run's
`harnesses/` and `frontier.json` itself (Read/Grep/Glob + Bash + jq +
WebSearch), decides which candidates to study, and records its chosen
priors in the new harness's `meta.json::parent_ids`. There is no
algorithmic selection — the frontier is a pure record store.

Two evaluation-efficiency mechanisms:

- **Staged evaluation** — each benchmark runs a stage1→2→3 promotion
  gauntlet with config thresholds; bad candidates die on a ~20-item stage1
  instead of consuming a full eval. Sampling is deterministic and *nested*
  (a smaller stage's task set is a strict subset of a larger one).
- **Cross-stage memory cache** — the memory a harness builds in Phase 1 is
  snapshotted (pickle; per checkpoint for DynamicMem) and reused at deeper
  stages instead of being rebuilt. See [`common/memory_cache.py`](common/memory_cache.py).

How much and which data each eval covers is controlled by three flags,
honored by forge AND every baseline (alma + the harness/ baselines):
`progressive` (default `true` — the staged gauntlet above vs. one
single-stage pass over the whole split; unifies/supersedes the older
`progressive: true|false`), `random_sample` (default `false` — whether each
search-loop step evaluates a different reproducibly-seeded task subset or
the same fixed one every step), and `sampling_seed` (default `42` — the
base seed for the per-step subset derivation). See
[`configs/search_example.yaml`](configs/search_example.yaml) for the fully
documented block. Held-out test evaluation (below) always forces full
the same single-stage pass regardless of these flags.

### What the proposer sees inside its container

Each propose call runs Claude Code in a fresh Singularity container with a
**deliberately selective** bind list — not a whole-repo mount:

```
/workspace                      (RW, cwd)  = workspace/<run_id>/ — THIS run only
├── harnesses/<int>_<hash8>/    every prior candidate: harness.py, meta.json,
│                                <dataset>/score.json + stages.json + traces/,
│                                and the exact prompts that produced it
│                                (.prompt_system.txt / .prompt_task.txt)
└── frontier.json               the population with per-benchmark scores

/app                            (RO, selective; PYTHONPATH=/app)
├── forge/harness_base.py       the MemoStructure base the new harness must inherit
├── common/{harness_base,recorder,llm,logger}.py
└── datasets/                   env/workflow/prompts per benchmark + raw data,
                                with the held-out TEST SPLIT PHYSICALLY ABSENT
                                (search-mode overlay binds shadow it — filtered
                                data files, gold answers stubbed out)
```

Not visible, by construction: `.env`, `.git`, other runs' workspaces,
`baselines/`, the host outer-loop code (`forge/orchestrator.py` etc.), and
the prompt-template package (prompts are rendered host-side and staged into
the harness dir). The evaluator container is similarly selective: full
`common/` + `datasets/` RO, `forge/launch.py` as entrypoint, the candidate
harness RO at `/harness`, output RW at `/out` — same search-split-only data
overlay during search.

---

## 3. The `MemoStructure` contract

Everything this repo evaluates — forge-evolved harnesses AND ready-made
baseline memory systems — is a subclass of
[`common.harness_base.MemoStructure`](common/harness_base.py) implementing
three optional-override hooks:

```python
class MyMemory(MemoStructure):

    async def build_memory_from_data(self, recorder) -> None:
        """BUILD. recorder.init holds the data newly visible for THIS call.
        Called once per visible-data batch (per checkpoint for DynamicMem);
        accumulate across calls and choose your own ingestion granularity."""

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        """RETRIEVE. recorder.init holds the query (+ per-benchmark metadata).
        Return the dict fed to the QA agent. Must be READ-ONLY w.r.t. memory
        state (DynamicMem interleaves queries with ingestion at checkpoints)."""

    async def use_memory_to_answer(self, recorder, retrieved, prompt) -> Optional[str]:
        """ANSWER (optional). Return the answer string to bypass the standard
        QA agent, or None (the default) to let it answer from `retrieved`.
        forge NEVER overrides this — the search optimizes memory, not the
        answerer. Agentic baselines may (e.g. Claude Code answers natively)."""
```

A fresh instance is created per user/sample — no cross-user state. The
`recorder` is the evaluation **data envelope**
([`common/recorder.py`](common/recorder.py)): the benchmark fills
`recorder.init` (each benchmark with a different shape — dispatch on its
keys), the workflow logs steps and reward for traces. One contract, two
consumers:

- **forge-evolved harnesses** implement build + retrieve (inheriting
  [`forge/harness_base.py`](forge/harness_base.py), a documented subclass);
- **baseline memory systems** implement the same hooks and are scored
  through the *same* per-dataset workflows — see
  [baselines/README.md](baselines/README.md) for the adaptation guide.

### Benchmarks

Three benchmarks are wired in (each runs as its own Singularity exec
against the same harness):

| Dataset | Source | Protocol | Split |
|---|---|---|---|
| **[DynamicMem](datasets/dynamicmem/)** | App-activity logs (~1500/user over 15 months) | Official **TCE v2 checkpoint protocol**: ingestion interleaved with tasks at 5 quarterly checkpoints; two task families (state completion + personalized service); official holistic Core+Detail judge, scores 0–1 | 6 users search / 4 test |
| **[LoCoMo](datasets/locomo/)** | Multi-session two-person conversations (~154 QA each after filtering) | Two-phase; binary CORRECT/WRONG judge (community-standard); QA **categories 1–4 only** (cat-5 adversarial excluded — the data carries no gold answers for them) | 6 conv search / 4 test |
| **[LongMemEval](datasets/longmemeval/)** | 500 questions, each with its own haystack of chat sessions (`s` ~48, `m` ~476) | Two-phase, 1 QA per question; binary yes/no judge (paper) | 300 search / 200 test (stratified by question type) |

Data files are **not** in the repo (DynamicMem `user_data/<user>/{app_log_large,task_packs}.json`,
~77 MB total; LongMemEval `m_cleaned.json` 2.6 GB). Acquire separately and
place under `datasets/<bench>/`; DynamicMem honors a `DYNAMICMEM_DATA`
env-var override.

---

## 4. Setup & quickstart

### Setup

**Requirements**: Python 3.12, an OpenAI API key, the host's `claude` CLI
(subscription login), and Singularity.

```bash
git clone https://github.com/ash-ding/memevol.git
cd memevol

# Forge's host env is intentionally minimal — only YAML parsing,
# the SDK proposer driver, and subprocess management. All ML
# dependencies (chromadb, sentence-transformers, ...) live in the
# Singularity images. Managed with uv (pyproject.toml + uv.lock):
uv sync            # → .venv/ (CPython 3.12 from .python-version)
# then run forge with `uv run python -m forge.orchestrator ...`

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
default is host-specific — override there if needed.)

### Part A — run the main method (search)

The search loop always runs on the **search split** (the held-out test
split is physically invisible inside search containers).

```bash
# Smoke test: ONE propose → eval → score round at sanity sizes
# (--smoke-test forces steps=1 / k_per_step=1 unless overridden; ~1-2 min)
uv run python -m forge.orchestrator --config configs/search_example.yaml --smoke-test

# Real search — copy the documented example config and edit it
# (steps, benchmark, stage sizes, models), then:
uv run python -m forge.orchestrator --config configs/my_search.yaml --run-name my_search

# CLI overrides (any YAML field has a matching CLI flag)
uv run python -m forge.orchestrator \
  --config configs/search_example.yaml \
  --steps 3 --datasets locomo --gpu
```

Everything the run produces lands under `workspace/<run_name>/`:
`harnesses/<int>_<hash8>/` (code + per-benchmark scores + traces),
`frontier.json` (the population), `orchestrator.log`.

### Part B — held-out test of a discovered harness

Held-out evaluation is deliberately a **separate entry point**
(`forge.heldout`): it runs frozen harnesses on the **test split**, whole
split by default (`progressive: false`),
with no proposer / sanity gate / frontier — running the search loop on
test data would optimize against the held-out split. Held-out evaluation
always uses one single-stage pass; the staged gauntlet
(`progressive: true`) samples subsets
and eliminates candidates early, which is invalid for final held-out
numbers — `forge.heldout` refuses to run (exits with an error) if either
is set.

```bash
# Config-first: list the harnesses in the YAML (see configs/test_example.yaml)
uv run python -m forge.heldout --config configs/test_example.yaml

# Or point at specific harness dir(s) from a finished search run
uv run python -m forge.heldout --config configs/test_example.yaml \
  --harness workspace/my_search/harnesses/3_9f00aa11

# → workspace/heldout_<ts>/heldout_results.json + per-benchmark artifacts
```

To compare against baselines on the same test split, see
**[baselines/README.md](baselines/README.md)** — baseline runs use the
same split definitions, workflows, and judges (literally the same code
path), so the numbers sit on the same axis.

---

## Method-design references

- [`docs/meta hearness.pdf`](docs/) — Meta-Harness paper (agent-driven
  parent selection, full filesystem feedback, no compressed per-candidate
  summaries).
- The mission framing inside the active prompt template under
  [`forge/prompts/templates/`](forge/prompts/templates/) decomposes
  "biological memory" into a 12-axis taxonomy — the search dimensions the
  proposer is asked to advance.

## License

MIT
