# Baselines

Comparison methods for the memory-architecture search. Each baseline runs
against the same benchmark set as the main method ([forge/](../forge/)),
producing comparable metrics: per-user reward, judge-scored accuracy, and
(for some) token / latency telemetry.

## Contents

1. [Layout — two kinds of baseline](#layout--two-kinds-of-baseline)
2. [Method-boundary conventions](#method-boundary-conventions)
3. [Shared progressive sampling & seeding](#shared-progressive-sampling--seeding)
4. [Configuration](#configuration)
5. [Existing baselines](#existing-baselines)
6. [**Adding a harness baseline** — adapting an existing memory system](#adding-a-harness-baseline--adapting-an-existing-memory-system)
7. [Adding an evolve baseline](#adding-an-evolve-baseline)
8. [Shared foundation](#shared-foundation)

## Layout — two kinds of baseline

`baselines/` is split by WHAT is being compared:

```
baselines/
├── registry.py          # shared dataset registry (both sides import it)
│                        # (set up any baseline with: cd <its dir> && uv sync;
│                        #  shared contract tests run in the repo-root uv project)
├── evolve/              # SEARCH-METHOD baselines — compared against forge ITSELF
│   ├── alma/            #   LLM-meta-agent search loop (memevol's original method)
│   │                    #     own pyproject.toml + .python-version + uv.lock + .venv/ (gitignored)
│   ├── evolvemem/       #   EvolveMem AutoResearch loop (arXiv:2605.13941) — STANDALONE
│   │                    #     reproduction: vendored upstream/ @ db80b6a incl. its own runners.
│   │                    #     Does NOT implement MemoClass; numbers not comparable with forge.
│   ├── meta-harness/    #   Meta-Harness coding-agent harness search (arXiv:2603.28052)
│   │                    #     proposer = the `claude` OR `codex` CLI, driven as a subprocess
│   └── memevolve/       #   (paper PDF; to be implemented under the conventions below)
└── harness/             # READY-MADE MEMORY SYSTEMS — compared against forge's
    ├── eval_harness.py  #   EVOLVED HARNESSES. THE one entrypoint (CLI + run_baseline())
    ├── config.example.yaml   # the one shared frame config (identical for all seven)
    ├── model_config.py  #   embedder factory + OpenAI param normalisation (two model arms)
    ├── hipporag2/       #   HippoRAG2 graph-RAG pipeline as retrieval memory
    │                    #     (vendored src/hipporag/ @ c617143)
    ├── amem/            #   A-mem agentic-notes memory as retrieval memory
    ├── lightmem/        #   LightMem compression + offline-update memory
    ├── simplemem/       #   SimpleMem semantic-compression memory as retrieval memory
    ├── mem0/            #   Mem0 LLM-extracted facts with ADD/UPDATE/DELETE consolidation
    ├── memoryos/        #   MemoryOS three-tier STM/MTM/LPM memory, heat-based promotion
    └── zep/             #   Zep/Graphiti temporal KG memory (embedded FalkorDB Lite)
                         #     Each harness/<name>/ (and evolve/alma/) also has
                         #     its own pyproject.toml + .python-version + uv.lock + .venv/ (gitignored)
```

- **`evolve/`** — methods that SEARCH over memory-structure code, like forge
  does. Their unit of comparison is the search loop itself (proposer quality,
  sample efficiency, final evolved-harness score vs forge's).
- **`harness/`** — fixed, hand-written memory systems implementing the same
  standardized 3-hook `MemoClass` contract
  (`build_memory_from_data` / `retrieve_memory_for_query` /
  `use_memory_to_answer`) that forge-evolved harnesses implement. Their unit
  of comparison is the harness artifact: they run through the SAME
  per-dataset workflows via `baselines.harness.eval_harness.run_baseline`, so
  their scores sit on the same axis as any evolved harness's.

## Method-boundary conventions

These rules keep scores comparable while keeping methods independent. They
bind every method here:

- **The eval surface is mandatorily shared.** A method's FINAL ARTIFACT is a
  `common.memo_class.MemoClass` subclass implementing the 3-hook
  contract, and it is scored ONLY through the shared registry/workflow path
  (`baselines/registry.py` → `benchmarks/<bench>/workflow.py` + the shared
  judge). No method ships its own scoring loop — otherwise its numbers stop
  being comparable.
- **Split discipline.** Development, tuning, and any internal
  search/self-improvement loop run on the **search split** only
  (`split: search` in a harness baseline's config, `--status search` for
  alma). The **test split** is touched exactly once per reported number
  (`split: test` in the shared `harness/config.example.yaml` — same
  data path as `forge.heldout`).
- **Dependency direction is one-way.** Methods import `common/` (and
  `benchmarks/` / `baselines/registry.py`); `common/` NEVER imports a method;
  methods NEVER import each other. Method-specific design vocabulary lives
  inside the method's own directory (e.g. alma's `Sub_memo_layer` in
  `evolve/alma/memo_layers.py`), never in `common/`.
- **Duplication over coupling.** If two methods need similar internal
  machinery, copy it. Duplicated internals are acceptable — preferred, even —
  because independence between methods matters more than DRY.

Each baseline is its **own standalone uv project** — `pyproject.toml` +
`.python-version` (3.12) + a committed `uv.lock`. Set one up by `cd`-ing into
its directory and running `uv sync`, which creates
`baselines/{evolve,harness}/<name>/.venv/`:

```bash
cd baselines/harness/amem && uv sync
```

`uv sync` is the whole setup for every baseline — none of them needs a second
step. (hipporag2 used to need an editable install of an external HippoRAG
checkout; its source is vendored under `src/` as of 2026-08, like every other
harness baseline.)

There is **no shared baselines dev/test venv** — the shared contract tests
(`tests/test_baselines_multidataset.py`, `test_config.py`, `test_sampling_plan.py`,
…) are baseline-free and run in the **repo-root uv project** (forge's project,
which already imports `common/` + `benchmarks/`): `uv sync` at the repo root,
then `uv run python tests/test_baselines_multidataset.py`.
A baseline's OWN behavioral test (`tests/test_<name>_baseline.py`) runs in that
baseline's own project (it needs that baseline's deps):
`uv run --project baselines/<evolve|harness>/<name> python tests/test_<name>_baseline.py`.
Every harness baseline writes one directory per run under its own `runs/`
(gitignored — see **Run records** below); alma writes `logs/` + `results/`.

## Shared progressive sampling & seeding

Every baseline (`evolve/alma`, `evolve/meta-harness`, and every `harness/*`)
shares the same
evaluation-sampling infrastructure as forge — the literal same `common/`
modules forge uses — via two flags/keys.
**alma** and **meta-harness** still expose these as CLI flags layered over
their config (see "Configuration"); **harness baselines** (2026-08-06) have NO CLI parameter
surface at all — every one of these is a config-file key only, validated
exactly by `common.config.validate_exact_config`:

- **`progressive`** (alma default `true`, harness config default `false` —
  matching each side's historical behavior): drives the candidate through
  the shared stage1→2→3 gauntlet (`common.evaluate.evaluate_memo`) instead
  of a single one-shot pass. Sizes come from the family `DEFAULT_STAGES`
  (in `common/evaluate.py`) unless overridden by a `stages:` block in the
  config — for BOTH alma and harness, sizing is config-file only (no
  `--stages`/`--stage-spec` JSON CLI flag anywhere; see "Configuration").
  The single-pass path (`progressive: false`) instead REQUIRES a
  `single_stage:` block in the config (a null/omitted field = the whole
  split for that dimension; an absent block raises a clear error, never a
  silent whole-split). `evaluate_memo`'s promotion/elimination logic,
  `stages.json` shape, and cost accounting are IDENTICAL to forge's — it is
  the literal same function; only the ISOLATION wrapper differs (forge runs
  evaluate_memo inside a Singularity container, alma inside a plain
  subprocess, harness baselines call it directly in-process).
- **`random_sample`** (alma + meta-harness; alma's CLI `--random_sample` — underscore,
  `argparse.BooleanOptionalAction` so the negation is
  `--no-random_sample`; harness baselines have no step loop, so this
  isn't exposed there at all — not as a flag, not as a config key; default
  `false`): whether each search STEP evaluates a different, reproducibly
  seeded task subset (via `common.sampling.derive_sample_seed`) instead of
  the same fixed subset every step.
- **`sampling_seed`** (default `42`; alma CLI `--sampling_seed`, harness
  config key `sampling_seed:`): the base seed. alma combines it with
  `(step_index, dataset)` when `random_sample` is on; harness baselines (no
  steps) use it directly as a single fixed seed for their one-shot sample.

There is no memory cache (removed 2026-09-14): every gauntlet stage builds its
users' Phase-1 memory from scratch, and a config that still lists
`memory_cache:` is rejected with an error naming the removal.

Surface per side:

```bash
# harness (one shared entrypoint for all seven) — exactly one flag, --config;
# harness / progressive / sampling_seed / stages / single_stage all live in the YAML
uv run --project baselines/harness/hipporag2 python -m baselines.harness.eval_harness --config my_hr.yaml
# my_hr.yaml: harness: hipporag2, dataset: locomo, progressive: true, sampling_seed: 42, ...

# alma (evolve/alma/run.py) — has a step loop and a real CLI, so
# random_sample applies and flags still layer over the config
cd baselines/evolve/alma && uv run python run.py \
    --status search --progressive --random_sample --sampling_seed 42 --steps 10
```

See `common/sampling.py` ("Shared progressive sampling") for the full
seed-derivation contract (`derive_sample_seed` / `combine_seed` /
`shuffle_prefix`, nesting guarantees, and the accepted overfitting-vs-
comparability tradeoff of `random_sample`) — not a link, since the
design docs under `docs/` are gitignored local scratch notes and never
committed (a link from a tracked file would be dead on a fresh clone).

## Configuration

The two sides of `baselines/` resolve their config differently (2026-08-06):
**alma** and **meta-harness** keep the original layered scheme (both have a
real CLI with runtime knobs like `--status`/`--steps`/`--run-name`); every
**harness baseline**
(`hipporag2`, `amem`, `lightmem`, `simplemem`, `zep`, `mem0`,
`memoryos`) runs through **one shared entrypoint**,
[`harness/eval_harness.py`](harness/eval_harness.py), which takes exactly one
flag, `--config <yaml>`, and reads a config that carries the **evaluation frame
only** — every method knob is a class-level default on the memo class.

### alma + meta-harness (evolve baselines) — layered, unchanged

- **[`common/config.py`](../common/config.py)** — `deep_merge(base, overlay)`
  (in-place recursive dict merge) + `resolve_config(defaults, config_path,
  cli_overrides)`. Precedence, lowest → highest: `run.py`'s own
  `DEFAULT_CONFIG` dict < the YAML at `--config` (deep-merged in, must be a
  mapping) < CLI flags (every flag defaults to `None`, a "not given"
  sentinel, so only flags actually passed on the command line override the
  YAML — an unset flag never clobbers a YAML value with `None`). forge
  reuses the same `deep_merge` (`forge/orchestrator.py` imports it as
  `common.config.deep_merge`), so there is exactly one merge implementation
  shared by forge and alma.
- **Strict config — no silent defaults** (2026-07-26, default ON): when
  alma is launched with `--config`, the config file MUST list EVERY
  parameter (every key in `DEFAULT_CONFIG`) — a `null` value counts as
  listed. A missing key aborts the run with a `ConfigCompletenessError`
  naming exactly which keys are absent, so a run can never silently pick up
  a hidden default (`sampling_seed`, `judge_model`, `model`, `progressive`,
  …) you didn't choose. [`evolve/alma/config.example.yaml`](evolve/alma/config.example.yaml)
  is an exhaustive template that passes strict as-is — copy it and edit.
  The active sizing block is checked to the LEAF (every native size field
  must be listed; a null leaf = whole split for that dimension). Escape
  hatch: `strict_config: false` in the YAML, or `--no-strict-config` on the
  CLI, disables the check for that run. Strict triggers ONLY with
  `--config` — pure-CLI runs are unaffected. (forge honors the same
  `strict_config` knob for `--config` runs, validated against its own
  nested required-schema; `forge.heldout` uses a smaller schema matching
  its reduced config surface.)

```bash
# alma — config file, with a CLI override (CLI wins on conflicts)
cd baselines/evolve/alma && uv run python run.py \
    --config config.example.yaml --sampling_seed 7
```

### harness baselines — one frame config, method knobs on the class

```bash
# from the repo root, inside the TARGET baseline's own uv project (each has
# its own venv, so --project is not optional)
uv run --project baselines/harness/<name> python -m baselines.harness.eval_harness \
    --config baselines/harness/config.example.yaml
```

[`harness/config.example.yaml`](harness/config.example.yaml) is the ONE config,
identical for all seven baselines. It carries the **evaluation frame** and
nothing else: `harness` (which memo class), `arm` (see **Two model arms**),
`dataset`, `split`, `progressive`, `sampling_seed`, `single_stage`, `stages`,
`llm_model`, `judge_model`, `max_sample_concurrent`, and an
optional `run_name`. No method parameter appears in it.

- **Method knobs live on the memo class**, declared ONCE as
  `CONFIG_DEFAULTS` in `harness/<name>/memo.py` — with the comment that
  justifies each value (paper section, upstream default, deliberate
  deviation) next to it — plus `UNIFIED_OVERRIDES`, the model swap
  `arm: unified` applies. The point of these baselines is to run each system
  as its authors specified and get one number, not to tune it, so there is no
  method-parameter surface in the documented path. Print a class's values
  without reading source:
  `uv run --project baselines/harness/<name> python -m baselines.harness.eval_harness --describe <name>`.
- **Exact validation is unchanged** —
  [`common/config.py::validate_exact_config`](../common/config.py): the YAML
  must list every frame key and NO other key (a `null` value counts as
  listed; a missing or unknown key — including a method knob typed at the top
  level — aborts before anything runs, naming every problem at once). The
  active sizing block is checked to the LEAF via
  `common.evaluate.missing_sizing_config`.
- **Ablation hatch (left out of the example on purpose)**: an optional
  `memo:` block overrides individual class defaults, e.g.
  `memo: {retrieve_k: 10}`. It is validated against that class's
  `CONFIG_DEFAULTS`, so a typo still aborts. It exists so an ablation never
  requires editing `memo.py` — which would silently change every later run.
- **`config.paper.yaml`** — `mem0` and `memoryos` keep one: the record of the
  run that produced the reproduction numbers in their READMEs (frame keys
  only; the method runs at its faithful defaults).
- **Sampling sizes are config-file only** across ALL baselines (and forge)
  — there is no `--stages` / `--stage-spec` JSON-string CLI flag anywhere,
  and no sizing CLI flag of any kind for harness baselines. Both
  `stages:` (progressive) and `single_stage:` (one-shot) are native YAML
  mappings: `stages: {stage1: {...}, ...}` drives the gauntlet;
  `single_stage: {n_qa: null, ...}` sizes the single pass (required when
  `progressive: false`; a null field = whole split for that dimension). The
  unified resolver is `common.evaluate.resolve_sampling_plan`
  (`progressive` → `stage_plan`; `not progressive` → the `single_stage`
  wire spec, raising if the block is absent) — shared verbatim by forge,
  alma, and every harness baseline.

```bash
# copy the shared example, edit it, point --config at your copy
cp baselines/harness/config.example.yaml my_hr.yaml
$EDITOR my_hr.yaml   # harness: hipporag2, dataset:, split:, sizing, ...
uv run --project baselines/harness/hipporag2 python -m baselines.harness.eval_harness --config my_hr.yaml
```

### Run records

Every run gets its own directory — nothing is ever overwritten, and every
number carries what produced it:

```
baselines/harness/<name>/runs/                       (gitignored)
├── index.jsonl                   one line per finished run: run_id, arm, dataset,
│                                 split, raw_score, stage, tokens, wall_clock_s, git_sha
├── latest                        the most recent run_id
└── <run_id>/                     <YYYYMMDD_HHMMSS>_<dataset>_<split>, or `run_name`
    ├── config.resolved.yaml      frame + the FULLY expanded memo config — the only
    │                             place a run's method parameters (internal LLM,
    │                             embedder, ...) are recorded; feeding it back as
    │                             --config reproduces the run
    ├── run.json                  harness, memo class, git sha, python, start/end,
    │                             wall clock, exit status
    ├── run.log                   the logger tape for this run
    └── score.json  token_usage.json  stages.json  traces/  stage*/   (evaluate_memo, unchanged)
```

### Two model arms — faithful vs unified

Every model a harness baseline touches — internal LLM, embedder, reranker,
compressor — is a class-level parameter, and the `arm:` frame key selects
between two settings of them:

| arm | where the values live | what it is |
|---|---|---|
| **faithful** | `CONFIG_DEFAULTS` | each method's own published models. **The default.** What every README's faithfulness table describes |
| **unified** | `CONFIG_DEFAULTS` + `UNIFIED_OVERRIDES` | one LLM (`gpt-5-mini`) + one embedder (`text-embedding-3-small`) everywhere. The arm to compare against the main method |

The faithful defaults are deliberately NOT replaced: the local embedders are
paper choices (zep's bge-m3, simplemem's Qwen3) and the READMEs make
faithfulness claims about them. Silently switching them would turn every
baseline into a variant its authors never published. Choosing an arm is a
config decision, and the two are directly comparable because everything else is
held fixed.

**Every faithful-arm model is traceable to a paper section**, cited inline in
each memo class's `CONFIG_DEFAULTS`:

| baseline | internal LLM | embedder | source |
|---|---|---|---|
| amem | gpt-4o-mini | all-MiniLM-L6-v2 | arXiv 2502.12110 §4.2 + Table 1 |
| hipporag2 | gpt-4o-mini ⚠️ | text-embedding-3-small ⚠️ | arXiv 2502.14802 §4.4 — **see below** |
| lightmem | gpt-4o-mini | all-MiniLM-L6-v2 | ICLR 2026 Table 5 (+ LLMlingua-2 compressor) |
| mem0 | gpt-4o-mini | text-embedding-3-small | arXiv 2504.19413 §2, §3.3 |
| memoryos | gpt-4o-mini | all-MiniLM-L6-v2 ⚠️ | arXiv 2506.06326 Tables 1-2; **paper states no embedder** |
| simplemem | gpt-4.1-mini | Qwen/Qwen3-Embedding-0.6B | arXiv (SimpleMem) §3.1 |
| zep | gpt-4o-mini-2024-07-18 | BAAI/bge-m3 | arXiv 2501.13956 §4.1 (+ bge-reranker-v2-m3) |

Three entries carry a caveat, each recorded in its own README rather than left
implicit:

- **zep** pins the **dated** `gpt-4o-mini-2024-07-18` because that is what §4.1
  names. The undated alias now resolves to a later snapshot, so leaving it
  undated would silently stop reproducing the paper.
- **memoryos**'s embedder comes from the vendored code, not the paper — §4.1
  gives every capacity and threshold but never names an embedding model.
- **hipporag2 is the one baseline whose default arm is not its paper's setup.**
  §4.4 specifies Llama-3.3-70B-Instruct (NER/OpenIE/triple filtering) and
  nvidia/NV-Embed-v2 (retriever) — a 70B model and a 7B embedder, both local and
  GPU-bound, which this API-based harness cannot run. The defaults are runnable
  API equivalents; the paper's models are named in the config and the README.

Where a paper and its shipped code disagree, **the paper wins** and the code's
value is named in the comment — memoryos (`short_term_capacity` 7 vs code 10,
`mid_term_capacity` 200 vs code 2000) and simplemem (`window_size` 20 vs code
40) both do this. Numbers collected under the other value are not comparable.

Two things stay local in BOTH arms because they have no API equivalent:
lightmem's **LLMlingua-2** prompt compressor (a BERT token classifier, produces
no vectors) and zep's **bge-reranker-v2-m3** (a cross-encoder scoring
`(query, doc)` pairs). They remain a real, untracked local compute cost.

**How the unified arm is possible without touching `src/`.** Both levers live in
[`harness/model_config.py`](harness/model_config.py) and act at boundaries the
vendored code already passes through, so every README's `diff -r` byte-identity
check still passes:

- **`get_embedder`**, one cached factory returning either a real
  sentence-transformer or an `.encode()`-compatible API adapter. It memoizes
  across users, which matters because a fresh MemoClass is built per user (up
  to ~0.6B of weights would otherwise reload per conversation). How each
  baseline reaches it differs:
  - **amem, simplemem, lightmem** build their embedder internally with no
    injection point, so `install_embedder_factory()` patches the
    `sentence_transformers` constructor they all funnel through. The
    *configured* name is also the name they request, so the factory just
    dispatches on it.
  - **memoryos** is the exception: its `get_embedding()` carries the model name
    as a default argument, so the requested name is never the configured one.
    It calls `get_embedder()` directly and seeds its own vendored model cache
    under the requested name — one dict entry, no global constructor patch.
  - **zep** needs none of it: Graphiti accepts an injected `EmbedderClient`.
    lightmem's API arm likewise uses its own vendored `TextEmbedderOpenAI`.
- **the OpenAI param normalisation** patches `chat.completions.create` to drop
  `temperature`/`top_p`/the penalties and rename `max_tokens` →
  `max_completion_tokens` for reasoning models. **Five of the seven baselines
  could not run a gpt-5 model at all without it** — amem, lightmem, simplemem
  and memoryos hardcode `temperature` + `max_tokens`, and zep's graphiti still
  sends `max_tokens` even though it already drops `temperature`.

**Dimension coupling.** The API embedder is 1536-dim against local defaults of
384 (MiniLM) / 1024 (bge-m3, Qwen3). Only lightmem carries an explicit
`embedding_dims` knob that must move with it; the others size their index from
the embedder itself. In every case, switching arms invalidates vector stores
built at the other width.

## Existing baselines

| Baseline | Kind | Approach | Optimization | Best for |
|---|---|---|---|---|
| **[evolve/alma](evolve/alma/)** | search method | LLM-meta-agent search loop | Yes — propose / select / evolve over harness code | Established baseline; the framework's "v1" memory-architecture search |
| **[evolve/evolvemem](evolve/evolvemem/)** | **standalone reproduction** (not a contract baseline) | EvolveMem AutoResearch loop (arXiv:2605.13941): LLM diagnoses per-question failure logs and proposes guarded edits to a ~42-field retrieval config | Yes — but over a CONFIG, never over code | Reproduces the paper (59.1 F1 vs its reported 54.3) by running upstream's own runners on upstream's benchmark. **Implements no `MemoClass` and never calls `evaluate_memo`, so its numbers do NOT sit on the same axis as forge's or any harness baseline's — never table them together.** It also evolves over the full LoCoMo-10, held-out conversations included |
| **[evolve/meta-harness](evolve/meta-harness/)** | search method | Meta-Harness coding-agent harness search (arXiv:2603.28052): a `claude`/`codex` session reads prior harness code, scores and QA traces off the run's filesystem and writes new `MemoClass` candidates; no parent selection, no compressed feedback | Yes — propose / evaluate / log over harness code | Agentic-proposer comparison point for forge's own proposer; Pareto over (score, context cost) |
| **[harness/hipporag2](harness/hipporag2/)** | ready-made harness | Graph-based RAG pipeline as retrieval memory (OpenIE → KG → PPR retrieval → passages; shared QA agent answers) | None (fixed pipeline) | Hand-designed memory architecture comparison point, multi-dataset |
| **[harness/amem](harness/amem/)** | ready-made harness | A-mem agentic-notes memory (per-note LLM analysis + memory evolution → keyword-rewrite retrieval; shared QA agent answers) | None (fixed pipeline) | Agentic note-graph memory comparison point, multi-dataset |
| **[harness/lightmem](harness/lightmem/)** | ready-made harness | LightMem compression + offline-update memory (LLMlingua-2 pre-compression → topic segmentation → LLM metadata/summary extraction → Qdrant index → per-entry LLM offline update; `LightMemory.retrieve` → passages; shared QA agent answers) | None (fixed pipeline) | Compression + offline-refinement memory comparison point, multi-dataset |
| **[harness/simplemem](harness/simplemem/)** | ready-made harness | SimpleMem semantic-compression memory (LLM window compression → self-contained memory units → intent-aware multi-view retrieval; shared QA agent answers) | None (fixed pipeline) | Compression-first memory comparison point, multi-dataset |
| **[harness/zep](harness/zep/)** | ready-made harness | Zep/Graphiti temporal knowledge-graph memory (episodes → LLM entity/fact/temporal-edge extraction; hybrid BM25+cosine+BFS search, BGE cross-encoder rerank; shared QA agent answers) | None (fixed pipeline) | Temporal-KG memory comparison point; embedded FalkorDB Lite backend, multi-dataset |

### evolve/alma — meta-learning loop

The original method memevol shipped with: an LLM meta-agent reads sampled
QA trajectories, identifies failure patterns, and proposes new memory-
structure code. Sanity-checks each candidate, then evaluates on the search
split. Softmax-weighted parent selection over reward.

```bash
# Smoke — tiny staged gauntlet, 2 steps (evaluation sizes come from the
# shared `stages` schema now, not flat eval_n_*/check_n_* flags — see
# "Shared progressive sampling" below)
cd baselines/evolve/alma && uv run python run.py \
    --status search --progressive --steps 2

# Full training — stage1->2->3 gauntlet (default DEFAULT_STAGES sizes), 10 steps
uv run python run.py \
    --status search --progressive --steps 10

# Held-out evaluation of a saved memo
uv run python run.py \
    --status test --memo_SHA <SHA> --progressive
```

Artifacts: `baselines/evolve/alma/{logs/, memo_archive/, results/}`. See
[evolve/alma/README.md](evolve/alma/README.md) for layout details.

**Key difference from forge**: alma's proposer is a single LLM call with
compressed feedback (sampled trajectories + meta-prompt), whereas forge's
proposer is an agentic CC SDK call with full filesystem access to all
prior code, traces, and scores. alma runs the shared per-dataset workflows
(including the official DynamicMem TCE v2 checkpoint protocol) AND the same
`common.evaluate.evaluate_memo` evaluator forge runs in-container, so its numbers ARE
comparable with forge.

### evolve/evolvemem — standalone reproduction of the EvolveMem paper

[EvolveMem](https://arxiv.org/abs/2605.13941), vendored whole from
[aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem) @ `db80b6a` —
the same commit `harness/simplemem` is pinned to — **including its own entry
points**. A closed loop (EVALUATE → DIAGNOSE → PROPOSE → GUARD) optimises a
~42-field retrieval configuration θ: an LLM reads per-question failure logs,
proposes at most two field edits per round, and an elitist guard reverts
regressions, explores on plateaus, and stops on convergence.

```bash
cd baselines/evolve/evolvemem/upstream
python run_benchmark.py locomo --data ../../../../benchmarks/locomo/locomo10.json \
    --initial weak --max-rounds 7 --embed-model BAAI/bge-base-en-v1.5
```

**This one is scoped differently from every other entry here, by owner decision
(2026-08-15).** It is a reproduction of someone else's paper, not a baseline on
this repo's contract: it implements no `MemoClass`, never calls
`common.evaluate.evaluate_memo`, and evolves over the full LoCoMo-10 — held-out
conversations included, with the adversarial category scored against
`adversarial_answer`, which this repo's own LoCoMo drops. It reaches **F1 59.1**
against the paper's reported 54.3.

Consequences, both load-bearing: its numbers **must never be tabled next to a
forge or harness-baseline number**, and everything under its
`evolution_results/` is test-touching and cannot be used to tune anything. An
earlier version did satisfy the evolve-baseline contract (θ-loading `MemoClass`
+ `evaluate_memo`) and scored 41.5 under this repo's rules; it lives in git
history (PR #33, before 2026-08-15). See
[evolve/evolvemem/README.md](evolve/evolvemem/README.md) for the trade-off and
for what the adversarial category actually scores.

### evolve/meta-harness — coding-agent search over harness code

[Meta-Harness](https://github.com/stanford-iris-lab/meta-harness)
(arXiv:2603.28052) delegates diagnosis and proposal to a **coding agent** with
filesystem access to the whole search history — every prior candidate's source,
score, and execution traces, read with `grep`/`cat` rather than compressed into
a prompt. There is no parent-selection rule and no summary channel. Each
iteration the agent writes N harnesses into `harnesses/` plus a
`pending_eval.json`; the loop import-checks them and scores each one through
`common.evaluate.evaluate_memo` on the search split, then appends the results
to the same filesystem the next iteration reads.

The proposer backend is one config key: **`claude_code`** (the `claude` CLI) or
**`codex`** (the `codex` CLI), both driven as stream-json subprocesses. Neither
is a Python dependency — install and log into the one you use.

```bash
cd baselines/evolve/meta-harness && uv sync

# search-split evolution, resumable under a fixed run name
uv run python run.py --config config.example.yaml --run-name run1

# same loop with Codex proposing
uv run python run.py --config config.example.yaml --run-name run1     --agent codex --agent-model gpt-5-codex

# ONE held-out evaluation of run1's Pareto frontier, then the run is frozen
uv run python run.py --config config.test.yaml --run-name run1
```

Artifacts: `baselines/evolve/meta-harness/{logs/<run>/, results/<dataset>/test/}`.
`logs/<run>/` IS the proposer's feedback channel — `evolution_summary.jsonl`,
`frontier_val.json`, `evals/<system>/traces/`, and the agent's own sessions
under `proposer/iter<N>/`. Two calibration harnesses ship tracked
(`no_memory`, `full_context`); everything else there is proposer-written and
gitignored.

**Key difference from alma and forge**: alma's proposer is one LLM call over
compressed feedback and forge's is an agentic CC SDK call inside a container;
meta-harness's is a host-side agent CLI session whose only structure is "here
is the run directory". Its candidates go through the SAME
`evaluate_memo` gauntlet, so its scores ARE comparable with forge's. Evolution
runs on `split: search` only; `--status test` spends the test split exactly once
and writes `finalized.json`, after which the run refuses to evolve further. See
[evolve/meta-harness/README.md](evolve/meta-harness/README.md) for the
faithfulness boundary (one benchmark per run, the staged gauntlet).

### harness/hipporag2 — graph-based RAG pipeline as retrieval memory

`HippoRAGMemo` ([harness/hipporag2/memo.py](harness/hipporag2/memo.py))
wraps [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s pipeline:

- **BUILD**: converts the visible data into text passages and indexes them
  into a per-user HippoRAG graph (OpenIE → NER + triples → knowledge graph +
  entity embeddings). Indexing is additive across calls, so DynamicMem's
  per-checkpoint segments accumulate correctly.
- **RETRIEVE**: fact retrieval → reranking → personalized PageRank → top-k
  passages, returned as `{"passages": [...]}`. The **shared QA agent**
  answers from those passages — a fair "HippoRAG-as-memory" comparison, not
  an end-to-end HippoRAG pipeline comparison.

```bash
# OpenAI API embedding (no GPU needed) — the class default
uv run --project baselines/harness/hipporag2 python -m baselines.harness.eval_harness \
    --config baselines/harness/config.example.yaml      # harness: hipporag2

# Local GPU embedding (NVIDIA) — override the class defaults in your config copy:
#   dataset: longmemeval_s
#   memo: {embedding: nvidia/NV-Embed-v2, embedding_batch_size: 2, embedding_dtype: float16}
```

Artifacts: `baselines/harness/hipporag2/{outputs/, runs/<run_id>/}`.

### harness/lightmem — compression + offline-update memory as retrieval memory

`LightMemMemo` ([harness/lightmem/memo.py](harness/lightmem/memo.py)) wraps
[LightMem](https://github.com/zjunlp/LightMem)'s vendored text pipeline
(`src/lightmem/{configs,factory,memory}`, byte-identical @ `34410f4`):

- **BUILD**: maps the visible data to LightMem turns (a `[user, assistant]` pair
  per unit) and feeds them one turn at a time to `add_memory` — LLMlingua-2
  pre-compression → attention topic segmentation → LLM metadata/summary
  extraction (`gpt-4o-mini`) → HF embedding → per-user Qdrant index
  (`update="offline"`). Additive across DynamicMem checkpoints. When
  `offline_update` is on (default), the LoCoMo-paper refinement
  (`construct_update_queue_all_entries` + `offline_update_all_entries`) runs after
  the last turn of a build call.
- **RETRIEVE**: `LightMemory.retrieve(query, limit)` — embed + Qdrant search,
  returned as `{"passages": [...]}`. The **shared QA agent** answers — a fair
  "LightMem-as-memory" comparison.

```bash
# faithful defaults (LLMlingua-2 pre-compress + topic-seg, MiniLM embedder; cuda
# if a GPU is visible, else cpu — slow)
uv run --project baselines/harness/lightmem python -m baselines.harness.eval_harness \
    --config baselines/harness/config.example.yaml      # harness: lightmem
```

Artifacts: `baselines/harness/lightmem/{outputs/, runs/<run_id>/}`.
See [harness/lightmem/README.md](harness/lightmem/README.md) for the faithfulness
boundary and provenance.

### harness/simplemem — semantic-compression memory as retrieval memory

`SimpleMemMemo` ([harness/simplemem/memo.py](harness/simplemem/memo.py)) wraps
[SimpleMem](https://github.com/aiming-lab/SimpleMem)'s vendored text pipeline
(`src/simplemem/{core,text}`, byte-identical @ `db80b6a`):

- **BUILD**: maps the visible data to SimpleMem `Dialogue`s and runs its
  compression pipeline — LLM window-compression (`WINDOW_SIZE=40`) into
  self-contained `MemoryEntry` units (coref-resolved restatement + keywords +
  timestamp/persons/entities/topic), indexed into a per-user LanceDB multi-view
  store. Additive across DynamicMem checkpoints.
- **RETRIEVE**: intent-aware planning + semantic/keyword/structured multi-view
  search (+ optional reflection), returned as `{"passages": [...]}`. The
  **shared QA agent** answers — a fair "SimpleMem-as-memory" comparison.

```bash
# faithful Qwen3-0.6B embedder (benefits from GPU)
uv run --project baselines/harness/simplemem python -m baselines.harness.eval_harness \
    --config baselines/harness/config.example.yaml      # harness: simplemem

# light MiniLM fallback embedder (no GPU) — override in your config copy:
#   memo: {embedding_model: all-MiniLM-L6-v2}
```

Artifacts: `baselines/harness/simplemem/{outputs/, runs/<run_id>/}`.
See [harness/simplemem/README.md](harness/simplemem/README.md) for the
faithfulness boundary and provenance.

### harness/zep — Zep/Graphiti temporal knowledge-graph memory

`ZepMemo` ([harness/zep/memo.py](harness/zep/memo.py)) vendors and drives
[Graphiti](https://github.com/getzep/graphiti) (@ `4f62cfe`, byte-identical under
`src/graphiti_core/`), the engine behind [Zep](https://arxiv.org/abs/2501.13956)
(arXiv:2501.13956):

- **BUILD**: each ingestion unit becomes one Graphiti *episode* via `add_episode`
  — LLM entity/fact extraction, entity resolution/dedup, bi-temporal edge
  extraction + invalidation, BGE-m3 embedding. Additive across calls (persistent
  per-user store), so DynamicMem's per-checkpoint deltas accumulate.
- **RETRIEVE**: `search_` with the paper's `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`
  recipe (BM25 + cosine + BFS over edges & nodes, BGE cross-encoder rerank), top-20
  facts + entity summaries formatted into the paper's FACTS/ENTITIES context string
  (`{"inline_memory_blocks": [...]}`). The **shared QA agent** answers.

Backend is **embedded FalkorDB Lite** (`falkordblite`, in-process, on-disk, no
server) scoped per-user by uuid — the amem/simplemem/lightmem operational model,
not Graphiti's default Neo4j server. Embedder/reranker default to paper-faithful
**BGE-m3** (config-toggleable to OpenAI). Requires **Python 3.12+**.

```bash
uv run --project baselines/harness/zep python -m baselines.harness.eval_harness \
    --config baselines/harness/config.example.yaml      # harness: zep
# device is auto-detected (cuda if visible, else cpu); pin with memo: {device: cpu}
```

Artifacts: `baselines/harness/zep/{outputs/, runs/<run_id>/}`. See
[harness/zep/README.md](harness/zep/README.md) for the faithfulness boundary
(FalkorDB Lite vs Neo4j-Lucene, the custom BGE-m3 embedder adapter) and cost
caveats (Graphiti's internal gpt-4o-mini calls bypass `common.tokens`).

---

## Adding a harness baseline — adapting an existing memory system

This is the recipe for evaluating an existing, human-crafted memory system
(mem0, letta, zep, MemGPT, your own prototype, ...) under this repo's
protocol. The whole adaptation is ONE file, `baselines/harness/<name>/memo.py`,
plus a registry entry; everything else — the CLI, config resolution, the run
directory, split resolution, the per-dataset evaluation protocol (including
DynamicMem's checkpoint interleaving), judging, scoring, trace persistence —
comes from the shared entrypoint and is byte-identical to what forge-evolved
harnesses get.

### Step 0 — understand what you're adapting to

Your system is driven through three async hooks on a
`common.memo_class.MemoClass` subclass. The evaluation lifecycle per
user/sample:

```
fresh instance created                       # NO cross-user state — ever
  → build_memory_from_data(recorder)          # 1..N times (N>1 only for DynamicMem:
                                              #   one call per checkpoint, DELTA data)
  → per query:  retrieve_memory_for_query(recorder)   # MUST be read-only
                use_memory_to_answer(recorder, retrieved, prompt)  # optional
```

Three lifecycle rules that trip up adapters:

1. **Fresh instance per user.** If your system persists state on disk,
   scope it per instance — e.g. `HippoRAGMemo` creates
   `uuid.uuid4().hex[:12]`-suffixed save dirs in `__init__`. Do NOT key
   state on `recorder.user_id` (it is always `""` at memo call sites).
2. **BUILD accumulates.** For LoCoMo/LongMemEval you get ONE build call
   with everything; for DynamicMem you get FIVE calls, each with only that
   checkpoint's new log segment. Your ingestion must be additive.
3. **RETRIEVE is read-only.** DynamicMem interleaves queries with
   ingestion at checkpoints — a retrieve that mutates memory corrupts
   checkpoint isolation.

### Step 1 — `memo.py`: the adapter class

`recorder.init` shapes per benchmark (dispatch on the keys):

| Benchmark | BUILD `recorder.init` | RETRIEVE `recorder.init` |
|---|---|---|
| dynamicmem | `{"app_logs": [log, ...]}` — each log: `app_log_id, timestamp, app_name, api_name, request, response`. Per-checkpoint DELTA | `{"app_logs": [...visible prefix...], "query": str}` |
| locomo | `{"conversation": {...}}` — `speaker_a/b`, `session_1..N` (turn lists: `speaker, dia_id, text`), `session_N_date_time` | `{"conversation": {...}, "query": str}` |
| longmemeval_s/m | `{"sessions": [session, ...]}` — each: `session_id, date, messages[{role, content}]` | `{"sessions": [...], "query": str, "question_date": str}` |

Skeleton:

```python
# baselines/harness/<name>/memo.py
import uuid
from typing import Dict, Optional
from common.memo_class import MemoClass

class MyMemo(MemoClass):
    # EVERY method knob, declared once, with the reason for its value. This is
    # the faithful arm; `--describe <name>` prints it, and every run records the
    # merged result in runs/<run_id>/config.resolved.yaml.
    CONFIG_DEFAULTS = {
        "my_llm_model": "gpt-4o-mini",      # PAPER §x.y: ...
        "embedding_model": "all-MiniLM-L6-v2",
        "top_k": 10,                        # PAPER §x.y: k=10
    }
    # What `arm: unified` changes — the fleet's one LLM + one API embedder.
    UNIFIED_OVERRIDES = {
        "my_llm_model": "gpt-5-mini",
        "embedding_model": "text-embedding-3-small",
    }

    def __init__(self, config=None):
        super().__init__(config)        # self.config = CONFIG_DEFAULTS overlaid with config
        self._instance_id = uuid.uuid4().hex[:12]   # per-user state scoping
        self._system = ...              # construct the wrapped memory system from
                                        #   self.config["top_k"] etc. — no second inline default

    async def build_memory_from_data(self, recorder) -> None:
        init = recorder.init
        if "app_logs" in init:          # dynamicmem (per-checkpoint delta)
            texts = [render_log(l) for l in init["app_logs"]]
        elif "conversation" in init:    # locomo
            texts = [render_turn(t) for t in iter_turns(init["conversation"])]
        elif "sessions" in init:        # longmemeval
            texts = [render_msg(s, m) for s in init["sessions"] for m in s["messages"]]
        self._system.add(texts)         # ADDITIVE — never reset here

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        hits = self._system.search(recorder.init["query"], top_k=...)
        return {"passages": hits}       # read-only w.r.t. memory state

    # OPTIONAL — only if your system answers natively (agentic systems).
    # Return None / omit entirely to let the shared QA agent answer.
    # async def use_memory_to_answer(self, recorder, retrieved, prompt) -> Optional[str]:
    #     return await self._system.answer(prompt)
```

Notes on the RETRIEVE return dict:

- Any dict works — it is rendered into the benchmark's QA prompt. A flat
  `{"passages": [...]}` is the safe generic choice.
- **DynamicMem tip**: `{"inline_memory_blocks": [str, ...]}` renders each
  block verbatim into the official TCE answer prompt's `[Memory]` section;
  any other dict shape is serialized as one JSON block. Surface the source
  logs *with their `app_log_id`* — evidence citation is scored.
- Override `use_memory_to_answer` ONLY for systems whose value proposition
  includes answering (an agentic system that answers natively). Retrieval-style
  systems should let the shared QA agent answer — that keeps the comparison
  about *memory*, not about who has the better answerer.

### Step 2 — register it

Add one line to `MEMOS` in [harness/eval_harness.py](harness/eval_harness.py):

```python
MEMOS = {
    ...
    "<name>": "baselines.harness.<name>.memo:MyMemo",
}
```

The registry maps a name to a *module path string* and imports lazily, only
for the harness a run selects — each baseline has its own venv, so a direct
import of every memo class would make the entrypoint unimportable in all of
them. That is also why `--project` is not optional in the launch command.

That is the whole integration. There is no per-baseline `run.py` and no
per-baseline config file: the frame config is the shared
[harness/config.example.yaml](harness/config.example.yaml) (set
`harness: <name>`), and your method knobs are `CONFIG_DEFAULTS` on the class.
The framework creates a FRESH memo per user/conversation as
`MyMemo(config=resolved_memo_config)`; each instance keeps its own private
copy at `self.config` (never share mutable state across users). No
`__init__.py` files needed (namespace packages).

### Step 3 — dependencies

Give your baseline its own self-contained `baselines/harness/<name>/pyproject.toml`
(+ `.python-version` pinned to 3.12), listing only your system's extra
packages on top of the shared core deps (see any existing
`harness/*/pyproject.toml` for the pattern). Build its `.venv/`:

```bash
cd baselines/harness/<name> && uv sync
```

This runs `uv sync` in `baselines/harness/<name>/`, creating
`baselines/harness/<name>/.venv/` and a committed `uv.lock`. Heavy or
mutually-conflicting deps are exactly why each baseline gets its own project
now — no need to reconcile them against any other baseline's dependencies.
Note anything unusual in your baseline's own README; scoring still MUST go
through `eval_harness`.

### Step 4 — validate on the SEARCH split (cheap, iterate freely)

Sizing and split are config keys, not flags — write a small smoke config
per dataset and point `--config` at it:

```yaml
# smoke_locomo.yaml — one conversation, a handful of QAs — does the
# adapter run end-to-end?  (copy harness/config.example.yaml and change:)
harness: <name>
dataset: locomo
split: search
single_stage: {n_conversations: 1, n_qa: 3}
```

```yaml
# smoke_dm.yaml — DynamicMem protocol check: 1 user exercises the
# checkpoint interleaving
harness: <name>
dataset: dynamicmem
split: search
single_stage: {n_users: 1, n_checkpoints: 1, n_task_a: 1, n_task_c: 1}
```

```bash
uv run --project baselines/harness/<name> python -m baselines.harness.eval_harness --config smoke_locomo.yaml
uv run --project baselines/harness/<name> python -m baselines.harness.eval_harness --config smoke_dm.yaml
```

Iterate here as much as you like — this is the split the main method
searches on. Read `runs/<run_id>/traces/<user>.json`: each QA
step records the query, your retrieved dict, the answer, and
`judge_reason` — the fastest way to see whether your retrieval is
surfacing the right memory.

### Step 5 — final numbers on the TEST split

Set `split: test` with an all-null `single_stage` (whole split; =
`forge.heldout` `progressive=false`) in your copy of the shared example and
run it per dataset (edit `dataset:` between runs, or keep one config per
dataset):

```bash
uv run --project baselines/harness/<name> python -m baselines.harness.eval_harness --config my_test.yaml
```

Outputs land in `baselines/harness/<name>/runs/<run_id>/` (and one line in
`runs/index.jsonl`):
`score.json` (mean reward = the number you report, same 0–1 scale as
forge's `accuracy_<dataset>`), `token_usage.json`, `traces/`. Because the
task list, workflow, judge, and scoring are the main method's own code,
these numbers are directly comparable to any forge-evolved harness's
held-out score — same code, not "comparable" code. Touch the test split
once per reported number; tuning happens in Step 4.

---

## Adding an evolve baseline

Evolve-framework baselines (a search loop that *produces* memory systems —
e.g. the EvolveMem / MemEvolve papers staged under `evolve/`) differ too
much internally for a step-by-step recipe: each has its own proposer,
feedback signal, and population management. What binds them is the
convention set above, concretely:

1. **Own directory, internal freedom.** Everything method-specific —
   search loop, prompts, checkpointing, its own base classes — lives in
   `baselines/evolve/<name>/`. Copy machinery from alma if useful; do not
   import it.
2. **The final artifact is a 3-hook `MemoClass`.** Whatever the search
   produces must be loadable as a `common.memo_class.MemoClass`
   subclass (directly, or via a thin adapter) so it can be scored through
   the shared workflow path. If the method's native artifact is not a
   Python class, the adapter is part of the baseline.
3. **Internal loops stay on the search split.** Every self-improvement
   iteration evaluates on `status/split = search`. The test split is used
   exactly once, for the final frozen artifact — mirroring how forge's
   search never touches test data (`forge.heldout` is the only test entry).
4. **Reference implementations**: [evolve/alma/](evolve/alma/) — its
   `run.py --status {search,test}` split handling, `eval_runner.py` →
   shared-workflow scoring, and `memo_archive/` artifact management are the
   patterns to mirror (by copying, not importing).
   [evolve/meta-harness/](evolve/meta-harness/) is the same skeleton with an
   AGENTIC proposer: `proposer.py` drives the `claude` or `codex` CLI as a
   stream-json subprocess, `prompts/proposer_system.md` is the agent-neutral
   prior both backends receive, and `state.py` shows a run whose test split is
   locked behind an explicit finalization step. Note what it does NOT have:
   forge's container. Its gold-data rule is a prompt instruction, not an
   enforced bind list.

## Shared foundation

All baselines (and forge) build on the same dataset adapters and judge:

- **[`baselines/registry.py`](registry.py)** — dataset name → (workflow,
  env module, recorder) resolution, shared by BOTH `evolve/` and `harness/`
  (the shared `benchmarks/registry.py`, re-exported as `baselines/registry.py`).
- **[`benchmarks/<bench>/env.py`](../benchmarks/)** — `load_user_data`,
  `get_task_list` (the single source of truth for the search/test split),
  per-benchmark Recorder.
- **[`common/metric.py`](../common/metric.py)** — LLM-as-judge with
  configurable prompt template and score range (DynamicMem uses the
  official TCE holistic judge in `benchmarks/dynamicmem/tce_prompts.py`).
- **[`common/llm.py`](../common/llm.py)** — `Agent` / `Embedding`
  wrappers with automatic token tracking; baselines use these so their
  cost numbers are comparable to forge's.

## License

MIT (same as the main project).
