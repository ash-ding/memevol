# evolvemem — self-evolving retrieval configuration (AutoResearch)

[EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents](evolvemem.pdf)
(arXiv:2605.13941) as an **evolve baseline** — a search loop that produces memory
systems, compared against forge itself rather than against forge's evolved harnesses.

**Provenance**: `src/evolvemem/` is vendored VERBATIM (byte-identical, all 32 files,
no exclusions inside the package) from <https://github.com/aiming-lab/SimpleMem> @
`db80b6a7c591e0ea730a058e9f5fc4eb06572299`, subdirectory `EvolveMem/evolvemem/`.
That is the same repo and the same commit `baselines/harness/simplemem` is pinned
to — simplemem excludes the `evolver/` subtree, which is a near-duplicate of this
one; the paper's README points at `EvolveMem/`, so this is the copy vendored.
Upstream's `run_evolution.py` / `run_benchmark.py` are deliberately NOT vendored:
they are replaced by `evolve.py` / `run.py`, because their scoring is exactly what
this repo's conventions forbid. No file under `src/` is edited — provenance lives
here, not in per-file headers, to preserve byte-identity:

    git -C <simplemem-clone> archive db80b6a EvolveMem/evolvemem | tar -x -C /tmp/em
    diff -r /tmp/em/EvolveMem/evolvemem src/evolvemem        # 0 diffs

Nothing inside the package is pruned: upstream's `__init__.py` eagerly imports 30
of its 32 modules, so pruning would mean editing a vendored file. The package is
740K and its dependency closure is small (numpy eager; rank-bm25 /
sentence-transformers / rouge-score lazy).

## What the method is

Every other memory system in this repo evolves what it *stores*. EvolveMem evolves
how it *retrieves* — and it evolves a **configuration, not code**, which is what
makes it an interesting comparison point against forge (which evolves harness
source). The retrieval configuration θ (`RetrievalConfig`, ~42 fields) is the
action space; a closed loop optimises it:

| Step | What happens |
|---|---|
| **Evaluate** | answer the split's questions under θ, write per-question failure logs |
| **Diagnose** | an LLM reads those logs and identifies root causes |
| **Propose** | targeted θ adjustments (max 2 fields per round) |
| **Guard** | elitist acceptance; revert on regression, explore on plateau, converge on ε |

The artifact is therefore a θ JSON, not a Python class — so `memo.py` **is** the
artifact loader that `baselines/README.md` § "Adding an evolve baseline" point 2
requires.

## Integration

| hook | what runs |
|---|---|
| `build_memory_from_data` | upstream `MemoryExtractor.extract_sessions` (sliding window + retry + coverage verify) over the ingested unit |
| `retrieve_memory_for_query` | upstream's own retrieve path under θ, via `EvolutionEngine._evaluate_qa` → `{"passages": [...]}` |
| `use_memory_to_answer` | **overridden** — answers under θ's evolved answer policy |

**This is the only baseline in the repo that overrides the third hook**, and the
choice is deliberate. θ's action space is not retrieval-only: it carries
`answer_style`, `enable_answer_verification` / `verification_style`, `answer_model`
/ `answer_model_ensemble` and per-category answer overrides — the paper's two
largest gains (R4 per-category styles, R7 answer verifier, +8.9pp) are exactly
those fields. The loop generates answers with them applied, so deferring to the
shared QA agent would mean **scoring an artifact different from the one the loop
optimised**. Set `honor_answer_policy: false` for the ablation; retrieval is
identical in that mode, so the delta isolates answer policy.

Consequence to keep in mind when reading numbers: this baseline's score is not on
the same "memory only, shared answerer" axis as the harness baselines. It is on the
evolve axis, where the unit of comparison is the search method's final artifact.

### Measured: honoring the answer policy HURTS on dynamicmem

Smoke-scale ablation (theta evolved on locomo/search for 2 rounds from `weak`;
1 unit per dataset, so these are sanity signals, NOT benchmark numbers):

| dataset | `honor_answer_policy: true` | `false` (shared QA agent) |
|---|---|---|
| locomo | 0.6 | 0.8 |
| dynamicmem | **0.0** | **0.167** |

The locomo gap is one question out of five — noise at this size, decide nothing
from it. **The dynamicmem gap is structural and reproducible**: with the answer
policy on, the prediction is the literal string `"null"` and the holistic judge
scores 0/3 fields; with it off, the shared agent emits a proper state template
(`{"schedule": {...}, "timing": {...}}`) and scores 1/3. The mechanism is not
subtle — upstream ships no dynamicmem answer adapter, so theta falls back to the
engine's generic prompt ("Professional Q&A assistant. Concise answers."), which
cannot satisfy DynamicMem's TCE structured-output contract. EvolveMem's answer
policy assumes short-form QA.

**So: use `honor_answer_policy: false` on dynamicmem** unless and until someone
writes a dynamicmem `BenchmarkAdapter` (which would be a faithful extension —
upstream's adapter interface is exactly the extension point for it). On locomo and
longmemeval_s, where upstream adapters exist, `true` is the faithful default.

`memo.py` drives the vendored engine rather than reimplementing its retrieval
ladder (intent planning > query decomposition > plain multiview, plus coverage
reflection / reflection rounds / verification, each gated by a θ field the loop can
flip). Re-implementing that here would let the scored artifact drift from the
optimised one. To see the FINAL retrieved list without retrieving twice, `memo.py`
temporarily wraps the engine's `_generate_answer` — integration glue in our file,
no vendored file touched.

**Gold never reaches inference.** Upstream's `_evaluate_qa` takes QA dicts carrying
the reference (it scores in the same pass) and forwards them to a benchmark adapter.
The dict `memo.py` builds carries only question / category / meta — never `answer`
or `adversarial_answer` — so leakage is impossible by construction. Locked in by
`tests/test_evolvemem_baseline.py::test_gold_answer_never_reaches_the_engine`.

## Setup

Its own uv project; `uv sync` alone is the whole setup:

```bash
cd baselines/evolve/evolvemem && uv sync
```

The env is ~4.9G, essentially all torch pulled in by sentence-transformers. A
`keyword_only` θ (including the default `weak` start) never loads an embedder, but
the dependency is declared because the search loop is free to evolve the semantic
view on.

## Usage

```bash
# SEARCH — evolve a theta on the search split
cd baselines/evolve/evolvemem
uv run python evolve.py --config config.example.yaml            # or --max-rounds 7

# SCORE — evaluate a theta through the shared eval path
uv run python run.py --config config.example.yaml               # or --theta <path>
```

Both entry points read the same config file and the same key set (`config_schema.py`),
so a θ cannot be evolved under one set of assumptions and scored under another.
`config.example.yaml` documents every key and passes strict validation unedited.
The CLI carries only the two genuine runtime knobs an evolve baseline is allowed:
`--max-rounds` and `--theta`.

**Split discipline is enforced in code**: `evolve.py` refuses any split but
`search`. Held-out numbers come from a frozen θ, and only with manager authorization.

Artifacts: `memo_archive/<dataset>/theta_<stamp>.json` +
`evolution_summary_<stamp>.json` (per-round scores, accept/revert decisions, applied
changes), and `results/<dataset>/<split>/` for scored runs.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | the whole `evolvemem` package (@ db80b6a, 32 files, 0 diffs): the EVALUATE→DIAGNOSE→PROPOSE→GUARD loop, elitist acceptance + revert + explore + convergence, the diagnosis and meta-analysis prompts, the multi-view retriever (BM25 / semantic / structured, fusion modes, entity-swap, intent planning, coverage reflection, query decomposition), the extractor's sliding window + retry + chunk-split + coverage verification, the consolidator, and the `weak` / `strong` initial configs |
| Integration adaptations (not algorithm) | data comes from this repo's registry + `env.load_user_data` instead of upstream's loaders; `evolve.py` / `run.py` replace upstream's `run_evolution.py` / `run_benchmark.py`; `llm_call` is injected via `llm_bridge.py` so internal calls go through `common.llm`; per-dataset ingestion mapping (`init_to_sessions`: locomo turns natively, longmemeval role/content → speaker/text, dynamicmem app-logs via hipporag2's shared `app_log_to_passage`); final scoring through `common.evaluate.evaluate_memo` (via `baselines.harness.eval_utility.run_baseline`) rather than upstream's own metric |
| Reduced by this repo's eval surface | **per-category θ fields are inert at scoring time.** A `MemoClass` sees only what `build_query_recorder_init` puts in `recorder.init` — locomo `{conversation, query}`, longmemeval `{sessions, query, question_date}`, dynamicmem `{app_logs, query}` — none carries the question category. So `per_category_overrides` and per-category answer styles fall back to the global config when scored, even though the search loop CAN evolve them (it reads the raw split, where categories exist). Widening `build_query_recorder_init` is a shared-eval-surface change and therefore a manager decision, not something to work around here. longmemeval's `question_date` IS available and feeds θ's time-decay anchor |
| Reduced, dynamicmem only | no upstream answer-prompt adapter exists for dynamicmem (upstream ships locomo / longmemeval / membench), so θ's `locomo_*`-style prompt flags have no counterpart and the engine's generic answer prompt is used. The evolution loop's internal signal also uses `env.load_user_data`, which for dynamicmem is the two-phase compat shim (last checkpoint's items); final scoring still uses the official TCE checkpoint protocol |
| Reduced, dynamicmem only (metric) | **the loop optimises a different metric than we score on.** The paper's objective (Eq. 3) is a task-specific metric — F1 in their experiments — and the code reads it from `adapter.primary_metric` (locomo/longmemeval `f1`, membench `accuracy`). Every guard hangs off that scalar: revert when `f_{r-1} − f_r > τ_rev`, explore when `\|f_r − f_{r-1}\| < ε` twice, converge when the round-over-round gain drops below ε. With no dynamicmem adapter the engine falls back to `_compute_f1`, i.e. plain token-F1 against a STRINGIFIED state template, while this repo scores dynamicmem with the TCE holistic judge. So a dynamicmem evolution run hill-climbs a weak proxy. Writing a dynamicmem `BenchmarkAdapter` (`score`/`primary_metric` + `build_answer_prompt`) fixes this and the answer-prompt gap above in one go — it is the single highest-value follow-up for this baseline |
| Upstream quirks preserved | `max_changes_per_round=2` and elitist acceptance are upstream defaults, kept; a failed LLM call returns `""`, which is what upstream's own retry / chunk-split / coverage fallbacks expect |

## Cost profile

Read this before raising `max_rounds`. Each round re-answers **every** question in
the sampled search split and then runs diagnosis + meta-analysis, so cost scales
with `rounds × split size`, and extraction (once, up front) scales with conversation
length.

Measured here (locomo, 1 conversation, 5 QA, `weak` start, gpt-4o-mini, 2 rounds):
19 sessions → 305 memories, **322s wall-clock** end to end. The paper's headline is
7 rounds over the full benchmark — budget accordingly, and get sign-off before a
full-scale run.

`max_rounds: 3` and a 1-conversation `single_stage` are the shipped defaults for
that reason; the paper's setting is documented, not defaulted.

**Token accounting actually works here**, unlike the harness baselines: EvolveMem
never imports an LLM SDK — every component takes an injected `llm_call` — so
`llm_bridge.py` routes internal extraction / diagnosis / answer calls through
`common.llm.Agent`, and they appear in `token_usage.json`. hipporag2 / mem0 / zep /
simplemem call the OpenAI SDK directly and their internal cost is invisible.

## Tests

```bash
uv run --project baselines/evolve/evolvemem python tests/test_evolvemem_baseline.py
```

11 tests, no LLM required (the engine is faked wherever a call would be made):
ingestion dispatch on all three datasets, θ resolution (weak / strong / file /
summary / unknown-key tolerance), the 3-hook contract, the ablation path, the
gold-leakage invariant, strict config validation, and the split-discipline guard.
