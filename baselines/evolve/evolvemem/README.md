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
uv run python evolve.py --config config.example.yaml            # paper setting by default (~$22, ~2h)
uv run python evolve.py --config config.example.yaml --max-rounds 2   # cheap probe

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

Artifacts are scoped **per run**, not per dataset —
`memo_archive/<dataset>/<initial>_r<rounds>[_resumed]_<model>/` holding `theta.json`,
`evolution_summary.json` (per-round F1, zero-F1 counts, per-category F1, applied
changes, token usage) and `rounds/round_N.json`. The engine renumbers its rounds from
zero on every run, so a shared directory would let a second run silently overwrite the
first one's trajectory. `cache/` is deliberately shared across runs at
`memo_archive/<dataset>/cache/`: extraction is keyed by session id and is the
expensive part to redo. Scored runs write `results/<dataset>/<split>/`.

## Reproduction check (2026-08-15)

**`config.example.yaml` ships the paper's configuration**, so an unedited
`evolve.py` run IS this reproduction: `weak` (BM25-only) start, gpt-4o backbone,
`BAAI/bge-base-en-v1.5` embedder, `max_rounds: 7`, whole search split, no sizing
caps. That costs ~$22 and ~2h — lower `max_rounds` or `single_stage` for a cheap
probe, and see the cost profile below before scaling up.
`tests/test_evolvemem_baseline.py` pins those four values so a well-meaning
"make the default cheaper" edit cannot silently turn the documented
reproduction into something else.

### First: does upstream's code reproduce the paper at all?

Before judging our own numbers, we ran **upstream's own entry point, unmodified,
on its own terms** — `run_benchmark.py locomo --initial weak --max-rounds 7
--embed-model BAAI/bge-base-en-v1.5`, full LoCoMo-10 (10 conversations, 272
sessions, 1,986 QA, cat-5 included), in a throwaway sandbox that touches nothing
in this repo. It reaches **F1 59.1** (R5), above the paper's reported 54.3:

| round | upstream, its own contract | paper (Table 4) | guard |
|---|---|---|---|
| R0 | **29.3** | 30.5 | start |
| R1 | 39.2 | 35.8 | accept +9.9 — `locomo_cat5_mcq` on |
| R2 | 53.0 | 34.8 | accept +13.8 — `enable_intent_planning` on |
| R3 | 50.3 | 37.2 | REJECT |
| R4 | 50.4 | 38.5 | REJECT |
| R5 | **59.1** | 38.1 | accept +6.1 — `enable_answer_verification` on, **best** |
| R6 | 58.5 | 45.4 | REJECT |

Three things follow, and they are the reason this section exists:

1. **The paper's headline is reproducible** — its own code exceeds it. So any
   shortfall on our side is ours to explain, not the paper's.
2. **The paper's per-round table does not reproduce.** Upstream's code is at 53.0
   by R2 where the paper reports 34.8, and its `--max-rounds 7` yields R0–R6
   (7 evaluations) while Table 4 lists R0–R7 (8 rows). Endpoint higher, path
   different, round count off by one. Treat Table 4 as illustrative.
3. **cat-5 is the single biggest lever, and the loop finds it first.** Upstream's
   largest early jump (+9.9 at R1) is `locomo_cat5_mcq`, and adversarial ends at
   85.9 — its top category. That category does not exist in our data at all, so
   our diagnosis module can never even propose it.

### Ours, under this repo's contract

Same loop, same code, our data: 6 conversations, 885 QA, 156 sessions, cat-5
excluded. Internal token-F1, i.e. the metric the loop optimises (Eq. 3), not our
judge.

| round | ours | paper | our guard |
|---|---|---|---|
| R0 | **33.5** | 30.5 | start (BM25-only, k=5, B_ctx=8 — same θ₀) |
| R1 | 32.9 | 35.8 | REJECT −0.006 → revert |
| R2 | 37.2 | 34.8 | accept +0.037 (`fusion_mode → rrf`, semantic view on) |
| R3 | 40.2 | 37.2 | accept +0.030 |
| R4 | **40.6** | 38.5 | accept +0.004 — **best**, returned |
| R5 | 40.3 | 38.1 | REJECT −0.003 |
| R6 | 39.9 | 45.4 | REJECT −0.007 |
| R7 | not run continuously — see below | **54.3** | — |

**The extra round, run afterwards by resuming from R4's θ** (`theta_path`, 2 further
evaluations): diagnosis proposed exactly the paper's R7 lever,
`enable_answer_verification`, and the guard accepted it — **+2.05 → F1 41.5**, our
best artifact. The paper's R7 gains +8.9 from the same lever. The gain concentrates
in raw cat 4 (0.441 → 0.479); cat 3 regresses (0.266 → 0.218) and the zero-F1 count
rises (244 → 263), which is what a second-pass rewriter does — it rescues some
answers and flattens others.

Resuming is measurably not free, so read that number with a handicap: re-evaluating
R4's own θ scored **39.4**, not the 40.6 it scored in place. The round counter and
`attempt_history` restart, and the ~350 memories that targeted re-extraction added
during the first run are not in the extraction cache, so the store restarts from the
base 2,580. A continuous 8th round would start 1.2 points higher.

**Verdict: the mechanism reproduces; the headline number does not, and most of the
difference is contract, not method.**

What reproduced: the starting point (33.5 vs 30.5 from an identical θ₀ — the
cleanest anchor available, and it says extraction → retrieval → answer → token-F1
line up end to end); the trajectory shape (a rejected round, then a climb); all
three guard branches firing as Algorithm 1 specifies; and targeted re-extraction
(Algorithm 1 lines 13–14) adding ~350 memories across rounds. R2–R5 ran above the
paper's curve at the same round index.

What did not: the paper's late jump (R6 45.4 → R7 54.3). Ours peaked at R4 and was
rejected twice after. Diagnosis proposed the right *kind* of change — per-category
overrides at R4/R5, a temporal-format flag at R6 — but they did not pay off here.

### Read this before comparing per-category numbers: the taxonomies disagree

**The paper's Table 2 column names do not follow LoCoMo's official category
numbering.** Its appendix calls raw category 4 "open-domain aggregation" (its
case-study probe `conv-26-95`, which is raw cat 4 in `locomo10.json`), while the
official taxonomy — and this repo, and the mem0 / MemoryOS reproduction tables —
call raw cat 4 **single-hop**. The data settles it: raw cat 4 is 54.6% of
non-adversarial questions, matching LoCoMo's documented ~55% single-hop share.

Weighting the paper's own columns by the true per-category counts confirms which
mapping its table uses: **its mapping reproduces its reported Overall (0.544 vs
0.543); the official mapping gives 0.480.** So pair by category id, never by name:

| raw cat | this repo's name | paper's column | ours (best) | paper | Δ | n (our split) |
|---|---|---|---|---|---|---|
| 1 | multi-hop | SingleHop | 33.1 | 32.9 | **+0.2** | 172 |
| 2 | temporal | Temporal | 38.1 | 38.4 | −0.3 | 180 |
| 3 | open-domain | MultiHop | 21.8 | 31.6 | −9.8 | 53 |
| 4 | single-hop | OpenDomain | 47.9 | 49.6 | −1.7 | 480 |
| 5 | adversarial | Adversarial | — excluded — | 93.6 | — | 0 |
| | | **weighted, cats 1–4** | **41.5** | **43.1** | **−1.6** | 885 |

"ours (best)" is the resumed answer-verification round. So on the four categories
both sides score, the gap is **1.6 points**, not 14 — and the one real deficit is
cat 3 (open-domain, 6% of questions). The
headline difference is dominated by **Adversarial at 93.6**, the paper's highest
cell, which this repo excludes entirely: 444 of 446 cat-5 items carry no `answer`
key — only `adversarial_answer`, the trap option — so before `7235255`
(2026-07-08) every one was judged against an empty gold, ~22% of LoCoMo scoring as
noise. Upstream instead reads `qa.get("answer") or qa.get("adversarial_answer")`
and evolves cat-5-specific machinery for it (the paper's R3 is "entity-swap for
Cat. 5"). **That cell is unreachable without changing the shared eval contract**,
which would invalidate every historical LoCoMo number in this repo.

The paper publishes exactly one per-category trajectory (Appendix C.1, raw cat 4):
41.0% at R0 → 49.6% at R7. Ours on the same category: 40.0% at R0 → 45.8% at R4.

### What we cannot match, and why

The knobs are now aligned — `config.example.yaml` ships the paper's backbone,
embedder, start config and round count, and the round count matches upstream's
code (its own `--max-rounds 7` yields R0–R6 too, so our earlier "off-by-one" was
faithful to the code, not to Table 4). What remains is **protocol**, and none of
it is ours to change unilaterally:

- **cat-5 (adversarial) does not exist in our LoCoMo.** 444 of 446 items carry no
  `answer` key, so `benchmarks/locomo/env.py` drops the category outright
  (`7235255`, 2026-07-08) — before that, ~22% of LoCoMo scored against an empty
  gold. Upstream instead reads `qa.get("answer") or qa.get("adversarial_answer")`
  and scores it, reaching 85.9 there. Its evolution loop finds this lever first
  (+9.9 at R1, `locomo_cat5_mcq`); ours cannot propose it, because no cat-5
  question is ever in the failure log it reads.
- **6 of 10 conversations.** We evolve on the search split; the other 4 are
  held out. Upstream evolves on all 10 — which for us would mean optimising on
  the test split, contaminating every held-out LoCoMo number in the repo. Not a
  cost decision, a protocol one.
- **Gold is `answer` only.** Even with cat-5 restored, our env would score it
  against an empty reference; matching upstream means changing the LoCoMo gold
  contract for every method in the repo.

All three live in `benchmarks/locomo/env.py` — shared surface, so a manager
decision, and one that invalidates every historical LoCoMo number. **The
honest reading is not "we scored lower", it is "we measure a different thing":
under our contract EvolveMem reaches 41.5, under its own it reaches 59.1, and
the delta is dominated by a question category we deliberately do not score.**

Charts and the full write-up: dashboard page **baselines #1**.

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

Read this before running the default config, which is no longer a toy: it is the
paper's setting (7 rounds over the whole search split), i.e. ~$22 and ~2h. Each
round re-answers **every** question in the split and then runs diagnosis +
meta-analysis, so cost scales with `rounds × split size`; extraction (once, up
front) scales with conversation length. `--max-rounds 2` with a
`single_stage: {n_conversations: 1, n_qa: 5}` is the ~$1 probe.

Measured, the paper-scale run (locomo/search, 6 conversations, 885 QA, 156
sessions, `weak` start, gpt-4o, 7 rounds — the Reproduction check above):
**7.14M tokens** (6.56M prompt / 585k completion) ≈ **$22** at gpt-4o list price,
**2h17m** wall-clock. The resumed answer-verification round added **3.48M tokens ≈
$10** for two evaluations — verification issues a second LLM pass per question, so a
round with it on costs roughly double. The engine answers questions **sequentially**,
so wall-clock scales with `rounds × split size` and cannot be parallelised away.

A cheap pilot for extrapolation (1 conversation, 152 QA, 3 rounds): 280k tokens,
$0.82, 5.6 min. Scaling that by answer-call count predicted the full run within a
factor of ~1.3 on cost — a pilot is the right way to size a run before paying for it.

For reference, the upstream sandbox run (full LoCoMo-10, 1,986 QA, 7 rounds) took
**~7.5h** and an estimated **$45–50**; upstream's own runner does not count tokens,
which is why that figure is an estimate and ours is not.

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
