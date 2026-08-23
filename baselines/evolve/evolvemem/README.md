# evolvemem — standalone reproduction of EvolveMem (arXiv:2605.13941)

[EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents](evolvemem.pdf).

> **Read this before quoting a number from here.** This module is **deliberately
> independent of the rest of the repo** — by owner decision, 2026-08-15. It runs
> upstream's code on upstream's benchmark under upstream's scoring, so its numbers
> reproduce the paper. That is exactly why they are **NOT comparable with forge's,
> alma's, or any harness baseline's**: those all go through
> `common.evaluate.evaluate_memo` and the shared judge, and nothing here does.
> Never put a number from this module in the same table as a forge number.
>
> It also **evolves over the full LoCoMo-10, which includes this repo's held-out
> test conversations** (`conv-47/48/49/50`). Treat every artifact under
> `upstream/evolution_results/` as test-touching: it must not be used to tune
> anything, and it is not a held-out number for anybody.

## What this is

`upstream/` is a **byte-identical copy of the whole `EvolveMem/` subtree** from
<https://github.com/aiming-lab/SimpleMem> @
`db80b6a7c591e0ea730a058e9f5fc4eb06572299` — the package *and* its entry points
(`run_benchmark.py`, `run_evolution.py`, `requirements.txt`, its own README).
That is the same repo and commit `baselines/harness/simplemem` is pinned to.
Nothing in it is edited, and the layout mirrors upstream exactly so one command
verifies the lot:

    git -C <simplemem-clone> archive db80b6a EvolveMem | tar -x -C /tmp/em
    diff -r -x __pycache__ /tmp/em/EvolveMem upstream        # 0 diffs

Everything outside `upstream/` is ours and touches nothing upstream: this README,
`pyproject.toml` / `.python-version` / `uv.lock`, and the test.

## The method in one paragraph

EvolveMem evolves a **configuration, not code**. The retrieval configuration θ
(`RetrievalConfig`, 42 fields — fusion mode, per-view top-k, context budget,
entity-swap, query decomposition, intent planning, answer style, answer
verification, per-category overrides) is the action space. A closed loop
optimises it: **EVALUATE** answers every question and writes per-question failure
logs → **DIAGNOSE** has an LLM read those logs and propose at most two field
edits → **GUARD** accepts only a strict improvement, reverts on regression,
perturbs on plateau, stops on convergence (paper Eq. 4 / Algorithm 1). The
objective is the benchmark's primary metric — token-F1 on LoCoMo.

## Setup

```bash
cd baselines/evolve/evolvemem && uv sync
```

~4.8G, essentially all torch via sentence-transformers. Dependencies are
upstream's `requirements.txt` plus `rank_bm25` and `rouge-score`, which its lazy
import paths need but it does not declare. No shared-core block — this module
imports nothing from `common/`, `benchmarks/` or `baselines/`, and a test
enforces that.

## Run it

The benchmark data is the repo's already-fetched copy (`tools/fetch_data.py`);
point `--data` at it rather than keeping a second 2.7MB copy:

```bash
cd baselines/evolve/evolvemem/upstream
export OPENAI_API_KEY=...            # or an openai_key.yaml, as upstream documents
export LLM_MODEL=gpt-4o              # the paper's backbone

# the paper's configuration, full LoCoMo-10
python run_benchmark.py locomo \
    --data ../../../../benchmarks/locomo/locomo10.json \
    --initial weak --max-rounds 7 \
    --embed-model BAAI/bge-base-en-v1.5
```

`--initial weak` is the paper's θ₀ (BM25-only, k_kw=5, B_ctx=8);
`--embed-model BAAI/bge-base-en-v1.5` is its embedder (§4.1). Output lands in
`upstream/evolution_results/locomo/<run_id>/` (gitignored): per-round
`round_N.json`, per-question `raw_results.jsonl`, and `evolution_summary.json`.

A cheap probe first — one conversation, two rounds, ~$1:

```bash
python run_benchmark.py locomo --data ../../../../benchmarks/locomo/locomo10.json \
    --samples 0 --max-rounds 2 --initial weak
```

## Reproduction result (2026-08-15)

Command above, unmodified code, full LoCoMo-10 (10 conversations, 272 sessions,
1,986 QA, all five categories), gpt-4o, bge-base-en-v1.5:

| round | this run | paper (Table 4) | guard |
|---|---|---|---|
| R0 | **29.3** | 30.5 | start (BM25-only) |
| R1 | 39.2 | 35.8 | accept +9.9 — `locomo_cat5_mcq` on |
| R2 | 53.0 | 34.8 | accept +13.8 — `enable_intent_planning` on |
| R3 | 50.3 | 37.2 | REJECT |
| R4 | 50.4 | 38.5 | REJECT |
| R5 | **59.1** | 38.1 | accept +6.1 — `enable_answer_verification` on, **best** |
| R6 | 58.5 | 45.4 | REJECT |

Per-category F1 at R5 (by raw LoCoMo category id): cat 1 39.8, cat 2 59.9,
cat 3 42.2, cat 4 53.0, **cat 5 (adversarial) 85.9**. Zero-F1 questions
396/1,986. ~7.5h, an estimated $45–50 (upstream's runner does not count tokens).

**Verdict: the paper's headline reproduces and is exceeded — 59.1 vs 54.3.**

Two caveats a reader should carry away:

- **The per-round table does not reproduce.** This run is at 53.0 by R2 where
  Table 4 reports 34.8, and `--max-rounds 7` yields R0–R6 (7 evaluations) where
  Table 4 lists R0–R7 (8 rows). Endpoint higher, path different, round count off
  by one. Treat Table 4 as illustrative, not as a trajectory to match.
- **The adversarial category carries the result, and it is worth knowing what it
  scores.** LoCoMo's cat-5 questions are name-swapped duplicates of real ones —
  "What does *Melanie*'s necklace symbolize?" against a cat-4 "What does
  *Caroline*'s necklace symbolize?", both with gold `love, faith, and strength`.
  Upstream takes gold as `qa.get("answer") or qa.get("adversarial_answer")`, so
  for cat-5 the gold IS the fact about the other person, and its
  `enable_entity_swap` "strips detected person names from the query and
  re-searches by topic". Scoring 85.9 there means answering questions about the
  wrong person, on purpose. Whether that is robustness or hallucination is a
  judgement call — this repo's own LoCoMo takes the other side and drops the
  category (`benchmarks/locomo/env.py`, commit `7235255`: 444 of 446 cat-5 items
  have no `answer` field at all).

## Why this module is scoped this way, and what it cost

`baselines/README.md` § "Adding an evolve baseline" and issue #21 require an
evolve baseline to produce a `MemoClass` artifact scored through
`evaluate_memo`, so its result sits on the same axis as forge's. An earlier
version of this module did exactly that — a θ-loading `MemoClass` adapter plus a
search loop reading the shared registry — and reached **41.5** under this repo's
contract (search split only, cat-5 excluded, gold from `answer` only).

That path was removed by owner decision in favour of reproducing the paper's
number. The trade is explicit: **41.5 was comparable with forge and not with the
paper; 59.1 is comparable with the paper and not with forge.** Nothing in this
repo can currently do both, because the two contracts disagree about which
questions exist and what counts as their gold.

If comparability with forge is wanted again, the removed adapter is in git
history (PR #33, commits before 2026-08-15) and issue #21's checklist describes
what it has to satisfy.

## Tests

```bash
uv run --project baselines/evolve/evolvemem python tests/test_evolvemem_baseline.py
```

Six checks, no LLM and no network: the vendored tree is whole (32 modules +
both runners), nothing under `upstream/` imports this repo, the entry points
parse and expose a CLI, the package imports and its θ₀ is the paper's, the README
pins the commit, and — when `EVOLVEMEM_UPSTREAM_CLONE` points at a clone — a full
`diff -r` against `db80b6a`.
