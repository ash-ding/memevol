# memevolve — meta-evolution of memory architectures (MemEvolve)

Native implementation of **MemEvolve** (OPPO AI Agent Team & LV-NUS, 2025 —
`memevolve.pdf` in this directory). The upstream code (bingreeky/MemEvolve /
EvolveLab) is coupled to web-agent frameworks and GAIA-style benchmarks, so
per the baselines convention the method is re-implemented against this
repo's shared workflow/judge path.

The claim being reproduced: agent memory systems evolve their *content* but
their *architecture* is static; MemEvolve meta-evolves the architecture
itself over a modular design space, jointly with the experience it stores:

```
iteration k:
  INNER (experience evolution)   each candidate Ω_j = (E,U,R,G) runs the
      benchmark from an EMPTY memory; its memory grows along the data
      stream; feedback F_j = (perf, −cost, −delay)
  OUTER (architectural evolution) F: Pareto non-dominated sort over F_j,
      Perf tiebreak, top-K parents → per parent DIAGNOSE (trajectory
      evidence → defect profile over the four components) then DESIGN
      (S constrained redesigns of the operator bodies) → next candidates
      = parents (elitism) + surviving variants
```

## The design space (EvolveLab's four operators, repo-shaped)

A genotype is four async operator bodies assembled onto an immutable
skeleton (`design_space.py`):

| operator | paper role | signature |
|---|---|---|
| `encode` | E: raw experience → memory units | `async def encode(items, state)` |
| `store` | U: integrate units into memory | `async def store(units, state)` |
| `retrieve` | R: query → context for the QA agent | `async def retrieve(query, state)` (read-only) |
| `manage` | G: consolidation / forgetting / reorganization | `async def manage(state)` |

The skeleton owns benchmark-init normalization (operators are
dataset-agnostic), the 3-hook `MemoStructure` adapter, and a helper toolkit
(`_tokenize`, `_parse_ts`, `_BM25`). The design LLM may only rewrite the
four bodies — that constraint is what keeps every descendant loadable by
the shared eval path ("structurally constrained within the unified design
space", §4.2).

## Layout

| file | role (paper §) |
|---|---|
| `design_space.py` | operator contract, skeleton, assembly + static validation |
| `seed_genotypes.py` | iteration-0 candidates (lexical-cheap vs dense-quality anchors) |
| `genotype_manager.py` | genotype archive + resumable population checkpoint |
| `meta_prompts.py` | DIAGNOSE / DESIGN / repair prompt builders (§4.2) |
| `meta_evolver.py` | Pareto selection, diagnose-and-design, sanity gate with repair |
| `eval_runner.py` / `launch.py` | subprocess eval through the shared registry/workflow/judge (copied from alma per convention) + F = (perf, cost, delay) assembly |
| `run_main.py` | dual-loop CLI: `--status search` / `--status test` |

Artifacts: `memo_archive/<dataset>/<sha>/` (four operator files +
`assembled.py` + meta.json with parent/defect-profile/rationale),
`memo_archive/<dataset>/population_<tag>.json` (resumable dual-loop record),
`results/<dataset>/<sha>_<status>_<mode>/` (score.json, traces/,
token_usage.json, timing.json), `logs/`.

## Usage

```bash
# Meta-evolution on the search split (paper default K_max=3; resumable)
baselines/venv/bin/python baselines/evolve/memevolve/run_main.py \
    --status search --dataset dynamicmem --iterations 3

# Cheap smoke: tiny population + tiny eval passes
baselines/venv/bin/python baselines/evolve/memevolve/run_main.py \
    --status search --dataset locomo --iterations 2 \
    --eval_n_samples 1 --eval_n_qa 5 --variants_per_parent 1

# Held-out test of the frozen best genotype (touch once)
baselines/venv/bin/python baselines/evolve/memevolve/run_main.py \
    --status test --dataset dynamicmem [--memo_SHA <sha>]
```

## Faithfulness notes (deviations from the paper, and why)

1. **The "agent framework" is this repo's QA workflow**, not SmolAgent /
   Flash-Searcher: the inner loop's task stream is the benchmark's
   ingestion+QA protocol, so perf is the shared judge's 0–1 reward —
   directly comparable with forge/alma, which is the point of the baseline.
2. **Feedback vector** F = (task score, total tokens, wall-clock seconds of
   the eval subprocess). The paper's per-trajectory latency becomes per-run
   wall-clock — same units across candidates, coarser granularity.
3. **Elitism**: parents are carried into the next candidate set alongside
   their descendants (the paper's F "retains the top-K" as parents; keeping
   them as candidates too makes best-so-far monotone and their feedback is
   reused, not re-evaluated).
4. **Replay interface** = the saved per-user trace files (worst-first
   sampled QA steps with gold evidence vs retrieved memory), not an
   interactive trajectory replayer.
5. **Seeds are two hand-written minimal genotypes** (EvolveLab's twelve
   re-implemented systems are out of scope here); the design space contract
   is what carries over, not the library.
