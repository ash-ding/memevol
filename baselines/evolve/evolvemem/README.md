# evolvemem — self-evolving retrieval configuration (EvolveMem)

Native implementation of **EvolveMem** (Liu et al., 2026 — `evolvemem.pdf`
in this directory; upstream code at aiming-lab/SimpleMem targets its own
eval stack, so per the baselines convention the method is re-implemented
against this repo's shared workflow/judge path — same discipline as alma).

The claim being reproduced: memory *content* evolution is not enough — the
**retrieval configuration itself** (view top-ks, fusion mode/weights,
context budget, augmentation toggles, answer style, per-category overrides)
should be a structured action space optimized by an LLM-driven closed loop:

```
round r:  EVALUATE (search split, full pass, per-question raw log)
       →  DIAGNOSE (LLM reads failure log + rubric → root causes + Δθ)
       →  PROPOSE  (structured adjustments incl. new `extras.*` dimensions)
       →  GUARD    (revert-on-regression τ_rev / explore-on-stagnation ε /
                    apply clamped Δθ) → θ_{r+1}
```

## Layout

| file | role (paper §) |
|---|---|
| `memo_evolvemem.py` | fixed 3-hook `MemoStructure`: typed store + consolidation (§3.1), multi-view retrieval + fusion + augmentation (§3.2). Reads θ from `$EVOLVEMEM_CONFIG`. |
| `action_space.py` | θ schema: defaults, bounds, clamping, `per_category` (θ_c), `extras` self-expansion surface, η_exp perturbation |
| `diagnosis.py` | failure-log assembly + rubric-driven diagnosis LLM call (§3.3) |
| `evolution.py` | guarded meta-analyzer (Eq. 4) + resumable `EvolutionState` |
| `eval_runner.py` / `launch.py` | subprocess eval through the shared registry/workflow/judge (copied from alma per convention) |
| `run_main.py` | CLI: `--status search` evolution loop / `--status test` frozen θ* |

Artifacts: `config_archive/<dataset>/evolution_log_<tag>.json` (every round's
θ, score, diagnosis, guard action — resumable), `results/<dataset>/<run>/`
(score.json, traces/, token_usage.json, the staged config.json), `logs/`.

## Usage

```bash
# Evolution on the search split (resumable; --rounds is a TOTAL target)
baselines/venv/bin/python baselines/evolve/evolvemem/run_main.py \
    --status search --dataset dynamicmem --rounds 8

# Cheap smoke: tiny eval passes
baselines/venv/bin/python baselines/evolve/evolvemem/run_main.py \
    --status search --dataset locomo --rounds 2 --eval_n_samples 1 --eval_n_qa 5

# Held-out test with the best evolved θ* (touch once)
baselines/venv/bin/python baselines/evolve/evolvemem/run_main.py \
    --status test --dataset dynamicmem
```

`memo_evolvemem.py` also runs standalone as a plain multi-view retrieval
memo (no `$EVOLVEMEM_CONFIG` → clamped defaults) — useful as the r=0
"minimal baseline" reference the paper measures relative gains against.

## Faithfulness notes (deviations from the paper, and why)

1. **Entity reinforcement ρ accumulates at BUILD time** (co-occurrence
   across ingested items), not from query hits — this repo's contract
   requires `retrieve_memory_for_query` to be read-only (DynamicMem
   checkpoint isolation).
2. **Answer generation stays with the benchmark's QA agent** so scores
   remain directly comparable with forge/alma. The answer-style dimension α
   is realized as an `answer_guidance` sentence inside the retrieved dict
   (the QA agent sees the dict verbatim).
3. **Question categories are regex-defined** (`per_category[].pattern` on
   the query text) because the benchmarks don't expose a category oracle at
   retrieve time; the diagnosis LLM invents and refines the patterns.
4. **Self-expanding dimensions are bounded by the memo's capability
   surface**: proposals for new `extras.*` keys are stored and logged, but
   only the hooks listed in `action_space.EXTRA_HOOKS` change behavior.
   (The paper's "entirely new configuration dimensions" likewise required
   the framework to implement the dimension before it took effect.)
5. **Store knobs are inside the action space** (`extraction_mode`
   raw|llm, window, granularity, dedup τ_J, decay α_d/ι_min) — default
   `raw` keeps round-0 cheap; the diagnosis loop can escalate to LLM
   extraction when the failure log shows store-quality gaps. The paper's
   coverage-verifier re-extraction pass is not implemented.
6. **Evolution scores are the benchmark judges' 0–1 rewards** (same as
   every other method here), not the paper's F1.

## simplemem/ — vendored upstream substrate

`simplemem/` vendors the real SimpleMem text core (the paper's actual base
system) as an evolvemem-internal substrate — fidelity reference for the
native `memo_evolvemem.py` approximation, and the future target for wiring
θ onto the genuine architecture. Evaluate it standalone via
`launch.py --substrate simplemem`. See [simplemem/README.md](simplemem/README.md).
