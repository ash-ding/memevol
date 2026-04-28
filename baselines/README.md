# Baselines

Comparison methods for the memory-architecture search. Each baseline runs
against the same benchmark set as the main method ([forge/](../forge/)),
producing comparable metrics: per-user reward, judge-scored accuracy, and
(for some) token / latency telemetry.

All baselines share **`baselines/venv/`** (full ML install:
`pip install -r baselines/requirements.txt`) and write artifacts under
each baseline's own `logs/` and `results/` directories (gitignored).

| Baseline | Approach | Optimization | Best for |
|---|---|---|---|
| **[alma](alma/)** | LLM-meta-agent search loop | Yes — propose / select / evolve over harness code | Established baseline; the framework's "v1" memory-architecture search |
| **[cc](cc/)** | Claude Code as direct QA agent | None (zero-shot) | What-if: just give CC the raw user data + tools and let it answer |
| **[hipporag2](hipporag2/)** | Graph-based RAG pipeline (OpenIE → KG → PPR retrieval) | None (fixed pipeline) | Hand-designed memory architecture comparison point |

---

## alma — meta-learning loop

The original method memevol shipped with: an LLM meta-agent reads sampled
QA trajectories, identifies failure patterns, and proposes new memory-
structure code. Sanity-checks each candidate, then evaluates on the search
split. Softmax-weighted parent selection over reward.

```bash
# Smoke — 2 users, 2 steps
baselines/venv/bin/python baselines/alma/run_main.py \
    --status search --eval_n_samples 2 --eval_n_qa 10 --steps 2

# Full training
baselines/venv/bin/python baselines/alma/run_main.py \
    --status search --eval_n_samples 6 --eval_n_qa 20 --steps 10

# Held-out evaluation of a saved memo
baselines/venv/bin/python baselines/alma/run_main.py \
    --status test --memo_SHA <SHA>
```

Artifacts: `baselines/alma/{logs/, memo_archive/, results/}`. See
[alma/README.md](alma/README.md) for layout details.

**Key difference from forge**: alma's proposer is a single LLM call with
compressed feedback (sampled trajectories + meta-prompt), whereas forge's
proposer is an agentic CC SDK call with full filesystem access to all
prior code, traces, and scores.

---

## cc — Claude Code as direct QA agent

Skips memory-architecture design entirely. Per question, Claude Code is
given the user's data files in a working directory and access to standard
SDK tools (Read, Grep, Glob), then asked to answer. Same judge as the rest
of the framework, so scores are directly comparable.

```bash
# Single QA dry run
baselines/venv/bin/python baselines/cc/eval_cc.py \
    --model claude-sonnet-4-20250514 --dry_run

# Full eval
baselines/venv/bin/python baselines/cc/eval_cc.py \
    --model claude-sonnet-4-20250514

# Both Sonnet and Opus side-by-side
baselines/venv/bin/python baselines/cc/eval_cc.py --model all
```

Useful as an upper-bound reference: how well does a strong agent do **with
no learned memory structure at all**, just raw access + tools?

Artifacts: `baselines/cc/{logs/, results/}`.

---

## hipporag2 — graph-based RAG pipeline

Runs [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s pipeline
on each user as a fixed memory architecture (no search):

- **Phase 1 (Index)**: app_logs → OpenIE (NER + triples) → knowledge graph
  + entity embeddings.
- **Phase 2 (QA)**: query → fact retrieval → reranking → personalized
  PageRank → top-k passages → LLM answer.

```bash
# OpenAI API embedding (no GPU needed)
baselines/venv/bin/python baselines/hipporag2/eval_hipporag2.py \
    --embedding text-embedding-3-small --dry_run

baselines/venv/bin/python baselines/hipporag2/eval_hipporag2.py \
    --embedding text-embedding-3-small

# Local GPU embedding (NVIDIA)
baselines/venv/bin/python baselines/hipporag2/eval_hipporag2.py \
    --embedding nvidia/NV-Embed-v2 \
    --embedding_batch_size 2 --embedding_dtype float16
```

Useful as a hand-designed comparison: how does a published RAG pipeline
perform on these benchmarks vs forge-evolved harnesses?

Artifacts: `baselines/hipporag2/{logs/, outputs/, results/}`.

---

## Shared foundation

All baselines (and forge) build on the same dataset adapters and judge:

- **[`datasets/<bench>/env.py`](../datasets/)** — `load_user_data`,
  `get_task_list`, per-benchmark Recorder.
- **[`common/judge.py`](../common/judge.py)** — LLM-as-judge with
  configurable prompt template and score range. Scores from baselines
  and forge are directly comparable when they use the same judge config.
- **[`common/llm.py`](../common/llm.py)** — `Agent` / `Embedding`
  wrappers with automatic token tracking; baselines use these so their
  cost numbers are comparable to forge's.

This separation means **adding a new baseline** only requires writing a
script that calls `load_user_data` + `Judge` — no framework changes.

## License

MIT (same as the main project).
