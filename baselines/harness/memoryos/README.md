# memoryos — three-tier heat-based memory OS

[MemoryOS](https://github.com/BAI-LAB/MemoryOS) (arXiv
[2506.06326](https://arxiv.org/abs/2506.06326), EMNLP 2025 Oral) as a
`MemoStructure`, scored through the shared registry/workflow/judge path like
every other baseline.

## What the method is

Memory is organised the way an OS organises pages, in three tiers:

- **STM** — a fixed-length queue of *dialogue pages*, a page being one
  `(user_input, agent_response)` pair. Full queue → FIFO-evict into MTM.
- **MTM** — segmented paging. A page joins a segment when
  `F_score = cos(e_s, e_p) + Jaccard(K_s, K_p) > θ`, keeping a segment
  topically coherent. Each segment carries a heat score
  `Heat = α·N_visit + β·L_interaction + γ·R_recency` (α=β=γ=1); crossing the
  threshold τ distils the segment into LPM, after which `L_interaction` resets.
- **LPM** — user profile, user knowledge base and assistant traits, the last two
  as fixed-capacity FIFO queues.

Retrieval reads all three: all of STM, a two-stage MTM search (top-m segments by
the same `F_score`, then top-k pages by semantic similarity, updating `N_visit`
and `R_recency` as a side effect), plus the LPM's top knowledge entries.

## Integration

| hook | what runs |
|---|---|
| `build_memory_from_data` | ingestion units → dialogue pages → `Memoryos.add_memory(user_input, agent_response, timestamp)`, which drives the whole STM→MTM→LPM update chain |
| `retrieve_memory_for_query` | `Retriever.retrieve_context(query)` + `short_term_memory.get_all()` → `{"passages": [...]}` |

MemoryOS ships its own answerer (`get_response`), which builds a prompt on top of
`retrieve_context`. It is deliberately UNUSED: `use_memory_to_answer` is not
overridden, so the SHARED QA agent answers and the comparison stays about memory.

**Page model.** A page is a PAIR, not a single utterance — the updater, the heat
accounting and the prompts all assume that shape. So ingestion pairs consecutive
turns (locomo), pairs user→assistant within a session (longmemeval), or emits a
single-sided page (dynamicmem, using hipporag2's `app_log_to_passage` text so
passage content matches the other baselines).

## Faithfulness notes

The implementation is VENDORED under `vendor/memoryos/` from the authors'
`memoryos-pro` wheel (0.1.0). Two reasons: its declared pins do not resolve on
this repo's Python (`numpy==1.24.*` has no 3.12 wheel and builds against the
removed `pkgutil.ImpImporter`; `faiss-gpu` needs CUDA), and vendoring keeps the
method's code byte-identical while we install only what its exercised path
imports.

**Name trap.** On PyPI, `memoryos` is a DIFFERENT system — MemTensor's MemOS
(memos.openmem.net, module `memos`). The paper's code is `memoryos-pro`.
`tests/test_memoryos_baseline.py` asserts we are running the former, not the latter.

Where the shipped code and the paper disagree, the code is the reference and the
config comment names the paper's value:

| knob | vendored default | paper §4.1 |
|---|---|---|
| `short_term_capacity` | 10 | **7** (used here) |
| `mid_term_capacity` | 2000 | 200 |
| `mid_term_heat_threshold` (τ) | 5.0 | 5 |
| `mid_term_similarity_threshold` (θ) | 0.6 | 0.6 |
| `long_term_knowledge_capacity` | 100 | 100 |
| recency time constant | `RECENCY_TAU_HOURS = 24` | µ = 1e7 s ≈ 2778 h |

The embedder is not a constructor argument in this build: `utils.get_embedding`
hardcodes `all-MiniLM-L6-v2` behind a process-global cache.

## Run

```bash
baselines/setup_venv.sh memoryos
baselines/harness/memoryos/venv/bin/python baselines/harness/memoryos/run.py \
    --config baselines/harness/memoryos/config.example.yaml
baselines/harness/memoryos/venv/bin/python tests/test_memoryos_baseline.py
```
