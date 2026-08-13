# memoryos — three-tier heat-based memory OS

[MemoryOS](https://github.com/BAI-LAB/MemoryOS) (arXiv
[2506.06326](https://arxiv.org/abs/2506.06326), EMNLP 2025 Oral) as a
`MemoClass`, scored through the shared registry/workflow/judge path like
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

The implementation is VENDORED under `src/memoryos/` from the authors'
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

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`config.example.yaml`) | unified arm (`config.unified.yaml`) |
|---|---|---|
| internal LLM (`memoryos_llm_model`) | `gpt-4o-mini` — the paper's headline LoCoMo setting | `gpt-5-mini` |
| embedder (`memoryos_embedding_model`) | `all-MiniLM-L6-v2`, local, 384-dim — the paper's | `text-embedding-3-small`, API, 1536-dim |

**The faithful arm is the default** (and `config.paper.yaml` additionally pins
the paper's gpt-4o-mini answerer). The unified arm puts all seven baselines on
one LLM and one embedder so the comparison against the main method is
like-for-like — it is a deliberate deviation from the paper, and its numbers
must not be quoted as MemoryOS's published result.

Both arms leave `src/` **byte-identical** — no vendored file is edited. Two
boundary levers in [`../model_config.py`](../model_config.py) make that
possible:

- **the embedder.** MemoryOS has no embedder constructor argument in this
  build: `utils.get_embedding` carries `all-MiniLM-L6-v2` as a DEFAULT
  ARGUMENT, behind a process-global cache. So the name the vendored code
  requests is never the configured one, and the key is applied by **overriding**
  the shared SentenceTransformer factory rather than dispatching on the
  requested name (the way amem and simplemem work). No dimension knob is needed
  — MemoryOS sizes its FAISS indexes from the embedding array itself
  (`dim = embeddings_np.shape[1]`), so a 1536-dim API embedder drops straight
  in. A `memory_cache: true` snapshot built at 384-dim is still invalid.
- **the LLM.** MemoryOS's vendored `chat_completion` hardcodes `temperature`
  and `max_tokens`, which the gpt-5 family rejects — so before this, MemoryOS
  could not run a gpt-5 model at all. The shim drops the rejected params and
  renames `max_tokens` → `max_completion_tokens` at the OpenAI-SDK boundary.

## Run

```bash
# build memoryos's isolated env
cd baselines/harness/memoryos && uv sync

# run it (from the baseline dir)
cd baselines/harness/memoryos && uv run python run.py --config config.example.yaml

# its unit test (from the repo root)
uv run --project baselines/harness/memoryos python tests/test_memoryos_baseline.py
```

## Reproduction check (2026-08-05)

`config.paper.yaml` — LoCoMo **search** split, all 6 conversations × 60
randomly-sampled QA = **360 questions**, answerer **gpt-4o-mini** (what the paper
reports on; `config.example.yaml` keeps the repo's shared gpt-5-mini agent), with
the paper's `retrieval_queue_capacity: 10` and `mid_term_capacity: 200`.
6/6 conversations, **3.5 hours**, **2,778 tokens/question**.

Scored with `baselines/harness/score_paper_metrics.py`, which recomputes the
papers' metrics — the shared judge is binary and is not what they report.

| category | our F1 | paper F1 | n |
|---|---|---|---|
| single-hop | **33.7** | **35.3** | 202 |
| multi-hop | 22.2 | 41.1 | 65 |
| temporal | 39.5 | 20.0 | 73 |
| open-domain | 12.1 | 48.6 | 20 |
| **unweighted mean** | **26.9** | **36.2** | |

**Verdict: the integration is correct, but two categories are not comparable.**

What reproduces: single-hop — the cleanest, largest cell (n=202) — lands at
**33.7 vs the paper's 35.3**, a precision no mis-wired integration produces. The
cost profile also reproduces: 2,778 tokens/question against the paper's reported
3,874 (Table 3), confirming the expense is intrinsic to the method rather than to
this integration. Head-to-head on identical questions, Mem0 (30.2) > MemoryOS
(26.9), matching the papers' own ordering (41.0 > 36.2).

What does not: **temporal is inverted.** The paper's signature is temporal being
MemoryOS's weakest category (20.02, its only cell below 25); here it is the
strongest (39.5). The cause is shared, not method-specific — `_LOCOMO_USER_CAT2`
supplies explicit date guidance, and under it Mem0 and MemoryOS converge to 35.9
and 39.5 while their papers report 48.9 and 20.0. Open-domain (12.1) is the same
story in a more extreme form; see mem0/README.md for the worked example. Neither
cell can be compared to a paper number without swapping in the original LoCoMo
prompts, which would invalidate every historical LoCoMo number in this repo.

### Cost and concurrency

Phase 1 is 20–43 min per conversation. `Memoryos.add_memory` is a blocking
synchronous call inside an async hook, so it holds the event loop and the
nominal `max_sample_concurrent: 3` does not actually overlap users — the build is
serial. It is left that way on purpose: `src/memoryos/utils.py`'s
`_embedding_cache` evicts by listing keys and deleting them one by one, which
races under threads, so wrapping the call in `asyncio.to_thread` would trade a
known cost for an unknown corruption.

`mid_term_capacity: 200` binds in **1 of 6** conversations (segment counts:
79 / 109 / 155 / 171 / 179 / **200** — landing exactly on the cap is the LFU
eviction signature). The vendored default of 2000 would never bind at this scale.
