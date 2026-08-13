# mem0 — LLM-extracted fact memory

[Mem0](https://github.com/mem0ai/mem0) as a `MemoClass`, scored through the
shared registry/workflow/judge path like every other baseline.

## What the method is

Mem0's contribution is at WRITE time. `Memory.add(messages)` sends a batch of
messages to an LLM that (a) extracts standalone facts from them and (b) decides,
per fact and against what is already stored, whether to **ADD** it, **UPDATE** an
existing fact, or **DELETE** one it contradicts. Memory therefore holds distilled,
de-conflicted facts — "User adopted a rescue dog named Pico, a beagle mix" — not
raw turns. Read time is a plain vector search over those facts.

That write-side consolidation is the whole point, so `infer: true` is the method;
`infer: false` degrades Mem0 to a vector store and is only meaningful as an
ablation.

## Integration

| hook | what runs |
|---|---|
| `build_memory_from_data` | ingestion units → chat messages → `Memory.add(..., user_id)` in batches of `add_batch_size` |
| `retrieve_memory_for_query` | `Memory.search(query, filters={user_id}, top_k, threshold)` → `{"passages": [...]}` |

`use_memory_to_answer` is NOT overridden: the SHARED QA agent answers from the
retrieved facts (hipporag2/amem/simplemem pattern), so the comparison is about
memory rather than about each method's own answerer.

Each user gets its own Mem0 instance with its own on-disk Qdrant collection and
history DB under `outputs/<instance>/`, so nothing leaks across conversations.

**Ingestion units.** locomo: one message per turn, with speaker and session date
folded INTO the content — Mem0's extractor only sees `role`/`content`, so
dropping them would make every "who said it / when" question unanswerable.
longmemeval: one message per message, roles preserved. dynamicmem: one message
per log entry using hipporag2's `app_log_to_passage` text, so passage content is
identical across baselines.

## Model configuration (two arms)

Every model this baseline touches is a config parameter, so it runs in two arms:

| | faithful arm (`config.example.yaml`) | unified arm (`config.unified.yaml`) |
|---|---|---|
| internal LLM (`mem0_llm_model`) | `gpt-4o-mini` — Mem0's own default | `gpt-5-mini` |
| embedder (`embedding_model`) | `text-embedding-3-small`, API, 1536-dim | **unchanged** |

Mem0 is one of only two baselines (with hipporag2) already on an API embedder,
so its embedder does not change between arms and no adapter is involved. **The
faithful arm is the default**, and `config.paper.yaml` additionally pins the
paper's gpt-4o-mini answerer. The unified arm puts all seven baselines on one
LLM and one embedder so the comparison against the main method is like-for-like
— a deliberate deviation from the paper, whose numbers must not be quoted as
Mem0's published result.

Mem0's OpenAI provider sends `temperature` and `max_tokens`, which the gpt-5
family rejects; the shim in [`../model_config.py`](../model_config.py) drops the
rejected params and renames `max_tokens` → `max_completion_tokens` at the
OpenAI-SDK boundary. `mem0ai` is a pinned PyPI package rather than vendored
source, so there is no `src/` byte-identity claim to preserve here — but the
same wrap-don't-rewrite technique is used, so the pin stays honest.

## Dependencies

`mem0ai` is PINNED in `pyproject.toml` rather than vendored: it is a maintained
package with a stable public API, and the pin is what makes a run reproducible
(same reasoning as hipporag2's external install). The exercised path is Mem0's
default stack — OpenAI LLM + OpenAI embedder + embedded Qdrant, no server.

## Run

```bash
# build mem0's isolated env
cd baselines/harness/mem0 && uv sync

# run it (from the baseline dir)
cd baselines/harness/mem0 && uv run python run.py --config config.example.yaml

# its unit test (from the repo root)
uv run --project baselines/harness/mem0 python tests/test_mem0_baseline.py
```

## Reproduction check (2026-08-05)

`config.paper.yaml` — LoCoMo **search** split, all 6 conversations × 60
randomly-sampled QA = **360 questions**, answerer **gpt-4o-mini** (what the paper
reports on; `config.example.yaml` keeps the repo's shared gpt-5-mini agent),
`top_k: 10` per the paper's "s=10 similar memories". 6/6 conversations, 22.5 min,
**796 tokens/question**.

Scored with `baselines/harness/score_paper_metrics.py`, which recomputes the
papers' metrics — the shared judge is binary and is not what they report.

| category | our F1 | paper F1 | n |
|---|---|---|---|
| single-hop | 46.7 | 38.7 | 202 |
| multi-hop | **32.7** | **28.6** | 65 |
| temporal | 35.9 | 48.9 | 73 |
| open-domain | 5.4 | 47.6 | 20 |
| **unweighted mean** | **30.2** | **41.0** | |

**Verdict: the integration reproduces.** Two of the paper's signatures hold —
multi-hop is the weakest real category (as in the paper), and excluding
open-domain the three-category mean is **38.4 vs the paper's 38.7**.

**The open-domain cell is a repo-level metric artifact, not a Mem0 defect.**
LoCoMo's category 3 is speculative yes/no ("Would Caroline want to move back?",
gold `"No; she's in the process of adopting children."`), but the shared prompt
in `benchmarks/locomo/prompts.py` asks for a short phrase "with exact words from
the context", so answers come back as descriptive statements that share almost no
tokens with the gold. It is not method-specific: a plain dense-retrieval memory
measured under the same prompt scores 12.3–14.3 open-domain F1, and MemoryOS
scores 12.1 — the cell is capped by the prompt, not by the memory. Comparing it
to a paper number is therefore meaningless without swapping in the original
LoCoMo category-3 prompt — which would invalidate every historical LoCoMo number
in this repo, so it is deliberately NOT done.

Temporal is likewise prompt-dominated: `_LOCOMO_USER_CAT2` supplies explicit date
guidance, and under it Mem0 (35.9) and MemoryOS (39.5) converge, while their
papers report 48.9 and 20.0 — a gap the shared prompt closes.
