# mem0 — LLM-extracted fact memory

[Mem0](https://github.com/mem0ai/mem0) as a `MemoClass`, scored through the
shared registry/workflow/judge path like every other baseline. The baseline
vendors and drives `mem0` directly.

**Provenance**: `src/mem0/` is vendored VERBATIM (byte-identical) from
<https://github.com/mem0ai/mem0> @ `12c47f524935692e27ad48d829f35fa1e4417181`
(tag `v2.0.17`, the version previously pinned as `mem0ai==2.0.17`), pruned to the
exercised slice — provenance lives here, not in per-file headers, to preserve
byte-identity:

    git -C <mem0-clone> archive 12c47f5 mem0 | tar -x -C /tmp/m0
    diff -r /tmp/m0/mem0 src/mem0        # only src/mem0/__init__.py differs (see below)

**Excluded** (upstream carries a very large provider surface; the baseline
exercises one LLM, one embedder, one vector store):

| excluded | why |
|---|---|
| `client/`, `proxy/`, `reranker/` | hosted-platform HTTP client, OpenAI-proxy shim, rerankers — none used |
| `llms/` — 17 of 18 providers | only `llms/openai.py` is configured (`provider: openai`) |
| `embeddings/` — 10 of 11 providers | only `embeddings/openai.py`; `mock.py` is kept because `utils/factory.py` imports it eagerly |
| `vector_stores/` + `configs/vector_stores/` — 24 of 25 each | only `qdrant.py` (embedded, on-disk, per user) |
| `utils/gcp_auth.py` | Vertex/GCP auth, unreachable from the exercised path |

Everything kept is on the eager import closure of `from mem0 import Memory` plus
those three factory-selected providers; the factory loads providers by dotted
string, so an excluded provider is a config that raises rather than a silent
fallback. `mem0/utils/` has no `__init__.py` upstream (namespace package) and is
vendored that way.

**The one file that is NOT byte-identical** is `src/mem0/__init__.py`: upstream
reads its version from installed `mem0ai` metadata
(`importlib.metadata.version("mem0ai")`), which raises `PackageNotFoundError` for
a vendored copy, and eagerly imports the un-vendored `client/` subtree. It is
replaced with a minimal initializer that inlines `__version__ = "2.0.17"` (read
by `memory/telemetry.py`) and re-exports only `Memory`/`AsyncMemory`. The reason
is restated in the file itself. All other integration code lives in `memo.py` /
`run.py`, never in `src/`.

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

## Dependencies

`mem0ai` is NOT a dependency — the source is vendored under `src/` (see
Provenance), so `uv sync` alone is the whole setup. `pyproject.toml` carries only
what the vendored slice imports: qdrant-client, pydantic, posthog (eagerly
imported by `memory/telemetry.py` even with `MEM0_TELEMETRY=False`, which
`memo.py` sets), plus openai/httpx from the shared-core block. Upstream's
sqlalchemy / pytz / protobuf / spacy are not imported by the slice (protobuf
arrives transitively via qdrant-client; spaCy only via the optional BM25
lemmatization path, which falls back to the raw text when it is absent).

The exercised path is Mem0's default stack — OpenAI LLM + OpenAI embedder +
embedded Qdrant, no server.

## Faithfulness boundary

| Category | Items |
|---|---|
| Verbatim | the whole vendored slice (@ v2.0.17) except `__init__.py`: the ADD/UPDATE/DELETE fact-consolidation pipeline, extraction + update prompts (`configs/prompts.py`), scoring/BM25 path, Qdrant store, OpenAI LLM + embedder clients |
| Integration adaptations (not algorithm) | `src/mem0/__init__.py` replaced (metadata version lookup + un-vendored `client/` import — see Provenance); embedded on-disk Qdrant + per-user instance for isolation rather than a shared server; `MEM0_TELEMETRY=False` set before import (upstream's telemetry store takes a process-global Qdrant lock that breaks concurrent users, and a benchmark should not phone home); batched `add()` (`add_batch_size`, Mem0's intended usage) instead of one call per turn; per-dataset ingestion-unit mappings (locomo folds speaker+date into content, longmemeval preserves roles, dynamicmem reuses hipporag2's `app_log_to_passage`); answering via the shared QA agent |
| Upstream quirks preserved | all extraction/update prompts, thresholds and the `infer=True` default path untouched; excluded providers fail loudly (factory loads by dotted string) rather than falling back |

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
