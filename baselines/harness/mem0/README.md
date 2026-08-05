# mem0 — LLM-extracted fact memory

[Mem0](https://github.com/mem0ai/mem0) as a `MemoStructure`, scored through the
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

## Dependencies

`mem0ai` is PINNED in `requirements.txt` rather than vendored: it is a maintained
package with a stable public API, and the pin is what makes a run reproducible
(same reasoning as hipporag2's external install). The exercised path is Mem0's
default stack — OpenAI LLM + OpenAI embedder + embedded Qdrant, no server.

## Run

```bash
baselines/setup_venv.sh mem0
baselines/harness/mem0/venv/bin/python baselines/harness/mem0/run.py \
    --config baselines/harness/mem0/config.example.yaml
baselines/harness/mem0/venv/bin/python tests/test_mem0_baseline.py
```
