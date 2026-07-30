# evolvemem/simplemem — vendored SimpleMem substrate

[SimpleMem](https://github.com/aiming-lab/SimpleMem) (AIMING Lab, MIT) —
the base memory system the EvolveMem paper evolves on top of. It lives
INSIDE evolvemem's directory (not under `baselines/harness/`) because it
exists for evolvemem's use: per the baselines method-boundary rules,
methods never import each other, so evolvemem owns its substrate.

Roles:
1. **Fidelity reference** for `../memo_evolvemem.py` (the native
   approximation built from the paper's description) — same subset, same
   eval path, diff the scores.
2. **Future evolution substrate**: wiring θ onto this real substrate makes
   evolvemem 100% faithful to the paper's setup (not wired yet — the
   adapter runs upstream's fixed default configuration).

## Evaluate standalone (through evolvemem's own launch path)

```bash
python baselines/evolve/evolvemem/launch.py \
    --memory_id simplemem_ref --substrate simplemem \
    --output_run_dir baselines/evolve/evolvemem/results/<ds>/simplemem_ref \
    --dataset locomo --eval_n_samples 1 --eval_n_qa 3 --status search
```

## Pipeline (upstream, unchanged)

- **BUILD** (paper §3.1–3.2): sliding windows (40 dialogues, overlap 2),
  one internal-LLM call per window → `MemoryEntry`s with *lossless
  restatement* (coreferences resolved, absolute timestamps), keyword +
  symbolic layers, multi-view indexed in LanceDB.
- **RETRIEVE** (§3.3): LLM query planning → parallel semantic/keyword/
  structured searches → LLM adequacy check + ≤2 reflection rounds
  (2–4+ internal LLM calls per query — a read-heavy cost shape).
- **ANSWER**: shared QA agent (upstream AnswerGenerator vendored, unused).

## Vendored code & patches

`vendor/simplemem/` = upstream `simplemem/{core,text}` only, imports
unchanged (adapter puts `vendor/` on sys.path). Patches marked `PATCH #n`
in-file:

1. minimal package `__init__` (drops router/multimodal surface + deps)
2. `core/utils/embedding.py` → OpenAI embeddings ($SIMPLEMEM_EMBED_MODEL,
   default text-embedding-3-small) replacing local Qwen3/torch — scores
   comparable within this repo, not with the paper's tables
3. `core/utils/llm_client.py` tallies token usage; the adapter drains it
   into GLOBAL_TOKEN_TRACKER (USE_STREAMING forced false)

Adapter-level deviations: `finalize()` after every build call (checkpoint
interleaving); internal LLM defaults to upstream's gpt-4.1-mini (its
LLMClient passes temperature — gpt-5 family rejects it); per-instance
LanceDB in a temp dir; lancedb imported with memevol's `datasets` package
masked (name collision with HF datasets).

Extra dependency: `lancedb`, `dateparser` (`pip install lancedb dateparser`).

## License

Upstream MIT (AIMING Lab).
