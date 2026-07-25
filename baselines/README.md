# Baselines

Comparison methods for the memory-architecture search. Each baseline runs
against the same benchmark set as the main method ([forge/](../forge/)),
producing comparable metrics: per-user reward, judge-scored accuracy, and
(for some) token / latency telemetry.

## Contents

1. [Layout — two kinds of baseline](#layout--two-kinds-of-baseline)
2. [Method-boundary conventions](#method-boundary-conventions)
3. [Shared progressive sampling, seeding & memory cache](#shared-progressive-sampling-seeding--memory-cache)
4. [Existing baselines](#existing-baselines)
5. [**Adding a harness baseline** — adapting an existing memory system](#adding-a-harness-baseline--adapting-an-existing-memory-system)
6. [Adding an evolve baseline](#adding-an-evolve-baseline)
7. [Shared foundation](#shared-foundation)

## Layout — two kinds of baseline

`baselines/` is split by WHAT is being compared:

```
baselines/
├── registry.py          # shared dataset registry (both sides import it)
├── requirements.txt     # shared full ML install
├── venv/                # shared Python 3.12 venv (gitignored)
├── evolve/              # SEARCH-METHOD baselines — compared against forge ITSELF
│   ├── alma/            #   LLM-meta-agent search loop (memevol's original method)
│   ├── evolvemem/       #   (paper PDF; to be implemented under the conventions below)
│   └── memevolve/       #   (paper PDF; to be implemented under the conventions below)
└── harness/             # READY-MADE MEMORY SYSTEMS — compared against forge's
    ├── eval_common.py   #   EVOLVED HARNESSES. Shared runner: run_baseline()
    ├── cc/              #   Claude Code as direct QA agent (native answer)
    └── hipporag2/       #   HippoRAG2 graph-RAG pipeline as retrieval memory
```

- **`evolve/`** — methods that SEARCH over memory-structure code, like forge
  does. Their unit of comparison is the search loop itself (proposer quality,
  sample efficiency, final evolved-harness score vs forge's).
- **`harness/`** — fixed, hand-written memory systems implementing the same
  standardized 3-hook `MemoStructure` contract
  (`build_memory_from_data` / `retrieve_memory_for_query` /
  `use_memory_to_answer`) that forge-evolved harnesses implement. Their unit
  of comparison is the harness artifact: they run through the SAME
  per-dataset workflows via `baselines.harness.eval_common.run_baseline`, so
  their scores sit on the same axis as any evolved harness's.

## Method-boundary conventions

These rules keep scores comparable while keeping methods independent. They
bind every method here:

- **The eval surface is mandatorily shared.** A method's FINAL ARTIFACT is a
  `common.harness_base.MemoStructure` subclass implementing the 3-hook
  contract, and it is scored ONLY through the shared registry/workflow path
  (`baselines/registry.py` → `datasets/<bench>/workflow.py` + the shared
  judge). No method ships its own scoring loop — otherwise its numbers stop
  being comparable.
- **Split discipline.** Development, tuning, and any internal
  search/self-improvement loop run on the **search split** only
  (`--split search` / `--status search`). The **test split** is touched
  exactly once per reported number (`--split test`, the default for
  harness runners — same data path as `forge.heldout`).
- **Dependency direction is one-way.** Methods import `common/` (and
  `datasets/` / `baselines/registry.py`); `common/` NEVER imports a method;
  methods NEVER import each other. Method-specific design vocabulary lives
  inside the method's own directory (e.g. alma's `Sub_memo_layer` in
  `evolve/alma/memo_layers.py`), never in `common/`.
- **Duplication over coupling.** If two methods need similar internal
  machinery, copy it. Duplicated internals are acceptable — preferred, even —
  because independence between methods matters more than DRY.

All baselines share **`baselines/venv/`** (full ML install:
`pip install -r baselines/requirements.txt`) and write artifacts under
each baseline's own `logs/` and `results/` directories (gitignored).

## Shared progressive sampling, seeding & memory cache

Every baseline (`evolve/alma` and every `harness/*`) now shares the same
evaluation-sampling infrastructure as forge, via two global flags plus a
cross-stage memory cache — not a re-implementation per baseline, but the
literal same `common/` modules forge uses:

- **`--progressive`** (alma default `true`, `harness/*/run.py` default
  `false` — matching each side's historical behavior): drives the candidate
  through the shared stage1→2→3 gauntlet (`common.staged_eval.run_gauntlet`)
  instead of a single one-shot pass. Sizes come from the family
  `DEFAULT_STAGES` (in `common/staged_eval.py`) unless overridden with
  `--stages '<json>'`. `run_gauntlet`'s promotion/elimination logic,
  `stages.json` shape, and cost accounting are IDENTICAL to forge's — only
  the stage-execution callback differs (forge: `singularity exec`; baselines:
  an in-process `run_all_users` runner).
- **`--random_sample`** / **`--random-sample`** (alma only — harness
  baselines have no step loop, so this flag isn't exposed there; default
  `false`): whether each search STEP evaluates a different, reproducibly
  seeded task subset (via `common.sampling.derive_sample_seed`) instead of
  the same fixed subset every step.
- **`--sampling_seed`** / **`--sampling-seed`** (default `42`): the base
  seed. alma combines it with `(step_index, dataset)` when `random_sample`
  is on; harness baselines (no steps) use it directly as a single fixed
  seed for their one-shot sample.
- **Memory cache** (`--memory_cache` / `--no-memory-cache`, default on):
  when `--progressive` is set, the SAME `common/memory_cache.py` mechanism
  forge's evaluator uses is mounted in the baseline's in-process stage
  runner too, so stage2/stage3 reuse stage1's built Phase-1 memory instead
  of re-ingesting from scratch — a real win for expensive builders (e.g.
  A-mem's per-note LLM analysis + evolution).

CLI surface per side:

```bash
# harness (cc / hipporag2 / amem run.py) — no --random-sample (no step loop)
baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset locomo --progressive --sampling-seed 42 \
    --stages '{"stage1": {"n_conversations": 2, "n_qa": 10, "threshold": 0.2}}'

# alma (evolve/alma/run_main.py) — has a step loop, so random_sample applies
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status search --progressive --random_sample --sampling_seed 42 --steps 10
```

See [CLAUDE.md](../CLAUDE.md) ("Shared progressive sampling") for the full
seed-derivation contract (`derive_sample_seed` / `combine_seed` /
`shuffle_prefix`, nesting guarantees, and the accepted overfitting-vs-
comparability tradeoff of `random_sample`).

## Existing baselines

| Baseline | Kind | Approach | Optimization | Best for |
|---|---|---|---|---|
| **[evolve/alma](evolve/alma/)** | search method | LLM-meta-agent search loop | Yes — propose / select / evolve over harness code | Established baseline; the framework's "v1" memory-architecture search |
| **[harness/cc](harness/cc/)** | ready-made harness | Claude Code as direct QA agent (native answer, multi-dataset) | None (zero-shot) | What-if: just give CC the raw user data + tools and let it answer |
| **[harness/hipporag2](harness/hipporag2/)** | ready-made harness | Graph-based RAG pipeline as retrieval memory (OpenIE → KG → PPR retrieval → passages; shared QA agent answers) | None (fixed pipeline) | Hand-designed memory architecture comparison point, multi-dataset |

### evolve/alma — meta-learning loop

The original method memevol shipped with: an LLM meta-agent reads sampled
QA trajectories, identifies failure patterns, and proposes new memory-
structure code. Sanity-checks each candidate, then evaluates on the search
split. Softmax-weighted parent selection over reward.

```bash
# Smoke — tiny staged gauntlet, 2 steps (evaluation sizes come from the
# shared `stages` schema now, not flat eval_n_*/check_n_* flags — see
# "Shared progressive sampling" below)
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status search --progressive --steps 2

# Full training — stage1->2->3 gauntlet (default DEFAULT_STAGES sizes), 10 steps
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status search --progressive --steps 10

# Held-out evaluation of a saved memo
baselines/venv/bin/python baselines/evolve/alma/run_main.py \
    --status test --memo_SHA <SHA> --progressive
```

Artifacts: `baselines/evolve/alma/{logs/, memo_archive/, results/}`. See
[evolve/alma/README.md](evolve/alma/README.md) for layout details.

**Key difference from forge**: alma's proposer is a single LLM call with
compressed feedback (sampled trajectories + meta-prompt), whereas forge's
proposer is an agentic CC SDK call with full filesystem access to all
prior code, traces, and scores. alma runs the shared per-dataset workflows
(including the official DynamicMem TCE v2 checkpoint protocol) AND the same
`common.staged_eval.run_gauntlet` driver forge uses, so its numbers ARE
comparable with forge.

### harness/cc — Claude Code as direct QA agent

Skips memory-architecture design entirely. `CCMemo`
([harness/cc/memo.py](harness/cc/memo.py)):

- **BUILD**: stashes the currently-visible data into a per-user temp
  directory as a single JSON file.
- **RETRIEVE**: returns `{}` — no separate retrieval step.
- **ANSWER** (`use_memory_to_answer`): runs Claude Code with tool access
  (Read, Grep, Glob) over the temp directory, on the workflow's exact
  formatted prompt — cc's own answer is judged verbatim, bypassing the
  shared QA agent. This is the one baseline that overrides the answer hook.

```bash
baselines/venv/bin/python baselines/harness/cc/run.py \
    --dataset locomo --model claude-sonnet-4-20250514

baselines/venv/bin/python baselines/harness/cc/run.py \
    --dataset dynamicmem --stage-spec '{"n_samples": 2}'
```

Useful as a reference point: how well does a strong agent do **with no
learned memory structure at all**, just raw access + tools?

Artifacts: `baselines/harness/cc/results/<dataset>/<split>/`.

### harness/hipporag2 — graph-based RAG pipeline as retrieval memory

`HippoRAGMemo` ([harness/hipporag2/memo.py](harness/hipporag2/memo.py))
wraps [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)'s pipeline:

- **BUILD**: converts the visible data into text passages and indexes them
  into a per-user HippoRAG graph (OpenIE → NER + triples → knowledge graph +
  entity embeddings). Indexing is additive across calls, so DynamicMem's
  per-checkpoint segments accumulate correctly.
- **RETRIEVE**: fact retrieval → reranking → personalized PageRank → top-k
  passages, returned as `{"passages": [...]}`. The **shared QA agent**
  answers from those passages — a fair "HippoRAG-as-memory" comparison, not
  an end-to-end HippoRAG pipeline comparison.

```bash
# OpenAI API embedding (no GPU needed)
baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset locomo --embedding text-embedding-3-small

# Local GPU embedding (NVIDIA)
baselines/venv/bin/python baselines/harness/hipporag2/run.py \
    --dataset longmemeval_s --embedding nvidia/NV-Embed-v2 \
    --embedding_batch_size 2 --embedding_dtype float16
```

Artifacts: `baselines/harness/hipporag2/{outputs/, results/<dataset>/<split>/}`.

---

## Adding a harness baseline — adapting an existing memory system

This is the recipe for evaluating an existing, human-crafted memory system
(mem0, letta, zep, MemGPT, your own prototype, ...) under this repo's
protocol. The whole adaptation is two files under
`baselines/harness/<name>/`; everything else — split resolution, the
per-dataset evaluation protocol (including DynamicMem's checkpoint
interleaving), judging, scoring, trace persistence — comes from the shared
runner and is byte-identical to what forge-evolved harnesses get.

### Step 0 — understand what you're adapting to

Your system is driven through three async hooks on a
`common.harness_base.MemoStructure` subclass. The evaluation lifecycle per
user/sample:

```
fresh instance created                       # NO cross-user state — ever
  → build_memory_from_data(recorder)          # 1..N times (N>1 only for DynamicMem:
                                              #   one call per checkpoint, DELTA data)
  → per query:  retrieve_memory_for_query(recorder)   # MUST be read-only
                use_memory_to_answer(recorder, retrieved, prompt)  # optional
```

Three lifecycle rules that trip up adapters:

1. **Fresh instance per user.** If your system persists state on disk,
   scope it per instance — e.g. `HippoRAGMemo` creates
   `uuid.uuid4().hex[:12]`-suffixed save dirs in `__init__`. Do NOT key
   state on `recorder.user_id` (it is always `""` at memo call sites).
2. **BUILD accumulates.** For LoCoMo/LongMemEval you get ONE build call
   with everything; for DynamicMem you get FIVE calls, each with only that
   checkpoint's new log segment. Your ingestion must be additive.
3. **RETRIEVE is read-only.** DynamicMem interleaves queries with
   ingestion at checkpoints — a retrieve that mutates memory corrupts
   checkpoint isolation (and the cross-stage memory cache).

### Step 1 — `memo.py`: the adapter class

`recorder.init` shapes per benchmark (dispatch on the keys):

| Benchmark | BUILD `recorder.init` | RETRIEVE `recorder.init` |
|---|---|---|
| dynamicmem | `{"app_logs": [log, ...]}` — each log: `app_log_id, timestamp, app_name, api_name, request, response`. Per-checkpoint DELTA | `{"app_logs": [...visible prefix...], "query": str}` |
| locomo | `{"conversation": {...}}` — `speaker_a/b`, `session_1..N` (turn lists: `speaker, dia_id, text`), `session_N_date_time` | `{"conversation": {...}, "query": str}` |
| longmemeval_s/m | `{"sessions": [session, ...]}` — each: `session_id, date, messages[{role, content}]` | `{"sessions": [...], "query": str, "question_date": str}` |

Skeleton:

```python
# baselines/harness/<name>/memo.py
import uuid
from typing import Dict, Optional
from common.harness_base import MemoStructure

class MyMemo(MemoStructure):
    _cfg: Dict = {}                     # filled by eval_common.make_memo_class

    def __init__(self):
        super().__init__()
        self._instance_id = uuid.uuid4().hex[:12]   # per-user state scoping
        self._system = ...              # construct the wrapped memory system
                                        #   using self._cfg (model names, top-k, ...)

    async def build_memory_from_data(self, recorder) -> None:
        init = recorder.init
        if "app_logs" in init:          # dynamicmem (per-checkpoint delta)
            texts = [render_log(l) for l in init["app_logs"]]
        elif "conversation" in init:    # locomo
            texts = [render_turn(t) for t in iter_turns(init["conversation"])]
        elif "sessions" in init:        # longmemeval
            texts = [render_msg(s, m) for s in init["sessions"] for m in s["messages"]]
        self._system.add(texts)         # ADDITIVE — never reset here

    async def retrieve_memory_for_query(self, recorder) -> Dict:
        hits = self._system.search(recorder.init["query"], top_k=...)
        return {"passages": hits}       # read-only w.r.t. memory state

    # OPTIONAL — only if your system answers natively (agentic systems).
    # Return None / omit entirely to let the shared QA agent answer.
    # async def use_memory_to_answer(self, recorder, retrieved, prompt) -> Optional[str]:
    #     return await self._system.answer(prompt)
```

Notes on the RETRIEVE return dict:

- Any dict works — it is rendered into the benchmark's QA prompt. A flat
  `{"passages": [...]}` is the safe generic choice.
- **DynamicMem tip**: `{"inline_memory_blocks": [str, ...]}` renders each
  block verbatim into the official TCE answer prompt's `[Memory]` section;
  any other dict shape is serialized as one JSON block. Surface the source
  logs *with their `app_log_id`* — evidence citation is scored.
- Override `use_memory_to_answer` ONLY for systems whose value proposition
  includes answering (like cc). Retrieval-style systems should let the
  shared QA agent answer — that keeps the comparison about *memory*, not
  about who has the better answerer.

### Step 2 — `run.py`: the CLI entry

Copy [harness/hipporag2/run.py](harness/hipporag2/run.py) (~40 lines) and
adjust the flags your system needs. Core shape:

```python
from baselines.registry import DATASETS
from baselines.harness.eval_common import make_memo_class, run_baseline, parse_stage_spec
from baselines.harness.<name>.memo import MyMemo

p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
p.add_argument("--split", default="test", choices=["test", "search"])
p.add_argument("--stage-spec", default=None)   # JSON size overrides, e.g. '{"n_samples": 2}'
...
memo_cls = make_memo_class(MyMemo, top_k=a.top_k, ...)   # → sets _cfg
result = asyncio.run(run_baseline(
    dataset=a.dataset, split=a.split,
    user_stage_spec=parse_stage_spec(a.stage_spec),
    memo_class=memo_cls, qa_model=a.model, judge_model=a.judge_model,
    out_dir=Path(__file__).resolve().parent / "results" / a.dataset / a.split,
))
```

`make_memo_class` exists because the workflow instantiates the memo class
with **no arguments** — your CLI config travels as the `_cfg` class
attribute. No `__init__.py` files needed (namespace packages).

### Step 3 — dependencies

Add your system's packages to [`baselines/requirements.txt`](requirements.txt)
and install into the shared venv:

```bash
baselines/venv/bin/pip install -r baselines/requirements.txt
```

(Heavy/conflicting deps? Note them in your baseline's own README; the
shared venv is the default, not a hard rule — but scoring still MUST go
through `run_baseline`.)

### Step 4 — validate on the SEARCH split (cheap, iterate freely)

```bash
# One conversation, a handful of QAs — does the adapter run end-to-end?
baselines/venv/bin/python baselines/harness/<name>/run.py \
    --dataset locomo --split search --stage-spec '{"n_samples": 1, "n_qa": 3}'

# DynamicMem protocol check — 1 user exercises the checkpoint interleaving
baselines/venv/bin/python baselines/harness/<name>/run.py \
    --dataset dynamicmem --split search --stage-spec '{"n_samples": 1}'
```

Iterate here as much as you like — this is the split the main method
searches on. Read `results/<dataset>/search/traces/<user>.json`: each QA
step records the query, your retrieved dict, the answer, and
`judge_reason` — the fastest way to see whether your retrieval is
surfacing the right memory.

### Step 5 — final numbers on the TEST split

```bash
# Whole test split (the default --split; = forge.heldout coverage=full)
baselines/venv/bin/python baselines/harness/<name>/run.py --dataset locomo
baselines/venv/bin/python baselines/harness/<name>/run.py --dataset dynamicmem
baselines/venv/bin/python baselines/harness/<name>/run.py --dataset longmemeval_s
```

Outputs land in `baselines/harness/<name>/results/<dataset>/test/`:
`score.json` (mean reward = the number you report, same 0–1 scale as
forge's `accuracy_<dataset>`), `token_usage.json`, `traces/`. Because the
task list, workflow, judge, and scoring are the main method's own code,
these numbers are directly comparable to any forge-evolved harness's
held-out score — same code, not "comparable" code. Touch the test split
once per reported number; tuning happens in Step 4.

---

## Adding an evolve baseline

Evolve-framework baselines (a search loop that *produces* memory systems —
e.g. the EvolveMem / MemEvolve papers staged under `evolve/`) differ too
much internally for a step-by-step recipe: each has its own proposer,
feedback signal, and population management. What binds them is the
convention set above, concretely:

1. **Own directory, internal freedom.** Everything method-specific —
   search loop, prompts, checkpointing, its own base classes — lives in
   `baselines/evolve/<name>/`. Copy machinery from alma if useful; do not
   import it.
2. **The final artifact is a 3-hook `MemoStructure`.** Whatever the search
   produces must be loadable as a `common.harness_base.MemoStructure`
   subclass (directly, or via a thin adapter) so it can be scored through
   the shared workflow path. If the method's native artifact is not a
   Python class, the adapter is part of the baseline.
3. **Internal loops stay on the search split.** Every self-improvement
   iteration evaluates on `status/split = search`. The test split is used
   exactly once, for the final frozen artifact — mirroring how forge's
   search never touches test data (`forge.heldout` is the only test entry).
4. **Reference implementation**: [evolve/alma/](evolve/alma/) — its
   `run_main.py --status {search,test}` split handling, `eval_runner.py` →
   shared-workflow scoring, and `memo_archive/` artifact management are the
   patterns to mirror (by copying, not importing).

## Shared foundation

All baselines (and forge) build on the same dataset adapters and judge:

- **[`baselines/registry.py`](registry.py)** — dataset name → (workflow,
  env module, recorder) resolution, shared by BOTH `evolve/` and `harness/`
  (mirrors `forge/launch.py::WORKFLOWS`; baselines never import forge).
- **[`datasets/<bench>/env.py`](../datasets/)** — `load_user_data`,
  `get_task_list` (the single source of truth for the search/test split),
  per-benchmark Recorder.
- **[`common/judge.py`](../common/judge.py)** — LLM-as-judge with
  configurable prompt template and score range (DynamicMem uses the
  official TCE holistic judge in `datasets/dynamicmem/tce_prompts.py`).
- **[`common/llm.py`](../common/llm.py)** — `Agent` / `Embedding`
  wrappers with automatic token tracking; baselines use these so their
  cost numbers are comparable to forge's.

## License

MIT (same as the main project).
