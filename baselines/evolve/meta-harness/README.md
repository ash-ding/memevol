# meta-harness — coding-agent search over memory harnesses

An evolve baseline: a search method compared against **forge itself**, not a
ready-made memory system. Adapts
[Meta-Harness](https://github.com/stanford-iris-lab/meta-harness)
([arXiv:2603.28052](https://arxiv.org/abs/2603.28052), paper PDF in this
directory) to this repo's `MemoClass` contract and evaluation protocol.

## The method

Meta-Harness searches over *harness code* — the program around a fixed base
model that decides what to store, retrieve, and show. Its one load-bearing
design choice is that the proposer is a **coding agent with filesystem access
to the entire search history**: every prior candidate's source, score, and
execution traces, read selectively with `grep`/`cat` rather than compressed
into a prompt. There is no parent-selection rule and no compressed feedback
channel — diagnosis and edit decisions are delegated to the agent.

One iteration:

1. **Propose** — the agent reads `evolution_summary.jsonl`, `frontier_val.json`,
   prior harness source, and the QA traces under `evals/<system>/traces/`, then
   writes `n_candidates` new harnesses into `harnesses/` and registers them in
   `pending_eval.json`.
2. **Gate** — two cheap checks before a candidate earns a real evaluation:
   an import check in a throwaway process, then ONE `sanity_check`-sized pass
   (`evaluate_memo(smoke=True)`). A harness fails sanity if any user errored or
   only partly completed — forge's rule. Import-clean code routinely crashes on
   first contact with real data, and this costs one tiny pass to find out.
3. **Evaluate** — each survivor is scored through
   `common.evaluate.evaluate_memo` on the **search** split, one subprocess per
   candidate.
4. **Log** — every outcome is appended to the same filesystem the next
   iteration reads, **rejections included** (marked `eliminated` with the
   error, so the frontier skips them but the proposer sees them). The Pareto
   front over (score up, context cost down) is recomputed.

`--status test` finalizes a named run: one held-out evaluation of the Pareto
frontier plus the baselines, after which the run is frozen against further
evolution.

## Setup

```bash
cd baselines/evolve/meta-harness && uv sync
```

The proposer is the **`claude` or `codex` CLI**, driven as a subprocess — not a
Python dependency. Install and log into whichever one you set as `agent:`
(`claude login` / `codex login`), and make sure it is on `PATH`.

**Model availability is account-scoped**, and there is no way to enumerate it:
a ChatGPT-account `codex login` rejects `gpt-5`, `gpt-5-codex` and `o3` with
`"not supported when using Codex with a ChatGPT account"`, accepting only what
the plan offers. So every search run **preflights the proposer** — one trivial
turn with the real argv — before it evaluates anything, and aborts with the
CLI's own error if that fails. Leaving `agent_model: null` sidesteps the whole
question by using whatever the CLI is already configured with.

## Usage

```bash
cd baselines/evolve/meta-harness

# Smoke: one iteration, one candidate, tiny gauntlet — does the loop close?
cp config.example.yaml smoke.yaml     # set iterations: 1, n_candidates: 1,
                                      # and shrink the `stages` block
uv run python run.py --config smoke.yaml --fresh

# Search: 10 iterations on the search split, resumable under a fixed run name
uv run python run.py --config config.example.yaml --run-name run1

# Same loop with Codex as the proposer
uv run python run.py --config config.example.yaml --run-name run1 --agent codex \
    --agent-model gpt-5-codex

# Finalize: ONE held-out evaluation of run1's frontier, then freeze it
uv run python run.py --config config.test.yaml --run-name run1
```

`run.py` takes `--config` plus a few runtime flags (`--status`, `--dataset`,
`--run-name`, `--iterations`, `--agent`, `--agent-model`, `--skip-baselines`,
`--fresh`); everything else is a config key. Precedence is
`DEFAULT_CONFIG < --config YAML < CLI`, and strict-config (on by default with
`--config`) requires the file to list every parameter and every sizing leaf.

Re-running with the same `--run-name` **resumes**: iteration numbering
continues from `evolution_summary.jsonl` and phase 0 can be skipped with
`--skip-baselines`.

## Layout

```
run.py            CLI + config resolution
loop.py           the outer search loop (propose -> validate -> evaluate -> log)
proposer.py       coding-agent driver: claude_code | codex, plus session logging
evaluator.py      one launch.py subprocess per candidate + import checking
launch.py         subprocess entry: load the MemoClass, call evaluate_memo
state.py          evolution_summary / Pareto frontier / finalization lock
history.py        CLI over a run's history: frontier / top / show / diff
prompts/          proposer_system.md — the proposer prior (system prompt)
harnesses/        the two tracked baselines, seeded into every run
logs/<run>/       per-run search filesystem, candidates included (gitignored)
results/<ds>/test/  held-out artifacts (gitignored)
```

Per run, under `logs/<run_name>/`:

| path | what it holds |
|---|---|
| `evolution_summary.jsonl` | one row per evaluated candidate: score, delta, context cost, stage, hypothesis |
| `frontier_val.json` | `best`, the `_pareto` front, and every system ranked |
| `evals/<system>/` | `score.json`, `metrics.json`, `stages.json`, `traces/<user>.json` |
| `evals/<system>/sanity/` | the pre-eval sanity pass — the first place to look when a candidate was rejected |
| `proposer/iter<N>/` | the agent session: `system_prompt.txt`, `task_prompt.txt`, `events.jsonl`, `response.md`, `meta.json` |
| `harnesses/` | this run's harnesses: the seeded baselines plus every candidate |
| `proposer_usage.jsonl` | one row per proposer session: tokens, duration, cost |
| `reports/` | proposer-written post-eval notes |
| `finalized.json` | the test-split lock |

**Harnesses are per run.** Candidates live in `logs/<run>/harnesses/`, not in a
shared directory, so concurrent or repeated runs never see each other's code and
`--fresh` on one run cannot delete another's. The tracked `harnesses/` at the
baseline root holds only the two baselines, copied into each run at start.

`evals/` is the point of the design: it is what the proposer reads, and the
traces are the highest-signal artifact in it.

`history.py` is the query layer over that tree — Appendix D recommends one, so
the proposer spends tokens on diagnosis rather than navigation. It is also the
fastest way for you to see what a run did:

```bash
cd baselines/evolve/meta-harness
uv run python history.py frontier          # Pareto front + best
uv run python history.py top -k 10         # ranked by score
uv run python history.py show <name>       # one harness: row, paths, artifacts
uv run python history.py diff <a> <b>      # results + code diff
uv run python history.py cost              # what the SEARCH itself cost
```

`--run <name>` picks the run; without it the most recent under `logs/` is used.

## Baseline harnesses

`harnesses/` ships two calibration points, both kept across `--fresh`:

- **`no_memory`** — stores nothing, retrieves nothing. The floor.
- **`full_context`** — renders every visible unit and returns the most recent
  30k characters verbatim. No selection, no compression, no LLM calls.

Together they bracket the accuracy/context-cost frontier the search optimizes
over.

## Comparability

The final artifact is a `common.memo_class.MemoClass` subclass scored through
`common.evaluate.evaluate_memo` — the literal same evaluator forge runs
in-container and alma runs in its subprocess, with the same split resolution,
per-dataset workflows, judge, and cost accounting. Only the isolation wrapper
differs (a plain subprocess here). So `score` sits on the same 0-1 axis as any
forge-evolved harness's `accuracy_<dataset>`, and the reported number is the
`--status test` score.

**The gold-data rule is honor-system.** forge runs its proposer inside a
Singularity sandbox whose bind list makes the raw benchmark files unreachable
at eval time. This proposer runs on the host with `bypassPermissions` /
`danger-full-access` and its cwd inside the repo, so
`benchmarks/dynamicmem/user_data/*/task_packs.json` (golden states),
`locomo10.json` and `longmemeval_*.json` are all readable. The proposer prior
forbids opening them at eval time and nothing else enforces it — upstream is
unsandboxed too. If a candidate's score jumps implausibly, read its source
before believing it.

**Split discipline.** Evolution only ever evaluates on `split: search`. The
test split is touched exactly once per run, by `--status test`, which writes
`finalized.json` before it starts and refuses to run twice; `--status search`
then refuses to add iterations to that run name. The proposer is never shown a
test result — it reads `logs/<run>/`, and test artifacts land under
`results/<dataset>/test/`.

## Faithfulness boundary

What is upstream's, and what changed to fit this repo:

| | upstream | here |
|---|---|---|
| proposer | coding agent with full filesystem access to prior code, scores, traces | same |
| feedback | raw artifacts, no compressed summaries, no parent selection | same |
| proposer prior | `.claude/skills/meta-harness/SKILL.md`, read whole and injected as the system prompt | `prompts/proposer_system.md`, injected the same way. Written to Appendix D's rule — it constrains outputs and forbidden behavior and names the objectives, and does NOT prescribe a diagnosis procedure. The text-classification skill's mandatory read order, prototyping step and rotation axes are deliberately absent (the terminal-bench skill has none of them either). Not a Claude Code skill: `codex` reads the identical file, so it carries no skill frontmatter |
| handoff | `pending_eval.json` written by the agent | same |
| candidate artifact | a `MemorySystem` with `predict` / `learn_from_batch` | a `MemoClass` with `build_memory_from_data` / `retrieve_memory_for_query` — this repo's contract |
| candidate dir | `agents/` | `harnesses/` |
| evaluation | its own `benchmark.py` sweep over text-classification datasets | `common.evaluate.evaluate_memo` — the shared evaluator, so scores are comparable |
| task scope | several datasets per run, per-dataset frontier | **one benchmark per run** (`dataset:`), like every other baseline here |
| frontier | Pareto over accuracy and context cost | same, over (score, `memory_tokens_per_query`) |
| finalization | `--test` on the frontier, then the run is frozen | same |
| proposer backend | Claude Code only; upstream suggests adapting its wrapper for others | **`claude_code` and `codex`**, one config key apart |

Two adaptations worth calling out:

- **One benchmark per run.** Upstream evolves against a set of classification
  datasets and keeps a frontier per dataset. Here sizing config is per
  benchmark family, so a run picks one `dataset:` — matching alma and every
  harness baseline. Run the loop once per benchmark.
- **The staged gauntlet.** `progressive: true` runs each candidate through the
  shared stage1→2→3 promotion gauntlet, so obviously broken candidates die
  after stage 1. Upstream evaluates every candidate at full size. The gauntlet
  is this repo's cost-control mechanism and keeps candidate scores comparable
  with forge's; `progressive: false` with a `single_stage` block reproduces
  upstream's flat behavior.

### Search cost

Proposer tokens do not pass through `common.llm`, so they cannot reach
`common.tokens` — the proposer is a CLI subprocess, and forge's containerized
proposer has the same boundary. Both CLIs do report usage on their own event
stream, so the numbers exist; `logs/<run>/proposer_usage.jsonl` keeps one row
per session and `history.py cost` totals them:

```
proposer sessions in run smoke3:
  iter   1  codex/(cli default)     795,508 in   13,202 out   340s  tools 35
  iter   2  codex/(cli default)   2,319,397 in   21,338 out   576s  tools 49
  iter   3  codex/(cli default)     731,413 in   11,529 out   281s  tools 31

total: 3 session(s)  3,846,318 in  46,069 out  1197s
  of which cached reads: 3,569,920 (93% of input)
```

Read that beside `results/<dataset>/test/<system>/token_usage.json`, which
covers evaluation only. The two are separate meters by construction, not an
oversight — but the search side is the larger one, so report both.

**Proposer spend is not billed through this repo.** Tokens the proposer burns are reported by
the agent CLI in `proposer/iter<N>/meta.json` and are **not** part of
`common.tokens` accounting — the same boundary forge's containerized proposer
has. This is the most expensive part of a run: the paper puts one Meta-Harness
iteration at roughly 10 MTok of proposer context (Table 1), orders of magnitude
above other text optimizers, which is the point of the method.

That is why `agent_auth: subscription` (the default) matters. It removes the
agent's own API key from the proposer's environment so the CLI falls back to
`claude login` / `codex login`, reproducing upstream's choice. `run.py` loads
the project `.env` before spawning anything, so without this the key you use
for evaluation reaches the proposer and can be billed for work no cost figure
in this repo will ever show. Set `agent_auth: api_key` only when you mean to
pay per token — and expect that spend to appear on your API bill and nowhere
else.

## License

MIT (same as the main project and as upstream).
