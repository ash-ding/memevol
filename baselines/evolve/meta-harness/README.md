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

`--status test` finalizes a named run: one held-out evaluation, after which the
run is frozen against further evolution. `finalize_systems` picks what gets
scored — `pareto` (the paper's rule: the whole frontier, which is what yields
an accuracy/context trade-off curve) or `best` (the top system only, cheaper,
one point). The baselines are scored either way; they are the calibration.

## Setup

```bash
cd baselines/evolve/meta-harness && uv sync
```

The proposer is the **`claude` or `codex` CLI**, driven as a subprocess — not a
Python dependency. Install and log into whichever one you set as `agent:`
(`claude login` / `codex login`), and make sure it is on `PATH`.

**Model availability follows `agent_auth`.** A ChatGPT-account `codex login`
serves only what the plan offers and refuses the rest outright — measured on
this repo: `gpt-5`, `gpt-5-codex` and `o3` all rejected, `gpt-5.5` fine. With
`agent_auth: api_key` the same three all run. So if the model you want is
refused, that is the auth mode talking, not the model.

`api_key` needs no change to your own login. Codex reads credentials from
`$CODEX_HOME` (`codex exec --help`: *"auth still uses CODEX_HOME"*), so the run
stages its own at `.codex_home/` from `OPENAI_API_KEY` and points the
subprocess there; `~/.codex` is untouched and stays whatever it was. That
directory holds a live API key and is gitignored — keep it that way.

Either way, every search run **preflights the proposer** — one trivial turn
with the real argv — before evaluating anything, and aborts with the CLI's own
error if it fails. That is what stops a wrong model id from costing you a phase
0 you then throw away.

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

A candidate the proposer writes somewhere else — its working directory, or the
path named in `pending_eval.json`'s `file` field — is **moved into the run and
evaluated** rather than dropped. Agents driving an unfamiliar shell misplace
files often enough that losing an iteration over it is not worth it. Resolution
never leaves the baseline directory, so a stray `file` field cannot pull in
arbitrary code.

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


## License

MIT (same as the main project and as upstream).
