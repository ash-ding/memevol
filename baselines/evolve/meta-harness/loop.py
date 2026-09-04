"""The Meta-Harness outer search loop.

Per iteration: a coding agent inspects the run's filesystem — prior harness
code, scores, and execution traces — and writes new candidates; the loop puts
each through two cheap gates (import, then a sanity-sized pass), scores the
survivors through the shared evaluator, and appends every outcome — rejections
included — back into that same filesystem. No parent selection, no compressed
feedback: the proposer decides what to read and what to change.

Evolution runs on the SEARCH split only. `--status test` finalizes a named run
with one held-out evaluation of the Pareto frontier and then freezes it.
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluator
import proposer
import state
from state import RunPaths

BASELINE_ROOT = Path(__file__).resolve().parent
PROPOSER_SYSTEM = BASELINE_ROOT / "prompts" / "proposer_system.md"
# Wall-clock cap for the sanity gate. Deliberately NOT `eval_timeout`: a sanity
# pass is a handful of QAs, so a candidate still running after this is hung, and
# waiting out a 14-hour eval budget to learn that would defeat the gate.
SANITY_TIMEOUT_S = 30 * 60

# Consecutive failed proposer sessions before the run gives up. Proposer
# failures are almost always configuration (bad model id, expired login, CLI
# not on PATH), which repeats identically on every iteration.
MAX_PROPOSER_FAILURES = 2

_interrupted = False


def _install_signal_handlers() -> None:
    def handler(_signum, _frame):
        global _interrupted
        _interrupted = True
        print("\ninterrupted — finishing the current step, then stopping", flush=True)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# task prompt
# --------------------------------------------------------------------------

def _task_prompt(paths: RunPaths, cfg: Dict[str, Any], iteration: int) -> str:
    frontier = state.read_frontier(paths)
    best = frontier.get("best")
    best_line = (
        f"{best['system']} at score {best['score']:.3f} "
        f"(context cost {best['context_cost']:.0f} tok/query)"
        if best else "none yet — the baselines are the only reference points"
    )
    return f"""Run iteration {iteration} of harness evolution.

Benchmark: {cfg['dataset']} (search split). Current best: {best_line}.
Propose {cfg['n_candidates']} new harnesses this iteration.

## Where everything is

This run owns its own directory; other runs are none of your business.

- `{paths.harnesses}/` — **every harness's source, and where your new ones go.**
  This run's directory, not a shared one — read and copy from it freely.
- `{paths.summary}` — every candidate evaluated so far
- `{paths.frontier}` — best + Pareto front
- `{paths.evals}/<system>/` — per-candidate artifacts:
  `score.json`, `metrics.json`, `stages.json`, `traces/<user>.json`
- `{paths.reports}/` — your notes, if you want them
- Write `pending_eval.json` to: `{paths.pending}`

Query it with `uv run python history.py --run {paths.run_name} <frontier|top|show|diff>`
from `{BASELINE_ROOT}` (also your working directory).
"""


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------

def _read_pending(paths: RunPaths, known: set) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Candidates the proposer registered, plus a reason if the handoff itself
    was unusable.

    Read as utf-8-SIG, not utf-8: the proposer writes this file with whatever
    shell it has, and Windows PowerShell's Set-Content / Out-File emit a UTF-8
    BOM by default. A strict utf-8 read rejects that outright, which silently
    discarded 16 written candidates across 5 iterations of the first real run.
    utf-8-sig strips a BOM when present and is identical to utf-8 when not.
    """
    if not paths.pending.exists():
        return [], "the proposer wrote no pending_eval.json"
    try:
        raw = paths.pending.read_text(encoding="utf-8-sig")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        head = paths.pending.read_bytes()[:160]
        _log(f"  pending_eval.json is unusable ({exc})")
        _log(f"  first bytes: {head!r}")
        return [], f"pending_eval.json could not be read: {exc}"

    candidates = []
    for entry in payload.get("candidates", []):
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        if name in known:
            _log(f"  SKIP {name}: a system with that name was already evaluated")
            continue
        if _resolve_candidate_file(paths, name, str(entry.get("file", ""))) is None:
            continue
        entry["name"] = name
        candidates.append(entry)
        known.add(name)
    if not candidates:
        return [], "pending_eval.json registered no usable candidate"
    return candidates, None


def _resolve_candidate_file(paths: RunPaths, name: str, declared: str) -> Optional[Path]:
    """Find a candidate's source wherever the proposer actually put it.

    The run's harness directory is the contract, but an agent driving an
    unfamiliar shell lands files in its working directory often enough to be
    worth recovering from — the first full LoCoMo run left two harnesses at the
    baseline root. A candidate whose code exists but sits one directory over is
    a lost iteration for no reason, so it is MOVED into the run and evaluated.

    Only paths under the baseline root are accepted, so a stray `file` field
    cannot pull arbitrary code in from elsewhere on the machine.
    """
    import shutil

    target = paths.harnesses / f"{name}.py"
    if target.exists():
        return target

    guesses = [paths.harnesses / declared, Path(declared), BASELINE_ROOT / declared] if declared else []
    guesses.append(BASELINE_ROOT / f"{name}.py")

    searched = [target]
    for guess in guesses:
        guess = Path(guess)
        if guess in searched:
            continue
        searched.append(guess)
        try:
            inside = guess.resolve().is_relative_to(BASELINE_ROOT.resolve())
        except (OSError, ValueError):
            continue
        if inside and guess.is_file() and guess.suffix == ".py":
            shutil.move(str(guess), str(target))
            _log(f"  MOVED {name}: written to {guess}, belongs in the run dir")
            return target

    _log(f"  SKIP {name}: no source found. Looked in: "
         + ", ".join(str(s) for s in searched))
    return None


def _reject(paths: RunPaths, iteration: int, candidate: Dict[str, Any],
            gate: str, error: str) -> None:
    """Record a candidate that never earned a full evaluation.

    Rejections go into evolution_summary.jsonl like any other row (eliminated,
    so the frontier ignores them) rather than being dropped: the proposer's only
    feedback channel is this filesystem, and "the code you wrote does not run"
    is exactly the kind of thing it needs to read next iteration.
    """
    _log(f"  FAIL {candidate['name']} [{gate}]: {error.splitlines()[-1][:160]}")
    state.append_row(paths, state.summary_row(
        iteration=iteration, system=candidate["name"], candidate=candidate,
        metrics={"raw_score": 0.0, "score_max": 1, "eliminated": True,
                 "error": f"{gate}: {error}"},
    ))


def _import_gate(candidates: List[Dict[str, Any]], paths: RunPaths,
                 iteration: int) -> List[Dict[str, Any]]:
    """Gate 1 — does the candidate import and expose a MemoClass?"""
    passed = []
    for candidate in candidates:
        error = evaluator.import_check(paths.harnesses / f"{candidate['name']}.py")
        if error is None:
            passed.append(candidate)
        else:
            _reject(paths, iteration, candidate, "import", error)
    return passed


async def _sanity_gate(candidates: List[Dict[str, Any]], paths: RunPaths,
                       cfg: Dict[str, Any], iteration: int) -> List[Dict[str, Any]]:
    """Gate 2 — ONE sanity_check-sized pass before a candidate earns a real one.

    Importing cleanly says nothing about surviving contact with real data. This
    is upstream's terminal-bench `smoke_test` and forge's sanity gate: the cost
    of finding out is one tiny pass instead of a full stage1, and it is what
    makes the `sanity_check` block in the `stages` config do something.
    """
    semaphore = asyncio.Semaphore(int(cfg["max_eval_concurrent"]))

    async def one(candidate: Dict[str, Any]) -> tuple:
        async with semaphore:
            name = candidate["name"]
            run_dir = paths.evals / name / "sanity"
            await evaluator.evaluate_candidate(
                name=name, harness_file=paths.harnesses / f"{name}.py",
                out_dir=run_dir, dataset=cfg["dataset"], split="search",
                cfg=cfg, step_index=iteration, smoke=True,
                timeout_s=SANITY_TIMEOUT_S,
            )
            return candidate, evaluator.sanity_errors(run_dir)

    passed = []
    for candidate, error in await asyncio.gather(*(one(c) for c in candidates)):
        if error is None:
            _log(f"  OK   {candidate['name']}")
            passed.append(candidate)
        else:
            _reject(paths, iteration, candidate, "sanity", error)
    return passed


async def _evaluate_all(
    *, names: List[str], paths: RunPaths, cfg: Dict[str, Any], split: str,
    out_root: Path, step_index: int,
) -> Dict[str, Dict[str, Any]]:
    """Score each named harness, at most `max_eval_concurrent` at a time."""
    semaphore = asyncio.Semaphore(int(cfg["max_eval_concurrent"]))

    async def one(name: str) -> tuple:
        async with semaphore:
            started = time.time()
            _log(f"  evaluating {name} ...")
            metrics = await evaluator.evaluate_candidate(
                name=name,
                harness_file=paths.harnesses / f"{name}.py",
                out_dir=out_root / name,
                dataset=cfg["dataset"], split=split, cfg=cfg,
                step_index=step_index, timeout_s=int(cfg["eval_timeout"]),
            )
            score = evaluator.normalized_score(metrics)
            note = f" [{metrics['error'][:80]}]" if metrics.get("error") else ""
            _log(f"  {name}: score={score:.3f} ({_elapsed(time.time() - started)}){note}")
            return name, metrics

    return dict(await asyncio.gather(*(one(n) for n in names)))


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------

def _seed_baselines(paths: RunPaths, cfg: Dict[str, Any]) -> None:
    """Copy the tracked baseline harnesses into this run's own harness dir.

    Every run gets its own copy so runs never share a directory: candidates from
    one run are invisible to another, and `--fresh` cannot delete a sibling
    run's work."""
    import shutil

    missing = []
    for name in cfg["baselines"]:
        src = paths.seed_harnesses / f"{name}.py"
        if not src.exists():
            missing.append(str(src))
            continue
        dst = paths.harnesses / f"{name}.py"
        if not dst.exists():
            shutil.copy2(src, dst)
    if missing:
        raise FileNotFoundError(f"baseline harness file(s) not found: {missing}")


async def _run_baselines(paths: RunPaths, cfg: Dict[str, Any]) -> None:
    names = list(cfg["baselines"])
    _log(f"phase 0: baselines {names}")
    results = await _evaluate_all(
        names=names, paths=paths, cfg=cfg, split="search",
        out_root=paths.evals, step_index=0,
    )
    for name in names:
        state.append_row(paths, state.summary_row(
            iteration=0, system=name, metrics=results[name],
        ))
    state.rebuild_frontier(paths)


async def _run_iteration(paths: RunPaths, cfg: Dict[str, Any], iteration: int,
                         known: set) -> Optional[str]:
    """Returns None when the proposer session succeeded, else why it failed."""
    best_before = state.best_score(paths)
    _log(f"iteration {iteration}  (best so far {best_before:.3f})")

    paths.pending.unlink(missing_ok=True)

    started = time.time()
    result = await proposer.propose(
        agent=cfg["agent"], model=cfg["agent_model"],
        system_prompt=PROPOSER_SYSTEM.read_text(encoding="utf-8"),
        task=_task_prompt(paths, cfg, iteration),
        cwd=BASELINE_ROOT, log_dir=paths.proposer_logs, name=f"iter{iteration}",
        timeout_s=int(cfg["propose_timeout"]), effort=cfg["agent_effort"],
        auth=cfg["agent_auth"],
    )
    usage = state.record_proposer_usage(
        paths, iteration, cfg["agent"], cfg["agent_model"], result)
    _log(f"  proposer finished in {_elapsed(time.time() - started)} "
         f"(exit={result.exit_code}, {len(result.tools)} tool calls, "
         f"{usage['input_tokens']:,} in / {usage['output_tokens']:,} out"
         + (f", ${usage['cost_usd']:.4f}" if usage["cost_usd"] else "") + ")")
    if not result.ok:
        # The reason usually arrives on the agent's event stream, not stderr.
        _log(f"  proposer FAILED: {result.failure}")
        _log(f"  session log: {result.log_dir}")

    registered, handoff_error = _read_pending(paths, known)
    if registered:
        _log(f"  gating {len(registered)} candidate(s): import, then sanity")
    candidates = await _sanity_gate(
        _import_gate(registered, paths, iteration), paths, cfg, iteration
    )
    if not candidates:
        state.rebuild_frontier(paths)   # rejections are rows too
        if not result.ok:
            return result.failure
        if handoff_error:
            # Nothing was recorded, so the proposer cannot learn from this and
            # will repeat it. Counts toward the consecutive-failure limit.
            _log(f"  HANDOFF FAILED: {handoff_error}")
            return handoff_error
        # Gate rejections ARE recorded, so the next iteration can read them.
        _log("  no candidate survived the gates — skipping to the next iteration")
        return None

    results = await _evaluate_all(
        names=[c["name"] for c in candidates], paths=paths, cfg=cfg, split="search",
        out_root=paths.evals, step_index=iteration,
    )
    for candidate in candidates:
        state.append_row(paths, state.summary_row(
            iteration=iteration, system=candidate["name"],
            metrics=results[candidate["name"]], candidate=candidate,
            best_score=best_before,
        ))
    frontier = state.rebuild_frontier(paths)
    best = frontier.get("best")
    if best and best["score"] > best_before:
        _log(f"  NEW BEST {best['system']} at {best['score']:.3f} "
             f"(+{best['score'] - best_before:.3f})")
    return None


async def evolve(paths: RunPaths, cfg: Dict[str, Any]) -> None:
    """Baselines, then `iterations` propose/evaluate rounds on the search split."""
    if state.is_finalized(paths):
        raise SystemExit(
            f"run '{paths.run_name}' is finalized — its test split is spent. "
            f"Use a new run_name to keep evolving."
        )
    paths.mkdirs()
    _log(f"meta-harness evolution  run={paths.run_name}  dataset={cfg['dataset']}  "
         f"agent={cfg['agent']}/{cfg['agent_model']}  iterations={cfg['iterations']}")

    # Before anything is evaluated: can the proposer run at all? Phase 0 costs
    # real tokens, and a run whose proposer cannot start is dead on arrival.
    _log(f"preflight: {cfg['agent']}/{cfg['agent_model'] or '(cli default)'} ...")
    error = await proposer.preflight(
        agent=cfg["agent"], model=cfg["agent_model"], cwd=BASELINE_ROOT,
        log_dir=paths.proposer_logs, effort=cfg["agent_effort"], auth=cfg["agent_auth"],
    )
    if error:
        raise SystemExit(
            f"proposer preflight FAILED — nothing was evaluated.\n"
            f"  {error}\n"
            f"  session log: {paths.proposer_logs / 'preflight'}\n"
            f"Check `agent_model` (model availability is account-scoped), that the "
            f"`{cfg['agent']}` CLI is on PATH, and that its login is current."
        )
    _log("preflight OK")

    _seed_baselines(paths, cfg)
    known = {row["system"] for row in state.read_rows(paths)}
    if not cfg["skip_baselines"]:
        await _run_baselines(paths, cfg)
        known |= set(cfg["baselines"])

    start = state.last_iteration(paths) + 1
    consecutive_failures = 0
    for offset in range(int(cfg["iterations"])):
        if _interrupted:
            break
        failure = await _run_iteration(paths, cfg, start + offset, known)
        # A bad model id, an expired login or a missing CLI fails identically
        # every time. Burning the whole iteration budget on it helps nobody.
        consecutive_failures = consecutive_failures + 1 if failure else 0
        if consecutive_failures >= MAX_PROPOSER_FAILURES:
            raise SystemExit(
                f"proposer failed {consecutive_failures} times in a row — stopping.\n"
                f"Last error: {failure}\n"
                f"Session logs: {paths.proposer_logs}"
            )

    best = state.rebuild_frontier(paths).get("best")
    if best:
        _log(f"evolution complete — best {best['system']} at {best['score']:.3f}")
    else:
        _log("evolution complete — nothing scored")
    total = state.proposer_usage_total(paths)
    if total["sessions"]:
        _log(f"search cost: {total['sessions']} proposer session(s), "
             f"{total['input_tokens']:,} in / {total['output_tokens']:,} out"
             + (f", ${total['cost_usd']:.2f}" if total["cost_usd"] else "")
             + f" ({paths.proposer_usage.name})")
    _log(f"finalize once with: uv run python run.py --config <cfg> "
         f"--status test --run-name {paths.run_name}")


async def finalize(paths: RunPaths, cfg: Dict[str, Any]) -> None:
    """One held-out evaluation of the Pareto frontier, then freeze the run.

    The test split is touched exactly once per run: this function refuses to
    run twice, and `evolve` refuses to run at all afterwards.
    """
    frontier = state.read_frontier(paths)
    if not frontier.get("all"):
        raise SystemExit(f"no search-split results for run '{paths.run_name}' — nothing to finalize")
    if state.is_finalized(paths):
        _log(f"run '{paths.run_name}' is already finalized:")
        _log(json.dumps(json.loads(paths.finalized.read_text(encoding="utf-8"))
                        .get("test_scores", {}), indent=2))
        return

    systems = sorted({e["system"] for e in frontier.get("_pareto", [])} | set(cfg["baselines"]))
    _log(f"finalizing run={paths.run_name} on the TEST split: {systems}")
    state.mark_finalizing(paths, systems)

    out_root = paths.test_results(cfg["dataset"])
    results = await _evaluate_all(
        names=systems, paths=paths, cfg=cfg, split="test",
        out_root=out_root, step_index=0,
    )
    scores = {name: evaluator.normalized_score(m) for name, m in results.items()}
    state.mark_finalized(paths, scores)

    _log("test results (mean reward, 0-1 — comparable with any forge harness):")
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        _log(f"  {score:.3f}  {name}")
    _log(f"artifacts: {out_root}")


def fresh_start(paths: RunPaths) -> None:
    """Delete this run. Harnesses live inside it, so nothing else is touched —
    a sibling run's candidates are unreachable from here by construction."""
    import shutil

    if paths.logs.exists():
        shutil.rmtree(paths.logs)
    _log(f"fresh start: removed {paths.logs}")


async def main(cfg: Dict[str, Any], run_name: str, fresh: bool) -> None:
    _install_signal_handlers()
    paths = RunPaths(root=BASELINE_ROOT, run_name=run_name)
    if fresh:
        fresh_start(paths)
    paths.mkdirs()
    if cfg["status"] == "search":
        await evolve(paths, cfg)
    else:
        await finalize(paths, cfg)
