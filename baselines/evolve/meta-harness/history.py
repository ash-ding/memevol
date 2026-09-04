"""Query a run's search history.

Appendix D of the paper recommends this: raw filesystem access alone gets
cumbersome as the history grows, and "a short CLI that lists the Pareto
frontier, shows top-k harnesses, and diffs code and results between pairs of
runs" saves the proposer tokens it would otherwise spend navigating. Querying a
CLI is also closer to what coding agents are trained on than walking a tree.

    uv run python history.py frontier            # Pareto front + best
    uv run python history.py top -k 10           # ranked by score
    uv run python history.py show <name>         # one harness: row, paths, code
    uv run python history.py diff <a> <b>        # code + results, side by side

`--run <name>` picks the run; without it, the most recently modified one under
logs/ is used.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parent
for _p in (str(BASELINE_ROOT.parents[2]), str(BASELINE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import state
from state import RunPaths


def _resolve_run(name: str | None) -> RunPaths:
    logs = BASELINE_ROOT / "logs"
    if name:
        return RunPaths(root=BASELINE_ROOT, run_name=name)
    runs = [d for d in logs.glob("*") if d.is_dir()] if logs.exists() else []
    if not runs:
        raise SystemExit(f"no runs under {logs} — pass --run <name>")
    return RunPaths(root=BASELINE_ROOT, run_name=max(runs, key=lambda d: d.stat().st_mtime).name)


def _rows_by_system(paths: RunPaths) -> dict:
    """Newest row per system."""
    return {row["system"]: row for row in state.read_rows(paths)}


def _table(rows: list) -> str:
    if not rows:
        return "  (none)"
    width = max(len(r["system"]) for r in rows)
    return "\n".join(
        f"  {r['system']:<{width}}  score {r.get('score', 0):.3f}"
        f"  ctx {r.get('context_cost', 0):>8.0f}"
        f"  iter {r.get('iteration', 0):>3}"
        + (f"  [{str(r['error'])[:60]}]" if r.get("error") else "")
        for r in rows
    )


def cmd_frontier(paths: RunPaths, _args) -> None:
    frontier = state.read_frontier(paths)
    if not frontier:
        raise SystemExit(f"no frontier yet for run {paths.run_name!r}")
    best = frontier.get("best")
    print(f"run {paths.run_name}  ({frontier.get('updated_at', '?')})")
    print(f"\nbest: {best['system']} at {best['score']:.3f}" if best else "\nbest: none")
    print("\npareto front (score up, context cost down):")
    print(_table(frontier.get("_pareto", [])))


def cmd_top(paths: RunPaths, args) -> None:
    rows = sorted(_rows_by_system(paths).values(),
                  key=lambda r: -r.get("score", 0.0))[: args.k]
    print(f"top {len(rows)} of run {paths.run_name} by score:")
    print(_table(rows))


def cmd_show(paths: RunPaths, args) -> None:
    row = _rows_by_system(paths).get(args.name)
    if row is None:
        raise SystemExit(f"no row for {args.name!r} in run {paths.run_name}")
    print(f"{args.name}")
    for key in ("iteration", "score", "delta", "context_cost", "stage",
                "eliminated", "tokens", "axis", "base_system"):
        if row.get(key) is not None:
            print(f"  {key:<13} {row[key]}")
    for key in ("hypothesis", "error"):
        if row.get(key):
            print(f"  {key:<13} {row[key]}")

    evals = paths.evals / args.name
    print(f"\n  source        {paths.harnesses / f'{args.name}.py'}")
    print(f"  artifacts     {evals}")
    for artifact in ("score.json", "metrics.json", "stages.json", "traces", "sanity"):
        if (evals / artifact).exists():
            print(f"                  {artifact}")


def cmd_diff(paths: RunPaths, args) -> None:
    rows = _rows_by_system(paths)
    print(f"results ({paths.run_name}):")
    print(_table([rows[n] for n in (args.a, args.b) if n in rows]))

    left, right = (paths.harnesses / f"{n}.py" for n in (args.a, args.b))
    missing = [str(p) for p in (left, right) if not p.exists()]
    if missing:
        raise SystemExit(f"\nmissing source: {', '.join(missing)}")

    print(f"\ncode diff ({args.a} -> {args.b}):")
    diff = difflib.unified_diff(
        left.read_text(encoding="utf-8").splitlines(),
        right.read_text(encoding="utf-8").splitlines(),
        fromfile=args.a, tofile=args.b, lineterm="",
    )
    body = "\n".join(diff)
    print(body if body else "  (identical)")


def main() -> None:
    p = argparse.ArgumentParser(description="Query a meta-harness run's history.")
    p.add_argument("--run", default=None, help="run name (default: most recent under logs/)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("frontier", help="Pareto front + best").set_defaults(fn=cmd_frontier)

    top = sub.add_parser("top", help="harnesses ranked by score")
    top.add_argument("-k", type=int, default=10)
    top.set_defaults(fn=cmd_top)

    show = sub.add_parser("show", help="one harness: row, paths, artifacts")
    show.add_argument("name")
    show.set_defaults(fn=cmd_show)

    diff = sub.add_parser("diff", help="two harnesses: results + code diff")
    diff.add_argument("a")
    diff.add_argument("b")
    diff.set_defaults(fn=cmd_diff)

    args = p.parse_args()
    args.fn(_resolve_run(args.run), args)


if __name__ == "__main__":
    main()
