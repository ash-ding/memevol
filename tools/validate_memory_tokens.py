#!/usr/bin/env python3
"""Validate the memory-token counter against traces produced by a real run.

    python tools/validate_memory_tokens.py <run_dir> [--model gpt-5-mini]
    python tools/validate_memory_tokens.py <run_dir> --json

`<run_dir>` is any directory containing `traces/*.json` (an evaluate_memo
output root, or one of its `<stage>/` subdirectories).

Why this exists
---------------
`common/workflow.py` measures memory tokens DIFFERENTIALLY at prompt
assembly: tokens(prompt with the retrieved memory) - tokens(the same prompt
with nothing retrieved). That is the number that is true by construction, but
it is computed live and there is nothing to check it against — except that
every trace also persists `steps[].retrieved_memory`, the payload itself.
This script re-counts that payload offline (definition (A)) and compares.

Reading the output
------------------
The two numbers are NOT expected to be equal, and a mismatch is not a bug:

  * the payload count omits the template scaffolding the benchmark wraps
    around the memory, and
  * splicing text into a template changes tokenization at the seams.

What validates the counter is that the two track each other: a high
correlation and a small, STABLE offset. What would indicate a real problem is
a near-zero correlation (the counter is measuring something unrelated), a
differential of 0 where a payload exists (the memory never reached the
prompt), or a wildly varying ratio (the measurement point drifted from where
the prompt is actually built).

Traces written before the memory-token counter landed carry no
`memory_tokens` field; those steps are reported as unmeasured rather than
counted as zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.tokens import count_text_tokens  # noqa: E402


def _payload_text(retrieved: Any) -> str:
    """Flatten a `retrieved_memory` value to the text a prompt would render.

    Harnesses return wildly different shapes (a string, {"memories": str},
    lists of chunks, dicts of scored facts). Strings and numbers are kept;
    container structure is walked. Keys are NOT included — they are addressing,
    not content, and a benchmark's template rarely renders them verbatim.
    """
    if retrieved is None:
        return ""
    if isinstance(retrieved, str):
        return retrieved
    if isinstance(retrieved, (int, float, bool)):
        return str(retrieved)
    if isinstance(retrieved, dict):
        return "\n".join(_payload_text(v) for v in retrieved.values())
    if isinstance(retrieved, (list, tuple, set)):
        return "\n".join(_payload_text(v) for v in retrieved)
    return str(retrieved)


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def collect(run_dir: Path, model: str) -> Dict[str, Any]:
    traces_dir = run_dir / "traces" if (run_dir / "traces").is_dir() else run_dir
    files = sorted(traces_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no trace files found under {traces_dir}")

    measured: List[Tuple[int, int]] = []   # (differential, payload), paired
    all_payloads: List[float] = []         # every step, measured or not
    unmeasured = 0
    unmeasured_with_payload = 0
    counted_but_empty_payload = 0
    n_steps = 0

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  skipping unreadable trace {f.name}: {exc}", file=sys.stderr)
            continue
        for step in data.get("steps", []):
            n_steps += 1
            payload = count_text_tokens(_payload_text(step.get("retrieved_memory")), model)
            all_payloads.append(float(payload))
            if "memory_tokens" not in step:
                unmeasured += 1
                if payload:
                    unmeasured_with_payload += 1
                continue
            diff = int(step["memory_tokens"])
            measured.append((diff, payload))
            if payload and diff == 0:
                counted_but_empty_payload += 1

    diffs = [float(d) for d, _ in measured]
    payloads = [float(p) for _, p in measured]
    ratios = [d / p for d, p in measured if p > 0]

    return {
        "run_dir": str(run_dir),
        "model": model,
        "n_traces": len(files),
        "n_steps": n_steps,
        "n_measured": len(measured),
        "n_unmeasured": unmeasured,
        "unmeasured_with_payload": unmeasured_with_payload,
        "counted_zero_despite_payload": counted_but_empty_payload,
        "differential_total": int(sum(diffs)),
        # Over the PAIRED steps (comparable to differential_total)...
        "payload_total": int(sum(payloads)),
        # ...and over every step, so old traces still yield an estimate.
        "payload_total_all_steps": int(sum(all_payloads)),
        "differential_per_query": (sum(diffs) / len(diffs)) if diffs else 0.0,
        "payload_per_query": (sum(payloads) / len(payloads)) if payloads else 0.0,
        "payload_per_query_all_steps": (
            (sum(all_payloads) / len(all_payloads)) if all_payloads else 0.0),
        "correlation": _pearson(diffs, payloads),
        "ratio_min": min(ratios) if ratios else None,
        "ratio_max": max(ratios) if ratios else None,
        "ratio_mean": (sum(ratios) / len(ratios)) if ratios else None,
    }


def report(r: Dict[str, Any]) -> int:
    print(f"run:            {r['run_dir']}")
    print(f"traces:         {r['n_traces']} files, {r['n_steps']} steps")
    print(f"measured:       {r['n_measured']}   unmeasured: {r['n_unmeasured']}"
          f" ({r['unmeasured_with_payload']} of those DID retrieve something)")
    print()
    if r["n_measured"]:
        print(f"differential (live counter):   {r['differential_total']:>10} total"
              f"  {r['differential_per_query']:>9.1f} / query")
        print(f"payload      (offline recount):{r['payload_total']:>10} total"
              f"  {r['payload_per_query']:>9.1f} / query   [paired steps only]")
    print(f"payload, ALL steps:            {r['payload_total_all_steps']:>10} total"
          f"  {r['payload_per_query_all_steps']:>9.1f} / query")
    corr = r["correlation"]
    print()
    print(f"correlation:    {'n/a' if corr is None else f'{corr:.4f}'}")
    if r["ratio_mean"] is not None:
        print(f"diff/payload:   mean {r['ratio_mean']:.3f}  "
              f"range {r['ratio_min']:.3f}–{r['ratio_max']:.3f}")

    # Verdict. These thresholds are diagnostics, not a spec — the two
    # definitions legitimately differ (see the module docstring).
    problems = []
    if r["n_measured"] == 0:
        problems.append("no step carries a memory_tokens field — traces predate "
                        "the counter, or the measurement point never ran")
    if corr is not None and corr < 0.9:
        problems.append(f"weak correlation ({corr:.3f}) — the counter may not be "
                        f"measuring the retrieved payload")
    if r["counted_zero_despite_payload"]:
        problems.append(f"{r['counted_zero_despite_payload']} step(s) retrieved a "
                        f"non-empty payload but added 0 tokens to the prompt — the "
                        f"memory may not be reaching the QA prompt")
    print()
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        return 1
    print("  OK — the live differential tracks the offline payload recount.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="dir containing traces/ (or the traces dir itself)")
    ap.add_argument("--model", default="gpt-5-mini",
                    help="QA model whose encoding to count with (default: gpt-5-mini)")
    ap.add_argument("--json", action="store_true", help="emit the raw numbers as JSON")
    args = ap.parse_args()

    r = collect(args.run_dir, args.model)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0
    return report(r)


if __name__ == "__main__":
    sys.exit(main())
