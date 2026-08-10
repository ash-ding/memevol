#!/usr/bin/env python3
"""Download the benchmark data files that are not tracked in git.

    python3 tools/fetch_data.py                  # all three benchmarks
    python3 tools/fetch_data.py --dataset locomo
    python3 tools/fetch_data.py --check          # report status, download nothing

Stdlib only — run it with any Python 3.9+, before or after `uv sync`.
Downloads are atomic (`.part` + rename) and idempotent: a file that already
exists and passes its structural check is skipped, so re-running is cheap and
never clobbers good data. Pass `--force` to re-download anyway.

Sources
-------
LoCoMo        github.com/snap-research/locomo, data/locomo10.json
              → benchmarks/locomo/locomo10.json          (~2.7 MB, verbatim)

DynamicMem    HF dataset xiewenya/dynamicmem (the official release used by the
              TCE v2 protocol), 10 users × {app_log_large,task_packs}.json
              → benchmarks/dynamicmem/user_data/<NNN_user_NNN>/   (~77 MB, verbatim)

LongMemEval   HF dataset xiaowu0162/longmemeval, file `longmemeval_s`
              → benchmarks/longmemeval/longmemeval_s_cleaned.json (~277 MB)
              NOT verbatim — see "The LongMemEval cleaning step" below.

Only the -S variant is fetched. The upstream -M variant (~2.7 GB) and the
oracle file are deliberately not used by this repo.

The LongMemEval cleaning step
-----------------------------
The filename this repo reads (`..._cleaned.json`) is a DERIVED artifact: the
raw upstream file contains haystack sessions with **zero messages**, carried in
the three parallel lists (`haystack_dates` / `haystack_session_ids` /
`haystack_sessions`). An empty session is not evidence and not a distractor —
it is a hole that every memory system has to special-case — so the cleaning
step drops those entries from all three lists in lockstep.

Honest caveat: this rule was recovered by differencing the repo's existing file
against upstream, not from a recorded preprocessing script (none exists in the
repo's history). On a 12-question sample it reproduces 11 exactly; one question
(`118b2229`) has a further session removed upstream-side that no rule derived
from the data explains. So a file produced here can differ from the one used
for previously recorded numbers by a small number of sessions. `--check`
reports how many empty sessions a local file still carries — a non-zero count
means it is raw upstream, not cleaned.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

DYNAMICMEM_REPO = "xiewenya/dynamicmem"
DYNAMICMEM_USERS = [f"{i:03d}_user_{i:03d}" for i in range(1, 11)]
DYNAMICMEM_FILES = ("app_log_large.json", "task_packs.json")

LONGMEMEVAL_REPO = "xiaowu0162/longmemeval"
LONGMEMEVAL_FILE = "longmemeval_s"

# Structural invariants — cheaper and more durable than pinning byte sizes,
# which would rot the moment upstream re-uploads an identical-content file.
EXPECT_LOCOMO_CONVERSATIONS = 10
EXPECT_LONGMEMEVAL_QUESTIONS = 500


# ---------------------------------------------------------------------------
# Download plumbing
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, label: str, retries: int = 3) -> None:
    """Fetch `url` to `dest` atomically, with a progress line and retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "memevol-fetch-data"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as fh:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total and sys.stderr.isatty():
                            pct = 100 * done / total
                            print(f"\r  {label}: {pct:5.1f}%  {_human(done)} / {_human(total)}",
                                  end="", file=sys.stderr)
            if sys.stderr.isatty():
                print("\r" + " " * 70 + "\r", end="", file=sys.stderr)
            if total and tmp.stat().st_size != total:
                raise IOError(f"short read: got {tmp.stat().st_size}, expected {total}")
            tmp.replace(dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                print(f"  {label}: attempt {attempt} failed ({exc}); retrying", file=sys.stderr)
    raise RuntimeError(f"{label}: download failed after {retries} attempts: {last_err}")


def _load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# LongMemEval cleaning
# ---------------------------------------------------------------------------

def clean_longmemeval(raw: List[Dict]) -> Tuple[List[Dict], int]:
    """Drop zero-message haystack sessions from every question.

    The three haystack lists are parallel, so an index is removed from all of
    them or none — dropping from one alone would silently mis-align dates and
    ids against sessions. Returns (cleaned, n_sessions_dropped).
    """
    dropped = 0
    out: List[Dict] = []
    for q in raw:
        dates = q.get("haystack_dates", [])
        ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])
        if not (len(dates) == len(ids) == len(sessions)):
            raise ValueError(
                f"{q.get('question_id')}: parallel haystack lists disagree "
                f"({len(dates)}/{len(ids)}/{len(sessions)}) — upstream format changed"
            )
        keep = [i for i, msgs in enumerate(sessions) if msgs]
        dropped += len(sessions) - len(keep)
        cleaned = dict(q)
        cleaned["haystack_dates"] = [dates[i] for i in keep]
        cleaned["haystack_session_ids"] = [ids[i] for i in keep]
        cleaned["haystack_sessions"] = [sessions[i] for i in keep]
        out.append(cleaned)
    return out, dropped


# ---------------------------------------------------------------------------
# Per-dataset fetchers — each returns a one-line status string
# ---------------------------------------------------------------------------

def _check_locomo(path: Path) -> str:
    data = _load_json(path)
    if len(data) != EXPECT_LOCOMO_CONVERSATIONS:
        raise ValueError(f"expected {EXPECT_LOCOMO_CONVERSATIONS} conversations, got {len(data)}")
    return f"{len(data)} conversations"


def fetch_locomo(force: bool) -> str:
    dest = BENCHMARKS / "locomo" / "locomo10.json"
    if dest.exists() and not force:
        return f"present ({_check_locomo(dest)})"
    _download(LOCOMO_URL, dest, "locomo10.json")
    return f"downloaded ({_check_locomo(dest)})"


def fetch_dynamicmem(force: bool) -> str:
    root = BENCHMARKS / "dynamicmem" / "user_data"
    got, skipped = 0, 0
    for user in DYNAMICMEM_USERS:
        for name in DYNAMICMEM_FILES:
            dest = root / user / name
            if dest.exists() and not force:
                skipped += 1
                continue
            url = HF_RESOLVE.format(repo=DYNAMICMEM_REPO, path=f"{user}/{name}")
            _download(url, dest, f"{user}/{name}")
            _load_json(dest)          # structural check: must parse
            got += 1
    missing = [u for u in DYNAMICMEM_USERS
               if not all((root / u / n).exists() for n in DYNAMICMEM_FILES)]
    if missing:
        raise RuntimeError(f"incomplete: {missing}")
    return f"{len(DYNAMICMEM_USERS)} users ({got} downloaded, {skipped} already present)"


def _check_longmemeval(path: Path) -> str:
    data = _load_json(path)
    if len(data) != EXPECT_LONGMEMEVAL_QUESTIONS:
        raise ValueError(f"expected {EXPECT_LONGMEMEVAL_QUESTIONS} questions, got {len(data)}")
    empties = sum(1 for q in data for s in q.get("haystack_sessions", []) if not s)
    note = f", {empties} empty sessions still present (NOT cleaned)" if empties else ""
    return f"{len(data)} questions{note}"


def fetch_longmemeval(force: bool) -> str:
    dest = BENCHMARKS / "longmemeval" / "longmemeval_s_cleaned.json"
    if dest.exists() and not force:
        return f"present ({_check_longmemeval(dest)})"

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HF_RESOLVE.format(repo=LONGMEMEVAL_REPO, path=LONGMEMEVAL_FILE)
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as td:
        raw_path = Path(td) / "longmemeval_s.raw.json"
        _download(url, raw_path, "longmemeval_s (~277 MB)")
        print("  parsing + cleaning (this needs ~3 GB RAM briefly) ...", file=sys.stderr)
        raw = _load_json(raw_path)
        cleaned, dropped = clean_longmemeval(raw)
        tmp = Path(td) / "cleaned.json"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cleaned, fh, indent=4, ensure_ascii=False)
        tmp.replace(dest)
    return f"downloaded + cleaned ({len(cleaned)} questions, {dropped} empty sessions dropped)"


FETCHERS: Dict[str, Callable[[bool], str]] = {
    "locomo": fetch_locomo,
    "dynamicmem": fetch_dynamicmem,
    "longmemeval": fetch_longmemeval,
}

CHECKS: Dict[str, Callable[[], str]] = {
    "locomo": lambda: _check_locomo(BENCHMARKS / "locomo" / "locomo10.json"),
    "dynamicmem": lambda: _check_dynamicmem_present(),
    "longmemeval": lambda: _check_longmemeval(
        BENCHMARKS / "longmemeval" / "longmemeval_s_cleaned.json"),
}


def _check_dynamicmem_present() -> str:
    root = BENCHMARKS / "dynamicmem" / "user_data"
    have = [u for u in DYNAMICMEM_USERS
            if all((root / u / n).is_file() for n in DYNAMICMEM_FILES)]
    if len(have) != len(DYNAMICMEM_USERS):
        raise ValueError(f"{len(have)}/{len(DYNAMICMEM_USERS)} users complete")
    return f"{len(have)} users"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download memevol's benchmark data (not tracked in git).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Disk: ~360 MB total (locomo 2.7 MB, dynamicmem 77 MB, longmemeval 277 MB).",
    )
    ap.add_argument("--dataset", default="all",
                    help="comma-separated: locomo,dynamicmem,longmemeval (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is already present and valid")
    ap.add_argument("--check", action="store_true",
                    help="report what is present and valid; download nothing")
    args = ap.parse_args()

    names = sorted(FETCHERS) if args.dataset == "all" else \
        [d.strip() for d in args.dataset.split(",") if d.strip()]
    unknown = [n for n in names if n not in FETCHERS]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {sorted(FETCHERS)}", file=sys.stderr)
        return 2

    failed = []
    for name in names:
        try:
            if args.check:
                print(f"  {name:12s} OK — {CHECKS[name]()}")
            else:
                print(f"{name} ...", file=sys.stderr)
                print(f"  {name:12s} {FETCHERS[name](args.force)}")
        except FileNotFoundError:
            print(f"  {name:12s} MISSING")
            failed.append(name)
        except Exception as exc:                     # noqa: BLE001 — report, keep going
            print(f"  {name:12s} FAILED — {exc}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} dataset(s) not ready: {failed}", file=sys.stderr)
        return 1
    print("\nAll requested data is in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
