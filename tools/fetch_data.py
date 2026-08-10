#!/usr/bin/env python3
"""Download the benchmark data files that are not tracked in git.

    python3 tools/fetch_data.py                  # all three benchmarks
    python3 tools/fetch_data.py --dataset locomo
    python3 tools/fetch_data.py --check          # report status, download nothing

Stdlib only — run it with any Python 3.9+, before or after `uv sync`.
Downloads are atomic (`.part` + rename) and idempotent: a file that already
exists and matches its expected digest is skipped, so re-running is cheap and
never clobbers good data. Pass `--force` to re-download anyway.

Every file is fetched VERBATIM from its upstream release — this script applies
no transformation of its own, and verifies every download against the
upstream's own digest (sha256 for HF LFS files, git blob sha1 otherwise).

Sources
-------
LoCoMo        github.com/snap-research/locomo, data/locomo10.json
              → benchmarks/locomo/locomo10.json                     (~2.7 MB)

DynamicMem    HF dataset xiewenya/dynamicmem — the official release the TCE v2
              protocol is defined against; 10 users × 2 files, and the repo
              layout maps 1:1 onto ours
              → benchmarks/dynamicmem/user_data/<NNN_user_NNN>/      (~77 MB)

LongMemEval   HF dataset xiaowu0162/longmemeval-**cleaned**
              → benchmarks/longmemeval/longmemeval_s_cleaned.json    (~277 MB)

Why the `-cleaned` LongMemEval repo, not `xiaowu0162/longmemeval`
-----------------------------------------------------------------
The `_cleaned` in the filename is the UPSTREAM author's, not ours. They publish
a second dataset that "replaces the original LongMemEval dataset ... removes
noisy history sessions that interfere with the answer correctness"
(https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). This repo has
always used that release verbatim — confirmed by sha256, see LONGMEMEVAL_SHA256
below. Fetching the raw `xiaowu0162/longmemeval` file instead would leave those
noisy sessions in the haystack and make scores incomparable with everything
recorded here.

Only the -S variant is fetched. The upstream -M variant (~2.7 GB) and the
oracle file are not used by this repo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO_ROOT / "benchmarks"

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
HF_PATHS_INFO = "https://huggingface.co/api/datasets/{repo}/paths-info/main"

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

DYNAMICMEM_REPO = "xiewenya/dynamicmem"
DYNAMICMEM_USERS = [f"{i:03d}_user_{i:03d}" for i in range(1, 11)]
DYNAMICMEM_FILES = ("app_log_large.json", "task_packs.json")

LONGMEMEVAL_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILE = "longmemeval_s_cleaned.json"
LONGMEMEVAL_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

# Pinned hashes are the point, not an optimization: they are what makes a fresh
# clone provably identical to the data behind every number recorded in this
# repo. A mismatch means upstream re-published — worth failing loudly over,
# not absorbing silently. DynamicMem's 20 hashes are fetched from the HF API at
# run time instead (too many to pin readably, and the API is authoritative).


# ---------------------------------------------------------------------------
# Download plumbing
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """Git's object id for a file: sha1(b"blob <size>\\0" + content).

    Needed because HF only reports a content sha256 for LFS-tracked files. A
    plain (non-LFS) file's `oid` is this git blob hash instead — same integrity
    guarantee, different preimage.
    """
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _hf_digests(repo: str, paths: List[str]) -> Dict[str, Tuple[str, str]]:
    """path -> (kind, expected_digest), kind in {"sha256", "git-blob-sha1"}."""
    req = urllib.request.Request(
        HF_PATHS_INFO.format(repo=repo),
        data=json.dumps({"paths": paths}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "memevol-fetch-data"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        entries = json.load(resp)
    out: Dict[str, Tuple[str, str]] = {}
    for e in entries:
        if e.get("lfs"):
            out[e["path"]] = ("sha256", e["lfs"]["oid"])
        elif e.get("oid"):
            out[e["path"]] = ("git-blob-sha1", e["oid"])
    return out


def _digest(path: Path, kind: str) -> str:
    return _sha256(path) if kind == "sha256" else _git_blob_sha1(path)


def _download(url: str, dest: Path, label: str, expect: Optional[str] = None,
              kind: str = "sha256", retries: int = 3) -> None:
    """Fetch `url` to `dest` atomically, verifying its digest before the rename."""
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
                            print(f"\r  {label}: {100 * done / total:5.1f}%  "
                                  f"{_human(done)} / {_human(total)}", end="", file=sys.stderr)
            if sys.stderr.isatty():
                print("\r" + " " * 72 + "\r", end="", file=sys.stderr)
            if total and tmp.stat().st_size != total:
                raise IOError(f"short read: got {tmp.stat().st_size}, expected {total}")
            if expect:
                got = _digest(tmp, kind)
                if got != expect:
                    raise IOError(
                        f"{kind} mismatch\n     expected {expect}\n     got      {got}\n"
                        f"     Upstream may have re-published this file. Do NOT mix it with "
                        f"existing results — the data behind them differs."
                    )
            tmp.replace(dest)
            return
        except (urllib.error.URLError, IOError, TimeoutError) as exc:
            last_err = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                print(f"  {label}: attempt {attempt} failed ({exc}); retrying", file=sys.stderr)
    raise RuntimeError(f"{label}: download failed after {retries} attempts: {last_err}")


def _verify(path: Path, expect: str, kind: str = "sha256") -> Tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    got = _digest(path, kind)
    return (True, f"{kind} OK") if got == expect else (False, f"{kind} differs ({got[:12]}…)")


# ---------------------------------------------------------------------------
# Per-dataset fetchers — each returns a one-line status string
# ---------------------------------------------------------------------------

def fetch_locomo(force: bool) -> str:
    dest = BENCHMARKS / "locomo" / "locomo10.json"
    ok, why = _verify(dest, LOCOMO_SHA256)
    if ok and not force:
        return f"present ({why})"
    _download(LOCOMO_URL, dest, "locomo10.json", LOCOMO_SHA256)
    return "downloaded (sha256 OK)"


def fetch_dynamicmem(force: bool) -> str:
    root = BENCHMARKS / "dynamicmem" / "user_data"
    rel = [f"{u}/{n}" for u in DYNAMICMEM_USERS for n in DYNAMICMEM_FILES]
    expected = _hf_digests(DYNAMICMEM_REPO, rel)
    absent = [p for p in rel if p not in expected]
    if absent:
        raise RuntimeError(f"HF metadata missing for {absent[:3]} — repo layout changed?")

    got, skipped = 0, 0
    for p in rel:
        dest = root / p
        kind, digest = expected[p]
        ok, _ = _verify(dest, digest, kind)
        if ok and not force:
            skipped += 1
            continue
        _download(HF_RESOLVE.format(repo=DYNAMICMEM_REPO, path=p), dest, p, digest, kind)
        got += 1
    return f"{len(DYNAMICMEM_USERS)} users, {len(rel)} files ({got} downloaded, {skipped} verified)"


def fetch_longmemeval(force: bool) -> str:
    dest = BENCHMARKS / "longmemeval" / LONGMEMEVAL_FILE
    ok, why = _verify(dest, LONGMEMEVAL_SHA256)
    if ok and not force:
        return f"present ({why})"
    url = HF_RESOLVE.format(repo=LONGMEMEVAL_REPO, path=LONGMEMEVAL_FILE)
    _download(url, dest, f"{LONGMEMEVAL_FILE} (~277 MB)", LONGMEMEVAL_SHA256)
    return "downloaded (sha256 OK)"


FETCHERS: Dict[str, Callable[[bool], str]] = {
    "locomo": fetch_locomo,
    "dynamicmem": fetch_dynamicmem,
    "longmemeval": fetch_longmemeval,
}


# ---------------------------------------------------------------------------
# --check: verify what is on disk, download nothing
# ---------------------------------------------------------------------------

def check_locomo() -> str:
    ok, why = _verify(BENCHMARKS / "locomo" / "locomo10.json", LOCOMO_SHA256)
    if not ok:
        raise ValueError(why)
    return why


def check_longmemeval() -> str:
    ok, why = _verify(BENCHMARKS / "longmemeval" / LONGMEMEVAL_FILE, LONGMEMEVAL_SHA256)
    if not ok:
        raise ValueError(why)
    return why


def check_dynamicmem() -> str:
    root = BENCHMARKS / "dynamicmem" / "user_data"
    rel = [f"{u}/{n}" for u in DYNAMICMEM_USERS for n in DYNAMICMEM_FILES]
    expected = _hf_digests(DYNAMICMEM_REPO, rel)
    bad = []
    for p in rel:
        kind, digest = expected.get(p, ("sha256", ""))
        if not _verify(root / p, digest, kind)[0]:
            bad.append(p)
    if bad:
        raise ValueError(f"{len(bad)}/{len(rel)} files missing or mismatched, e.g. {bad[:2]}")
    return f"{len(rel)} files, digest OK"


CHECKS: Dict[str, Callable[[], str]] = {
    "locomo": check_locomo,
    "dynamicmem": check_dynamicmem,
    "longmemeval": check_longmemeval,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download memevol's benchmark data (not tracked in git). "
                    "Every file is fetched verbatim and verified by sha256.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Disk: ~360 MB total (locomo 2.7 MB, dynamicmem 77 MB, longmemeval 277 MB).",
    )
    ap.add_argument("--dataset", default="all",
                    help="comma-separated: locomo,dynamicmem,longmemeval (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the local file already verifies")
    ap.add_argument("--check", action="store_true",
                    help="verify what is present; download nothing")
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
