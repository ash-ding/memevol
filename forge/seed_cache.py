"""`seeds/` — harnesses that have already been evaluated, and what they scored.

A search run throws its workspace away, so the same seed was re-evaluated by
every new run: the same code, the same benchmark, the same sizing, the same
models, and the same bill. This module keeps those results instead.

Layout — code once, results once per configuration::

    seeds/<harness hash>/
        harness.py  meta.json  requirements.txt   the code (identical to the
                                                  workspace dir it came from)
        index.json                                {harness_id, first_seen, evals}
        evals/<eval key>/
            manifest.json                         WHAT was run (see below)
            <dataset>/score.json, stages.json, token_usage.json,
                      run_record.json, traces/…   the complete artifacts

WHAT MAKES RESULTS REUSABLE is the whole question, so the eval key is a hash
of everything that can change a number:

  * the harness code            (already the directory name)
  * dataset, split              obviously
  * the sizing spec             10 queries and 200 queries are different runs,
                                and per-query cost is amortized over them
  * model, judge_model          a different answerer or judge is a different
                                score
  * progressive                 stage sizes differ from the single pass
  * random_sample, sampling_seed  which subset each step drew
  * the repo's git commit       the evaluation code itself — judging, prompts,
                                token accounting. Cheap to include and the
                                only honest way to avoid serving numbers that
                                predate a change in how they are computed

A dirty working tree gets `<sha>-dirty.<digest>`, where the digest covers the
uncommitted change itself — so two runs of the SAME edit reuse each other's
numbers, and a run of a DIFFERENT edit does not. A bare `-dirty` marker cannot
do that: it is one constant string, so every uncommitted state collides with
every other.

"Dirty" is judged over the evaluation code only (see `_CODE_PATHS`). The cache
writes `seeds/`, which is git-tracked by design but uncommitted until someone
commits it — counting that as an edit to the evaluation code would mark every
run after the first one dirty, for no reason anyone could act on.

`manifest.json` stores those inputs in readable form next to the hash, so a
cache entry can always be explained ("which run produced this, and under what
config") rather than just trusted.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from common.logger import get_logger
from forge.paths import PROJECT_ROOT, SEEDS_DIR

log = get_logger("main")



#: Per-dataset artifacts worth keeping. Traces are big and kept anyway — they
#: are the only record of what the harness actually answered, which is the
#: thing you want when a cached score looks surprising.
_RESULT_DIR_NAME = "evals"


#: What counts as "the evaluation code" when deciding whether the tree is
#: dirty. Everything else at the top level is inputs or outputs: `seeds/` is
#: this module's own output, `workspace/` is per-run scratch, and the data
#: files are not code. Naming the code explicitly keeps the cache from
#: declaring itself a change to the thing it is versioning.
_CODE_PATHS = ("forge", "common", "benchmarks", "baselines", "tools", "tracing")


def _git(*args: str, timeout: int = 30) -> Optional[str]:
    """Run one git command in the project root; None if git cannot answer."""
    try:
        r = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                           capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    return r.stdout if r.returncode == 0 else None


def repo_version() -> str:
    """The commit this evaluation code is at; `<sha>-dirty.<digest>` when edited.

    Falls back to "unknown" outside a git checkout (a tarball deployment),
    which then never matches another key — the safe direction.
    """
    sha = _git("rev-parse", "--short=12", "HEAD")
    if sha is None:
        return "unknown"
    sha = sha.strip()

    status = _git("status", "--porcelain", "--", *_CODE_PATHS)
    if status is None:
        return "unknown"
    if not status.strip():
        return sha

    # Identify the edit, not just its existence: the tracked diff, plus the
    # bytes of anything untracked (porcelain never lists ignored files, so
    # this is new source, not build output).
    h = hashlib.sha256()
    h.update(status.encode("utf-8", "replace"))
    h.update((_git("diff", "HEAD", "--", *_CODE_PATHS) or "").encode("utf-8", "replace"))
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = PROJECT_ROOT / line[3:].strip().strip('"')
        for f in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if not f.is_file():
                continue
            h.update(str(f.relative_to(PROJECT_ROOT)).encode("utf-8", "replace"))
            try:
                h.update(f.read_bytes())
            except OSError:
                h.update(b"<unreadable>")
    return f"{sha}-dirty.{h.hexdigest()[:12]}"


def eval_inputs(cfg: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    """The configuration a result depends on, in readable form."""
    ds_cfg = dict((cfg.get("datasets") or {}).get(dataset) or {})
    progressive = bool(cfg.get("progressive", False))
    return {
        "dataset": dataset,
        "split": cfg.get("split", "search"),
        "progressive": progressive,
        "sizing": ds_cfg.get("stages") if progressive else ds_cfg.get("single_stage"),
        "model": cfg.get("model"),
        "judge_model": ds_cfg.get("judge_model") or cfg.get("judge_model"),
        "random_sample": bool(cfg.get("random_sample", False)),
        "sampling_seed": cfg.get("sampling_seed") if cfg.get("random_sample") else None,
        "repo_version": repo_version(),
    }


def eval_key(inputs: Dict[str, Any]) -> str:
    """Stable 16-hex digest of `eval_inputs` (sorted JSON, so key order and
    dict insertion order cannot change it)."""
    blob = json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _entry_dir(harness_id: str) -> Path:
    return SEEDS_DIR / harness_id


def lookup(harness_id: str, key: str) -> Optional[Path]:
    """The cached result dir for this (code, configuration), or None."""
    candidate = _entry_dir(harness_id) / _RESULT_DIR_NAME / key
    return candidate if (candidate / "manifest.json").exists() else None


def _copy_code(harness_dir: Path, dst: Path) -> None:
    """Code in, results out — see `forge.orchestrator._copy_harness_code`,
    which owns the rule (a harness can be one file or a whole package tree)."""
    from forge.orchestrator import _copy_harness_code
    _copy_harness_code(harness_dir, dst)


def store(harness_dir: Path, harness_id: str, cfg: Dict[str, Any],
          per_ds: Dict[str, Dict[str, Any]], *, sanity_status: str,
          run_id: str) -> List[Path]:
    """Cache this harness's code and its per-dataset results. Returns the
    result dirs written (one per dataset).

    Never raises: a cache write failing must not fail a search that already
    produced its results — it is an optimisation, not the outcome. Failures
    are logged loudly enough to notice.

    One entry per (harness, dataset, configuration): the same harness
    evaluated on two benchmarks, or at two sizings, keeps both.
    """
    written: List[Path] = []
    try:
        entry = _entry_dir(harness_id)
        _copy_code(harness_dir, entry)

        for dataset, metrics in per_ds.items():
            src = harness_dir / dataset
            if not (src / "score.json").exists():
                continue                     # nothing was produced for it
            inputs = eval_inputs(cfg, dataset)
            key = eval_key(inputs)
            dst = entry / _RESULT_DIR_NAME / key
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
            manifest = {
                "harness_id": harness_id,
                "eval_key": key,
                "inputs": inputs,
                "raw_score": metrics.get("raw_score"),
                "score_max": metrics.get("score_max"),
                "stage": metrics.get("stage"),
                "cost_tokens_per_query": metrics.get("cost_tokens_per_query"),
                "sanity": sanity_status,
                "run_id": run_id,
                "stored_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            with (dst / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            written.append(dst)
            log.info(f"seed cache: stored {harness_id}/{dataset} (key {key})")

        if written:
            _write_index(entry, harness_id)
    except Exception as exc:                 # noqa: BLE001 — see docstring
        log.warning(f"seed cache: could not store {harness_id}: {exc}")
    return written


def _write_index(entry: Path, harness_id: str) -> None:
    """Rebuild `index.json` from the manifests actually on disk, so it can
    never drift from them."""
    evals = []
    for manifest_path in sorted((entry / _RESULT_DIR_NAME).glob("*/manifest.json")):
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        evals.append({
            "eval_key": manifest.get("eval_key", manifest_path.parent.name),
            "dataset": (manifest.get("inputs") or {}).get("dataset"),
            "raw_score": manifest.get("raw_score"),
            "stored_at": manifest.get("stored_at"),
        })
    index_path = entry / "index.json"
    first_seen = None
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                first_seen = json.load(f).get("first_seen")
        except Exception:
            first_seen = None
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({
            "harness_id": harness_id,
            "first_seen": first_seen or _dt.datetime.now().isoformat(timespec="seconds"),
            "evals": evals,
        }, f, indent=2, ensure_ascii=False)


def results_for(harness_id: str, cfg: Dict[str, Any],
                datasets: List[str]) -> Optional[Dict[str, Path]]:
    """Cached result dirs for EVERY dataset this run needs, or None.

    All-or-nothing on purpose: a harness with locomo cached but dynamicmem
    missing still has to be evaluated, and mixing a cached score with a fresh
    one would produce an entry whose numbers came from two different days.
    """
    found: Dict[str, Path] = {}
    for dataset in datasets:
        hit = lookup(harness_id, eval_key(eval_inputs(cfg, dataset)))
        if hit is None:
            return None
        found[dataset] = hit
    return found


def restore(harness_id: str, cfg: Dict[str, Any], datasets: List[str],
            dst: Path) -> Optional[Dict[str, Path]]:
    """Materialise a cached harness into `dst` — code plus its results.

    Returns the per-dataset result dirs that were restored, or None when the
    cache does not hold this exact configuration for every dataset (the caller
    then evaluates normally). `dst` is left as a working harness dir: the same
    layout an evaluation would have produced.
    """
    hits = results_for(harness_id, cfg, datasets)
    if hits is None:
        return None
    entry = _entry_dir(harness_id)
    _copy_code(entry, dst)
    for dataset, src in hits.items():
        target = dst / dataset
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(src, target)
        (target / "manifest.json").unlink(missing_ok=True)   # cache bookkeeping
    return hits


def describe(harness_id: str, result_dir: Path) -> str:
    """One line saying where reused numbers came from — a reuse must never be
    silent."""
    try:
        with (result_dir / "manifest.json").open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return f"{harness_id}: cached results (manifest unreadable)"
    inputs = manifest.get("inputs", {})
    return (f"{harness_id}: reusing {inputs.get('dataset')} results from run "
            f"{manifest.get('run_id')} ({manifest.get('stored_at')}, repo "
            f"{inputs.get('repo_version')}, key {manifest.get('eval_key')})")


def entries() -> List[Dict[str, Any]]:
    """Every cached harness, newest first — `{harness_id, first_seen, evals}`."""
    if not SEEDS_DIR.exists():
        return []
    out = []
    for index_path in SEEDS_DIR.glob("*/index.json"):
        try:
            with index_path.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as exc:             # noqa: BLE001
            log.warning(f"seed cache: unreadable index {index_path}: {exc}")
    out.sort(key=lambda e: e.get("first_seen") or "", reverse=True)
    return out
