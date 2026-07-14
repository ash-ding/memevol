"""Search-mode data isolation — make the TEST split technically invisible
inside both sandbox containers (proposer + evaluator).

Both containers bind the full `datasets/` package RO (the workflows need the
.py modules and the search-split raw data). Without isolation, the search
agent and evolved harness code could read the held-out test questions AND
gold answers — the only protection was a prompt-level policy. This module
turns that policy into a filesystem guarantee via Singularity's nested-bind
overlay (empirically verified: a later bind shadows a path inside an earlier
RO dir bind):

  LoCoMo       filtered locomo10.json (first TRAIN_SAMPLES=6 conversations,
               original order → the positional split derivation still holds:
               train = all 6, test = []) bound over the original file.
  DynamicMem   an empty dir bound over user_data/ hides all users; the
               search users are bound back individually (zero-copy).
  LongMemEval  filtered s/m jsons (the 300 search question_ids, from the
               env's own _compute_split) + a split_manifest.json the env
               prefers over recomputing (a filtered file would otherwise be
               re-stratified into a bogus 180/120 split); the unregistered
               oracle file (contains gold answers) is shadowed by a stub.

Isolation applies to every orchestrator run (the orchestrator only ever
drives the search split, including the sanity tier and smoke-test runs).
forge.heldout (test split) keeps the full bind: the leakage direction is
search→test (heldout harnesses are frozen), and the positional split
derivations don't survive test-only files.

KNOWN LIMIT (see PROGRESS): the eval container still exposes the CURRENT
split's gold answers to harness code — fixing that cheating channel needs a
data/answer separation refactor, out of scope here.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from forge.paths import PROJECT_ROOT, WORKSPACE_BASE

log = logging.getLogger("forge.data_isolation")

_DATASETS = PROJECT_ROOT / "datasets"

# Cross-run staging cache. The LongMemEval raw files are LARGE (m variant
# ~2.6 GB) — filtering them costs minutes of JSON parse/dump, so staged
# artifacts are fingerprinted against their sources (size + mtime_ns) and
# reused across runs. Lives under the gitignored workspace/ root.
DEFAULT_STAGING_DIR = WORKSPACE_BASE / ".data_staging_cache"


def _src_fingerprint(*srcs: Path) -> str:
    parts = []
    for src in srcs:
        st = Path(src).stat()
        parts.append(f"{src}:{st.st_size}:{st.st_mtime_ns}")
    return "|".join(parts)


def _fresh(out: Path, *srcs: Path) -> bool:
    """True iff `out` exists and its sidecar matches the sources' stats."""
    sidecar = out.with_name(out.name + ".src.json")
    if not out.exists() or not sidecar.exists():
        return False
    try:
        return json.loads(sidecar.read_text()) == _src_fingerprint(*srcs)
    except (OSError, json.JSONDecodeError):
        return False


def _mark(out: Path, *srcs: Path) -> None:
    out.with_name(out.name + ".src.json").write_text(
        json.dumps(_src_fingerprint(*srcs)), encoding="utf-8"
    )
# Container-side mount point of the datasets package (same in both sandboxes).
_APP_DATASETS = "/app/datasets"


def _stage_locomo(staging: Path) -> List[str]:
    from datasets.locomo.env import TRAIN_SAMPLES, DATA_PATH

    src = Path(DATA_PATH)
    out = staging / "locomo10.json"
    if not _fresh(out, src):
        data = json.loads(src.read_text(encoding="utf-8"))
        filtered = data[:TRAIN_SAMPLES]
        out.write_text(json.dumps(filtered, ensure_ascii=False), encoding="utf-8")
        _mark(out, src)
        log.info(
            f"data_isolation: locomo staged {len(filtered)}/{len(data)} conversations"
        )
    return [f"{out}:{_APP_DATASETS}/locomo/locomo10.json:ro"]


def _stage_dynamicmem(staging: Path) -> List[str]:
    from datasets.dynamicmem.env import TRAIN_USERS, _get_all_user_dirs

    empty = staging / "empty_user_data"
    empty.mkdir(parents=True, exist_ok=True)
    binds = [f"{empty}:{_APP_DATASETS}/dynamicmem/user_data:ro"]
    search_dirs = _get_all_user_dirs()[:TRAIN_USERS]
    for d in search_dirs:
        name = Path(d).name
        binds.append(
            f"{d}:{_APP_DATASETS}/dynamicmem/user_data/{name}:ro"
        )
    log.info(
        f"data_isolation: dynamicmem exposing {len(search_dirs)} search users "
        f"(test users hidden)"
    )
    return binds


def _stage_longmemeval(staging: Path) -> List[str]:
    from datasets.longmemeval.env import DATA_PATHS, _compute_split

    search_qids, _test_qids = _compute_split()
    keep = set(search_qids)

    binds: List[str] = []
    for variant in ("s", "m"):
        src = Path(DATA_PATHS[variant])
        out = staging / src.name
        if not _fresh(out, src):
            # The m variant is ~2.6 GB — this parse/dump is exactly what the
            # fingerprint cache exists to avoid repeating across runs.
            data = json.loads(src.read_text(encoding="utf-8"))
            filtered = [s for s in data if s.get("question_id") in keep]
            out.write_text(json.dumps(filtered, ensure_ascii=False), encoding="utf-8")
            _mark(out, src)
            log.info(f"data_isolation: staged {out.name} ({len(filtered)} questions)")
        binds.append(f"{out}:{_APP_DATASETS}/longmemeval/{src.name}:ro")

    # The env prefers this manifest over recomputing the stratified split —
    # a 300-question file would otherwise be re-split into a bogus 180/120.
    manifest = staging / "split_manifest.json"
    manifest.write_text(
        json.dumps({"search": sorted(search_qids), "test": []}),
        encoding="utf-8",
    )
    # Container-only path — binding to a target that doesn't pre-exist inside
    # the host-bound /app/datasets dir would make Singularity create an empty
    # placeholder file on the HOST (which then poisons host-side split
    # computation). /staging lives in the container overlay, never on disk.
    binds.append(f"{manifest}:/staging/split_manifest.json:ro")

    # The oracle variant is not registered in any workflow but carries gold
    # answers for all 500 questions — shadow it with a stub.
    stub = staging / "longmemeval_oracle.json"
    stub.write_text("[]", encoding="utf-8")
    binds.append(f"{stub}:{_APP_DATASETS}/longmemeval/longmemeval_oracle.json:ro")

    log.info(
        f"data_isolation: longmemeval staged {len(keep)} search questions "
        f"(test + oracle hidden)"
    )
    return binds


def stage_search_data(staging_dir: Optional[Path] = None) -> List[str]:
    """Write search-split-only data under `staging_dir` (default: the shared
    cross-run cache workspace/.data_staging_cache) and return the singularity
    `--bind` values that overlay it onto the container's /app/datasets.
    Deterministic; artifacts are fingerprinted against their source files and
    reused until the sources change. Derived from each env's OWN split
    functions (never re-implements a split). ALL benchmarks are filtered
    regardless of which are enabled for the run: un-enabled test data is
    equally unnecessary to the agent.
    """
    if staging_dir is None:
        staging_dir = DEFAULT_STAGING_DIR
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    binds: List[str] = []
    binds += _stage_locomo(staging_dir)
    binds += _stage_longmemeval(staging_dir)
    # DynamicMem raw data is gitignored and may be absent on a fresh checkout;
    # skip its overlay when there is nothing to hide.
    if (_DATASETS / "dynamicmem" / "user_data").is_dir():
        binds += _stage_dynamicmem(staging_dir)
    return binds
