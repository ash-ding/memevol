"""Cross-stage memory cache — persist a harness's Phase-1 memory to disk so
later evaluation stages can reuse it instead of rebuilding.

Design (borrowed from the official DynamicMem baselines' snapshot machinery —
per-checkpoint bundles, confirmed-depth metadata, load-only consumption,
atomic writes; see Amem/simplemem in the upstream repo):

  <cache_dir>/<key>.pkl          the pickled memo object (default storage)
  <cache_dir>/<key>.meta.json    sidecar: fingerprints + ingest config

- Sidecar-per-entry instead of a central manifest: multiple users run
  concurrently inside one evaluator exec; per-file sidecars are race-free.
- Both files are written atomically (`.tmp` → os.replace).
- A load is a HIT only when every field the caller passes in `expect_meta`
  matches the sidecar exactly (harness fingerprint, model, ...).
  Anything else — missing files, corrupt pickle, meta drift, unpicklable
  state — degrades to a miss (rebuild); the cache must never break an eval.

Optional harness hooks (see forge/memo_class.py): a memo may implement
`save_memory(path) -> bool` / `load_memory(path) -> bool` to handle backends
the default pickle can't (the hook owns the file format at `path.*`). The
sidecar meta is still written/validated by this module.

The cache directory is ``<out_dir>/memory_cache`` — keyed on baseline + dataset
+ split, with nothing run-specific in it — so entries persist ACROSS runs and a
re-run reuses the previous one's Phase-1 memory. The ``memory_cache`` config key
chooses what to do about that; see ``CACHE_MODES`` / ``resolve_cache_mode``.

Security note: pickles are produced and consumed ONLY inside the harness's
own evaluation container, which already runs the harness's arbitrary code —
no new attack surface. The host never unpickles cache files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("main")

CACHE_VERSION = 1

_KEY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def user_key(user_dir: str) -> str:
    """Filename-safe cache key component for a user/sample identifier."""
    name = Path(str(user_dir)).name or str(user_dir)
    return _KEY_SANITIZE_RE.sub("_", name)


#: Accepted values for the `memory_cache` config key.
#:   True      read + write — reuse a previous run's snapshot (the default)
#:   False     off entirely — every stage rebuilds
#:   "rebuild" write-only — build Phase 1 fresh ONCE this run, then let the
#:             later gauntlet stages reuse that build and overwrite the stale
#:             entry. The middle ground `False` cannot express.
CACHE_MODES = (True, False, "rebuild")


def resolve_cache_mode(memory_cache) -> tuple:
    """Normalise the `memory_cache` config value to (enabled, read_allowed).

    Raises ValueError on anything else, so a typo (`rebiuld`) fails loudly
    instead of being truthy and silently behaving like `True`.
    """
    if isinstance(memory_cache, bool):
        return memory_cache, memory_cache
    if isinstance(memory_cache, str) and memory_cache.strip().lower() == "rebuild":
        return True, False
    raise ValueError(
        f"memory_cache must be one of {CACHE_MODES!r}, got {memory_cache!r}"
    )


def config_fingerprint(config: Optional[Dict[str, Any]]) -> str:
    """sha256[:16] over a memo's CONFIG — the knobs that change what gets built.

    `harness_fingerprint` covers the memo's source, so editing memo.py
    invalidates a cache entry. It does NOT cover configuration, and the config
    is where the interesting differences live: swap `embedding_model` from a
    384-dim MiniLM to a 1536-dim API embedder, or change simplemem's
    `window_size`, and the memory that comes out is materially different while
    the source is byte-identical. Without this, the entry would be reused and
    the run would silently answer from memory built under the OTHER settings.

    Values that cannot be serialised (a factory callable injected by a test)
    contribute their TYPE NAME, never their repr — a repr carries a memory
    address, which would change every process and make the entry permanently
    unreusable rather than merely correct.
    """
    if not config:
        return "none"
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False,
                      default=lambda o: f"<{type(o).__name__}>")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def harness_fingerprint(harness_dir: Path) -> str:
    """sha256[:16] over the harness's code files (harness.py + helper *.py +
    requirements.txt), name + content. Mirrors the scope of the orchestrator's
    content hash (meta.json / artifacts excluded) — a cache entry written by
    different code must never be reused."""
    h = hashlib.sha256()
    harness_dir = Path(harness_dir)
    files = sorted(p for p in harness_dir.glob("*.py"))
    req = harness_dir / "requirements.txt"
    if req.exists():
        files.append(req)
    for p in files:
        try:
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes())
        except OSError:
            continue
    return h.hexdigest()[:16]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _paths(cache_dir: Path, key: str):
    cache_dir = Path(cache_dir)
    return cache_dir / f"{key}.pkl", cache_dir / f"{key}.meta.json", cache_dir / key


def save_memo(memo: Any, cache_dir: Path, key: str, meta: Dict[str, Any]) -> bool:
    """Persist one memory snapshot. Returns True on success; False (with a
    warning) when the memo can't be serialized — the eval continues uncached."""
    pkl_path, meta_path, hook_path = _paths(cache_dir, key)
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        storage = None
        hook = getattr(memo, "save_memory", None)
        if callable(hook):
            try:
                if hook(hook_path) is True:
                    storage = "hook"
            except Exception as exc:
                log.warning(f"[memcache] save_memory hook failed for {key}: {exc!r} "
                            f"— falling back to pickle")

        if storage is None:
            data = pickle.dumps(memo, protocol=pickle.HIGHEST_PROTOCOL)
            _atomic_write_bytes(pkl_path, data)
            storage = "pickle"

        sidecar = dict(meta)
        sidecar.update({
            "version": CACHE_VERSION,
            "storage": storage,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        })
        _atomic_write_json(meta_path, sidecar)
        return True
    except Exception as exc:
        log.warning(f"[memcache] could not save {key}: {exc!r} — continuing uncached")
        # Best-effort cleanup so a half-written entry can never be loaded.
        for p in (pkl_path, meta_path):
            try:
                p.unlink(missing_ok=True)
                p.with_name(p.name + ".tmp").unlink(missing_ok=True)
            except OSError:
                pass
        return False


def load_memo(
    cache_dir: Path,
    key: str,
    expect_meta: Dict[str, Any],
    memo_factory: Optional[Callable[[], Any]] = None,
) -> Optional[Any]:
    """Load one memory snapshot; None on any miss/mismatch/error.

    Every key in `expect_meta` must equal the stored sidecar value — a
    fingerprint or ingest-config drift silently invalidates the entry.
    `memo_factory` is required to consume hook-stored entries (a fresh memo
    instance whose `load_memory(path)` is called)."""
    pkl_path, meta_path, hook_path = _paths(cache_dir, key)
    if not meta_path.exists():
        return None
    try:
        sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning(f"[memcache] unreadable sidecar for {key}: {exc!r}")
        return None

    if sidecar.get("version") != CACHE_VERSION:
        return None
    for field, expected in expect_meta.items():
        if sidecar.get(field) != expected:
            log.info(f"[memcache] {key}: {field} mismatch "
                     f"(cached={sidecar.get(field)!r}, expected={expected!r}) — miss")
            return None

    storage = sidecar.get("storage")
    try:
        if storage == "hook":
            if memo_factory is None:
                return None
            memo = memo_factory()
            loader = getattr(memo, "load_memory", None)
            if callable(loader) and loader(hook_path) is True:
                return memo
            return None
        if storage == "pickle":
            if not pkl_path.exists():
                return None
            # Safe by construction (see module docstring): this runs only in
            # the harness's own eval container, deserializing a file that the
            # same harness's code wrote — the container already executes that
            # harness's arbitrary Python, so pickle adds no new attack surface.
            # The host process never unpickles cache files.
            return pickle.loads(pkl_path.read_bytes())
    except Exception as exc:
        log.warning(f"[memcache] could not load {key}: {exc!r} — rebuilding")
        return None
    return None
