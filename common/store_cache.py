"""Reusable `save_memory` / `load_memory` for memos whose state is on disk.

`common/memory_cache.py` persists a user's Phase-1 memory so later gauntlet
stages reuse it instead of rebuilding. Its default mechanism is
`pickle.dumps(memo)`, which fails for any memo holding a live resource — a DB
connection, an HTTP pool, a thread pool, a subprocess all contain
`_thread.RLock`, which is unpicklable. The eval then degrades to a miss and
rebuilds Phase-1 from scratch at EVERY stage. That is the expensive failure this
module removes: one simplemem LoCoMo user costs ~450s of Phase-1 build.

Most memory systems already solve their own persistence — they keep everything
in a per-instance directory (LanceDB, Qdrant, a graph store, a save_dir). For
those, `save_memory`/`load_memory` reduce to "copy that path out, copy it back
in", which is what :class:`DiskStoreCache` implements once for all of them.

    class MyMemo(DiskStoreCache, MemoClass):
        _store_handle = "_system"          # the lazily-built system attribute

        def _store_path(self):
            return OUTPUTS_DIR / self._instance_id

Two contract points a baseline must respect:

* **The handle is reset, not rebuilt.** `load_memory` only restores the bytes
  and sets ``_store_handle`` to None; the next `_ensure_*()` reopens the system
  against the restored directory. Nothing here knows how to construct a
  SimpleMemSystem or a Memoryos.
* **Do not wipe a restored store.** Several baselines clear their directory when
  they construct (mem0 and memoryos `rmtree`, simplemem passes `clear_db=True`).
  That would erase the restore, so they must honour ``self.restored_from_cache``
  — which this mixin sets — and skip the wipe.

A fresh instance gets a NEW `_instance_id`, so the restore lands under the new
instance's path, not the one it was saved from. Restoring is therefore safe
under concurrent per-user evaluation.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from common.logger import get_logger

log = get_logger("main")

# Suffix for the snapshot. `path` from memory_cache is a filename PREFIX that
# the hook owns; keeping the suffix here means the sidecar and the payload can
# never collide.
_SUFFIX = ".store"


class DiskStoreCache:
    """Mixin implementing the memory-cache hooks by copying a directory/file."""

    #: Name of the attribute holding the lazily-built system. Reset to None on
    #: restore so the baseline's own `_ensure_*()` reopens it.
    _store_handle: str = "_system"

    #: Set by `load_memory`. Baselines that clear their store on construction
    #: MUST check this and skip the wipe.
    restored_from_cache: bool = False

    def _store_path(self) -> Optional[Path]:
        """The directory (or file) holding everything this memo must persist."""
        raise NotImplementedError(
            f"{type(self).__name__} uses DiskStoreCache but does not implement "
            f"_store_path()"
        )

    # -- hooks -------------------------------------------------------------

    def save_memory(self, path) -> bool:
        src = self._store_path()
        if src is None or not Path(src).exists():
            # Nothing built yet, or an in-memory backend — let the caller fall
            # back to pickle rather than writing an empty snapshot.
            return False
        dst = Path(str(path) + _SUFFIX)
        try:
            _replace(Path(src), dst)
        except Exception as exc:
            log.warning(f"[store_cache] save failed for {type(self).__name__}: {exc!r}")
            return False
        return True

    def load_memory(self, path) -> bool:
        src = Path(str(path) + _SUFFIX)
        if not src.exists():
            return False
        dst = self._store_path()
        if dst is None:
            return False
        try:
            _replace(src, Path(dst))
        except Exception as exc:
            log.warning(f"[store_cache] load failed for {type(self).__name__}: {exc!r}")
            return False
        # Drop the handle so the baseline's own _ensure_*() reopens against the
        # restored bytes, and tell it not to wipe them on the way.
        setattr(self, self._store_handle, None)
        self.restored_from_cache = True
        return True


def _replace(src: Path, dst: Path) -> None:
    """Copy `src` over `dst`, handling both a directory and a single file.

    zep's store is one `.db` file where the others are directories, so this has
    to cover both. `dst` is removed first so a restore can never merge into a
    half-built store.
    """
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
