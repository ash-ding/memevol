"""Tests for the reusable disk-store memory-cache hooks (common/store_cache.py).

Zero-dependency runner (no pytest), root project env:

    uv run python tests/test_store_cache.py
"""
import sys, tempfile, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import memory_cache as mc          # noqa: E402
from common.memo_class import MemoClass        # noqa: E402
from common.store_cache import DiskStoreCache  # noqa: E402


class DirMemo(DiskStoreCache, MemoClass):
    """A memo whose whole state is a directory — the shape 5 baselines have."""
    _store_handle = "_system"

    def __init__(self, root, config=None):
        super().__init__(config)
        self._root = Path(root)
        self._system = None          # the unpicklable handle, once opened
        self.wiped = False

    def _store_path(self):
        return self._root / "store"

    def open(self, contents=None):
        """Stand-in for _ensure_system: wipes unless restored, then opens."""
        d = self._store_path()
        if d.exists() and not self.restored_from_cache:
            import shutil; shutil.rmtree(d); self.wiped = True
        d.mkdir(parents=True, exist_ok=True)
        if contents is not None:
            (d / "data.txt").write_text(contents, encoding="utf-8")
        import threading
        self._system = threading.RLock()      # exactly what breaks pickle
        return self


def _meta():
    return {"model": "m", "max_logs": None, "harness_fingerprint": "fp"}


def test_hooks_beat_pickle_for_an_unpicklable_memo():
    """The whole point: this memo cannot be pickled, but caches fine."""
    import pickle
    with tempfile.TemporaryDirectory() as d:
        memo = DirMemo(d).open("built")
        try:
            pickle.dumps(memo)
            raise AssertionError("expected an unpicklable memo")
        except (TypeError, AttributeError):
            pass
        assert mc.save_memo(memo, Path(d) / "cache", "u__final", _meta()) is True
        sidecar = (Path(d) / "cache" / "u__final.meta.json").read_text(encoding="utf-8")
        assert '"storage": "hook"' in sidecar, sidecar


def test_round_trip_restores_the_store_into_a_fresh_instance():
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a", Path(d) / "b"
        cache = Path(d) / "cache"
        mc.save_memo(DirMemo(src).open("hello"), cache, "u__final", _meta())

        fresh = mc.load_memo(cache, "u__final", _meta(),
                             memo_factory=lambda: DirMemo(dst))
        assert fresh is not None, "hook-stored entry must load"
        assert (fresh._store_path() / "data.txt").read_text(encoding="utf-8") == "hello"
        # handle reset so the baseline's own _ensure_*() reopens it
        assert fresh._system is None
        assert fresh.restored_from_cache is True


def test_restore_survives_a_baseline_that_wipes_on_open():
    """mem0/memoryos rmtree their store and simplemem passes clear_db=True; a
    restore is worthless if opening then erases it."""
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "cache"
        mc.save_memo(DirMemo(Path(d) / "a").open("keepme"), cache, "u__final", _meta())
        fresh = mc.load_memo(cache, "u__final", _meta(),
                             memo_factory=lambda: DirMemo(Path(d) / "b"))
        fresh.open()                      # would wipe, were it not restored
        assert fresh.wiped is False
        assert (fresh._store_path() / "data.txt").read_text(encoding="utf-8") == "keepme"


def test_nothing_built_yet_falls_back_to_pickle():
    """No store on disk => the hook declines and the default path takes over."""
    with tempfile.TemporaryDirectory() as d:
        memo = DirMemo(Path(d) / "a")            # never opened
        assert memo.save_memory(Path(d) / "snap") is False
        assert mc.save_memo(memo, Path(d) / "cache", "u__final", _meta()) is True
        sidecar = (Path(d) / "cache" / "u__final.meta.json").read_text(encoding="utf-8")
        assert '"storage": "pickle"' in sidecar, sidecar


def test_missing_snapshot_is_a_miss_not_a_crash():
    with tempfile.TemporaryDirectory() as d:
        assert DirMemo(Path(d)).load_memory(Path(d) / "nope") is False


def test_restore_overwrites_a_partly_built_store():
    """A restore must replace, never merge into, whatever is already there."""
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "cache"
        mc.save_memo(DirMemo(Path(d) / "a").open("good"), cache, "u__final", _meta())
        stale = DirMemo(Path(d) / "b").open("stale")
        (stale._store_path() / "junk.txt").write_text("x", encoding="utf-8")
        assert stale.load_memory(cache / "u__final") is True
        assert (stale._store_path() / "data.txt").read_text(encoding="utf-8") == "good"
        assert not (stale._store_path() / "junk.txt").exists()


def test_single_file_store_round_trips():
    """zep's store is one .db FILE, not a directory."""
    class FileMemo(DirMemo):
        def _store_path(self):
            return self._root / "graph.db"

    with tempfile.TemporaryDirectory() as d:
        a = FileMemo(Path(d) / "a")
        a._store_path().parent.mkdir(parents=True, exist_ok=True)
        a._store_path().write_text("bytes", encoding="utf-8")
        assert a.save_memory(Path(d) / "snap") is True
        b = FileMemo(Path(d) / "b")
        assert b.load_memory(Path(d) / "snap") is True
        assert b._store_path().read_text(encoding="utf-8") == "bytes"


def test_every_wired_baseline_declares_a_store_path():
    """A baseline mixing in DiskStoreCache without _store_path would raise at
    cache time, not import time — catch it here instead."""
    import ast
    wired = []
    for f in sorted((PROJECT_ROOT / "baselines" / "harness").glob("*/memo.py")):
        src = f.read_text(encoding="utf-8", errors="ignore")
        if "DiskStoreCache" not in src:
            continue
        wired.append(f.parent.name)
        assert "def _store_path" in src, f"{f.parent.name} mixes in the cache but has no _store_path"
        assert "_store_handle" in src, f"{f.parent.name} does not name its handle"
        # the declared handle must be a real attribute the class assigns
        handle = src.split('_store_handle = "', 1)[1].split('"', 1)[0]
        assert f"self.{handle}" in src, f"{f.parent.name}: {handle} is never assigned"
    assert len(wired) >= 5, f"expected the disk-backed baselines to be wired, got {wired}"


def test_lock_files_are_never_snapshotted():
    """REGRESSION (found by a real lightmem run): Qdrant holds `.lock` OPEN for
    the life of the client, so copying it failed with `[Errno 13] Permission
    denied` on Windows and took the whole snapshot down — lightmem then fell
    through to pickle, which also fails for it, leaving it uncached.

    A lock is process-liveness, not memory; restoring a stale one would also
    stop the backend taking its own.
    """
    with tempfile.TemporaryDirectory() as d:
        memo = DirMemo(Path(d) / "a").open("real-data")
        store = memo._store_path()
        (store / ".lock").write_text("held", encoding="utf-8")
        (store / "seg.pid").write_text("123", encoding="utf-8")
        (store / "nested").mkdir()
        (store / "nested" / "x.lock").write_text("held", encoding="utf-8")

        assert memo.save_memory(Path(d) / "snap") is True
        snap = Path(str(Path(d) / "snap") + ".store")
        copied = {p.name for p in snap.rglob("*") if p.is_file()}
        assert "data.txt" in copied, copied
        assert not any(n.endswith((".lock", ".pid")) for n in copied), copied

        # and the restore brings the real data back, still lock-free
        fresh = DirMemo(Path(d) / "b")
        assert fresh.load_memory(Path(d) / "snap") is True
        assert (fresh._store_path() / "data.txt").read_text(encoding="utf-8") == "real-data"
        assert not (fresh._store_path() / ".lock").exists()


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn(); print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}"); traceback.print_exc(); failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
