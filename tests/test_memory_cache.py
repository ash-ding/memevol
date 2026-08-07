"""Tests for cross-stage memory caching (common/memory_cache.py + workflow
instrumentation).

Zero-dependency runner:

    uv run python tests/test_memory_cache.py
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

REPO = Path(__file__).resolve().parent.parent

import common.memory_cache as mc


# ---------------- fake memo classes (module-level so pickle can find them) ----------------

class FakeMemo:
    """Mimics an evolved harness's state shape: dicts + list + private attrs."""
    def __init__(self):
        self.events = []
        self.profile = {}
        self._private_index = {}

    async def build_memory_from_data(self, recorder):  # pragma: no cover - shape only
        pass

    async def retrieve_memory_for_query(self, recorder):  # pragma: no cover
        return {}


class HookedMemo(FakeMemo):
    """Overrides the optional save/load hooks."""
    def __init__(self):
        super().__init__()
        self.hook_saved = False

    def save_memory(self, path) -> bool:
        Path(str(path) + ".hooked").write_text(json.dumps({"events": self.events}))
        self.hook_saved = True
        return True

    def load_memory(self, path) -> bool:
        p = Path(str(path) + ".hooked")
        if not p.exists():
            return False
        self.events = json.loads(p.read_text())["events"]
        return True


class UnpicklableMemo(FakeMemo):
    def __init__(self):
        super().__init__()
        self.bad = lambda x: x  # lambdas don't pickle


# Module-level (pickle resolves classes by module+qualname; evolved harness
# classes are module-level too). Ingest counts recorded in a global list.
INGESTED_BATCHES = []


class CountingMemo(FakeMemo):
    async def build_memory_from_data(self, recorder):
        INGESTED_BATCHES.append(len(recorder.init.get("app_logs", [])))

    async def retrieve_memory_for_query(self, recorder):
        return {}


class CountingConvMemo(FakeMemo):
    async def build_memory_from_data(self, recorder):
        INGESTED_BATCHES.append(1)

    async def retrieve_memory_for_query(self, recorder):
        return {}


def _meta(**over):
    m = {"model": "gpt-5-mini",
         "max_logs": None, "harness_fingerprint": "abc123def456"}
    m.update(over)
    return m


# ---------------- save / load roundtrip ----------------

def test_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        memo = FakeMemo()
        memo.events = [{"app_log_id": "log_1"}, {"app_log_id": "log_2"}]
        memo.profile = {"habit": "runs daily"}
        memo._private_index = {"log_1": 0}
        ok = mc.save_memo(memo, Path(d), "user1__final", _meta())
        assert ok, "save failed"
        assert (Path(d) / "user1__final.pkl").exists()
        assert (Path(d) / "user1__final.meta.json").exists()

        loaded = mc.load_memo(Path(d), "user1__final", _meta())
        assert loaded is not None, "load missed"
        assert loaded.events == memo.events
        assert loaded.profile == memo.profile
        assert loaded._private_index == {"log_1": 0}  # private attrs preserved


def test_miss_when_absent():
    with tempfile.TemporaryDirectory() as d:
        assert mc.load_memo(Path(d), "nope__final", _meta()) is None


def test_meta_mismatch_is_miss():
    with tempfile.TemporaryDirectory() as d:
        memo = FakeMemo()
        mc.save_memo(memo, Path(d), "u__final", _meta())
        # each guarded field mismatching → miss
        for key, bad in [("model", "gpt-4.1"), ("max_logs", 100),
                         ("harness_fingerprint", "ffffffffffffffff")]:
            assert mc.load_memo(Path(d), "u__final", _meta(**{key: bad})) is None, \
                f"mismatched {key} must be a miss"
        # matching meta still hits
        assert mc.load_memo(Path(d), "u__final", _meta()) is not None


def test_corrupt_pickle_returns_none():
    with tempfile.TemporaryDirectory() as d:
        memo = FakeMemo()
        mc.save_memo(memo, Path(d), "u__final", _meta())
        (Path(d) / "u__final.pkl").write_bytes(b"corrupted garbage")
        assert mc.load_memo(Path(d), "u__final", _meta()) is None


def test_unpicklable_memo_save_returns_false():
    with tempfile.TemporaryDirectory() as d:
        ok = mc.save_memo(UnpicklableMemo(), Path(d), "u__final", _meta())
        assert ok is False
        # no partial artifacts left behind
        leftovers = [p.name for p in Path(d).iterdir()]
        assert not leftovers, f"partial artifacts remain: {leftovers}"


def test_no_tmp_leftovers_after_save():
    with tempfile.TemporaryDirectory() as d:
        mc.save_memo(FakeMemo(), Path(d), "u__final", _meta())
        tmps = [p.name for p in Path(d).iterdir() if ".tmp" in p.name]
        assert not tmps, f"tmp files remain: {tmps}"


# ---------------- optional harness hooks ----------------

def test_hooks_take_priority():
    with tempfile.TemporaryDirectory() as d:
        memo = HookedMemo()
        memo.events = [{"x": 1}]
        ok = mc.save_memo(memo, Path(d), "u__final", _meta())
        assert ok and memo.hook_saved
        assert not (Path(d) / "u__final.pkl").exists(), "hook save must skip default pickle"
        assert (Path(d) / "u__final.meta.json").exists(), "meta sidecar still required"

        fresh = HookedMemo()
        loaded = mc.load_memo(Path(d), "u__final", _meta(), memo_factory=lambda: fresh)
        assert loaded is fresh and loaded.events == [{"x": 1}]


# ---------------- helpers ----------------

def test_user_key_sanitization():
    k = mc.user_key("/app/benchmarks/dynamicmem/user_data/001_user_001")
    assert k == "001_user_001"
    k2 = mc.user_key("conv-26")
    assert k2 == "conv-26"
    assert "/" not in mc.user_key("a/b/c weird:name")


def test_harness_fingerprint_stable_and_sensitive():
    with tempfile.TemporaryDirectory() as d:
        hd = Path(d)
        (hd / "harness.py").write_text("class A: pass\n")
        (hd / "helper.py").write_text("x = 1\n")
        f1 = mc.harness_fingerprint(hd)
        f2 = mc.harness_fingerprint(hd)
        assert f1 == f2 and len(f1) == 16
        (hd / "harness.py").write_text("class A: pass  # edited\n")
        assert mc.harness_fingerprint(hd) != f1
        # meta.json changes must NOT affect it (matches _compute_content_hash scope)
        f3 = mc.harness_fingerprint(hd)
        (hd / "meta.json").write_text("{}")
        assert mc.harness_fingerprint(hd) == f3


# ---------------- dynamicmem workflow-level: prefix cps not re-ingested ----------------

def test_dynamicmem_cache_skips_ingested_checkpoints():
    from benchmarks.dynamicmem.workflow import DynamicMemWorkflow

    user_dir = str(REPO / "benchmarks" / "dynamicmem" / "user_data" / "001_user_001")

    ingested_batches = INGESTED_BATCHES
    ingested_batches.clear()

    async def run(stage, spec, cache_dir):
        wf = DynamicMemWorkflow(
            memo_class=CountingMemo, model="gpt-5-mini/low", judge_model="gpt-5-mini",
        )
        wf.status = "search"
        wf.memory_cache_dir = cache_dir
        # answer path stubbed out: no items sampled → pure Phase-1 exercise
        rec = await wf.run_single_user(user_dir, stage=stage, stage_spec=spec)
        return rec

    with tempfile.TemporaryDirectory() as d:
        cache_dir = Path(d)
        spec1 = {"n_samples": 1, "n_checkpoints": 1, "n_task_a": 0, "n_task_c": 0}
        spec2 = {"n_samples": 1, "n_checkpoints": 3, "n_task_a": 0, "n_task_c": 0}

        asyncio.run(run("stage1", spec1, cache_dir))
        stage1_calls = list(ingested_batches)
        assert sum(stage1_calls) == 180, f"stage1 should ingest cp1 prefix (180 logs), got {sum(stage1_calls)}"
        # snapshot for cp1 exists
        snaps = sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".pkl")
        assert any("cp1" in s for s in snaps), snaps

        ingested_batches.clear()
        asyncio.run(run("stage2", spec2, cache_dir))
        # cp1 loaded from cache → only cp2+cp3 segments ingested (466+250 logs for user001)
        assert sum(ingested_batches) == (466 - 180) + (716 - 466), \
            f"stage2 must skip cached cp1; ingested {sum(ingested_batches)}"
        snaps = sorted(p.name for p in cache_dir.iterdir() if p.suffix == ".pkl")
        assert sum(1 for s in snaps if "__cp" in s) == 3, snaps

        # third run at same depth: everything cached, zero ingestion
        ingested_batches.clear()
        asyncio.run(run("stage2", spec2, cache_dir))
        assert sum(ingested_batches) == 0, f"full cache hit expected, ingested {ingested_batches}"


def test_base_workflow_cache_skips_phase1():
    """LoCoMo-style: second run with cache loads final memory, no ingest."""
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from benchmarks.locomo.env import get_task_list

    conv = get_task_list("search", 1)[0]
    ingest_calls = INGESTED_BATCHES
    ingest_calls.clear()

    async def run(cache_dir):
        wf = LoCoMoWorkflow(memo_class=CountingConvMemo, model="gpt-5-mini/low")
        wf.status = "search"
        wf.memory_cache_dir = cache_dir
        return await wf.run_single_user(conv, stage="stage1",
                                        stage_spec={"n_samples": 1, "n_qa": 0})

    with tempfile.TemporaryDirectory() as d:
        asyncio.run(run(Path(d)))
        assert sum(ingest_calls) >= 1, "first run must ingest"
        ingest_calls.clear()
        asyncio.run(run(Path(d)))
        assert sum(ingest_calls) == 0, "second run must hit the final-memory cache"


# ---------------- runner ----------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
