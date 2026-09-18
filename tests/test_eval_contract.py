"""Tests for the standardized memory-system eval contract.

Zero-dependency runner:
    uv run python tests/test_eval_contract.py
"""
import asyncio
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_build_and_retrieve_are_required_answer_is_optional():
    from common.memo_class import MemoClass

    class Bare(MemoClass):
        pass

    class Misspelled(MemoClass):
        async def build_memory_from_data(self, recorder): return None
        async def retrieve_memory_for_querry(self, recorder): return {}   # typo

    for cls, missing in ((Bare, "build_memory_from_data"), (Misspelled, "retrieve_memory_for_query")):
        try:
            cls()
        except TypeError as e:
            assert missing in str(e), str(e)
        else:
            raise AssertionError(f"{cls.__name__} must not be instantiable")

    class Minimal(MemoClass):
        async def build_memory_from_data(self, recorder): return None
        async def retrieve_memory_for_query(self, recorder): return {}

    m = Minimal()
    assert not hasattr(m, "database")
    # Answering is NOT part of the contract: the benchmark's shared QA agent
    # answers for every memo, so a memo cannot substitute its own answerer.
    assert not hasattr(m, "use_memory_to_answer")


def test_harness_loaders_name_the_unimplemented_hook():
    """Every loader of candidate / generated harness files picks the concrete
    class and, when there is none, says which hook is missing."""
    import tempfile
    from pathlib import Path
    from common.memo_select import select_memo_class
    from forge.launch import _load_harness_class
    from forge.contract import HarnessError, load_harness_class

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "memo.py").write_text(
            "from forge.memo_class import MemoClass\n"
            "class Half(MemoClass):\n"
            "    async def build_memory_from_data(self, recorder): return None\n",
            encoding="utf-8")
        for load, exc in ((_load_harness_class, ImportError), (load_harness_class, HarnessError)):
            try:
                load(d)
            except exc as e:
                assert "retrieve_memory_for_query" in str(e) and "Half" in str(e), str(e)
            else:
                raise AssertionError(f"{load.__name__} accepted an abstract harness")

    from common.memo_class import MemoClass

    class _AbstractHelper(MemoClass):
        async def build_memory_from_data(self, recorder): return None

    class _Concrete(_AbstractHelper):
        async def retrieve_memory_for_query(self, recorder): return {}

    # an abstract helper base listed first is skipped, not returned
    assert select_memo_class([MemoClass, _AbstractHelper, _Concrete], "x.py") is _Concrete


def test_forge_loads_the_harness_class_not_the_base():
    """The class-discriminator must pick the harness's own class, not the
    imported base."""
    from pathlib import Path
    from forge.launch import _load_harness_class
    from forge.contract import load_harness_class
    seed = Path(__file__).resolve().parents[1] / "baselines" / "harness" / "no_memory"
    assert _load_harness_class(seed).__name__ == "NoMemoryHarness"
    assert load_harness_class(seed).__name__ == "NoMemoryHarness"


def test_phase1_update_calls_build_memory_once():
    """The workflow hands the whole data in ONE build_memory_from_data call."""
    from common.workflow import BaseWorkflow
    from common.memo_class import MemoClass

    calls = []

    class _Memo(MemoClass):
        async def build_memory_from_data(self, recorder):
            calls.append(list(recorder.init.get("items", [])))
        async def retrieve_memory_for_query(self, recorder): return {}

    # minimal BaseWorkflow with the 7 abstract hooks stubbed + a recorder that
    # stores init under "items" via phase1_log_init.
    class _Rec:
        def __init__(self): self.init = {}
    class _W(BaseWorkflow):
        recorder_class = _Rec
        async def load_user_data(self, *a, **k): return None
        async def phase1_log_init(self, r, chunk): r.init = {"items": chunk}
        def build_query_recorder_init(self, *a, **k): return {}
        def build_qa_prompt(self, *a, **k): return [{"content": ""}, {"content": ""}]
        def extract_relevant_context(self, *a, **k): return None
        def build_qa_metadata(self, *a, **k): return {}
        async def log_qa_step(self, *a, **k): return None
    import asyncio
    w = _W(memo_class=_Memo, model="gpt-5-mini")
    m = _Memo()
    asyncio.new_event_loop().run_until_complete(w._phase1_update(m, [1, 2, 3, 4, 5]))
    assert calls == [[1, 2, 3, 4, 5]]   # ONE call, whole data (not chunked by the workflow)


def test_memo_class_is_pure_contract():
    """common/ purification guard (2026-07-16): Sub_memo_layer must not flow
    back into common.memo_class (it is alma-owned design vocabulary, at
    baselines/evolve/alma/memo_layers.py), and Basic_Recorder is DEFINED in
    common/recorder.py while the legacy import path keeps working."""
    import common.memo_class as hb
    import common.recorder as rec

    # Sub_memo_layer is gone from memo_class (not defined, not re-exported)
    assert not hasattr(hb, "Sub_memo_layer"), \
        "Sub_memo_layer leaked back into common.memo_class"

    # legacy path still importable, and it is the SAME object as common.recorder's
    from common.memo_class import Basic_Recorder as legacy_recorder
    assert rec.Basic_Recorder is legacy_recorder
    assert rec.Basic_Recorder is hb.Basic_Recorder
    # ... and it is defined in common.recorder, only re-exported by memo_class
    assert rec.Basic_Recorder.__module__ == "common.recorder"

    # alma's memo_layers owns Sub_memo_layer now
    from baselines.evolve.alma.memo_layers import Sub_memo_layer
    assert Sub_memo_layer.__module__ == "baselines.evolve.alma.memo_layers"


def test_no_answer_hook_anywhere_in_the_eval_path():
    """Answering is the benchmarks' own shared QA agent, not a memo hook: the
    contract must not declare `use_memory_to_answer` and neither answer site
    may call one (a memo that answered itself would also hide its cost from
    the answer-phase token accounting)."""
    import inspect
    from common import workflow as bw
    from common.memo_class import MemoClass
    from benchmarks.dynamicmem import workflow as dw

    assert not hasattr(MemoClass, "use_memory_to_answer")
    for src in (inspect.getsource(bw.BaseWorkflow.run_single_user),
                inspect.getsource(dw.DynamicMemWorkflow._run_item)):
        assert "use_memory_to_answer" not in src


def test_dynamicmem_ingests_each_checkpoint_segment_from_scratch():
    """DynamicMem's per-checkpoint loop hands BUILD exactly the newly visible
    log segment at each checkpoint (checkpoint isolation), on a memo built
    with `memo_config`, and a second run rebuilds everything — no memory is
    carried across runs or stages."""
    import os
    os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
    from benchmarks.dynamicmem.workflow import DynamicMemWorkflow
    from common.memo_class import MemoClass

    batches, configs = [], []

    class _CountingMemo(MemoClass):
        async def build_memory_from_data(self, recorder):
            configs.append(self.config.get("tag"))
            batches.append(len(recorder.init.get("app_logs", [])))

        async def retrieve_memory_for_query(self, recorder): return {}

    user_dir = str(PROJECT_ROOT / "benchmarks" / "dynamicmem" / "user_data" / "001_user_001")
    # no items sampled → pure Phase-1 exercise, no QA/judge calls
    spec = {"n_samples": 1, "n_checkpoints": 3, "n_task_a": 0, "n_task_c": 0}

    for _ in range(2):
        batches.clear()
        wf = DynamicMemWorkflow(memo_class=_CountingMemo, model="gpt-5-mini",
                                memo_config={"tag": "cfg"})
        wf.status = "search"
        asyncio.run(wf.run_single_user(user_dir, stage="stage2", stage_spec=spec))
        # user_001's first three checkpoints end at logs 180 / 466 / 716
        assert batches == [180, 466 - 180, 716 - 466], batches
    assert set(configs) == {"cfg"}, configs


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
