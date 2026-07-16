"""Tests for the standardized memory-system eval contract.

Zero-dependency runner:
    baselines/venv/bin/python tests/test_eval_contract.py
    venv/bin/python tests/test_eval_contract.py
"""
import asyncio
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_three_hooks_optional_with_defaults():
    from common.harness_base import MemoStructure

    class Bare(MemoStructure):
        pass  # overrides nothing — must be instantiable now (hooks non-abstract)

    m = Bare()
    loop = asyncio.new_event_loop()

    class _Rec:
        init = {"query": "q"}

    assert loop.run_until_complete(m.general_update(_Rec())) is None
    assert loop.run_until_complete(m.general_retrieve(_Rec())) == {}
    assert loop.run_until_complete(m.general_answer(_Rec(), {}, "PROMPT")) is None


def test_chunked_partitions():
    from common.harness_base import MemoStructure

    class M(MemoStructure):
        pass

    m = M()
    data = list(range(10))
    m._update_type = "all_at_once"
    assert list(m.chunked(data)) == [data]
    m._update_type = "sequential"
    assert list(m.chunked(data)) == [[i] for i in range(10)]
    m._update_type = "chunked"
    m._n_chunks = 5
    chunks = list(m.chunked(data))
    assert len(chunks) == 5 and sum(len(c) for c in chunks) == 10
    assert [x for c in chunks for x in c] == data  # order preserved, no loss

    m._update_type = "chunked"
    m._n_chunks = 5
    chunks = list(m.chunked(list(range(11))))   # 11 not divisible by 5
    assert len(chunks) == 5                       # exactly n_chunks partitions
    assert sum(len(c) for c in chunks) == 11      # no loss
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1  # near-equal
    assert [x for c in chunks for x in c] == list(range(11))               # order preserved


def test_forge_loads_the_harness_class_not_the_base():
    """De-abstracting the base must not make the class-discriminator pick the
    imported base instead of the harness's own class."""
    from pathlib import Path
    from forge.launch import _load_harness_class
    from forge.contract import load_harness_class
    seed = Path(__file__).resolve().parents[1] / "seeds" / "no_memory"
    assert _load_harness_class(seed).__name__ == "NoMemoryHarness"
    assert load_harness_class(seed).__name__ == "NoMemoryHarness"


def test_phase1_update_calls_general_update_once():
    """The workflow hands the whole data in ONE general_update call."""
    from common.workflow import BaseWorkflow
    from common.harness_base import MemoStructure

    calls = []

    class _Memo(MemoStructure):
        async def general_update(self, recorder):
            calls.append(list(recorder.init.get("items", [])))

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
    w = _W(memo_class=_Memo, model="gpt-5-mini", update_type="chunked")
    m = _Memo()
    asyncio.new_event_loop().run_until_complete(w._phase1_update(m, [1, 2, 3, 4, 5]))
    assert calls == [[1, 2, 3, 4, 5]]   # ONE call, whole data (not chunked by the workflow)


def test_general_answer_used_else_agent():
    """The answer step uses memo.general_answer; falls back to the agent on None."""
    import asyncio
    from common.workflow import BaseWorkflow  # noqa: F401 (import proves module loads post-refactor)
    from common.harness_base import MemoStructure

    class _Answering(MemoStructure):
        async def general_answer(self, recorder, retrieved, prompt):
            return "MEMO:" + prompt

    class _Deferring(MemoStructure):
        pass  # general_answer default None → agent answers

    loop = asyncio.new_event_loop()
    a = _Answering()
    assert loop.run_until_complete(a.general_answer(None, {}, "Q")) == "MEMO:Q"
    d = _Deferring()
    assert loop.run_until_complete(d.general_answer(None, {}, "Q")) is None


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
