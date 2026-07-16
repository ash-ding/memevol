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

    assert loop.run_until_complete(m.build_memory_from_data(_Rec())) is None
    assert loop.run_until_complete(m.retrieve_memory_for_query(_Rec())) == {}
    assert loop.run_until_complete(m.use_memory_to_answer(_Rec(), {}, "PROMPT")) is None


def test_forge_loads_the_harness_class_not_the_base():
    """De-abstracting the base must not make the class-discriminator pick the
    imported base instead of the harness's own class."""
    from pathlib import Path
    from forge.launch import _load_harness_class
    from forge.contract import load_harness_class
    seed = Path(__file__).resolve().parents[1] / "seeds" / "no_memory"
    assert _load_harness_class(seed).__name__ == "NoMemoryHarness"
    assert load_harness_class(seed).__name__ == "NoMemoryHarness"


def test_phase1_update_calls_build_memory_once():
    """The workflow hands the whole data in ONE build_memory_from_data call."""
    from common.workflow import BaseWorkflow
    from common.harness_base import MemoStructure

    calls = []

    class _Memo(MemoStructure):
        async def build_memory_from_data(self, recorder):
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
    w = _W(memo_class=_Memo, model="gpt-5-mini")
    m = _Memo()
    asyncio.new_event_loop().run_until_complete(w._phase1_update(m, [1, 2, 3, 4, 5]))
    assert calls == [[1, 2, 3, 4, 5]]   # ONE call, whole data (not chunked by the workflow)


def test_use_memory_to_answer_used_else_agent():
    """The answer step uses memo.use_memory_to_answer; falls back to the agent on None."""
    import asyncio
    from common.workflow import BaseWorkflow  # noqa: F401 (import proves module loads post-refactor)
    from common.harness_base import MemoStructure

    class _Answering(MemoStructure):
        async def use_memory_to_answer(self, recorder, retrieved, prompt):
            return "MEMO:" + prompt

    class _Deferring(MemoStructure):
        pass  # use_memory_to_answer default None → agent answers

    loop = asyncio.new_event_loop()
    a = _Answering()
    assert loop.run_until_complete(a.use_memory_to_answer(None, {}, "Q")) == "MEMO:Q"
    d = _Deferring()
    assert loop.run_until_complete(d.use_memory_to_answer(None, {}, "Q")) is None


def test_use_memory_to_answer_gets_query_scoped_recorder():
    """Both answer sites pass the query-scoped retrieve_recorder to use_memory_to_answer,
    not the phase-1 recorder — so recorder.init means the same thing everywhere."""
    import inspect
    from common import workflow as bw
    from datasets.dynamicmem import workflow as dw

    base_src = inspect.getsource(bw.BaseWorkflow.run_single_user)
    dm_src = inspect.getsource(dw.DynamicMemWorkflow._run_item)

    # Both should pass retrieve_recorder (spaces removed for robustness)
    assert "use_memory_to_answer(retrieve_recorder" in base_src.replace(" ", ""), \
        "BaseWorkflow.run_single_user must pass retrieve_recorder to use_memory_to_answer"
    assert "use_memory_to_answer(retrieve_recorder" in dm_src.replace(" ", ""), \
        "DynamicMemWorkflow._run_item must pass retrieve_recorder to use_memory_to_answer"


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
