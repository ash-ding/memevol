"""Tests for cc + hipporag2 multi-dataset support (shared registry, runner,
memo structures). Zero-dependency runner:

    baselines/venv/bin/python tests/test_baselines_multidataset.py
    venv/bin/python tests/test_baselines_multidataset.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_shared_registry_resolves_all_four():
    from baselines.registry import REGISTRY, DATASETS, resolve
    from datasets.locomo.workflow import LoCoMoWorkflow
    from datasets.locomo.env import LoCoMoRecorder
    assert DATASETS == ["dynamicmem", "locomo", "longmemeval_m", "longmemeval_s"]
    wf, env, rec = resolve("locomo")
    assert wf is LoCoMoWorkflow and rec is LoCoMoRecorder and hasattr(env, "get_task_list")


def test_alma_still_imports_shared_registry():
    # ALMA re-points to the shared registry; its own module still exposes resolve.
    from baselines.alma.registry import resolve as alma_resolve
    from baselines.registry import resolve as shared_resolve
    assert alma_resolve is shared_resolve


def test_answer_query_hook_default_and_override():
    from common.workflow import BaseWorkflow
    from datasets.dynamicmem.workflow import DynamicMemWorkflow
    # both classes expose the hook
    assert hasattr(BaseWorkflow, "_answer_query")
    assert hasattr(DynamicMemWorkflow, "_answer_query")

    # a pass-through mixin overriding the hook returns the method's answer
    import asyncio
    class _PassThrough:
        async def _answer_query(self, agent, system_msg, user_msg, retrieved):
            return retrieved.get("cc_answer", "")
    class _W(_PassThrough, BaseWorkflow):
        # BaseWorkflow is an ABC with several other abstract hooks unrelated
        # to _answer_query; stub them so the class can be instantiated (none
        # of these are exercised by this test — only _answer_query is).
        async def load_user_data(self, user_dir, eval_n_qa): return (None, [])
        async def phase1_log_init(self, recorder, chunk): return None
        def build_query_recorder_init(self, init_data, qa): return {}
        def build_qa_prompt(self, query, retrieved, qa_metadata, reference=""): return [
            {"role": "system", "content": ""}, {"role": "user", "content": ""}
        ]
        def extract_relevant_context(self, qa, init_data): return None
        def build_qa_metadata(self, qa): return {}
        async def log_qa_step(self, **kwargs): return None
    # instantiate minimally: BaseWorkflow needs memo_class; use a dummy
    from common.harness_base import MemoStructure
    class _Memo(MemoStructure):
        async def general_update(self, r): return None
        async def general_retrieve(self, r): return {}
    w = _W(memo_class=_Memo, model="gpt-5-mini", update_type="all_at_once")
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            w._answer_query(agent=None, system_msg="s", user_msg="u", retrieved={"cc_answer": "HELLO"})
        )
    finally:
        loop.close()
    assert out == "HELLO"


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
