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


def test_base_workflow_answer_query_default_call_signature():
    """BaseWorkflow._answer_query default MUST set the system message and call
    agent.ask(user_msg, with_history=False, reasoning_effort=...)."""
    import asyncio
    from common.workflow import BaseWorkflow
    from common.harness_base import MemoStructure

    class _Memo(MemoStructure):
        async def general_update(self, r): return None
        async def general_retrieve(self, r): return {}

    calls = {}
    class _StubAgent:
        def __init__(self): self.messages = None
        async def ask(self, user_msg, **kwargs):
            calls["user_msg"] = user_msg
            calls["kwargs"] = kwargs
            calls["messages"] = self.messages
            return "ANSWER"

    # minimal BaseWorkflow: implement the 7 abstract hooks as no-ops
    class _W(BaseWorkflow):
        async def load_user_data(self, *a, **k): return None, []
        async def phase1_log_init(self, *a, **k): return None
        def build_query_recorder_init(self, *a, **k): return {}
        def build_qa_prompt(self, *a, **k): return [{"role": "system", "content": ""}, {"role": "user", "content": ""}]
        def extract_relevant_context(self, *a, **k): return None
        def build_qa_metadata(self, *a, **k): return {}
        async def log_qa_step(self, *a, **k): return None

    w = _W(memo_class=_Memo, model="gpt-5-mini", update_type="all_at_once")
    agent = _StubAgent()
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            w._answer_query(agent, "SYS", "USR", {}))
    finally:
        loop.close()
    assert out == "ANSWER"
    assert calls["messages"] == [{"role": "system", "content": "SYS"}]
    assert calls["user_msg"] == "USR"
    assert calls["kwargs"].get("with_history") is False
    assert "reasoning_effort" in calls["kwargs"]


def test_dynamicmem_answer_query_default_call_signature():
    """DynamicMemWorkflow._answer_query default MUST call
    agent.ask(user_msg, reasoning_effort=...) with NO system set and NO with_history."""
    import asyncio
    from datasets.dynamicmem.workflow import DynamicMemWorkflow
    from common.harness_base import MemoStructure

    class _Memo(MemoStructure):
        async def general_update(self, r): return None
        async def general_retrieve(self, r): return {}

    calls = {}
    class _StubAgent:
        def __init__(self): self.messages = "UNSET"
        async def ask(self, user_msg, **kwargs):
            calls["user_msg"] = user_msg
            calls["kwargs"] = kwargs
            calls["messages"] = self.messages
            return "ANSWER"

    w = DynamicMemWorkflow(memo_class=_Memo, model="gpt-5-mini", update_type="all_at_once")
    agent = _StubAgent()
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            w._answer_query(agent, "", "PROMPT", {}))
    finally:
        loop.close()
    assert out == "ANSWER"
    assert calls["user_msg"] == "PROMPT"
    assert "reasoning_effort" in calls["kwargs"]
    assert "with_history" not in calls["kwargs"]      # DynamicMem default does NOT pass with_history
    assert calls["messages"] == "UNSET"               # DynamicMem default does NOT set agent.messages


# -------------------- eval_common (shared runner + data-alignment) --------------------

def test_parse_stage_spec_raw_user_overrides():
    from baselines.eval_common import parse_stage_spec
    assert parse_stage_spec(None) == {}
    assert parse_stage_spec('{"n_samples": 3, "n_qa": 5}') == {"n_samples": 3, "n_qa": 5}


def test_family_full_spec_matches_main_method():
    """The default (no --stage-spec) effective spec MUST equal the main
    method's coverage=full wire spec — otherwise DynamicMem silently drops out
    of the TCE checkpoint path (its branch keys on the n_checkpoints KEY)."""
    from baselines.eval_common import family_full_spec, effective_stage_spec
    from forge.orchestrator import full_wire_spec   # test-only import (baselines never import forge)
    for ds in ("dynamicmem", "locomo", "longmemeval_s", "longmemeval_m"):
        assert family_full_spec(ds) == full_wire_spec(ds), ds
        assert effective_stage_spec(ds, None) == full_wire_spec(ds), ds
    # dynamicmem full spec carries the n_checkpoints KEY → TCE path
    assert "n_checkpoints" in family_full_spec("dynamicmem")
    # user override merges over the family-full base (keeps the TCE keys present)
    merged = effective_stage_spec("dynamicmem", {"n_samples": 2})
    assert merged["n_samples"] == 2 and "n_checkpoints" in merged


def test_task_list_identical_to_main_method():
    """The baseline's split derivation MUST equal the main method's
    (forge/launch.py:185-189 calls the SAME env.get_task_list)."""
    from baselines.eval_common import resolve_task_list, effective_stage_spec
    from datasets.locomo import env as locomo_env
    from datasets.longmemeval import env as lme_env
    from datasets.dynamicmem import env as dm_env
    # whole test split (default) == get_task_list("test", None) == the heldout split
    assert resolve_task_list("locomo", "test", effective_stage_spec("locomo", None)) == locomo_env.get_task_list("test", None)
    assert resolve_task_list("longmemeval_s", "test", effective_stage_spec("longmemeval_s", None)) == lme_env.get_task_list("test", None)
    assert resolve_task_list("dynamicmem", "test", effective_stage_spec("dynamicmem", None)) == dm_env.get_task_list("test", None)
    # capped units == get_task_list("test", N)
    assert resolve_task_list("locomo", "test", effective_stage_spec("locomo", {"n_samples": 2})) == locomo_env.get_task_list("test", 2)


def test_make_memo_class_no_arg_instantiable():
    from baselines.eval_common import make_memo_class
    from common.harness_base import MemoStructure
    class Base(MemoStructure):
        async def general_update(self, r): return None
        async def general_retrieve(self, r): return {"cfg": self._cfg}
    Cls = make_memo_class(Base, model="x", k=3)
    inst = Cls()  # workflow instantiates with NO args
    assert inst._cfg == {"model": "x", "k": 3}
    assert isinstance(inst, MemoStructure)


# -------------------- hipporag2 (retrieval MemoStructure) --------------------

def test_hipporag_memo_passage_conversion():
    from baselines.hipporag2.memo import _init_to_passages
    # dynamicmem
    ap = _init_to_passages({"app_logs": [{"timestamp": "T", "app_name": "A",
          "api_name": "x", "request": {}, "response": {}, "metadata": {"domain": "d"}}]})
    assert len(ap) == 1 and "App: A" in ap[0]
    # locomo
    lp = _init_to_passages({"conversation": {"speaker_a": "Amy", "speaker_b": "Bob",
          "session_1": [{"speaker": "Amy", "dia_id": "D1:1", "text": "hi"}],
          "session_1_date_time": "2023/01/01"}})
    assert any("hi" in p for p in lp)
    # longmemeval
    sp = _init_to_passages({"sessions": [{"session_id": "s1", "date": "2023/05/20",
          "messages": [{"role": "user", "content": "hello"}]}]})
    assert any("hello" in p for p in sp)


def test_hipporag_memo_retrieve_returns_passages(monkeypatch=None):
    """general_retrieve returns retrieved passages as the context dict; the
    shared QA agent (not HippoRAG's own reader) answers."""
    import asyncio
    from baselines.hipporag2.memo import HippoRAGMemo
    from baselines.eval_common import make_memo_class

    class _FakeHippo:
        def __init__(self, **kw): pass
        def index(self, docs): self._docs = docs
        def retrieve(self, queries, num_to_retrieve=5):
            class _S: docs = ["passage about hi"]
            return [_S()]
    Cls = make_memo_class(HippoRAGMemo, embedding="e", llm_model="m", judge_model="j",
                          _hippo_factory=lambda **kw: _FakeHippo())
    memo = Cls()
    class _Rec:  # minimal recorder
        user_id = "u1"
        init = {"conversation": {"speaker_a": "A", "speaker_b": "B",
                "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"}],
                "session_1_date_time": "d"}}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.general_update(_Rec()))
    _Rec.init = {"conversation": _Rec.init["conversation"], "query": "what did A say?"}
    out = loop.run_until_complete(memo.general_retrieve(_Rec()))
    assert "passages" in out and out["passages"]


# -------------------- cc (native-answer MemoStructure) --------------------

def test_cc_passthrough_returns_native_answer():
    import asyncio
    from baselines.cc.memo import CCPassThroughMixin
    from common.workflow import BaseWorkflow
    from common.harness_base import MemoStructure
    class _Memo(MemoStructure):
        async def general_update(self, r): return None
        async def general_retrieve(self, r): return {}
    class _W(CCPassThroughMixin, BaseWorkflow):
        # BaseWorkflow is an ABC with several other abstract hooks unrelated
        # to _answer_query; stub them so the class can be instantiated (none
        # of these are exercised by this test — only _answer_query is). Same
        # pattern as test_answer_query_hook_default_and_override above.
        async def load_user_data(self, user_dir, eval_n_qa): return (None, [])
        async def phase1_log_init(self, recorder, chunk): return None
        def build_query_recorder_init(self, init_data, qa): return {}
        def build_qa_prompt(self, query, retrieved, qa_metadata, reference=""): return [
            {"role": "system", "content": ""}, {"role": "user", "content": ""}
        ]
        def extract_relevant_context(self, qa, init_data): return None
        def build_qa_metadata(self, qa): return {}
        async def log_qa_step(self, **kwargs): return None
    w = _W(memo_class=_Memo, model="gpt-5-mini", update_type="all_at_once")
    out = asyncio.new_event_loop().run_until_complete(
        w._answer_query(agent=None, system_msg="s", user_msg="u",
                        retrieved={"cc_answer": "NATIVE"}))
    assert out == "NATIVE"


def test_cc_memo_writes_context_and_answers(monkeypatch=None):
    """general_retrieve writes the visible data + runs cc (stubbed) and returns
    its answer under cc_answer."""
    import asyncio
    from baselines.cc.memo import CCMemo
    from baselines.eval_common import make_memo_class
    async def _fake_ask(question, tmp_dir, model, max_turns):
        return ("STUB ANSWER", {}, [{"role": "assistant", "type": "text", "content": "STUB ANSWER"}])
    Cls = make_memo_class(CCMemo, model="sonnet", max_turns=5, _ask_cc=_fake_ask)
    memo = Cls()
    class _Rec:
        user_id = "u1"
        init = {"sessions": [{"session_id": "s", "date": "d",
                "messages": [{"role": "user", "content": "hi"}]}], "query": "q?"}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.general_update(_Rec()))
    out = loop.run_until_complete(memo.general_retrieve(_Rec()))
    assert out["cc_answer"] == "STUB ANSWER"


# -------------------- runner --------------------

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
