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
    from baselines.evolve.alma.registry import resolve as alma_resolve
    from baselines.registry import resolve as shared_resolve
    assert alma_resolve is shared_resolve


def test_base_workflow_default_answer_call_signature():
    """When memo.use_memory_to_answer defers (returns None, the default), the
    answer step in run_single_user MUST set agent.messages=[{system}] then
    call agent.ask(user_msg, with_history=False, reasoning_effort=...) —
    exact byte-identity of the pre-refactor _answer_query default, now
    inlined at the call site instead of a separate overridable hook."""
    import asyncio
    import common.llm as llm_mod
    from common.workflow import BaseWorkflow
    from common.harness_base import MemoStructure

    calls = {}

    async def _fake_ask(self, user_msg, **kwargs):
        calls["user_msg"] = user_msg
        calls["kwargs"] = kwargs
        calls["messages"] = self.messages
        return "ANSWER"

    class _Memo(MemoStructure):
        async def retrieve_memory_for_query(self, r): return {}
        # use_memory_to_answer NOT overridden -> defaults to None (defers to agent)

    class _Rec:
        def __init__(self):
            self.init = {}
            self.steps = []
            self.user_id = ""
            self.failure_info = None
        async def set_reward(self, r): self.reward = r

    class _W(BaseWorkflow):
        recorder_class = _Rec
        judge_score_max = 1
        async def load_user_data(self, user_dir, eval_n_qa):
            return [], [{"query": "USR", "reference": "r", "metadata": {}}]
        async def phase1_log_init(self, recorder, chunk): return None
        def build_query_recorder_init(self, init_data, qa): return {}
        def build_qa_prompt(self, query, retrieved, qa_metadata, reference=""):
            return [{"role": "system", "content": "SYS"}, {"role": "user", "content": query}]
        def extract_relevant_context(self, qa, init_data): return None
        def build_qa_metadata(self, qa): return {}
        async def log_qa_step(self, recorder, query, predicted, reference, score,
                              judge_reason, qa_metadata, retrieved_memory, relevant_context):
            recorder.steps.append({"query": query, "predicted": predicted, "score": score})
        async def judge(self, query, predicted, reference, qa_metadata=None):
            return 1, "fake-judge"

    orig_ask = llm_mod.Agent.ask
    llm_mod.Agent.ask = _fake_ask
    try:
        w = _W(memo_class=_Memo, model="gpt-5-mini")
        loop = asyncio.new_event_loop()
        rec = loop.run_until_complete(
            w.run_single_user("user1", stage="sanity", stage_spec={"n_qa": 1}))
    finally:
        llm_mod.Agent.ask = orig_ask

    assert calls["messages"] == [{"role": "system", "content": "SYS"}]
    assert calls["user_msg"] == "USR"
    assert calls["kwargs"].get("with_history") is False
    assert "reasoning_effort" in calls["kwargs"]
    assert rec.steps[0]["predicted"] == "ANSWER"


def test_dynamicmem_default_answer_call_signature():
    """When memo.use_memory_to_answer defers, DynamicMem's answer step in
    _run_item MUST call agent.ask(prompt, reasoning_effort=...) with NO
    with_history kwarg and NO system message set — exact byte-identity of
    the pre-refactor _answer_query default, now inlined in _run_item."""
    import asyncio
    import common.llm as llm_mod
    from datasets.dynamicmem.workflow import DynamicMemWorkflow
    from datasets.dynamicmem.env import DynamicMemRecorder
    from common.harness_base import MemoStructure

    calls = {}

    async def _fake_ask(self, user_msg, **kwargs):
        calls["user_msg"] = user_msg
        calls["kwargs"] = kwargs
        calls["messages"] = self.messages
        return "ANSWER"

    async def _fake_judge_item(self, item, raw_answer):
        return 1.0, "fake-judge", raw_answer, {}

    class _Memo(MemoStructure):
        async def retrieve_memory_for_query(self, r): return {}
        # use_memory_to_answer NOT overridden -> defaults to None (defers to agent)

    item = {
        "task_family": "apply_service",   # != TASK_FAMILY_STATE_COMPLETION -> Task C branch
        "query": "PROMPT",
        "service_family": "user_communication",
        "output_template": None,
        "checkpoint_id": "cp1",
        "state_key": "",
    }

    orig_ask = llm_mod.Agent.ask
    orig_judge_item = DynamicMemWorkflow._judge_item
    llm_mod.Agent.ask = _fake_ask
    DynamicMemWorkflow._judge_item = _fake_judge_item
    try:
        w = DynamicMemWorkflow(memo_class=_Memo, model="gpt-5-mini")
        memo = _Memo()
        recorder = DynamicMemRecorder()
        loop = asyncio.new_event_loop()
        err = loop.run_until_complete(w._run_item(memo, recorder, item, []))
    finally:
        llm_mod.Agent.ask = orig_ask
        DynamicMemWorkflow._judge_item = orig_judge_item

    assert err is None
    assert "PROMPT" in calls["user_msg"]   # item["query"] (task_body) embedded in the built TCE prompt
    assert "reasoning_effort" in calls["kwargs"]
    assert "with_history" not in calls["kwargs"]      # DynamicMem default does NOT pass with_history
    assert calls["messages"] == [{"role": "system", "content": ""}]  # untouched — no system override
    assert recorder.steps[0]["predicted"] == "ANSWER"


# -------------------- eval_common (shared runner + data-alignment) --------------------

def test_parse_stage_spec_raw_user_overrides():
    from baselines.harness.eval_common import parse_stage_spec
    assert parse_stage_spec(None) == {}
    assert parse_stage_spec('{"n_samples": 3, "n_qa": 5}') == {"n_samples": 3, "n_qa": 5}


def test_family_full_spec_matches_main_method():
    """The default (no --stage-spec) effective spec MUST equal the main
    method's coverage=full wire spec — otherwise DynamicMem silently drops out
    of the TCE checkpoint path (its branch keys on the n_checkpoints KEY)."""
    from baselines.harness.eval_common import family_full_spec, effective_stage_spec
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
    from baselines.harness.eval_common import resolve_task_list, effective_stage_spec
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
    from baselines.harness.eval_common import make_memo_class
    from common.harness_base import MemoStructure
    class Base(MemoStructure):
        async def build_memory_from_data(self, r): return None
        async def retrieve_memory_for_query(self, r): return {"cfg": self._cfg}
    Cls = make_memo_class(Base, model="x", k=3)
    inst = Cls()  # workflow instantiates with NO args
    assert inst._cfg == {"model": "x", "k": 3}
    assert isinstance(inst, MemoStructure)


# -------------------- hipporag2 (retrieval MemoStructure) --------------------

def test_hipporag_memo_passage_conversion():
    from baselines.harness.hipporag2.memo import _init_to_passages
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
    """retrieve_memory_for_query returns retrieved passages as the context dict; the
    shared QA agent (not HippoRAG's own reader) answers."""
    import asyncio
    from baselines.harness.hipporag2.memo import HippoRAGMemo
    from baselines.harness.eval_common import make_memo_class

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
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))
    _Rec.init = {"conversation": _Rec.init["conversation"], "query": "what did A say?"}
    out = loop.run_until_complete(memo.retrieve_memory_for_query(_Rec()))
    assert "passages" in out and out["passages"]


# -------------------- cc (native-answer MemoStructure) --------------------

def test_cc_use_memory_to_answer_runs_cc():
    import asyncio
    from baselines.harness.cc.memo import CCMemo
    from baselines.harness.eval_common import make_memo_class
    async def _fake_ask(question, tmp_dir, model, max_turns, system_prompt=None):
        return ("CCANS:" + question, {}, [])
    Cls = make_memo_class(CCMemo, model="sonnet", max_turns=5, _ask_cc=_fake_ask)
    memo = Cls()
    class _Rec:
        user_id = ""
        init = {"sessions": [{"session_id": "s", "date": "d",
                "messages": [{"role": "user", "content": "hi"}]}], "query": "q?"}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))   # writes context to tmp_dir
    ans = loop.run_until_complete(memo.use_memory_to_answer(_Rec(), {}, "FORMATTED PROMPT"))
    assert ans == "CCANS:FORMATTED PROMPT"


def test_cc_memo_retrieve_empty_and_run_cc_answers():
    import asyncio
    from baselines.harness.cc.memo import CCMemo
    from baselines.harness.eval_common import make_memo_class
    async def _fake_ask(question, tmp_dir, model, max_turns, system_prompt=None):
        return ("STUB:" + question, {}, [])
    Cls = make_memo_class(CCMemo, model="sonnet", max_turns=5, _ask_cc=_fake_ask)
    memo = Cls()
    class _Rec:
        user_id = ""
        init = {"sessions": [{"session_id": "s", "date": "d",
                "messages": [{"role": "user", "content": "hi"}]}], "query": "q?"}
    loop = asyncio.new_event_loop()
    loop.run_until_complete(memo.build_memory_from_data(_Rec()))
    out = loop.run_until_complete(memo.retrieve_memory_for_query(_Rec()))
    assert out == {}                                   # cc injects no memory; answers via files
    ans, _u, _t = loop.run_until_complete(memo._run_cc("FORMATTED PROMPT"))
    assert ans == "STUB:FORMATTED PROMPT"


# -------------------- integration: run_baseline end-to-end (locomo) --------------------

def test_run_baseline_locomo_end_to_end():
    """Deterministic full drive of baselines.harness.eval_common.run_baseline on ONE
    real locomo test-split unit — not a scoped-down slice. This exercises:
    registry.resolve -> resolve_task_list (real locomo10.json split) ->
    LoCoMoWorkflow construction -> run_all_users -> run_single_user (REAL
    Phase 1 ingestion against the actual conv-47 sessions, REAL
    build_qa_prompt/build_qa_metadata/log_qa_step dispatch) ->
    save_full_traces -> alma's _build_score_json -> score.json /
    token_usage.json / traces/*.json written to disk.

    Only the two legs that would hit a real network are stubbed, using the
    SAME monkeypatch pattern as tests/test_phase2_failures.py:
      - common.llm.Agent.ask (the shared QA agent) -> a canned answer
      - LoCoMoWorkflow.judge (the judge)           -> a forced score=1

    A stub MemoStructure supplies retrieve_memory_for_query's canned passages;
    build_memory_from_data is a no-op (Phase 1 ingestion still runs for real, it
    just has nothing to persist).
    """
    import asyncio
    import json
    import shutil
    import tempfile
    from pathlib import Path

    import common.llm as llm_mod
    from baselines.harness.eval_common import effective_stage_spec, resolve_task_list, run_baseline
    from common.harness_base import MemoStructure
    from datasets.locomo.workflow import LoCoMoWorkflow

    class _StubMemo(MemoStructure):
        async def build_memory_from_data(self, r):
            return None

        async def retrieve_memory_for_query(self, r):
            return {"passages": ["Amy told Bob she adopted a new puppy."]}

    async def _fake_ask(self, user_input, **kw):
        return "a canned answer"

    async def _fake_judge(self, query, predicted, reference, qa_metadata=None):
        return 1, "fake-judge: forced pass"

    # Sanity-check the exact unit this test is deterministic over — a
    # regression in the split logic would silently change WHICH sample gets
    # exercised, so pin it explicitly rather than only trusting resolve_task_list.
    expected = resolve_task_list("locomo", "test", effective_stage_spec("locomo", {"n_samples": 1}))
    assert expected == ["conv-47"], expected

    orig_ask, orig_judge = llm_mod.Agent.ask, LoCoMoWorkflow.judge
    llm_mod.Agent.ask, LoCoMoWorkflow.judge = _fake_ask, _fake_judge
    out_dir = Path(tempfile.mkdtemp(prefix="test_run_baseline_locomo_"))
    try:
        try:
            score = asyncio.run(run_baseline(
                dataset="locomo", split="test",
                user_stage_spec={"n_samples": 1, "n_qa": 1},
                memo_class=_StubMemo,
                qa_model="gpt-5-mini", judge_model="gpt-5-mini",
                out_dir=out_dir, max_sample_concurrent=1,
            ))
        finally:
            llm_mod.Agent.ask, LoCoMoWorkflow.judge = orig_ask, orig_judge

        assert (out_dir / "score.json").exists()
        assert (out_dir / "token_usage.json").exists()
        trace_files = sorted((out_dir / "traces").glob("*.json"))
        assert len(trace_files) == 1, trace_files
        trace = json.loads(trace_files[0].read_text())
        assert trace["user_id"] == "conv-47"
        assert trace["n_qa"] == 1
        assert trace["failure_info"] is None
        assert trace["steps"][0]["predicted"] == "a canned answer"
        assert trace["steps"][0]["score"] == 1

        assert score["benchmark_eval_score"]["benchmark_overall_eval_score"] == 1.0
        assert score["per_user"]["conv-47"]["n_qa"] == 1
        assert score["invalid_users"] == []
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


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
