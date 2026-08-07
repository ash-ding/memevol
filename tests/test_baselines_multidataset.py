"""Shared-contract tests for the baseline harness layer (registry resolution,
BaseWorkflow/DynamicMemWorkflow default-answer signature, sizing wire specs,
task-list derivation, eval_utility's run_baseline). This file
is BASELINE-FREE — it must NOT import any concrete baseline's memo (cc,
hipporag2, amem); those live in their own tests/test_<name>_baseline.py, run
in that baseline's own venv. This file runs in the shared dev/test env:

    uv run python tests/test_baselines_multidataset.py
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
    from common.memo_class import MemoClass

    calls = {}

    async def _fake_ask(self, user_msg, **kwargs):
        calls["user_msg"] = user_msg
        calls["kwargs"] = kwargs
        calls["messages"] = self.messages
        return "ANSWER"

    class _Memo(MemoClass):
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
        async def load_user_data(self, user_dir, eval_n_qa, sample_seed=None):
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
    from common.memo_class import MemoClass

    calls = {}

    async def _fake_ask(self, user_msg, **kwargs):
        calls["user_msg"] = user_msg
        calls["kwargs"] = kwargs
        calls["messages"] = self.messages
        return "ANSWER"

    async def _fake_judge_item(self, item, raw_answer):
        return 1.0, "fake-judge", raw_answer, {}

    class _Memo(MemoClass):
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


# -------------------- eval_utility (shared runner + data-alignment) --------------------

def test_single_stage_whole_split_matches_main_method():
    """An all-null (or omitted-field) `single_stage` MUST size to the main
    method's progressive=false wire spec — otherwise DynamicMem silently drops out
    of the TCE checkpoint path (its branch keys on the n_checkpoints KEY).
    Both sides go through common.evaluate (shared with forge)."""
    from common.evaluate import single_stage_wire_spec
    from forge.orchestrator import full_wire_spec   # test-only import (baselines never import forge)
    for ds in ("dynamicmem", "locomo", "longmemeval_s", "longmemeval_m"):
        assert single_stage_wire_spec(ds, {}) == full_wire_spec(ds), ds
    # dynamicmem single_stage carries the n_checkpoints KEY → TCE path
    assert "n_checkpoints" in single_stage_wire_spec("dynamicmem", {})
    # a sized single_stage keeps the TCE keys present (native field n_users → n_samples)
    sized = single_stage_wire_spec("dynamicmem", {"n_users": 2})
    assert sized["n_samples"] == 2 and "n_checkpoints" in sized


def test_task_list_identical_to_main_method():
    """The wire spec's n_samples field must map onto env.get_task_list exactly —
    evaluate_memo (shared by forge's container AND every baseline) derives its
    task list as get_task_list(split, spec["n_samples"], seed) — so an all-null
    single_stage is the whole split and a capped one is the same prefix the
    main method sees."""
    from common.evaluate import single_stage_wire_spec
    from datasets.locomo import env as locomo_env
    from datasets.longmemeval import env as lme_env
    from datasets.dynamicmem import env as dm_env
    def task_list(env, ds, single_stage):
        n = single_stage_wire_spec(ds, single_stage)["n_samples"]
        return env.get_task_list("test", None if n is None else int(n))
    # whole test split (all-null single_stage) == get_task_list("test", None) == heldout split
    assert task_list(locomo_env, "locomo", {}) == locomo_env.get_task_list("test", None)
    assert task_list(lme_env, "longmemeval_s", {}) == lme_env.get_task_list("test", None)
    assert task_list(dm_env, "dynamicmem", {}) == dm_env.get_task_list("test", None)
    # capped units == get_task_list("test", N)
    assert task_list(locomo_env, "locomo", {"n_conversations": 2}) == locomo_env.get_task_list("test", 2)


def test_memo_constructor_config():
    """Config reaches memo instances through the CONSTRUCTOR (2026-08-06,
    replacing the make_memo_class dynamic-subclass injection): each instance
    gets its own private copy at self.config, mutations never leak across
    instances (the fresh-instance-per-user guarantee extends to config), and
    plain instances pickle without any class-anchoring magic."""
    import pickle
    from common.memo_class import MemoClass

    cfg = {"model": "x", "k": 3}
    a = MemoClass(config=cfg)
    b = MemoClass(config=cfg)
    assert a.config == {"model": "x", "k": 3}
    # per-instance copy: mutating one instance (or the caller dict) leaks nowhere
    a.config["k"] = 99
    cfg["model"] = "mutated"
    assert b.config == {"model": "x", "k": 3}
    # zero-arg still works (forge-evolved harnesses are never handed a config)
    assert MemoClass().config == {}
    # plain instances pickle round-trip, config included (memcache relies on it)
    restored = pickle.loads(pickle.dumps(a))
    assert restored.config["k"] == 99


# -------------------- integration: run_baseline end-to-end (locomo) --------------------

def test_run_baseline_locomo_end_to_end():
    """Deterministic full drive of baselines.harness.eval_utility.run_baseline on ONE
    real locomo test-split unit — not a scoped-down slice. This exercises:
    registry.resolve -> the shared task-list derivation (real locomo10.json split) ->
    LoCoMoWorkflow construction -> run_all_users -> run_single_user (REAL
    Phase 1 ingestion against the actual conv-47 sessions, REAL
    build_qa_prompt/build_qa_metadata/log_qa_step dispatch) ->
    save_full_traces -> alma's _build_score_json -> score.json /
    token_usage.json / traces/*.json written to disk.

    Only the two legs that would hit a real network are stubbed, using the
    SAME monkeypatch pattern as tests/test_phase2_failures.py:
      - common.llm.Agent.ask (the shared QA agent) -> a canned answer
      - LoCoMoWorkflow.judge (the judge)           -> a forced score=1

    A stub MemoClass supplies retrieve_memory_for_query's canned passages;
    build_memory_from_data is a no-op (Phase 1 ingestion still runs for real, it
    just has nothing to persist).
    """
    import asyncio
    import json
    import shutil
    import tempfile
    from pathlib import Path

    import common.llm as llm_mod
    from baselines.harness.eval_utility import run_baseline
    from common.memo_class import MemoClass
    from common.evaluate import single_stage_wire_spec
    from common.sampling import derive_sample_seed
    from datasets.locomo.workflow import LoCoMoWorkflow

    class _StubMemo(MemoClass):
        async def build_memory_from_data(self, r):
            return None

        async def retrieve_memory_for_query(self, r):
            return {"passages": ["Amy told Bob she adopted a new puppy."]}

    async def _fake_ask(self, user_input, **kw):
        return "a canned answer"

    async def _fake_judge(self, query, predicted, reference, qa_metadata=None):
        return 1, "fake-judge: forced pass"

    # progressive=False sizes from single_stage; run_baseline threads a fixed
    # step-0 sample_seed (honoring sampling_seed=42) into the split derivation,
    # so pin the exact deterministic unit via the SAME seeded spec run_baseline
    # builds internally — a split-logic regression still changes WHICH unit runs.
    single_stage = {"n_conversations": 1, "n_qa": 1}
    seed = derive_sample_seed(42, 0, "locomo")
    spec = {**single_stage_wire_spec("locomo", single_stage), "sample_seed": seed}
    from datasets.locomo.env import get_task_list as _locomo_tasks
    expected = _locomo_tasks("test", int(spec["n_samples"]), seed=spec["sample_seed"])
    assert len(expected) == 1, expected
    the_user = expected[0]

    orig_ask, orig_judge = llm_mod.Agent.ask, LoCoMoWorkflow.judge
    llm_mod.Agent.ask, LoCoMoWorkflow.judge = _fake_ask, _fake_judge
    out_dir = Path(tempfile.mkdtemp(prefix="test_run_baseline_locomo_"))
    try:
        try:
            score = asyncio.run(run_baseline(
                dataset="locomo", split="test",
                single_stage=single_stage,
                memo_class=_StubMemo,
                qa_model="gpt-5-mini", judge_model="gpt-5-mini",
                out_dir=out_dir, max_sample_concurrent=1,
                progressive=False, sampling_seed=42,
            ))
        finally:
            llm_mod.Agent.ask, LoCoMoWorkflow.judge = orig_ask, orig_judge

        assert (out_dir / "score.json").exists()
        assert (out_dir / "token_usage.json").exists()
        # Unified: single pass runs through evaluate_memo's one-item plan, so traces
        # land under out_dir/single/traces/ (like forge's progressive=false), and
        # run_baseline returns the shared metrics dict.
        trace_files = sorted((out_dir / "single" / "traces").glob("*.json"))
        assert len(trace_files) == 1, trace_files
        trace = json.loads(trace_files[0].read_text())
        assert trace["user_id"] == the_user
        assert trace["n_qa"] == 1
        assert trace["failure_info"] is None
        assert trace["steps"][0]["predicted"] == "a canned answer"
        assert trace["steps"][0]["score"] == 1

        assert score["locomo"]["raw_score"] == 1.0, score
        # The detailed score.json (per_user / invalid_users) is on disk at the root.
        disk_score = json.loads((out_dir / "score.json").read_text())
        assert disk_score["benchmark_eval_score"]["benchmark_overall_eval_score"] == 1.0
        assert disk_score["per_user"][the_user]["n_qa"] == 1
        assert disk_score["invalid_users"] == []
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
