"""Tests for Phase-2 failure visibility (audit M1) — a harness whose
`general_retrieve` (or the QA agent) fails must repopulate
`recorder.failure_info` so the orchestrator's sanity gate can see it,
instead of silently scoring 0 and burning a full stage1 eval.

Zero-dependency runner (no network — the QA agent is never reached or is
monkeypatched):

    venv/bin/python tests/test_phase2_failures.py
"""
import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from common.workflow import BaseWorkflow
from forge.orchestrator import _collect_sanity_errors


# ---------------- scaffolding ----------------

class FakeRecorder:
    def __init__(self):
        self.init: Dict = {}
        self.steps: List[Dict] = []
        self.user_id = ""
        self.reward = 0.0
        self.failure_info: Optional[str] = None

    async def set_reward(self, r: float) -> None:
        self.reward = r


class GoodMemo:
    async def general_update(self, recorder):
        pass

    async def general_retrieve(self, recorder):
        return {"context": "ok"}


class BadRetrieveMemo(GoodMemo):
    async def general_retrieve(self, recorder):
        raise NameError("name 'chroma_db' is not defined")  # classic LLM-written bug


class FakeWorkflow(BaseWorkflow):
    recorder_class = FakeRecorder  # type: ignore[assignment]
    judge_score_max = 1

    def __init__(self, memo_class, n_qa=3, **kw):
        super().__init__(memo_class=memo_class, model="test-model", **kw)
        self._n_qa = n_qa

    async def load_user_data(self, user_dir, eval_n_qa):
        n = self._n_qa if eval_n_qa is None else int(eval_n_qa)
        qa = [{"query": f"q{i}", "reference": f"r{i}", "metadata": {}}
              for i in range(n)]
        return [], qa

    async def phase1_log_init(self, recorder, chunk):
        recorder.init = {"chunk": chunk}

    def build_query_recorder_init(self, init_data, qa):
        return {"query": qa["query"]}

    def build_qa_prompt(self, query, retrieved, qa_metadata, reference=""):
        return [{"role": "system", "content": ""}, {"role": "user", "content": query}]

    def extract_relevant_context(self, qa, init_data):
        return None

    def build_qa_metadata(self, qa):
        return dict(qa.get("metadata", {}))

    async def log_qa_step(self, recorder, query, predicted, reference, score,
                          judge_reason, qa_metadata, retrieved_memory,
                          relevant_context):
        recorder.steps.append({
            "query": query, "predicted": predicted, "score": score,
            "judge_reason": judge_reason, "qa_metadata": qa_metadata,
            "retrieved_memory": retrieved_memory,
        })

    async def judge(self, query, predicted, reference, qa_metadata=None):
        return 1, "fake-judge: correct"


# ---------------- failure_info repopulation ----------------

def test_retrieve_failure_sets_failure_info():
    wf = FakeWorkflow(BadRetrieveMemo)
    rec = asyncio.run(wf.run_single_user("user_x", stage="sanity",
                                         stage_spec={"n_qa": 3}))
    assert rec.failure_info, "failure_info must be repopulated (audit M1)"
    assert "retrieve=3/3" in rec.failure_info, rec.failure_info
    assert "NameError" in rec.failure_info
    assert len(rec.steps) == 3
    assert all(s["score"] == 0 for s in rec.steps)
    assert all("[Phase2_Retrieve_ERROR]" in s["judge_reason"] for s in rec.steps)


def test_agent_failure_sets_failure_info_and_marker():
    import common.llm as llm_mod

    async def _boom(self, *a, **kw):
        raise TimeoutError("simulated transport outage")

    orig = llm_mod.Agent.ask
    llm_mod.Agent.ask = _boom
    try:
        wf = FakeWorkflow(GoodMemo)
        rec = asyncio.run(wf.run_single_user("user_y", stage="sanity",
                                             stage_spec={"n_qa": 2}))
    finally:
        llm_mod.Agent.ask = orig

    assert rec.failure_info and "answer=2/2" in rec.failure_info, rec.failure_info
    assert all("[Phase2_Answer_ERROR]" in s["judge_reason"] for s in rec.steps)
    # retrieved memory is preserved in the step even when answering failed
    assert all(s["retrieved_memory"] == {"context": "ok"} for s in rec.steps)


def test_clean_run_leaves_failure_info_none():
    import common.llm as llm_mod

    async def _ok(self, *a, **kw):
        return "a fine answer"

    orig = llm_mod.Agent.ask
    llm_mod.Agent.ask = _ok
    try:
        wf = FakeWorkflow(GoodMemo)
        rec = asyncio.run(wf.run_single_user("user_z", stage="sanity",
                                             stage_spec={"n_qa": 3}))
    finally:
        llm_mod.Agent.ask = orig
    assert rec.failure_info is None
    assert rec.reward == 1.0
    assert all(s["score"] == 1 for s in rec.steps)


# ---------------- judge-outage abort (audit D2) ----------------

class OutageJudgeWorkflow(FakeWorkflow):
    """Every judge call fails at the transport level."""
    async def judge(self, query, predicted, reference, qa_metadata=None):
        return 0, "Judge error: Request timed out."


class FlakyJudgeWorkflow(FakeWorkflow):
    """Judge fails on every other step (50% — at the threshold, not above)."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._n = 0

    async def judge(self, query, predicted, reference, qa_metadata=None):
        self._n += 1
        if self._n % 2 == 0:
            return 0, "Judge error: Request timed out."
        return 1, "fake-judge: correct"


def _run_all(wf, n_users=2, n_qa=2):
    import common.llm as llm_mod

    async def _ok(self, *a, **kw):
        return "answer"

    orig = llm_mod.Agent.ask
    llm_mod.Agent.ask = _ok
    try:
        return asyncio.run(wf.run_all_users(
            task_list=[f"user_{i}" for i in range(n_users)],
            stage="stage1", stage_spec={"n_qa": n_qa},
        ))
    finally:
        llm_mod.Agent.ask = orig


def test_judge_outage_aborts_eval():
    wf = OutageJudgeWorkflow(GoodMemo)
    try:
        _run_all(wf)
        assert False, "majority judge transport failure must abort the eval"
    except RuntimeError as exc:
        assert "judge outage" in str(exc), exc


def test_judge_half_failures_do_not_abort():
    wf = FlakyJudgeWorkflow(GoodMemo)
    results, n = _run_all(wf)
    assert n == 2  # completes normally: 50% is not ABOVE the threshold


def test_phase2_error_steps_dont_count_as_judge_outage():
    """A broken harness (all retrieve errors) means the judge never ran —
    that's a sanity-gate matter, not a judge outage."""
    wf = FakeWorkflow(BadRetrieveMemo)
    results, n = _run_all(wf)
    assert n == 2
    assert all(r.failure_info for r in results)


# ---------------- sanity gate consumes failure_info ----------------

def _write_sanity_score(harness_dir: Path, ds: str, per_user: Dict) -> None:
    d = harness_dir / ds / "sanity"
    d.mkdir(parents=True, exist_ok=True)
    (d / "score.json").write_text(json.dumps({
        "benchmark_eval_score": {"benchmark_overall_eval_score": 0.0},
        "per_user": per_user,
        "invalid_users": [],
    }))


def test_sanity_gate_fails_on_failure_info():
    with tempfile.TemporaryDirectory() as d:
        hd = Path(d)
        _write_sanity_score(hd, "locomo", {
            "user_x": {"reward": 0.0, "n_qa": 3,
                       "failure_info": "phase2_errors: retrieve=3/3 answer=0/3; "
                                       "first: NameError: ..."},
        })
        passed, trace = _collect_sanity_errors(hd, {"locomo": {}})
        assert not passed, "sanity must fail when failure_info is non-empty"
        assert "phase2_errors" in trace


def test_sanity_gate_passes_clean_users():
    with tempfile.TemporaryDirectory() as d:
        hd = Path(d)
        _write_sanity_score(hd, "locomo", {
            "user_x": {"reward": 1.0, "n_qa": 3, "failure_info": None},
        })
        passed, trace = _collect_sanity_errors(hd, {"locomo": {}})
        assert passed, trace


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
