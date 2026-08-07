"""evaluate_memo (the pure, execution-independent evaluator) promotes/eliminates
through the staged gauntlet; sample_seed rides in every stage spec.
    uv run python tests/test_evaluate_memo.py
"""
import sys, asyncio, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_gauntlet_eliminates_below_threshold():
    # reward 0.0 → normalized 0.0 < locomo stage1 threshold 0.30 → eliminated
    seen = []
    out = _run_evaluate_memo(0.0, seen, progressive=True)
    assert seen == ["stage1"], seen                 # stopped after stage1
    assert out["eliminated"] is True and out["stage"] == 1.0


def test_gauntlet_passes_seed_into_every_stage_spec():
    # sample_seed rides every stage spec (constant → nesting holds); captured
    # off the workflow's stage_spec by the fake run_all_users.
    specs = []
    async def _capture(self, task_list, *, stage="stage3", stage_spec=None,
                       max_sample_concurrent=6):
        specs.append((stage, (stage_spec or {}).get("sample_seed")))
        recs = [_FakeRec(u, 1.0) for u in task_list]
        return recs, len(recs)
    import tempfile, shutil
    from datasets.locomo.workflow import LoCoMoWorkflow
    from common.evaluate import evaluate_memo
    from common.memo_class import MemoClass
    class _M(MemoClass):
        async def retrieve_memory_for_query(self, r): return {}
    out_dir = Path(tempfile.mkdtemp(prefix="test_seed_spec_"))
    orig = LoCoMoWorkflow.run_all_users
    LoCoMoWorkflow.run_all_users = _capture
    try:
        out = asyncio.run(evaluate_memo(
            memo_class=_M, dataset="locomo", split="test", progressive=True,
            out_dir=out_dir, qa_model="m", judge_model="m",
            max_sample_concurrent=1, memory_cache=False, sample_seed="SEED7"))
    finally:
        LoCoMoWorkflow.run_all_users = orig
        shutil.rmtree(out_dir, ignore_errors=True)
    assert [st for st, _ in specs] == ["stage1", "stage2", "stage3"]
    assert all(seed == "SEED7" for _, seed in specs)
    assert out["eliminated"] is False


class _FakeRec:
    def __init__(self, user_id, reward):
        self.user_id = user_id
        self.reward = float(reward)
        self.steps = [{"q": "x", "score": reward}]
        self.failure_info = None


def _run_evaluate_memo(reward, seen, **kwargs):
    """Drive the pure (execution-independent) evaluate_memo with LoCoMo's
    run_all_users monkeypatched — no network, no phases; just the plan /
    promotion / artifact plumbing."""
    import tempfile, shutil
    from pathlib import Path
    from datasets.locomo.workflow import LoCoMoWorkflow
    from common.evaluate import evaluate_memo
    from common.memo_class import MemoClass

    class _StubMemo(MemoClass):
        async def retrieve_memory_for_query(self, r): return {}

    async def _fake(self, task_list, *, stage="stage3", stage_spec=None,
                    max_sample_concurrent=6):
        seen.append(stage)
        recs = [_FakeRec(u, reward) for u in task_list]
        return recs, len(recs)

    out_dir = Path(tempfile.mkdtemp(prefix="test_evaluate_memo_"))
    orig = LoCoMoWorkflow.run_all_users
    LoCoMoWorkflow.run_all_users = _fake
    try:
        return asyncio.run(evaluate_memo(
            memo_class=_StubMemo, dataset="locomo", split="test",
            out_dir=out_dir, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            max_sample_concurrent=1, memory_cache=False, **kwargs))
    finally:
        LoCoMoWorkflow.run_all_users = orig
        shutil.rmtree(out_dir, ignore_errors=True)


def test_evaluate_memo_progressive_gauntlet():
    # evaluate_memo(progressive=True) runs the staged gauntlet in-process —
    # no executor parameter; isolation is the CALLER's wrapper.
    seen = []
    out = _run_evaluate_memo(1.0, seen, progressive=True)
    assert seen == ["stage1", "stage2", "stage3"], seen
    assert out["eliminated"] is False and out["stage"] == 3.0


def test_evaluate_memo_single_stage_one_pass():
    # evaluate_memo(progressive=False) → ONE 'single' pass sized by single_stage.
    from common.evaluate import FULL_STAGE
    seen = []
    out = _run_evaluate_memo(
        1.0, seen, progressive=False,
        single_stage={"n_conversations": 1, "n_qa": 1})
    assert seen == ["single"], seen
    assert out["stage"] == FULL_STAGE and out["eliminated"] is False


def test_evaluate_memo_smoke_sanity_pass():
    # smoke=True → ONE sanity_check-sized pass, stage telemetry 0.0.
    seen = []
    out = _run_evaluate_memo(1.0, seen, progressive=True, smoke=True)
    assert seen == ["sanity"], seen
    assert out["stage"] == 0.0 and out["eliminated"] is False


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
