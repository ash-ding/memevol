"""run_gauntlet promotes/eliminates via injected fakes; sample_seed rides in spec.
    uv run python tests/test_run_gauntlet.py
"""
import sys, asyncio, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cfg():
    from common.evaluate import DEFAULT_STAGES
    return {"locomo": {"stages": DEFAULT_STAGES["locomo"]}}


def test_gauntlet_eliminates_below_threshold():
    from common.evaluate import run_gauntlet
    seen = []
    async def run_stage(ds, stage, spec):
        seen.append(stage); return None
    # stage1 normalized 0.0 < locomo stage1 threshold 0.30 → eliminated after stage1
    def read_metrics(ds, stage):
        return {"raw_score": 0.0, "score_max": 1, "tokens": {"x": 1}}
    out = asyncio.run(run_gauntlet(
        datasets_config=_cfg(), progressive=True, smoke=False,
        sample_seed_for=lambda ds: None, run_stage_fn=run_stage, read_metrics_fn=read_metrics))
    assert seen == ["stage1"]                       # stopped after stage1
    assert out["locomo"]["eliminated"] is True


def test_gauntlet_full_runs_all_stages_and_passes_seed():
    from common.evaluate import run_gauntlet
    specs = []
    async def run_stage(ds, stage, spec):
        specs.append((stage, spec.get("sample_seed"))); return None
    def read_metrics(ds, stage):
        return {"raw_score": 1.0, "score_max": 1, "tokens": {}}
    out = asyncio.run(run_gauntlet(
        datasets_config=_cfg(), progressive=True, smoke=False,
        sample_seed_for=lambda ds: "SEED7", run_stage_fn=run_stage, read_metrics_fn=read_metrics))
    assert [s for s, _ in specs] == ["stage1", "stage2", "stage3"]   # promoted through
    assert all(seed == "SEED7" for _, seed in specs)                 # seed injected into every stage spec
    assert out["locomo"]["eliminated"] is False


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
    from common.harness_base import MemoStructure

    class _StubMemo(MemoStructure):
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
