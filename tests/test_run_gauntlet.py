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


def test_evaluate_memo_progressive_forwards_executor():
    # evaluate_memo(progressive=True) → the staged gauntlet, forwarding the
    # Executor's callbacks to run_gauntlet.
    from common.evaluate import evaluate_memo, Executor
    seen = []
    async def run_stage(ds, stage, spec):
        seen.append(stage); return None
    def read_metrics(ds, stage):
        return {"raw_score": 1.0, "score_max": 1, "tokens": {}}
    out = asyncio.run(evaluate_memo(
        datasets_config=_cfg(), progressive=True,
        executor=Executor(run_stage=run_stage, read_metrics=read_metrics)))
    assert seen == ["stage1", "stage2", "stage3"], seen   # progressive=True → the staged gauntlet
    assert out["locomo"]["eliminated"] is False


def test_evaluate_memo_single_stage_one_pass():
    # evaluate_memo(progressive=False) → ONE ('single', spec, None) pass
    # (progressive=False), sized by single_stage. No gauntlet.
    from common.evaluate import evaluate_memo, Executor, FULL_STAGE
    seen = []
    async def run_stage(ds, stage, spec):
        seen.append(stage); return None
    def read_metrics(ds, stage):
        return {"raw_score": 1.0, "score_max": 1, "tokens": {}}
    cfg = {"locomo": {"single_stage": {"n_conversations": 1, "n_qa": 1}}}
    out = asyncio.run(evaluate_memo(
        datasets_config=cfg, progressive=False,
        executor=Executor(run_stage=run_stage, read_metrics=read_metrics)))
    assert seen == ["single"], seen
    assert out["locomo"]["stage"] == FULL_STAGE
    assert out["locomo"]["eliminated"] is False


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
