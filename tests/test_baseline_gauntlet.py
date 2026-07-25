"""Progressive gauntlet + memcache + seed for the harness baselines
(baselines/harness/eval_common.py::run_baseline).

Zero-dependency runner (baselines venv; no network — the QA agent + judge are
stubbed, run_all_users is faked for the gauntlet-logic tests):

    baselines/venv/bin/python tests/test_baseline_gauntlet.py

Covers:
  (a) progressive=True runs stages in order + returns a stages.json-shaped result
  (b) a below-threshold stage stops early (eliminated)
  (c) progressive=False still takes the single-stage path UNCHANGED
  (d) whole-split seed no-op (n=None → seeded task SET == unseeded)
  (e) REAL memcache reuse — stage2/stage3 load stage1's pickled Phase-1 memory
      (real LoCoMoWorkflow, only the network legs stubbed)
"""
import sys, asyncio, json, shutil, tempfile, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.harness_base import MemoStructure


# --------------------------------------------------------------------------
# Fakes for the gauntlet-logic tests (a/b/c) — run_all_users is monkeypatched
# so no Phase-1/Phase-2 network work happens; only the promotion/elimination
# plumbing in run_baseline(progressive=True) is exercised.
# --------------------------------------------------------------------------

class _FakeRec:
    def __init__(self, user_id, reward):
        self.user_id = user_id
        self.reward = float(reward)
        self.steps = [{"q": "x", "score": reward}]
        self.failure_info = None


class _StubMemo(MemoStructure):
    async def build_memory_from_data(self, r):
        return None

    async def retrieve_memory_for_query(self, r):
        return {}


def _patch_run_all_users(reward, seen):
    """A fake BaseWorkflow.run_all_users that records the stage it ran and
    returns canned recorders carrying `reward` (constant → controls the
    normalized stage score fed to the threshold gate)."""
    async def _fake(self, task_list, *, stage="stage3", stage_spec=None,
                    max_sample_concurrent=6):
        seen.append(stage)
        recs = [_FakeRec(u, reward) for u in task_list]
        return recs, len(recs)
    return _fake


def _run_with_fake_run_all_users(reward, seen, **run_kwargs):
    from datasets.locomo.workflow import LoCoMoWorkflow
    from baselines.harness.eval_common import run_baseline
    orig = LoCoMoWorkflow.run_all_users
    LoCoMoWorkflow.run_all_users = _patch_run_all_users(reward, seen)
    try:
        return asyncio.run(run_baseline(**run_kwargs))
    finally:
        LoCoMoWorkflow.run_all_users = orig


# --------------------------------------------------------------------------
# (a) progressive=True runs the stage1→2→3 gauntlet in order
# --------------------------------------------------------------------------

def test_progressive_promotes_through_all_stages():
    seen = []
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_promote_"))
    try:
        result = _run_with_fake_run_all_users(
            reward=1.0, seen=seen,
            dataset="locomo", split="test", user_stage_spec=None,
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            progressive=True, memory_cache=False,
        )
        # promoted through every stage, in order
        assert seen == ["stage1", "stage2", "stage3"], seen
        # per-dataset metrics dict (run_gauntlet shape)
        assert result["locomo"]["eliminated"] is False, result
        assert result["locomo"]["stage"] == 3.0, result
        # stages.json written at the out_dir root
        stages_json = out_dir / "stages.json"
        assert stages_json.exists(), "stages.json not written"
        summary = json.loads(stages_json.read_text())
        assert summary["reached"] == "stage3"
        assert summary["eliminated"] is False
        assert list(summary["stages"].keys()) == ["stage1", "stage2", "stage3"]
        # final-stage score.json copied to the dataset root
        assert (out_dir / "score.json").exists()
        # per-stage artifacts exist
        for s in ("stage1", "stage2", "stage3"):
            assert (out_dir / s / "score.json").exists(), s
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# (b) a below-threshold stage stops early (eliminated)
# --------------------------------------------------------------------------

def test_progressive_eliminates_below_threshold():
    seen = []
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_eliminate_"))
    try:
        # reward 0.0 → normalized 0.0 < locomo stage1 threshold 0.30 → eliminated
        result = _run_with_fake_run_all_users(
            reward=0.0, seen=seen,
            dataset="locomo", split="test", user_stage_spec=None,
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            progressive=True, memory_cache=False,
        )
        assert seen == ["stage1"], seen                 # stopped after stage1
        assert result["locomo"]["eliminated"] is True, result
        assert result["locomo"]["stage"] == 1.0, result
        summary = json.loads((out_dir / "stages.json").read_text())
        assert summary["reached"] == "stage1"
        assert summary["eliminated"] is True
        # never ran stage2/stage3
        assert not (out_dir / "stage2").exists()
        assert not (out_dir / "stage3").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# (c) progressive=False takes the single-stage path UNCHANGED
# --------------------------------------------------------------------------

def test_progressive_false_single_stage_unchanged():
    seen = []
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_single_"))
    try:
        score = _run_with_fake_run_all_users(
            reward=1.0, seen=seen,
            dataset="locomo", split="test",
            user_stage_spec={"n_samples": 1, "n_qa": 1},
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            # progressive defaults to False
        )
        assert seen == ["full"], seen                   # single full pass
        # single-stage return is the _build_score_json shape, NOT the gauntlet dict
        assert "benchmark_eval_score" in score, score
        assert score["benchmark_eval_score"]["benchmark_overall_eval_score"] == 1.0
        assert (out_dir / "score.json").exists()
        assert (out_dir / "token_usage.json").exists()
        # single-stage path writes NO stages.json and NO per-stage subdirs
        assert not (out_dir / "stages.json").exists()
        assert not (out_dir / "stage1").exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# (d) whole-split seed no-op: at n=None the seeded task SET equals unseeded
# --------------------------------------------------------------------------

def test_whole_split_seed_is_noop():
    from baselines.harness.eval_common import resolve_task_list
    from common.sampling import derive_sample_seed
    seed = derive_sample_seed(42, 0, "locomo")
    assert seed  # non-empty seed derived
    seeded = resolve_task_list("locomo", "test", {"n_samples": None, "sample_seed": seed})
    unseeded = resolve_task_list("locomo", "test", {"n_samples": None})
    # n=None → whole pool regardless of seed: same SET (order may differ under a
    # shuffle, but the aggregate score is order-independent, so results are
    # unchanged). This is why existing whole-split baseline numbers hold.
    assert sorted(seeded) == sorted(unseeded), (seeded, unseeded)
    assert len(seeded) == len(unseeded)


# --------------------------------------------------------------------------
# (e) REAL memcache reuse: stage2/stage3 load stage1's pickled Phase-1 memory
#     Real LoCoMoWorkflow; only the two network legs (QA agent + judge) stubbed.
# --------------------------------------------------------------------------

# Module-level counter + memo base so pickle can resolve the class across the
# save/load roundtrip within this process.
_BUILD_CALLS = []


class _CountingMemo(MemoStructure):
    """Picklable stub memo: no per-instance state (empty __dict__), so pickle
    of the built memo is trivial. Phase-1 bumps a module counter so we can
    detect that stage2/stage3 SKIP the build (cache hit)."""
    async def build_memory_from_data(self, r):
        _BUILD_CALLS.append(1)
        return None

    async def retrieve_memory_for_query(self, r):
        return {"passages": ["stub passage"]}


def test_progressive_real_memcache_reuse():
    import common.llm as llm_mod
    from datasets.locomo.workflow import LoCoMoWorkflow
    from baselines.harness.eval_common import run_baseline, make_memo_class

    _BUILD_CALLS.clear()

    async def _fake_ask(self, user_input, **kw):
        return "a canned answer"

    async def _fake_judge(self, query, predicted, reference, qa_metadata=None):
        return 1, "fake-judge: forced pass"

    # make_memo_class wrapper must be picklable (the fix in eval_common):
    memo_class = make_memo_class(_CountingMemo)

    # thresholds 0.0 → promote through all 3 stages; n_conversations=1 with a
    # constant per-run seed → the SAME single user each stage → same cache key.
    stages = {
        "stage1": {"n_conversations": 1, "n_qa": 2, "threshold": 0.0},
        "stage2": {"n_conversations": 1, "n_qa": 2, "threshold": 0.0},
        "stage3": {"n_conversations": 1, "n_qa": 2},
    }

    orig_ask, orig_judge = llm_mod.Agent.ask, LoCoMoWorkflow.judge
    llm_mod.Agent.ask, LoCoMoWorkflow.judge = _fake_ask, _fake_judge
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_memcache_"))
    try:
        result = asyncio.run(run_baseline(
            dataset="locomo", split="test", user_stage_spec=None,
            memo_class=memo_class, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            progressive=True, memory_cache=True, stages=stages,
        ))
    finally:
        llm_mod.Agent.ask, LoCoMoWorkflow.judge = orig_ask, orig_judge

    # promoted through all three stages
    assert result["locomo"]["stage"] == 3.0, result
    assert result["locomo"]["eliminated"] is False, result

    # ONE build across three stages (1 user) — stage2 + stage3 hit the cache.
    assert len(_BUILD_CALLS) == 1, (
        f"expected exactly 1 Phase-1 build (stage1 only); got {len(_BUILD_CALLS)} "
        f"— cross-stage memcache reuse is NOT working"
    )

    # the pickled snapshot + sidecar were actually written to the shared dir
    memcache_dir = out_dir / "memory_cache"
    pkls = list(memcache_dir.glob("*__final.pkl"))
    metas = list(memcache_dir.glob("*__final.meta.json"))
    assert pkls, f"no memcache pkl written under {memcache_dir}"
    assert metas, f"no memcache sidecar written under {memcache_dir}"
    # the sidecar records the harness fingerprint the loads validate against
    side = json.loads(metas[0].read_text())
    assert side.get("harness_fingerprint"), side
    assert side.get("storage") == "pickle", side

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
