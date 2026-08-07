"""Progressive gauntlet + memcache + seed for the harness baselines
(baselines/harness/eval_utility.py::run_baseline).

Zero-dependency runner (baselines venv; no network — the QA agent + judge are
stubbed, run_all_users is faked for the gauntlet-logic tests):

    uv run python tests/test_baseline_gauntlet.py

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

from common.memo_class import MemoClass


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


class _StubMemo(MemoClass):
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
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from baselines.harness.eval_utility import run_baseline
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
            dataset="locomo", split="test",
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            progressive=True, memory_cache=False,
        )
        # promoted through every stage, in order
        assert seen == ["stage1", "stage2", "stage3"], seen
        # per-dataset metrics dict (shared metrics shape)
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
            dataset="locomo", split="test",
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
# (c) progressive=False takes the single-stage path (stage label "single")
# --------------------------------------------------------------------------

def test_progressive_false_single_stage_path():
    seen = []
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_single_"))
    try:
        score = _run_with_fake_run_all_users(
            reward=1.0, seen=seen,
            dataset="locomo", split="test",
            single_stage={"n_conversations": 1, "n_qa": 1},
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            # progressive defaults to False
        )
        assert seen == ["single"], seen                 # single pass, stage label "single"
        # Unified 2026-08: progressive=False now runs through evaluate_memo's one-item
        # plan, so it returns the shared metrics dict (same shape as
        # progressive), NOT the old flat score.json dict.
        assert "locomo" in score, score
        assert score["locomo"]["raw_score"] == 1.0
        assert score["locomo"]["stage"] == 4.0          # FULL_STAGE (single pass)
        assert not score["locomo"]["eliminated"]
        # Artifacts now match forge's progressive=false / the progressive path: the
        # reached-stage score.json + token_usage.json copied to the out_dir root, the
        # pass's own copy under out_dir/single/, and a stages.json at the root.
        assert (out_dir / "score.json").exists()
        assert (out_dir / "token_usage.json").exists()
        assert (out_dir / "stages.json").exists()
        assert (out_dir / "single").is_dir()
        assert not (out_dir / "stage1").exists()        # single plan, not a gauntlet
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# (c2) progressive=False sizes the single pass from single_stage_wire_spec:
#      the stage_spec handed to run_all_users == single_stage_wire_spec(...)
#      plus the derived per-run sample_seed, and n_samples caps the task list.
# --------------------------------------------------------------------------

def test_progressive_false_sizes_from_single_stage():
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from baselines.harness.eval_utility import run_baseline
    from common.evaluate import single_stage_wire_spec
    from common.sampling import derive_sample_seed

    captured = {}

    async def _fake(self, task_list, *, stage="stage3", stage_spec=None,
                    max_sample_concurrent=6):
        captured["stage"] = stage
        captured["stage_spec"] = dict(stage_spec or {})
        captured["task_list"] = list(task_list)
        recs = [_FakeRec(u, 1.0) for u in task_list]
        return recs, len(recs)

    single_stage = {"n_conversations": 2, "n_qa": 5}
    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_single_sized_"))
    orig = LoCoMoWorkflow.run_all_users
    LoCoMoWorkflow.run_all_users = _fake
    try:
        score = asyncio.run(run_baseline(
            dataset="locomo", split="test", single_stage=single_stage,
            memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
            out_dir=out_dir, max_sample_concurrent=1,
            progressive=False, sampling_seed=42,
        ))
    finally:
        LoCoMoWorkflow.run_all_users = orig
        shutil.rmtree(out_dir, ignore_errors=True)

    assert captured["stage"] == "single", captured["stage"]
    expected = single_stage_wire_spec("locomo", single_stage)   # {"n_samples": 2, "n_qa": 5}
    for k, v in expected.items():
        assert captured["stage_spec"].get(k) == v, (k, captured["stage_spec"])
    # the per-run seed rides along (fixed step 0, honoring sampling_seed)
    assert captured["stage_spec"]["sample_seed"] == derive_sample_seed(42, 0, "locomo")
    # n_samples caps the task list (2 conversations of the 4-conv locomo test split)
    assert len(captured["task_list"]) == 2, captured["task_list"]
    # Unified: returns the shared metrics dict (not the flat score dict).
    assert score["locomo"]["raw_score"] == 1.0, score


# --------------------------------------------------------------------------
# (c3) progressive=False with NO single_stage raises a clear ValueError
#      (no silent whole-split — sizing is REQUIRED).
# --------------------------------------------------------------------------

def test_progressive_false_requires_single_stage():
    from baselines.harness.eval_utility import run_baseline

    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_no_single_"))
    raised = None
    try:
        try:
            asyncio.run(run_baseline(
                dataset="locomo", split="test", single_stage=None,
                memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
                out_dir=out_dir, max_sample_concurrent=1,
                progressive=False,
            ))
        except ValueError as e:
            raised = e
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    assert raised is not None, "expected ValueError when single_stage absent"
    assert "single_stage" in str(raised), str(raised)


# --------------------------------------------------------------------------
# (c4) progressive=False with an UNKNOWN single_stage field raises (typo guard)
#      — validated through the SAME resolver the progressive path uses, so a
#      mis-spelled size field errors instead of silently mis-sizing.
# --------------------------------------------------------------------------

def test_progressive_false_rejects_unknown_single_stage_field():
    from baselines.harness.eval_utility import run_baseline

    out_dir = Path(tempfile.mkdtemp(prefix="test_gauntlet_bad_field_"))
    raised = None
    try:
        try:
            asyncio.run(run_baseline(
                dataset="locomo", split="test",
                single_stage={"n_conversations": 2, "bogus": 1},   # `bogus` is not a locomo size field
                memo_class=_StubMemo, qa_model="gpt-5-mini", judge_model="gpt-5-mini",
                out_dir=out_dir, max_sample_concurrent=1,
                progressive=False,
            ))
        except ValueError as e:
            raised = e
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    assert raised is not None, "expected ValueError for an unknown single_stage field"
    assert "bogus" in str(raised), str(raised)


# --------------------------------------------------------------------------
# (d) whole-split seed no-op: at n=None the seeded task SET equals unseeded
# --------------------------------------------------------------------------

def test_whole_split_seed_is_noop():
    # Directly against the primitive evaluate_memo calls (env.get_task_list →
    # shuffle_prefix): a seed at whole-split n=None must not change the SET.
    from benchmarks.locomo.env import get_task_list
    from common.sampling import derive_sample_seed
    seed = derive_sample_seed(42, 0, "locomo")
    assert seed  # non-empty seed derived
    seeded = get_task_list("test", None, seed=seed)
    unseeded = get_task_list("test", None)
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


class _CountingMemo(MemoClass):
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
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from baselines.harness.eval_utility import run_baseline

    _BUILD_CALLS.clear()

    async def _fake_ask(self, user_input, **kw):
        return "a canned answer"

    async def _fake_judge(self, query, predicted, reference, qa_metadata=None):
        return 1, "fake-judge: forced pass"

    # plain memo classes pickle without any wrapper magic now (constructor
    # config injection, 2026-08-06):
    memo_class = _CountingMemo

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
            dataset="locomo", split="test",
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
