"""Tests for ALMA multi-dataset support (registry + dataset_info + prompt builders).

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_alma_multidataset.py
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_registry_resolves_all_four():
    from baselines.evolve.alma.registry import REGISTRY, DATASETS, resolve
    from benchmarks.dynamicmem.workflow import DynamicMemWorkflow
    from benchmarks.locomo.workflow import LoCoMoWorkflow
    from benchmarks.longmemeval.workflow import LongMemEvalSWorkflow, LongMemEvalMWorkflow
    from benchmarks.dynamicmem.env import DynamicMemRecorder
    from benchmarks.locomo.env import LoCoMoRecorder
    from benchmarks.longmemeval.env import LongMemEvalRecorder

    assert DATASETS == ["dynamicmem", "locomo", "longmemeval_m", "longmemeval_s"]

    wf, env, rec = resolve("dynamicmem")
    assert wf is DynamicMemWorkflow and rec is DynamicMemRecorder
    assert hasattr(env, "get_task_list")

    wf, env, rec = resolve("locomo")
    assert wf is LoCoMoWorkflow and rec is LoCoMoRecorder

    wf, _, rec = resolve("longmemeval_s")
    assert wf is LongMemEvalSWorkflow and rec is LongMemEvalRecorder

    wf, _, rec = resolve("longmemeval_m")
    assert wf is LongMemEvalMWorkflow and rec is LongMemEvalRecorder


def test_registry_unknown_raises():
    from baselines.evolve.alma.registry import resolve
    try:
        resolve("nope")
    except ValueError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown dataset")


_REQUIRED_INFO_KEYS = {
    "task_description", "gen_intro", "recorder_env_import", "recorder_class_name",
    "gen_protocol", "code_usage", "design_goals", "reflection_protocol",
    "reflection_code_usage", "analysis_protocol", "analysis_shape_a", "evidence_key",
}


def test_dataset_info_has_all_datasets_and_keys():
    from baselines.evolve.alma.dataset_info import DATASET_INFO
    from baselines.evolve.alma.registry import DATASETS
    assert set(DATASET_INFO) == set(DATASETS)
    for ds, info in DATASET_INFO.items():
        missing = _REQUIRED_INFO_KEYS - set(info)
        assert not missing, f"{ds} missing keys: {missing}"
        for k in _REQUIRED_INFO_KEYS:
            assert isinstance(info[k], str) and info[k], f"{ds}.{k} empty"


def test_dataset_info_evidence_keys_and_recorders():
    from baselines.evolve.alma.dataset_info import DATASET_INFO
    assert DATASET_INFO["dynamicmem"]["evidence_key"] == "relevant_app_logs"
    assert DATASET_INFO["locomo"]["evidence_key"] == "relevant_turns"
    assert DATASET_INFO["longmemeval_s"]["evidence_key"] == "relevant_sessions"
    assert DATASET_INFO["longmemeval_m"]["evidence_key"] == "relevant_sessions"
    assert DATASET_INFO["locomo"]["recorder_class_name"] == "LoCoMoRecorder"
    # each dataset's protocol names its own recorder.init shape
    assert "app_logs" in DATASET_INFO["dynamicmem"]["gen_protocol"]
    assert "conversation" in DATASET_INFO["locomo"]["gen_protocol"]
    assert "sessions" in DATASET_INFO["longmemeval_s"]["gen_protocol"]
    assert "question_date" in DATASET_INFO["longmemeval_s"]["gen_protocol"]


def test_dynamicmem_prompts_byte_identical():
    """DynamicMem prompt text is locked byte-identically against the fixture.
    Fixture regenerated 2026-08-06 (contract rename): the embedded contract
    source is now common/memo_class.py (class MemoClass + MemoClass alias) —
    alma prompt text intentionally changed at that point; comparisons to
    pre-rename alma runs are prompt-version-crossing."""
    import json
    from pathlib import Path
    from baselines.evolve.alma import meta_agent_prompt as m
    from benchmarks.dynamicmem.env import DynamicMemRecorder

    fixture = json.loads(
        (PROJECT_ROOT / "tests/fixtures/alma_dynamicmem_prompts.json").read_text(encoding="utf-8")
    )
    memo_info = {
        "source_code": "class Foo: pass",
        "examples": [{"user_id": "user_001", "query": "q", "retrieved_memory": {},
                      "predicted": "p", "reference": "r", "score": 0.5,
                      "judge_reason": "jr", "relevant_app_logs": []},
                     {"error_info": "User user_002 failed: [Phase1_Update] KeyError: x", "score": 0.0}],
        "benchmark_eval_score": {"benchmark_overall_eval_score": 0.42},
        "improve_example": {"source_code": "class Bar: pass", "suggestion": {}, "improve_score": 0.1},
    }
    rec = DynamicMemRecorder
    a_sys, a_user, _ = m.build_analysis_prompt(memo_info, dataset="dynamicmem")
    g_sys, g_user = m.build_generate_new_code_prompt(
        memo_info,
        {"suggested_changes": [{"priority": "High", "what": "w", "why": "y"}],
         "trajectory_score_assessment": [], "execution_errors": []},
        rec, dataset="dynamicmem",
    )
    r_sys, r_user = m.build_reflection_prompt("class Baz: pass", rec, "boom", dataset="dynamicmem")
    assert a_sys == fixture["a_sys"], "analysis system prompt drifted"
    assert a_user == fixture["a_user"], "analysis user prompt drifted"
    assert g_sys == fixture["g_sys"], "gen system prompt drifted"
    assert g_user == fixture["g_user"], "gen user prompt drifted"
    assert r_sys == fixture["r_sys"], "reflection system prompt drifted"
    assert r_user == fixture["r_user"], "reflection user prompt drifted"


def test_prompts_render_for_all_datasets():
    from baselines.evolve.alma import meta_agent_prompt as m
    from baselines.evolve.alma.registry import resolve, DATASETS
    memo_info = {"source_code": "class Foo: pass", "examples": [],
                 "benchmark_eval_score": {"benchmark_overall_eval_score": 0.0}}
    for ds in DATASETS:
        _, _, rec = resolve(ds)
        a_sys, _, _ = m.build_analysis_prompt(memo_info, dataset=ds)
        g_sys, _ = m.build_generate_new_code_prompt(memo_info, {}, rec, dataset=ds)
        r_sys, _ = m.build_reflection_prompt("class B: pass", rec, "e", dataset=ds)
        # dataset-appropriate shape word shows up in the rendered gen prompt
        shape_word = {"dynamicmem": "app_logs", "locomo": "conversation",
                      "longmemeval_s": "sessions", "longmemeval_m": "sessions"}[ds]
        assert shape_word in g_sys


def test_launch_dispatches_via_registry():
    """launch.py routes everything through the shared execution-independent
    evaluate_memo (which resolves workflows via benchmarks.registry) — no
    hardcoded DynamicMem import (main() can't be called in-process — it
    os._exit(0)s)."""
    import inspect
    import baselines.evolve.alma.launch as launch
    src = inspect.getsource(launch)
    # dispatches through the shared evaluator (registry resolution lives there)
    assert "evaluate_memo" in src
    # no hardcoded DynamicMem workflow/get_task_list import survives
    assert "from benchmarks.dynamicmem.workflow import DynamicMemWorkflow" not in src
    assert "from benchmarks.dynamicmem.env import get_task_list" not in src
    # main() takes a dataset param defaulting to dynamicmem (back-compat)
    sig = inspect.signature(launch.main)
    assert sig.parameters["dataset"].default == "dynamicmem"


def test_output_run_dir_templated_by_dataset():
    from baselines.evolve.alma.eval_runner import get_output_run_dir
    p_dm = get_output_run_dir("abc123", "search", "eval")  # default dynamicmem
    assert p_dm.parent.name == "dynamicmem" and p_dm.name == "abc123_search_eval"
    p_lc = get_output_run_dir("abc123", "search", "eval", dataset="locomo")
    assert p_lc.parent.name == "locomo"


def test_memo_manager_archive_root_by_dataset():
    from baselines.evolve.alma.memo_manager import Memo_Manager
    mm = Memo_Manager(dataset="locomo")
    assert mm.ARCHIVE_ROOT.name == "locomo"
    assert mm.ARCHIVE_ROOT.parent.name == "memo_archive"
    mm_dm = Memo_Manager()  # default
    assert mm_dm.ARCHIVE_ROOT.name == "dynamicmem"
    # baseline seed dir is shared across datasets (same parent as any dataset archive)
    from baselines.evolve.alma.memo_manager import Memo_Manager as _MM
    assert _MM(dataset="locomo").ARCHIVE_ROOT.parent == _MM(dataset="dynamicmem").ARCHIVE_ROOT.parent


def test_meta_agent_recorder_by_dataset():
    from baselines.evolve.alma.meta_agent import MetaAgent
    from benchmarks.locomo.env import LoCoMoRecorder
    from benchmarks.dynamicmem.env import DynamicMemRecorder
    ma = MetaAgent(dataset="locomo")
    assert ma._get_recorder_class() is LoCoMoRecorder
    assert ma.dataset == "locomo"
    ma_dm = MetaAgent()  # default
    assert ma_dm._get_recorder_class() is DynamicMemRecorder


def test_run_parses_dataset():
    import importlib
    rm = importlib.import_module("baselines.evolve.alma.run")
    import sys as _sys
    argv = ["run.py", "--dataset", "longmemeval_m", "--steps", "1"]
    old = _sys.argv
    _sys.argv = argv
    try:
        cfg = rm.build_cfg(rm.parse_args())
        assert cfg["dataset"] == "longmemeval_m"
    finally:
        _sys.argv = old
    # default: CLI None sentinel resolves to DEFAULT_CONFIG's dynamicmem
    _sys.argv = ["run.py"]
    try:
        assert rm.build_cfg(rm.parse_args())["dataset"] == "dynamicmem"
    finally:
        _sys.argv = old


def test_sampling_reads_dataset_evidence_key():
    import json, tempfile, os
    from baselines.evolve.alma.sampling import build_analysis_artifact
    d = tempfile.mkdtemp()
    # minimal score.json + one trace with a locomo-style evidence key
    json.dump({"benchmark_eval_score": {"benchmark_overall_eval_score": 1.0,
               "benchmark_overall_eval_standard_deviation": 0.0},
               "per_user": {"conv_1": {"reward": 1.0, "n_qa": 1, "failure_info": None}},
               "invalid_users": []},
              open(os.path.join(d, "score.json"), "w"))
    os.makedirs(os.path.join(d, "traces"))
    json.dump({"user_id": "conv_1", "failure_info": None,
               "steps": [{"query": "q", "predicted": "p", "reference": "r",
                          "score": 1.0, "judge_reason": "jr",
                          "retrieved_memory": {}, "relevant_turns": [{"t": 1}]}]},
              open(os.path.join(d, "traces", "conv_1.json"), "w"))
    from pathlib import Path
    art = build_analysis_artifact(Path(d), evidence_key="relevant_turns")
    ex = [e for e in art["examples"] if "error_info" not in e]
    assert ex and ex[0]["relevant_turns"] == [{"t": 1}]


def test_locomo_end_to_end_fake():
    """Fake-evaluator integration test for the non-DynamicMem ALMA path
    (Task 10 brief, Step 1 fallback — no live LLM calls, no network, no cost).

    Drives ``MetaAgent(dataset="locomo").run_single_memo(..., status="test")``
    end-to-end. status="test" takes the non-search branch in run_single_memo,
    which skips analyze_memo_structure / generate_new_code / examine_new_code
    (each of those calls a real meta-model LLM via common.llm.Agent) and goes
    straight to memo_manager.execute_memo_structure -> eval_runner.run_evaluation
    -> sampling.build_analysis_artifact. That is exactly the registry ->
    dataset_info -> memo_manager -> sampling wiring the task exercises for a
    non-DynamicMem dataset, driven deterministically by monkeypatching
    run_evaluation (modeled on tests/test_heldout.py's fake-evaluator pattern)
    to drop a canned score.json + traces/conv_1.json (with a `relevant_turns`
    step, LoCoMo's evidence key) instead of spawning the real subprocess.

    Asserts: (a) no exception anywhere in that call chain, and (b) the actual
    build_analysis_artifact call made from inside execute_memo_structure used
    LoCoMo's evidence_key ("relevant_turns", read from DATASET_INFO — not
    hardcoded) and its output example carries the relevant_turns payload.
    """
    import asyncio
    import json
    import shutil
    import tempfile
    from pathlib import Path

    from baselines.evolve.alma.meta_agent import MetaAgent
    from baselines.evolve.alma import memo_manager as MM
    from baselines.evolve.alma.dataset_info import DATASET_INFO

    sha = "faketest01"
    archive_dir = MM.ALMA_ROOT / "memo_archive" / "locomo"
    archive_dir.mkdir(parents=True, exist_ok=True)
    memo_file = archive_dir / f"memo_structure_{sha}.py"
    memo_file.write_text("# fake memo source for integration test\n", encoding="utf-8")

    fake_out_dir = Path(tempfile.mkdtemp(prefix="alma_fake_eval_"))

    async def fake_run_evaluation(**kwargs):
        traces_dir = fake_out_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        with (fake_out_dir / "score.json").open("w", encoding="utf-8") as f:
            json.dump({
                "benchmark_eval_score": {
                    "benchmark_overall_eval_score": 0.66,
                    "benchmark_overall_eval_standard_deviation": 0.0,
                },
                "per_user": {"conv_1": {"reward": 0.66, "n_qa": 1, "failure_info": None}},
                "invalid_users": [],
            }, f)
        with (traces_dir / "conv_1.json").open("w", encoding="utf-8") as f:
            json.dump({
                "user_id": "conv_1", "failure_info": None,
                "steps": [{"query": "q", "predicted": "p", "reference": "r",
                           "score": 0.66, "judge_reason": "jr", "retrieved_memory": {},
                           "relevant_turns": [{"speaker": "A", "text": "hi", "turn_id": 3}]}],
            }, f)
        return fake_out_dir

    captured = {}
    real_build_artifact = MM.build_analysis_artifact

    def spy_build_analysis_artifact(*args, **kwargs):
        art = real_build_artifact(*args, **kwargs)
        captured["evidence_key"] = kwargs.get("evidence_key")
        captured["artifact"] = art
        return art

    orig_run_eval = MM.run_evaluation
    orig_build_artifact = MM.build_analysis_artifact
    MM.run_evaluation = fake_run_evaluation
    MM.build_analysis_artifact = spy_build_analysis_artifact
    try:
        ma = MetaAgent(dataset="locomo")
        asyncio.run(ma.run_single_memo(
            memo_SHA=sha, status="test",
            max_sample_concurrent=1, judge_model="gpt-5-mini",
        ))
    finally:
        MM.run_evaluation = orig_run_eval
        MM.build_analysis_artifact = orig_build_artifact
        shutil.rmtree(fake_out_dir, ignore_errors=True)
        memo_file.unlink(missing_ok=True)

    assert captured.get("evidence_key") == "relevant_turns" == DATASET_INFO["locomo"]["evidence_key"]
    examples = [e for e in captured["artifact"]["examples"] if "error_info" not in e]
    assert examples and examples[0]["relevant_turns"] == [{"speaker": "A", "text": "hi", "turn_id": 3}]


def test_history_ckpt_filename_includes_dataset():
    from baselines.evolve.alma.meta_agent import MetaAgent
    ma = MetaAgent(dataset="locomo")
    fn = ma._history_ckpt_filename("check", 10, "TS")
    assert "locomo" in fn and fn == "check_locomo_10_TS.json"
    assert MetaAgent()._history_ckpt_filename("check", 10, "TS") == "check_dynamicmem_10_TS.json"


# ---------------- Task 9: progressive gauntlet + per-step seed migration ----------------


def test_launch_main_signature_migrated():
    """launch.main drops the flat sizing knobs (eval_n_*/check_n_*) and gains
    the progressive / per-step-seed knobs (Task 9 clean break)."""
    import inspect
    import baselines.evolve.alma.launch as launch
    params = inspect.signature(launch.main).parameters
    for p in ("progressive", "random_sample", "sampling_seed", "stages", "step_index"):
        assert p in params, f"launch.main missing new param {p!r}"
    for p in ("eval_n_samples", "eval_n_qa", "check_n_samples", "check_n_qa"):
        assert p not in params, f"launch.main still accepts removed flat knob {p!r}"
    assert params["progressive"].default is True
    assert params["random_sample"].default is False
    assert params["sampling_seed"].default == 42
    # dataset param survives the migration (registry dispatch relies on it)
    assert params["dataset"].default == "dynamicmem"


def test_meta_agent_search_loop_signature_migrated():
    """MetaAgent.forward / run_single_memo thread the progressive knobs and no
    longer carry the removed flat sizing knobs."""
    import inspect
    from baselines.evolve.alma.meta_agent import MetaAgent
    for meth in (MetaAgent.forward, MetaAgent.run_single_memo):
        params = inspect.signature(meth).parameters
        for p in ("progressive", "random_sample", "sampling_seed"):
            assert p in params, f"{meth.__name__} missing {p!r}"
        for p in ("eval_n_samples", "eval_n_qa", "check_n_samples", "check_n_qa"):
            assert p not in params, f"{meth.__name__} still carries removed {p!r}"
    # step_index threads into the per-candidate eval so consecutive steps seed
    # differently.
    assert "step_index" in inspect.signature(MetaAgent.run_single_memo).parameters


def test_run_cli_migrated_flags():
    """run CLI exposes the new flags and rejects the removed flat ones."""
    import contextlib
    import importlib
    import io
    import sys as _sys
    rm = importlib.import_module("baselines.evolve.alma.run")
    old = _sys.argv
    try:
        _sys.argv = ["run.py", "--dataset", "locomo", "--progressive",
                     "--random_sample", "--sampling_seed", "7", "--steps", "1"]
        args = rm.parse_args()
        assert args.progressive is True
        assert args.random_sample is True
        assert args.sampling_seed == 7
        # `stages` / `single_stage` are config-file-only now — NOT CLI args.
        assert not hasattr(args, "stages")
        assert not hasattr(args, "single_stage")

        _sys.argv = ["run.py", "--no-progressive"]
        assert rm.parse_args().progressive is False

        # removed flat flags AND the removed `--stages` sizing flag are rejected
        # (argparse -> SystemExit); sizing now lives in the --config YAML.
        for bad in ("--eval_n_samples", "--check_n_samples",
                    "--eval_n_qa", "--check_n_qa", "--stages", "--single_stage"):
            _sys.argv = ["run.py", bad, "6"]
            raised = False
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    rm.parse_args()
                except SystemExit:
                    raised = True
            assert raised, f"{bad} should be rejected after removal"
    finally:
        _sys.argv = old


def test_per_step_seed_changes_task_subset():
    """random_sample=True → consecutive search steps derive different seeds,
    and those seeds select different task subsets from get_task_list — the
    exact mechanism alma threads into evaluate_memo's sample_seed. Uses
    LoCoMo (git-tracked locomo10.json; no network)."""
    from common.sampling import derive_sample_seed
    from benchmarks.locomo.env import get_task_list
    ds = "locomo"
    s0 = derive_sample_seed(42, 0, ds)
    s1 = derive_sample_seed(42, 1, ds)
    assert s0 != s1, "per-step seeds must differ across steps"
    # the seed reaches get_task_list and reorders the pool per step
    assert get_task_list("search", None, seed=s0) != get_task_list("search", None, seed=s1)
    # and picks a different capped subset per step
    assert get_task_list("search", 2, seed=s0) != get_task_list("search", 2, seed=s1)
    # random_sample=False → seed None → historical deterministic prefix (stable)
    assert get_task_list("search", 2, seed=None) == get_task_list("search", 2, seed=None)


# ---------------- Task 4: shared --config / DEFAULT_CONFIG ----------------


def test_alma_default_config_roundtrips():
    from baselines.evolve.alma.run import DEFAULT_CONFIG
    from common.config import resolve_config
    d = DEFAULT_CONFIG
    assert d["progressive"] is True and d["random_sample"] is False and d["sampling_seed"] == 42
    assert d["steps"] == 10 and d["dataset"] == "dynamicmem"
    # `stages`/`single_stage` are config-file-only knobs (None in DEFAULT_CONFIG);
    # the removed `--stages` CLI flag no longer surfaces here.
    assert d["stages"] is None and d["single_stage"] is None
    assert resolve_config(d, None, {k: None for k in d}) == d


# ---------------- Task 4: single_stage sizing for the progressive=false pass ----------------
#
# launch.main mode=eval + progressive=false sizes its single pass from the
# REQUIRED `single_stage` block (was the terminal `stage3` block). Mirrors the
# harness reference tests (tests/test_baseline_gauntlet.py c2/c3/c4): sizes from
# single_stage, ValueError when absent, ValueError on an unknown field.

_STUB_MEMO_SRC = '''\
from common.memo_class import MemoClass


class StubMemo(MemoClass):
    async def build_memory_from_data(self, recorder):
        return None

    async def retrieve_memory_for_query(self, recorder):
        return {}
'''


class _StopEval(Exception):
    """Sentinel raised by the fake evaluate_memo to halt launch.main just
    before its terminal os._exit(0) (which would otherwise kill the test)."""


def _run_launch_main_eval(single_stage, dataset="locomo"):
    """Drive launch.main through the progressive=false (single-pass) routing with
    common.evaluate.evaluate_memo patched by a fake that (a) captures the kwargs
    launch hands over, (b) runs the SAME shared single_stage resolution the real
    evaluate_memo runs (resolve_single_stage_spec — so validation ValueErrors
    propagate exactly like the real path), then (c) raises _StopEval to stop
    before any real workflow work + os._exit(0). Returns the captured dict."""
    import asyncio
    import os as _os
    import shutil as _shutil
    import tempfile

    import common.evaluate as ce
    import baselines.evolve.alma.launch as launch

    captured = {}

    async def _fake_evaluate_memo(**kwargs):
        captured.update(kwargs)
        if not kwargs.get("smoke") and not kwargs.get("progressive"):
            captured["spec"] = ce.resolve_single_stage_spec(
                kwargs["dataset"], kwargs.get("single_stage"))
        raise _StopEval()

    fd = tempfile.NamedTemporaryFile(
        prefix="alma_stub_memo_", suffix=".py", delete=False, mode="w", encoding="utf-8"
    )
    fd.write(_STUB_MEMO_SRC)
    fd.close()
    memo_path = fd.name
    out_dir = tempfile.mkdtemp(prefix="alma_launch_main_")
    orig = ce.evaluate_memo
    ce.evaluate_memo = _fake_evaluate_memo
    try:
        asyncio.run(launch.main(
            module_path=memo_path, memory_id="stub", output_run_dir=out_dir,
            dataset=dataset, mode="eval", progressive=False,
            single_stage=single_stage,
        ))
    except _StopEval:
        pass
    finally:
        ce.evaluate_memo = orig
        _os.unlink(memo_path)
        _shutil.rmtree(out_dir, ignore_errors=True)
    return captured


def test_alma_non_progressive_sizes_from_single_stage():
    """mode=eval + progressive=false hands the raw single_stage block to the
    shared evaluate_memo, whose resolution sizes the pass
    (spec == single_stage_wire_spec) — NOT from stage3."""
    from common.evaluate import single_stage_wire_spec
    single_stage = {"n_conversations": 2, "n_qa": 5}
    captured = _run_launch_main_eval(single_stage, dataset="locomo")
    assert captured.get("progressive") is False and captured.get("smoke") is False
    assert captured.get("single_stage") == single_stage
    expected = single_stage_wire_spec("locomo", single_stage)   # {"n_samples": 2, "n_qa": 5}
    for k, v in expected.items():
        assert captured["spec"].get(k) == v, (k, captured["spec"])


def test_alma_non_progressive_requires_single_stage():
    """progressive=false with NO single_stage → clear ValueError (no silent
    whole-split, no stage3 fallback)."""
    raised = None
    try:
        _run_launch_main_eval(None, dataset="locomo")
    except ValueError as e:
        raised = e
    assert raised is not None, "expected ValueError when single_stage absent"
    assert "single_stage" in str(raised), str(raised)


def test_alma_non_progressive_rejects_unknown_single_stage_field():
    """An unknown single_stage field is rejected via the shared validation
    (_resolve_dataset_stages) — a typo can't silently mis-size."""
    raised = None
    try:
        _run_launch_main_eval({"n_conversations": 2, "bogus": 1}, dataset="locomo")
    except ValueError as e:
        raised = e
    assert raised is not None, "expected ValueError for an unknown single_stage field"
    assert "bogus" in str(raised), str(raised)


# ---------------- runner ----------------

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
