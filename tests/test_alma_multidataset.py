"""Tests for ALMA multi-dataset support (registry + dataset_info + prompt builders).

Zero-dependency runner (no pytest in the venvs):

    baselines/venv/bin/python tests/test_alma_multidataset.py
    venv/bin/python tests/test_alma_multidataset.py
"""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_registry_resolves_all_four():
    from baselines.evolve.alma.registry import REGISTRY, DATASETS, resolve
    from datasets.dynamicmem.workflow import DynamicMemWorkflow
    from datasets.locomo.workflow import LoCoMoWorkflow
    from datasets.longmemeval.workflow import LongMemEvalSWorkflow, LongMemEvalMWorkflow
    from datasets.dynamicmem.env import DynamicMemRecorder
    from datasets.locomo.env import LoCoMoRecorder
    from datasets.longmemeval.env import LongMemEvalRecorder

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
    """The dataset_info extraction must not change DynamicMem prompt text."""
    import json
    from pathlib import Path
    from baselines.evolve.alma import meta_agent_prompt as m
    from datasets.dynamicmem.env import DynamicMemRecorder

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
    """launch.py resolves its workflow/env from the registry, not a hardcoded
    DynamicMem import (main() can't be called in-process — it os._exit(0)s)."""
    import inspect
    import baselines.evolve.alma.launch as launch
    src = inspect.getsource(launch)
    # dispatches through the shared registry
    assert "from baselines.registry import" in src
    assert "resolve(dataset)" in src
    # no hardcoded DynamicMem workflow/get_task_list import survives
    assert "from datasets.dynamicmem.workflow import DynamicMemWorkflow" not in src
    assert "from datasets.dynamicmem.env import get_task_list" not in src
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
    from datasets.locomo.env import LoCoMoRecorder
    from datasets.dynamicmem.env import DynamicMemRecorder
    ma = MetaAgent(dataset="locomo")
    assert ma._get_recorder_class() is LoCoMoRecorder
    assert ma.dataset == "locomo"
    ma_dm = MetaAgent()  # default
    assert ma_dm._get_recorder_class() is DynamicMemRecorder


def test_run_main_parses_dataset():
    import importlib
    rm = importlib.import_module("baselines.evolve.alma.run_main")
    import sys as _sys
    argv = ["run_main.py", "--dataset", "longmemeval_m", "--steps", "1"]
    old = _sys.argv
    _sys.argv = argv
    try:
        args = rm.parse_args()
        assert args.dataset == "longmemeval_m"
    finally:
        _sys.argv = old
    # default
    _sys.argv = ["run_main.py"]
    try:
        assert rm.parse_args().dataset == "dynamicmem"
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
            memo_SHA=sha, status="test", eval_n_samples=1,
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
