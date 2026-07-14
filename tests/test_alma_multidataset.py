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
    from baselines.alma.registry import REGISTRY, DATASETS, resolve
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
    from baselines.alma.registry import resolve
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
    from baselines.alma.dataset_info import DATASET_INFO
    from baselines.alma.registry import DATASETS
    assert set(DATASET_INFO) == set(DATASETS)
    for ds, info in DATASET_INFO.items():
        missing = _REQUIRED_INFO_KEYS - set(info)
        assert not missing, f"{ds} missing keys: {missing}"
        for k in _REQUIRED_INFO_KEYS:
            assert isinstance(info[k], str) and info[k], f"{ds}.{k} empty"


def test_dataset_info_evidence_keys_and_recorders():
    from baselines.alma.dataset_info import DATASET_INFO
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
    from baselines.alma import meta_agent_prompt as m
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
    from baselines.alma import meta_agent_prompt as m
    from baselines.alma.registry import resolve, DATASETS
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
