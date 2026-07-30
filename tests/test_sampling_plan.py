"""Tests for the unified sampling-plan resolution in common/staged_eval.py.
Zero-dep runner — both venvs:
    venv/bin/python tests/test_sampling_plan.py
    baselines/venv/bin/python tests/test_sampling_plan.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_single_stage_wire_spec_null_is_whole():
    from common.staged_eval import single_stage_wire_spec
    # locomo family: n_conversations (sample) + n_qa (extra)
    assert single_stage_wire_spec("locomo", {"n_conversations": 2, "n_qa": 20}) == {"n_samples": 2, "n_qa": 20}
    assert single_stage_wire_spec("locomo", {"n_conversations": None, "n_qa": None}) == {"n_samples": None, "n_qa": None}


def test_resolve_plan_progressive_true_is_stage_plan():
    from common.staged_eval import resolve_sampling_plan, DEFAULT_STAGES, stage_plan
    params = {"stages": DEFAULT_STAGES["locomo"]}
    got = resolve_sampling_plan("locomo", params, progressive=True)
    assert [p[0] for p in got] == ["stage1", "stage2", "stage3"]
    assert got == stage_plan("locomo", params)


def test_resolve_plan_progressive_false_uses_single_stage():
    from common.staged_eval import resolve_sampling_plan
    params = {"single_stage": {"n_conversations": 2, "n_qa": 20}}
    got = resolve_sampling_plan("locomo", params, progressive=False)
    assert got == [("single", {"n_samples": 2, "n_qa": 20}, None)]


def test_resolve_plan_progressive_false_missing_single_stage_raises():
    from common.staged_eval import resolve_sampling_plan
    raised = False
    try:
        resolve_sampling_plan("locomo", {"stages": {}}, progressive=False)
    except ValueError as e:
        raised = "single_stage" in str(e)
    assert raised


def test_resolve_plan_single_stage_all_null_is_whole_split():
    from common.staged_eval import resolve_sampling_plan
    got = resolve_sampling_plan("locomo", {"single_stage": {"n_conversations": None, "n_qa": None}}, progressive=False)
    assert got == [("single", {"n_samples": None, "n_qa": None}, None)]


def test_resolve_dataset_stages_validates_single_stage_block():
    # single_stage with an unknown field errors; null/"full" normalize to None.
    from common.staged_eval import _resolve_dataset_stages
    p = {"single_stage": {"n_conversations": "full", "n_qa": None}}
    _resolve_dataset_stages("locomo", p)   # should not raise; normalizes
    assert p["single_stage"]["n_conversations"] is None
    bad = {"single_stage": {"n_conversations": 2, "threshold": 0.3}}  # threshold not allowed on single_stage
    raised = False
    try:
        _resolve_dataset_stages("locomo", bad)
    except ValueError:
        raised = True
    assert raised


def test_resolve_plan_direct_caller_rejects_unknown_field():
    # Minor-B fix (2026-07-26 review): resolve_sampling_plan self-validates the
    # single_stage block — a field typo is rejected even WITHOUT a prior
    # _resolve_dataset_stages call. Before the fix this silently returned a
    # whole-split plan (the typo'd field → None → whole). Regression anchor.
    from common.staged_eval import resolve_sampling_plan
    raised = False
    try:
        # `n_conversation` (missing the trailing s) is not a locomo size field.
        resolve_sampling_plan("locomo", {"single_stage": {"n_conversation": 2}}, progressive=False)
    except ValueError as e:
        raised = "unknown field" in str(e)
    assert raised


def test_resolve_plan_empty_single_stage_is_whole_not_error():
    # Minor-C fix: an empty `{}` block counts as PRESENT (= all-null = whole
    # split), NOT absent — consistent with forge's normalization. Before the fix
    # the `if not single` check treated `{}` as absent and raised. Regression
    # anchor for the whole-split-not-error behavior.
    from common.staged_eval import resolve_sampling_plan
    got = resolve_sampling_plan("locomo", {"single_stage": {}}, progressive=False)
    assert got == [("single", {"n_samples": None, "n_qa": None}, None)]


def test_resolve_single_stage_spec_absent_vs_empty():
    # The shared resolver: None (truly absent) → ValueError; {} (present) → whole
    # split. This is the single source both forge and the baselines call, so the
    # absent-vs-empty distinction is uniform across every progressive=false path.
    from common.staged_eval import resolve_single_stage_spec
    raised = False
    try:
        resolve_single_stage_spec("locomo", None)
    except ValueError as e:
        raised = "single_stage" in str(e)
    assert raised
    assert resolve_single_stage_spec("locomo", {}) == {"n_samples": None, "n_qa": None}


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
