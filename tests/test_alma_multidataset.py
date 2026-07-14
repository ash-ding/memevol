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
