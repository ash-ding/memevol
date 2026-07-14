"""Tests for cc + hipporag2 multi-dataset support (shared registry, runner,
memo structures). Zero-dependency runner:

    baselines/venv/bin/python tests/test_baselines_multidataset.py
    venv/bin/python tests/test_baselines_multidataset.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_shared_registry_resolves_all_four():
    from baselines.registry import REGISTRY, DATASETS, resolve
    from datasets.locomo.workflow import LoCoMoWorkflow
    from datasets.locomo.env import LoCoMoRecorder
    assert DATASETS == ["dynamicmem", "locomo", "longmemeval_m", "longmemeval_s"]
    wf, env, rec = resolve("locomo")
    assert wf is LoCoMoWorkflow and rec is LoCoMoRecorder and hasattr(env, "get_task_list")


def test_alma_still_imports_shared_registry():
    # ALMA re-points to the shared registry; its own module still exposes resolve.
    from baselines.alma.registry import resolve as alma_resolve
    from baselines.registry import resolve as shared_resolve
    assert alma_resolve is shared_resolve


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
