"""common/staged_eval.py owns the stage config, and forge re-exports it identically.
    venv/bin/python tests/test_staged_eval_shared.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_staged_eval_module_owns_config():
    from common import staged_eval as se
    assert set(se.DEFAULT_STAGES) == {"dynamicmem", "locomo", "longmemeval"}
    plan = se.stage_plan("locomo", {"stages": se.DEFAULT_STAGES["locomo"]})
    assert [p[0] for p in plan] == ["stage1", "stage2", "stage3"]
    assert plan[0][2] == 0.30   # locomo stage1 threshold


def test_forge_reexports_are_identical_objects():
    from common import staged_eval as se
    from forge import orchestrator as orch
    assert orch.DEFAULT_STAGES is se.DEFAULT_STAGES
    assert orch.stage_plan is se.stage_plan
    assert orch.stage_wire_spec is se.stage_wire_spec


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
