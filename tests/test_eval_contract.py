"""Tests for the standardized memory-system eval contract.

Zero-dependency runner:
    baselines/venv/bin/python tests/test_eval_contract.py
    venv/bin/python tests/test_eval_contract.py
"""
import asyncio
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_three_hooks_optional_with_defaults():
    from common.harness_base import MemoStructure

    class Bare(MemoStructure):
        pass  # overrides nothing — must be instantiable now (hooks non-abstract)

    m = Bare()
    loop = asyncio.new_event_loop()

    class _Rec:
        init = {"query": "q"}

    assert loop.run_until_complete(m.general_update(_Rec())) is None
    assert loop.run_until_complete(m.general_retrieve(_Rec())) == {}
    assert loop.run_until_complete(m.general_answer(_Rec(), {}, "PROMPT")) is None


def test_chunked_partitions():
    from common.harness_base import MemoStructure

    class M(MemoStructure):
        pass

    m = M()
    data = list(range(10))
    m._update_type = "all_at_once"
    assert list(m.chunked(data)) == [data]
    m._update_type = "sequential"
    assert list(m.chunked(data)) == [[i] for i in range(10)]
    m._update_type = "chunked"
    m._n_chunks = 5
    chunks = list(m.chunked(data))
    assert len(chunks) == 5 and sum(len(c) for c in chunks) == 10
    assert [x for c in chunks for x in c] == data  # order preserved, no loss


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
