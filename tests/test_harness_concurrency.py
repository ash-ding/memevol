"""Thread-safety helpers for running harness baselines' synchronous vendored
code on worker threads (baselines/harness/concurrency.py).
Zero-dependency runner (no pytest); needs only the ROOT project env:

    uv run python tests/test_harness_concurrency.py
"""
import io
import sys
import threading
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.harness.concurrency import quiet_stdout, serialize_calls  # noqa: E402


def test_quiet_stdout_is_safe_across_overlapping_threads():
    # The failure this replaces: per-call redirect_stdout blocks overlapping on
    # different threads restore each other's streams out of order, leaving
    # sys.stdout pointing at a closed devnull.
    original = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    errors = []

    def worker(i):
        try:
            for _ in range(50):
                with quiet_stdout():
                    print(f"vendored noise {i} — naïve č")   # non-Latin-1 must not raise
                    time.sleep(0.0005)
        except Exception as exc:   # pragma: no cover - surfaced below
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert sys.stdout is captured, "the original stream must be restored after the last exit"
        assert "vendored noise" not in captured.getvalue()
        print("still works")
        assert "still works" in captured.getvalue()
    finally:
        sys.stdout = original


def test_serialize_calls_never_overlaps_and_keeps_the_type():
    class Model:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def encode(self, x):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.005)
            self.active -= 1
            return x

    m = Model()
    assert serialize_calls(m, "encode") is m and isinstance(m, Model)
    held = m.encode                      # a reference taken after wrapping (like vendored code)
    serialize_calls(m, "encode")         # idempotent: no double wrapping
    assert m.encode is held
    threads = [threading.Thread(target=lambda: [held(i) for i in range(10)]) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.max_active == 1


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
