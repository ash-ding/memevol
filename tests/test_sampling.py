"""Tests for common/sampling.py (seed derivation + shuffle-prefix primitive).
Zero-dependency runner — run under BOTH venvs:
    venv/bin/python tests/test_sampling.py
    baselines/venv/bin/python tests/test_sampling.py
"""
import sys, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_shuffle_prefix_none_seed_is_raw_prefix():
    from common.sampling import shuffle_prefix
    pool = [f"c{i}" for i in range(10)]
    assert shuffle_prefix(pool, 3, None) == ["c0", "c1", "c2"]   # raw prefix (back-compat)
    assert shuffle_prefix(pool, None, None) == pool               # None n = all
    assert shuffle_prefix(pool, 3, None) is not pool              # copy, not alias


def test_shuffle_prefix_seed_shuffles_but_deterministic():
    from common.sampling import shuffle_prefix
    pool = [f"c{i}" for i in range(10)]
    a = shuffle_prefix(pool, 4, "seedA")
    b = shuffle_prefix(pool, 4, "seedA")
    assert a == b                       # deterministic for same seed
    assert len(a) == 4
    assert set(a) <= set(pool)
    assert a != ["c0", "c1", "c2", "c3"]   # actually shuffled (not raw prefix)
    assert pool == [f"c{i}" for i in range(10)]   # input untouched


def test_shuffle_prefix_nesting_same_seed():
    # THE nesting invariant staged eval depends on: same seed → smaller ⊂ larger.
    from common.sampling import shuffle_prefix
    pool = [f"c{i}" for i in range(20)]
    small = shuffle_prefix(pool, 3, "s")
    large = shuffle_prefix(pool, 8, "s")
    assert small == large[:3]           # prefix-nested


def test_derive_sample_seed_reproducible_and_varies():
    from common.sampling import derive_sample_seed
    assert derive_sample_seed(42, 0, "locomo") == derive_sample_seed(42, 0, "locomo")  # reproducible
    assert derive_sample_seed(42, 0, "locomo") != derive_sample_seed(42, 1, "locomo")  # varies by step
    assert derive_sample_seed(42, 0, "locomo") != derive_sample_seed(7, 0, "locomo")   # varies by base
    assert derive_sample_seed(42, 0, "locomo") != derive_sample_seed(42, 0, "dynamicmem")  # varies by ds
    assert isinstance(derive_sample_seed(42, 0, "locomo"), str)


def test_combine_seed_backcompat_and_step():
    from common.sampling import combine_seed
    assert combine_seed(None, "conv-26") == "conv-26"          # back-compat: exactly user_dir
    got = combine_seed("STEPSEED", "conv-26")
    assert got != "conv-26" and "conv-26" in got               # step-varying, still user-scoped


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
