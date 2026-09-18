"""Tests for common/sampling.py (seed derivation + shuffle-prefix primitive).
Zero-dependency runner — run under BOTH venvs:
    uv run python tests/test_sampling.py
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


def test_get_task_list_seed_none_is_unchanged_all_datasets():
    # Back-compat anchor: seed=None must equal the historical raw-prefix output.
    from benchmarks.locomo import env as lc
    base = lc.get_task_list(status="search", eval_n_samples=2)
    assert lc.get_task_list(status="search", eval_n_samples=2, seed=None) == base
    # the pool's raw prefix is what "no seed" means:
    full = lc.get_task_list(status="search", eval_n_samples=None, seed=None)
    assert base == full[:2]


def test_get_task_list_seed_varies_and_nests():
    from benchmarks.longmemeval import env as lme
    full = lme.get_task_list(status="search", eval_n_samples=None, seed=None)
    if len(full) >= 8:
        s1 = lme.get_task_list(status="search", eval_n_samples=3, seed="STEP1")
        s2 = lme.get_task_list(status="search", eval_n_samples=6, seed="STEP1")
        assert s1 == s2[:3]                          # nested for same seed
        assert s1 != lme.get_task_list(status="search", eval_n_samples=3, seed=None)  # differs from raw prefix
        assert set(s1) <= set(full)


def test_get_task_list_dynamicmem_seed_none_and_nesting():
    from benchmarks.dynamicmem import env as dm
    base = dm.get_task_list(status="search", eval_n_samples=2)
    assert dm.get_task_list(status="search", eval_n_samples=2, seed=None) == base   # back-compat
    full = dm.get_task_list(status="search", eval_n_samples=None, seed=None)
    assert base == full[:2]
    if len(full) >= 4:
        s1 = dm.get_task_list(status="search", eval_n_samples=2, seed="S")
        s2 = dm.get_task_list(status="search", eval_n_samples=3, seed="S")
        assert s1 == s2[:2]                       # nested for same seed
        assert set(s1) <= set(full)


def test_locomo_qa_sampling_honors_stage_sample_seed():
    # load_user_data must seed QA sampling on combine_seed(sample_seed, user_dir).
    from benchmarks.locomo import env as lc
    from common.sampling import combine_seed
    tasks = lc.get_task_list(status="search", eval_n_samples=1)
    uid = tasks[0]
    _c, _p, base = lc.load_user_data(uid, eval_n_qa=5)                 # historical seed (user_dir)
    _c, _p, none_seed = lc.load_user_data(uid, eval_n_qa=5, sample_seed=None)
    assert [q["query"] for q in none_seed] == [q["query"] for q in base]   # back-compat
    _c, _p, stepped = lc.load_user_data(uid, eval_n_qa=5, sample_seed="STEP9")
    # different step seed → (very likely) different QA subset, same size
    assert len(stepped) == len(base)
    assert [q["query"] for q in stepped] != [q["query"] for q in base]


def test_the_split_is_twenty_percent_search():
    """The search/held-out ratio is a research decision, not an incidental
    number: 20% to search on, 80% held out (changed from 60/40 on 2026-09-18).
    Scores never cross a change to this — the membership differs."""
    from benchmarks.locomo.env import TRAIN_SAMPLES, EVAL_SAMPLES
    from benchmarks.dynamicmem.env import TRAIN_USERS, EVAL_USERS
    from benchmarks.longmemeval.env import SEARCH_SIZE

    assert (TRAIN_SAMPLES, EVAL_SAMPLES) == (2, 8)      # 10 conversations
    assert (TRAIN_USERS, EVAL_USERS) == (2, 8)          # 10 users
    assert SEARCH_SIZE == 100                            # of 500 questions


def test_no_gauntlet_stage_asks_for_more_units_than_the_search_split_holds():
    """A stage that exceeds the pool is silently clamped to it, which would
    make two stages identical and the promotion gate between them a no-op."""
    from common.evaluate import DEFAULT_STAGES
    from benchmarks.locomo.env import TRAIN_SAMPLES
    from benchmarks.dynamicmem.env import TRAIN_USERS
    from benchmarks.longmemeval.env import SEARCH_SIZE

    caps = {"locomo": ("n_conversations", TRAIN_SAMPLES),
            "dynamicmem": ("n_users", TRAIN_USERS),
            "longmemeval": ("n_questions", SEARCH_SIZE)}
    for family, (field, cap) in caps.items():
        sizes = [DEFAULT_STAGES[family][s][field]
                 for s in ("sanity_check", "stage1", "stage2", "stage3")]
        assert all(n <= cap for n in sizes), (family, sizes, cap)
        assert sizes == sorted(sizes), f"{family} stages must not shrink: {sizes}"


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
