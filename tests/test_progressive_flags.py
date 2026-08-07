"""progressive/random_sample/sampling_seed resolve with the right defaults,
CLI/YAML precedence.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_progressive_flags.py
"""
import os
import sys
import tempfile
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")


def _resolve(yaml_text, extra_argv=None):
    """Run _resolve_config against a temp YAML with a CLI Namespace built the
    same way tests/test_evaluate.py does: build_arg_parser().parse_args()
    over an argv list (--config path [+ extra flags]).

    Always passes --no-strict-config: this file's fixtures are deliberately
    partial (only the progressive-relevant keys), and this file
    tests progressive semantics, not strict-config completeness
    (that's tests/test_strict_config_forge.py's job)."""
    from forge.orchestrator import _resolve_config, build_arg_parser
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        argv = ["--config", path, "--no-strict-config"] + list(extra_argv or [])
        args = build_arg_parser().parse_args(argv)
        return _resolve_config(args)
    finally:
        os.unlink(path)


_BASE_YAML = "datasets:\n  locomo: {}\n"


# ---------------- defaults ----------------

def test_defaults_preserve_current_behavior():
    cfg = _resolve(_BASE_YAML)
    assert cfg["progressive"] is True
    assert cfg["random_sample"] is False
    assert cfg["sampling_seed"] == 42
    # `coverage` is gone (2026-08) — `progressive` is the single knob.


# ---------------- progressive from CLI / YAML ----------------

def test_no_progressive_cli_flag():
    cfg = _resolve(_BASE_YAML, ["--no-progressive"])
    assert cfg["progressive"] is False


def test_progressive_cli_overrides_yaml():
    cfg = _resolve("datasets:\n  locomo: {}\nprogressive: false\n", ["--progressive"])
    assert cfg["progressive"] is True


def test_yaml_progressive_false():
    cfg = _resolve("datasets:\n  locomo: {}\nprogressive: false\n")
    assert cfg["progressive"] is False


# ---------------- random_sample / sampling_seed ----------------

def test_random_sample_cli_flag():
    cfg = _resolve(_BASE_YAML)
    assert cfg["random_sample"] is False
    cfg = _resolve(_BASE_YAML, ["--random-sample"])
    assert cfg["random_sample"] is True


def test_sampling_seed_cli_override():
    cfg = _resolve(_BASE_YAML, ["--sampling-seed", "7"])
    assert cfg["sampling_seed"] == 7


def test_sampling_seed_yaml_value():
    cfg = _resolve("datasets:\n  locomo: {}\nsampling_seed: 123\nrandom_sample: true\n")
    assert cfg["sampling_seed"] == 123
    assert cfg["random_sample"] is True


# ---------------- per-step seed derivation (Task 1 integration sanity) ----------------

def test_derive_sample_seed_used_by_orchestrator_import():
    # Sanity: orchestrator imports derive_sample_seed (used for per-step
    # wiring in propose_eval_one) — regression guard against a broken import.
    from forge.orchestrator import derive_sample_seed
    a = derive_sample_seed(42, 1, "locomo")
    b = derive_sample_seed(42, 2, "locomo")
    assert a != b
    assert derive_sample_seed(42, 1, "locomo") == a


# ---------------- Task 6b: forge/launch.py forwards sample_seed to get_task_list ----------------

def test_launch_style_seed_forwarding_varies_task_list():
    # Mirrors forge/launch.py's actual call (stage_spec.get("sample_seed") ->
    # get_task_list(..., seed=...)) without needing the container: proves the
    # per-step seed actually reaches task-LIST selection, not just per-user
    # QA/item sampling. Regression target for the gap found in Task 7 review.
    from datasets.longmemeval import env as lme

    def pick(spec):
        n = spec.get("n_samples")
        return lme.get_task_list(
            status="search",
            eval_n_samples=None if n is None else int(n),
            seed=spec.get("sample_seed"),
        )

    raw = pick({"n_samples": 20})  # no seed -> raw prefix (today's behavior)
    step0 = pick({"n_samples": 20, "sample_seed": "SEEDA"})
    step1 = pick({"n_samples": 20, "sample_seed": "SEEDB"})

    assert raw == pick({"n_samples": 20})  # seed absent = deterministic raw prefix
    assert step0 != raw  # a seed actually changes WHICH tasks are selected
    assert step0 != step1  # different step seeds -> different subsets
    assert len(step0) == len(step1) == 20  # same size regardless of seed
    assert set(step0) <= set(pick({"n_samples": None}))  # still within the split


def test_plan_kinds_recognized_by_evaluator_timeouts():
    """Regression lock (single-container rework, 2026-08): the evaluator picks
    its wall-clock cap from the plan kind — every kind the orchestrator can
    emit (smoke / gauntlet / single) must have a timeout entry, or a
    progressive=false run would silently fall back to the default cap."""
    import forge.evaluator as evaluator_mod
    for kind in ("smoke", "gauntlet", "single"):
        assert kind in evaluator_mod.SUBPROCESS_TIMEOUT, kind
    # gauntlet cap must cover the whole stage1+2+3 budget in one container
    assert evaluator_mod.SUBPROCESS_TIMEOUT["gauntlet"] >= (2 + 4 + 8) * 3600


def test_memcache_mounted_for_eval_never_smoke():
    """The cross-stage memory cache must be wired for gauntlet + single evals
    and NEVER for smoke/sanity (harness code can still change during the
    sanity-fix retry loop) — both on the host (orchestrator mounts the
    persistent dir) and inside evaluate_memo (skips smoke)."""
    import inspect
    import forge.orchestrator as orchestrator_mod
    import common.evaluate as evaluate_mod
    orch_src = inspect.getsource(orchestrator_mod.evaluate_harness)
    assert "if memory_cache and not smoke:" in orch_src
    em_src = inspect.getsource(evaluate_mod.evaluate_memo)
    assert "if memory_cache and not smoke:" in em_src


def test_evaluate_memo_forwards_sample_seed_to_get_task_list():
    # Lightweight source check: guard against someone reverting the shared
    # evaluate_memo's task-list call back to the old no-seed form.
    import inspect
    import common.evaluate as evaluate_mod
    src = inspect.getsource(evaluate_mod.evaluate_memo)
    call_start = src.index("task_list = env_module.get_task_list(")
    call_snippet = src[call_start:call_start + 300]
    assert "seed=spec.get(\"sample_seed\")" in call_snippet


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
