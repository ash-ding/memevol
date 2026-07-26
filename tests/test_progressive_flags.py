"""progressive/random_sample/sampling_seed resolve with the right defaults,
CLI/YAML precedence, and the coverage<->progressive alias mapping.

Zero-dependency runner (no pytest in the venvs):

    venv/bin/python tests/test_progressive_flags.py
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
    same way tests/test_staged_eval.py does: build_arg_parser().parse_args()
    over an argv list (--config path [+ extra flags])."""
    from forge.orchestrator import _resolve_config, build_arg_parser
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        argv = ["--config", path] + list(extra_argv or [])
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
    assert cfg["coverage"] == "sample"  # derived, stays in sync


# ---------------- --coverage alias ----------------

def test_coverage_full_maps_to_non_progressive():
    cfg = _resolve(_BASE_YAML, ["--coverage", "full"])
    assert cfg["progressive"] is False
    assert cfg["coverage"] == "full"


def test_coverage_sample_maps_to_progressive():
    cfg = _resolve(_BASE_YAML, ["--coverage", "sample"])
    assert cfg["progressive"] is True
    assert cfg["coverage"] == "sample"


def test_yaml_coverage_full_without_progressive_key_maps_too():
    """The alias also fires from a YAML-only `coverage: full` (no CLI flag
    at all) — matches today's configs that pre-date `progressive`."""
    cfg = _resolve("datasets:\n  locomo: {}\ncoverage: full\n")
    assert cfg["progressive"] is False
    assert cfg["coverage"] == "full"


# ---------------- explicit progressive always wins over coverage ----------------

def test_explicit_progressive_wins_over_coverage_cli():
    cfg = _resolve(_BASE_YAML, ["--coverage", "full", "--progressive"])
    assert cfg["progressive"] is True
    assert cfg["coverage"] == "sample"  # re-derived from the winning flag


def test_no_progressive_wins_over_coverage_sample():
    cfg = _resolve(_BASE_YAML, ["--coverage", "sample", "--no-progressive"])
    assert cfg["progressive"] is False
    assert cfg["coverage"] == "full"


def test_yaml_progressive_false_wins_over_default_coverage():
    cfg = _resolve("datasets:\n  locomo: {}\nprogressive: false\n")
    assert cfg["progressive"] is False
    assert cfg["coverage"] == "full"


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


def test_single_stage_name_recognized_by_all_container_wire_gates():
    """Fix-round regression lock (2026-07-26 review): resolve_sampling_plan
    (progressive=false) emits the plan/stage name "single" — every place
    that gates on the stage-name literal (memcache mount + wall-clock
    timeout, on BOTH the host orchestrator side and the in-container
    launch.py side) must recognize it, or memory_cache=True silently
    becomes a no-op / the container crashes with an argparse error. This is
    a static/source guard on top of the behavioral
    test_evaluate_harness_full_wires_memcache_dir in test_heldout.py, which
    exercises the actual gate end-to-end."""
    import inspect
    import forge.launch as launch_mod
    import forge.evaluator as evaluator_mod
    import forge.orchestrator as orchestrator_mod

    launch_src = inspect.getsource(launch_mod)
    # --stage argparse choices
    choices_start = launch_src.index('choices=["sanity", "stage1", "stage2", "stage3"')
    assert '"single"' in launch_src[choices_start:choices_start + 200]
    # memcache mount gate
    gate_start = launch_src.index("args.stage in (")
    assert '"single"' in launch_src[gate_start:gate_start + 100]

    assert "single" in evaluator_mod.SUBPROCESS_TIMEOUT

    orch_src = inspect.getsource(orchestrator_mod.evaluate_harness)
    gate_start = orch_src.index("stage_name in (")
    assert '"single"' in orch_src[gate_start:gate_start + 100]


def test_launch_py_forwards_sample_seed_to_get_task_list():
    # Lightweight source check: guard against someone reverting the
    # forge/launch.py forwarding line back to the old no-seed call.
    import inspect
    import forge.launch as launch_mod
    src = inspect.getsource(launch_mod)
    call_start = src.index("task_list = env_module.get_task_list(")
    call_snippet = src[call_start:call_start + 300]
    assert "seed=stage_spec.get(\"sample_seed\")" in call_snippet


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
