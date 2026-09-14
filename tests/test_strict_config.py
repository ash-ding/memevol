"""Config-completeness tests for the baseline entrypoints.

Two schemes coexist (2026-08-06):

- HARNESS baselines (hipporag2/amem/...) share ONE entrypoint
  (baselines/harness/eval_harness.py) and ONE frame config, validated by
  `common.config.validate_exact_config` against eval_harness.FRAME_KEYS
  (missing keys AND unknown keys both abort; sizing checked to the leaf).
  Method knobs are each memo.py's module-level CONFIG_DEFAULTS, resolved by
  eval_harness.resolve_memo_config; `memo:` overrides are validated.
- alma (evolve baseline) keeps the layered scheme (DEFAULT_CONFIG < YAML < CLI
  + strict_on gate) because its CLI carries genuine runtime knobs
  (--status/--steps/--memo_SHA).

Runs in the repo-root venv (or any venv): a baseline whose isolated
per-baseline-venv deps aren't installed in the CURRENT venv (e.g. amem's
sentence-transformers in the root venv) is SKIPPED, not failed — its own venv
verifies it. Same skip-on-missing-deps pattern as tests/test_config.py.
    uv run python tests/test_strict_config.py
"""
import sys, traceback, importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import (
    ConfigCompletenessError, load_config_file, validate_exact_config,
)

HARNESS_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "config.example.yaml"
ALMA_EXAMPLE = PROJECT_ROOT / "baselines" / "evolve" / "alma" / "config.example.yaml"


class _SkipTest(Exception):
    """This baseline's deps aren't installed in the current venv — skip it here
    (its own per-baseline venv verifies its config wiring)."""


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ImportError as e:  # ModuleNotFoundError included — baseline deps absent
        raise _SkipTest(f"{rel}: deps unavailable in this venv ({e})")
    return m


# ---------------------------------------------------------------------------
# Harness baselines — one shared frame config, exact validation
# ---------------------------------------------------------------------------

def _frame_cfg(**changes):
    cfg = dict(load_config_file(HARNESS_EXAMPLE))
    cfg.update(changes)
    return cfg


def test_shared_example_passes_exactly():
    from baselines.harness.eval_harness import load_frame_config
    load_frame_config(HARNESS_EXAMPLE)   # no raise


def test_missing_key_raises():
    from baselines.harness.eval_harness import FRAME_KEYS
    cfg = _frame_cfg(); del cfg["llm_model"]; del cfg["run_name"]
    try:
        validate_exact_config(cfg, FRAME_KEYS, "harness")
    except ConfigCompletenessError as e:
        assert "llm_model" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_unknown_key_raises():
    # Typo protection: an unknown key must abort, not silently ride along —
    # and a METHOD knob at the top level is an unknown key now (it belongs
    # under `memo:`).
    from baselines.harness.eval_harness import FRAME_KEYS
    for bad in ("llm_modle", "retrieve_k"):
        cfg = _frame_cfg(**{bad: 1}); del cfg["run_name"]
        try:
            validate_exact_config(cfg, FRAME_KEYS, "harness")
        except ConfigCompletenessError as e:
            assert bad in str(e)
        else:
            raise AssertionError("expected ConfigCompletenessError")


def test_missing_sizing_leaf_raises():
    # Second check layer: every top-level key present, but the active
    # single_stage block misses one native sizing leaf — must still raise,
    # naming the sizing path.
    from baselines.harness.eval_harness import FRAME_KEYS
    cfg = _frame_cfg(progressive=False, dataset="locomo",
                     single_stage={"n_conversations": 2})   # n_qa leaf missing
    del cfg["run_name"]
    try:
        validate_exact_config(cfg, FRAME_KEYS, "harness")
    except ConfigCompletenessError as e:
        assert "single_stage" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_memo_override_of_unknown_key_raises():
    # The `memo:` block is validated against CONFIG_DEFAULTS: a typo must
    # abort rather than silently become a no-op.
    from baselines.harness.eval_harness import resolve_memo_config

    defaults, model_keys = {"top_k": 5, "llm": "m"}, {"llm": ("llm",)}
    assert resolve_memo_config(defaults, model_keys, arm="faithful", overrides={"top_k": 9}) == {"top_k": 9, "llm": "m"}
    try:
        resolve_memo_config(defaults, model_keys, arm="faithful", overrides={"topk": 9})
    except KeyError as e:
        assert "topk" in str(e)
    else:
        raise AssertionError("expected KeyError")


# ---------------------------------------------------------------------------
# alma — keeps the layered DEFAULT_CONFIG < YAML < CLI + strict_on scheme
# ---------------------------------------------------------------------------

def _alma_strict_check(default_cfg, file_cfg, cli, dataset, progressive, context):
    """Mirror of the strict block alma's run.py runs after `resolve_config`."""
    from common.config import provided_keys, require_present_keys
    from common.evaluate import missing_sizing_config
    require_present_keys(provided_keys(file_cfg, cli), set(default_cfg) - {"strict_config"}, context)
    miss = missing_sizing_config(dataset, file_cfg, progressive, path_prefix="")
    if miss:
        raise ConfigCompletenessError(f"{context}: missing sizing leaf(s): {sorted(miss)}")


def test_alma_missing_key_raises():
    alma = _load("_alma_run_missing", "baselines/evolve/alma/run.py")
    fc = {"dataset": "locomo"}  # almost everything missing
    cli = {k: None for k in alma.DEFAULT_CONFIG}
    try:
        _alma_strict_check(alma.DEFAULT_CONFIG, fc, cli, "locomo", False, "alma")
    except ConfigCompletenessError:
        return
    raise AssertionError("expected ConfigCompletenessError")


def test_alma_complete_passes():
    alma = _load("_alma_run_complete", "baselines/evolve/alma/run.py")
    fc = load_config_file(ALMA_EXAMPLE)
    cli = {k: None for k in alma.DEFAULT_CONFIG}
    _alma_strict_check(alma.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "alma")  # no raise


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []; skipped = []
    for n, f in tests:
        try: f(); print(f"  PASS  {n}")
        except _SkipTest as e: print(f"  SKIP  {n}  ({e})"); skipped.append(n)
        except Exception: print(f"  FAIL  {n}"); traceback.print_exc(); failed.append(n)
    print(f"\n{len(tests)-len(failed)-len(skipped)}/{len(tests)} passed, {len(skipped)} skipped")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
