"""Config-completeness tests for the baseline entrypoints.

Two schemes coexist (2026-08-06):

- HARNESS baselines (cc/hipporag2/amem/...) are config-file-ONLY: run.py has
  no DEFAULT_CONFIG and no CLI parameter flags — just `--config`, validated by
  `common.config.validate_exact_config` against run.py's REQUIRED_KEYS
  (missing keys AND unknown keys both abort; sizing checked to the leaf).
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

CC_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "cc" / "config.example.yaml"
HIPPORAG2_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "hipporag2" / "config.example.yaml"
AMEM_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "amem" / "config.example.yaml"
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
# Harness baselines — config-file-only exact validation
# ---------------------------------------------------------------------------

def _harness_required(name):
    mod = _load(f"_{name}_run", f"baselines/harness/{name}/run.py")
    assert not hasattr(mod, "DEFAULT_CONFIG"), f"{name}: DEFAULT_CONFIG must be gone"
    return mod.REQUIRED_KEYS


def test_cc_example_passes_exactly():
    req = _harness_required("cc")
    validate_exact_config(load_config_file(CC_EXAMPLE), req, "cc")  # no raise


def test_cc_missing_key_raises():
    req = _harness_required("cc")
    cfg = dict(load_config_file(CC_EXAMPLE))
    del cfg["model"]
    try:
        validate_exact_config(cfg, req, "cc")
    except ConfigCompletenessError as e:
        assert "model" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_cc_unknown_key_raises():
    # Typo protection: an unknown key must abort, not silently ride along.
    req = _harness_required("cc")
    cfg = dict(load_config_file(CC_EXAMPLE))
    cfg["modle"] = "typo"
    try:
        validate_exact_config(cfg, req, "cc")
    except ConfigCompletenessError as e:
        assert "modle" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


def test_hipporag2_example_passes_exactly():
    req = _harness_required("hipporag2")
    validate_exact_config(load_config_file(HIPPORAG2_EXAMPLE), req, "hipporag2")


def test_amem_example_passes_exactly():
    req = _harness_required("amem")
    validate_exact_config(load_config_file(AMEM_EXAMPLE), req, "amem")


def test_amem_missing_sizing_leaf_raises():
    # Second check layer: every top-level key present, but the active
    # single_stage block misses one native sizing leaf — must still raise,
    # naming the sizing path.
    req = _harness_required("amem")
    cfg = dict(load_config_file(AMEM_EXAMPLE))
    cfg["progressive"] = False
    cfg["single_stage"] = {"n_conversations": 2}   # n_qa leaf missing
    cfg["dataset"] = "locomo"
    try:
        validate_exact_config(cfg, req, "amem")
    except ConfigCompletenessError as e:
        assert "single_stage" in str(e)
    else:
        raise AssertionError("expected ConfigCompletenessError")


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
