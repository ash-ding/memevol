"""Tests for Task 2: strict-config wiring in the four baseline run.py entrypoints
(baselines/harness/{cc,hipporag2,amem}/run.py + baselines/evolve/alma/run.py).

Runs in the repo-root venv/ (or any venv): each baseline's run.py is loaded to
read its DEFAULT_CONFIG. A baseline whose isolated per-baseline-venv deps aren't
installed in the CURRENT venv (e.g. amem's sentence-transformers in the root
venv) is SKIPPED, not failed — its own venv verifies it. Same skip-on-missing-
deps pattern as tests/test_config.py.
    uv run python tests/test_strict_config.py

Each test loads a run.py as a standalone module (via importlib, custom module
name — never registered under its real dotted name, and `__name__ !=
"__main__"` so the `if __name__ == "__main__":` guard never fires) purely to
read its DEFAULT_CONFIG, then drives the SAME strict-completeness check the
module's own main()/build_cfg() runs, via `_strict_check_flat` (a literal
mirror of the run.py block — see task-2-brief.md).

Two kinds of cases per baseline:
  - test_<m>_missing_key_raises: an almost-empty raw file_cfg -> raises
    ConfigCompletenessError. Exercises real behavior, works today.
  - test_<m>_complete_passes: loads the REAL `config.example.yaml` for that
    baseline (via `common.config.load_config_file`) -> does NOT raise. This is
    the Task 4 regression anchor that keeps the shipped example configs
    exhaustive + strict-passing (every DEFAULT_CONFIG key present, full sizing
    leaves on the active stages/single_stage block).
"""
import sys, traceback, importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config_file

CC_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "cc" / "config.example.yaml"
HIPPORAG2_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "hipporag2" / "config.example.yaml"
AMEM_EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "amem" / "config.example.yaml"
ALMA_EXAMPLE = PROJECT_ROOT / "baselines" / "evolve" / "alma" / "config.example.yaml"


class _SkipTest(Exception):
    """This baseline's deps aren't installed in the current venv — skip it here
    (its own per-baseline venv verifies its strict-config wiring)."""


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except ImportError as e:  # ModuleNotFoundError included — baseline deps absent
        raise _SkipTest(f"{rel}: deps unavailable in this venv ({e})")
    return m


def _strict_check_flat(default_cfg, file_cfg, cli, dataset, progressive, context):
    """Mirror of the strict block each run.py runs after `resolve_config`."""
    from common.config import provided_keys, require_present_keys, ConfigCompletenessError
    from common.staged_eval import missing_sizing_config
    require_present_keys(provided_keys(file_cfg, cli), set(default_cfg) - {"strict_config"}, context)
    miss = missing_sizing_config(dataset, file_cfg, progressive, path_prefix="")
    if miss:
        raise ConfigCompletenessError(f"{context}: missing sizing leaf(s): {sorted(miss)}")


# ---------------------------------------------------------------------------
# Hand-built complete fixtures — kept only where still used by a case OTHER
# than test_<m>_complete_passes (which now loads the real config.example.yaml
# directly; see the CC_EXAMPLE / HIPPORAG2_EXAMPLE / AMEM_EXAMPLE / ALMA_EXAMPLE
# paths above). _complete_hipporag2_cfg / _complete_alma_cfg were dropped —
# nothing else referenced them.
# ---------------------------------------------------------------------------

def _complete_cc_cfg():
    return {
        "dataset": "dynamicmem", "split": "test",
        "progressive": False, "sampling_seed": 42,
        "single_stage": {"n_users": None, "n_checkpoints": None, "n_task_a": None, "n_task_c": None},
        "stages": None, "memory_cache": True,
        "model": "sonnet", "max_turns": 30, "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3, "strict_config": True,
    }


def _complete_amem_cfg():
    return {
        "dataset": "locomo", "split": "test",
        "progressive": False, "sampling_seed": 42,
        "single_stage": {"n_conversations": None, "n_qa": None},
        "stages": None, "memory_cache": True,
        "amem_llm_model": "gpt-4o-mini", "retrieve_k": 10,
        "llm_model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3, "strict_config": True,
    }


# ---------------------------------------------------------------------------
# cc
# ---------------------------------------------------------------------------

def test_cc_missing_key_raises():
    cc = _load("_cc_run_missing", "baselines/harness/cc/run.py")
    from common.config import ConfigCompletenessError
    fc = {"dataset": "locomo"}  # almost everything missing
    cli = {k: None for k in cc.DEFAULT_CONFIG}
    raised = False
    try:
        _strict_check_flat(cc.DEFAULT_CONFIG, fc, cli, "locomo", False, "cc")
    except ConfigCompletenessError:
        raised = True
    assert raised


# Repointed (Task 4): loads the REAL config.example.yaml -> must pass strict.
def test_cc_complete_passes():
    cc = _load("_cc_run_complete", "baselines/harness/cc/run.py")
    fc = load_config_file(CC_EXAMPLE)
    cli = {k: None for k in cc.DEFAULT_CONFIG}
    _strict_check_flat(cc.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "cc")  # no raise


def test_cc_cli_provided_key_counts_as_present():
    # A required top-level key omitted from the YAML but supplied via a
    # non-None CLI value must still count as "provided" — exercises
    # provided_keys's CLI-union branch (every other test's `cli` is all-None).
    cc = _load("_cc_run_cli_present", "baselines/harness/cc/run.py")
    fc = _complete_cc_cfg()
    del fc["judge_model"]  # omitted from the "YAML" entirely
    cli = {k: None for k in cc.DEFAULT_CONFIG}
    cli["judge_model"] = "gpt-5-mini"  # supplied on the CLI instead
    _strict_check_flat(cc.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "cc")  # no raise


# ---------------------------------------------------------------------------
# hipporag2
# ---------------------------------------------------------------------------

def test_hipporag2_missing_key_raises():
    hp = _load("_hipporag2_run_missing", "baselines/harness/hipporag2/run.py")
    from common.config import ConfigCompletenessError
    fc = {"dataset": "locomo"}
    cli = {k: None for k in hp.DEFAULT_CONFIG}
    raised = False
    try:
        _strict_check_flat(hp.DEFAULT_CONFIG, fc, cli, "locomo", False, "hipporag2")
    except ConfigCompletenessError:
        raised = True
    assert raised


# Repointed (Task 4): loads the REAL config.example.yaml -> must pass strict.
def test_hipporag2_complete_passes():
    hp = _load("_hipporag2_run_complete", "baselines/harness/hipporag2/run.py")
    fc = load_config_file(HIPPORAG2_EXAMPLE)
    cli = {k: None for k in hp.DEFAULT_CONFIG}
    _strict_check_flat(hp.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "hipporag2")  # no raise


# ---------------------------------------------------------------------------
# amem
# ---------------------------------------------------------------------------

def test_amem_missing_key_raises():
    am = _load("_amem_run_missing", "baselines/harness/amem/run.py")
    from common.config import ConfigCompletenessError
    fc = {"dataset": "locomo"}
    cli = {k: None for k in am.DEFAULT_CONFIG}
    raised = False
    try:
        _strict_check_flat(am.DEFAULT_CONFIG, fc, cli, "locomo", False, "amem")
    except ConfigCompletenessError:
        raised = True
    assert raised


# Repointed (Task 4): loads the REAL config.example.yaml -> must pass strict.
def test_amem_complete_passes():
    am = _load("_amem_run_complete", "baselines/harness/amem/run.py")
    fc = load_config_file(AMEM_EXAMPLE)
    cli = {k: None for k in am.DEFAULT_CONFIG}
    _strict_check_flat(am.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "amem")  # no raise


def test_amem_missing_sizing_leaf_raises():
    # Isolates the SECOND check (missing_sizing_config) firing on its own: every
    # top-level key is present (require_present_keys passes clean), but the
    # active `single_stage` block is missing one native sizing leaf (n_qa for
    # the locomo family) — must still raise, naming the sizing path.
    am = _load("_amem_run_sizing_missing", "baselines/harness/amem/run.py")
    from common.config import ConfigCompletenessError
    fc = _complete_amem_cfg()
    del fc["single_stage"]["n_qa"]  # all top-level keys still present; one leaf missing
    cli = {k: None for k in am.DEFAULT_CONFIG}
    raised = False
    msg = ""
    try:
        _strict_check_flat(am.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "amem")
    except ConfigCompletenessError as e:
        raised = True
        msg = str(e)
    assert raised
    assert "single_stage" in msg


# ---------------------------------------------------------------------------
# alma
# ---------------------------------------------------------------------------

def test_alma_missing_key_raises():
    alma = _load("_alma_run_missing", "baselines/evolve/alma/run.py")
    from common.config import ConfigCompletenessError
    fc = {"dataset": "locomo"}
    cli = {k: None for k in alma.DEFAULT_CONFIG}
    raised = False
    try:
        _strict_check_flat(alma.DEFAULT_CONFIG, fc, cli, "locomo", False, "alma")
    except ConfigCompletenessError:
        raised = True
    assert raised


# Repointed (Task 4): loads the REAL config.example.yaml -> must pass strict.
def test_alma_complete_passes():
    alma = _load("_alma_run_complete", "baselines/evolve/alma/run.py")
    fc = load_config_file(ALMA_EXAMPLE)
    cli = {k: None for k in alma.DEFAULT_CONFIG}
    _strict_check_flat(alma.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "alma")  # no raise


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
