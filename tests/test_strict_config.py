"""Tests for Task 2: strict-config wiring in the four baseline run.py entrypoints
(baselines/harness/{cc,hipporag2,amem}/run.py + baselines/evolve/alma/run.py).

Zero-dep runner — run under the baselines venv:
    baselines/venv/bin/python tests/test_strict_config.py

Each test loads a run.py as a standalone module (via importlib, custom module
name — never registered under its real dotted name, and `__name__ !=
"__main__"` so the `if __name__ == "__main__":` guard never fires) purely to
read its DEFAULT_CONFIG, then drives the SAME strict-completeness check the
module's own main()/build_cfg() runs, via `_strict_check_flat` (a literal
mirror of the run.py block — see task-2-brief.md).

Two kinds of cases per baseline:
  - test_<m>_missing_key_raises: an almost-empty raw file_cfg -> raises
    ConfigCompletenessError. Exercises real behavior, works today.
  - test_<m>_complete_passes: a HAND-BUILT complete fixture dict (every
    DEFAULT_CONFIG key present + full single_stage sizing leaves) -> does NOT
    raise. Fixtures are hand-built here (NOT the real config.example.yaml,
    which isn't exhaustive until Task 4) — each fixture carries a
    `# repointed to the real config in Task 4` comment.
"""
import sys, traceback, importlib.util
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _strict_check_flat(default_cfg, file_cfg, cli, dataset, progressive, context):
    """Mirror of the strict block each run.py runs after `resolve_config`."""
    from common.config import provided_keys, require_present_keys, ConfigCompletenessError
    from common.staged_eval import missing_sizing_config
    require_present_keys(provided_keys(file_cfg, cli), set(default_cfg) - {"strict_config"}, context)
    miss = missing_sizing_config(dataset, file_cfg, progressive, path_prefix="")
    if miss:
        raise ConfigCompletenessError(f"{context}: missing sizing leaf(s): {sorted(miss)}")


# ---------------------------------------------------------------------------
# Hand-built complete fixtures (Task 2 anchor; Task 4 repoints to the real
# config.example.yaml once those are made exhaustive).
# ---------------------------------------------------------------------------

def _complete_cc_cfg():
    # repointed to the real config in Task 4
    return {
        "dataset": "dynamicmem", "split": "test",
        "progressive": False, "sampling_seed": 42,
        "single_stage": {"n_users": None, "n_checkpoints": None, "n_task_a": None, "n_task_c": None},
        "stages": None, "memory_cache": True,
        "model": "sonnet", "max_turns": 30, "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3, "strict_config": True,
    }


def _complete_hipporag2_cfg():
    # repointed to the real config in Task 4
    return {
        "dataset": "dynamicmem", "split": "test",
        "progressive": False, "sampling_seed": 42,
        "single_stage": {"n_users": None, "n_checkpoints": None, "n_task_a": None, "n_task_c": None},
        "stages": None, "memory_cache": True,
        "embedding": "text-embedding-3-small", "llm_model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "embedding_batch_size": None, "embedding_dtype": None, "max_sample_concurrent": 3,
        "strict_config": True,
    }


def _complete_amem_cfg():
    # repointed to the real config in Task 4
    return {
        "dataset": "locomo", "split": "test",
        "progressive": False, "sampling_seed": 42,
        "single_stage": {"n_conversations": None, "n_qa": None},
        "stages": None, "memory_cache": True,
        "amem_llm_model": "gpt-4o-mini", "retrieve_k": 10,
        "llm_model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3, "strict_config": True,
    }


def _complete_alma_cfg():
    # repointed to the real config in Task 4
    return {
        "meta_model": "gpt-5", "execution_model": "gpt-5-mini", "steps": 10,
        "max_memo_concurrent": 2, "result_dir": "check", "status": "search",
        "dataset": "dynamicmem", "memo_SHA": None, "history_ckpt_path": None,
        "max_logs": None, "max_sample_concurrent": 3, "n_score_bins": 3,
        "samples_per_bin": 3, "judge_model": "gpt-5-mini",
        "progressive": False, "random_sample": False, "sampling_seed": 42,
        "stages": None,
        "single_stage": {"n_users": None, "n_checkpoints": None, "n_task_a": None, "n_task_c": None},
        "memory_cache": True, "strict_config": True,
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


# GREEN after Task 4 (hand-built fixture stands in for config.example.yaml)
def test_cc_complete_passes():
    cc = _load("_cc_run_complete", "baselines/harness/cc/run.py")
    fc = _complete_cc_cfg()
    cli = {k: None for k in cc.DEFAULT_CONFIG}
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


# GREEN after Task 4 (hand-built fixture stands in for config.example.yaml)
def test_hipporag2_complete_passes():
    hp = _load("_hipporag2_run_complete", "baselines/harness/hipporag2/run.py")
    fc = _complete_hipporag2_cfg()
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


# GREEN after Task 4 (hand-built fixture stands in for config.example.yaml)
def test_amem_complete_passes():
    am = _load("_amem_run_complete", "baselines/harness/amem/run.py")
    fc = _complete_amem_cfg()
    cli = {k: None for k in am.DEFAULT_CONFIG}
    _strict_check_flat(am.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "amem")  # no raise


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


# GREEN after Task 4 (hand-built fixture stands in for config.example.yaml)
def test_alma_complete_passes():
    alma = _load("_alma_run_complete", "baselines/evolve/alma/run.py")
    fc = _complete_alma_cfg()
    cli = {k: None for k in alma.DEFAULT_CONFIG}
    _strict_check_flat(alma.DEFAULT_CONFIG, fc, cli, fc["dataset"], fc["progressive"], "alma")  # no raise


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for n, f in tests:
        try: f(); print(f"  PASS  {n}")
        except Exception: print(f"  FAIL  {n}"); traceback.print_exc(); failed.append(n)
    print(f"\n{len(tests)-len(failed)}/{len(tests)} passed")
    if failed: sys.exit(1)


if __name__ == "__main__":
    main()
