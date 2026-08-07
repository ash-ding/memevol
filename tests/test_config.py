"""Tests for common/config.py (shared baseline config resolution).
Zero-dep runner — run under BOTH venvs:
    uv run python tests/test_config.py
"""
import sys, tempfile, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_deep_merge_nested():
    from common.config import deep_merge
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    deep_merge(base, {"b": {"y": 20, "z": 3}, "c": 4})
    assert base == {"a": 1, "b": {"x": 1, "y": 20, "z": 3}, "c": 4}


def test_resolve_defaults_only():
    from common.config import resolve_config
    d = {"dataset": "locomo", "seed": 42}
    got = resolve_config(d, None, {"dataset": None, "seed": None})
    assert got == d
    assert got is not d                      # a copy, not the same object


def test_resolve_cli_beats_yaml_beats_default():
    from common.config import resolve_config
    import yaml
    d = {"dataset": "locomo", "seed": 42, "split": "test"}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"seed": 7, "split": "search"}, f)
        p = f.name
    # YAML overrides default; CLI (non-None) overrides YAML; None CLI keeps YAML.
    got = resolve_config(d, p, {"dataset": "dynamicmem", "seed": None, "split": None})
    assert got == {"dataset": "dynamicmem", "seed": 7, "split": "search"}


def test_resolve_missing_file_raises():
    from common.config import resolve_config
    raised = False
    try:
        resolve_config({"a": 1}, "/no/such/config.yaml", {})
    except FileNotFoundError:
        raised = True
    assert raised


def test_resolve_non_mapping_yaml_raises():
    from common.config import resolve_config
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump([1, 2, 3], f)         # a list, not a mapping
        p = f.name
    raised = False
    try:
        resolve_config({"a": 1}, p, {})
    except ValueError:
        raised = True
    assert raised


def test_amem_config_file_only_surface():
    # Harness baselines are config-file-ONLY (2026-08-06): run.py exposes
    # REQUIRED_KEYS (no DEFAULT_CONFIG, no CLI overrides) and its
    # config.example.yaml passes the exact-key validation as-is.
    try:
        import importlib
        m = importlib.import_module("baselines.harness.amem.run")
    except ImportError:
        print("    SKIP (amem deps unavailable in this venv)"); return
    from common.config import load_config_file, validate_exact_config
    assert not hasattr(m, "DEFAULT_CONFIG")
    assert "dataset" in m.REQUIRED_KEYS and "single_stage" in m.REQUIRED_KEYS
    cfg = load_config_file("baselines/harness/amem/config.example.yaml")
    validate_exact_config(cfg, m.REQUIRED_KEYS, "amem")  # no raise


def test_require_present_keys_missing_raises():
    from common.config import require_present_keys, ConfigCompletenessError
    raised = False
    try:
        require_present_keys({"a", "b"}, {"a", "b", "c"}, context="x config")
    except ConfigCompletenessError as e:
        # Quoted-repr match (message renders sorted(missing) as a list repr)
        # avoids a loose substring match masking a wrong/missing key.
        raised = "'c'" in str(e) and "strict" in str(e).lower()
    assert raised

def test_require_present_keys_null_counts_as_present():
    # provided is a KEY set — a key whose YAML value is null is still "provided".
    from common.config import require_present_keys
    require_present_keys({"a", "b", "c"}, {"a", "b", "c"}, context="x")  # no raise

def test_provided_keys_unions_yaml_and_nonnull_cli():
    from common.config import provided_keys
    got = provided_keys({"a": 1, "b": None}, {"c": 5, "d": None})
    assert got == {"a", "b", "c"}   # d is None → not provided

def test_strict_on_gating():
    from common.config import strict_on
    assert strict_on("cfg.yaml", {"strict_config": True}) is True
    assert strict_on(None, {"strict_config": True}) is False        # no --config
    assert strict_on("cfg.yaml", {"strict_config": False}) is False # escape hatch
    assert strict_on("cfg.yaml", {}) is True                        # default True

def test_require_schema_nested_missing_and_conditional():
    from common.config import require_schema, REQUIRED, Cond, ConfigCompletenessError
    schema = {
        "agent": REQUIRED,
        "proposer": {
            "codex": Cond(lambda c: c.get("agent") == "codex", {"model": REQUIRED}),
            "claude_code": Cond(lambda c: c.get("agent", "claude_code") == "claude_code", {"model": REQUIRED}),
        },
    }
    # agent=codex → codex.model required, claude_code branch inactive
    file_cfg = {"agent": "codex", "proposer": {"codex": {}}}
    raised = False
    try:
        require_schema(file_cfg, schema, resolved_cfg=file_cfg, context="forge")
    except ConfigCompletenessError as e:
        raised = "proposer.codex.model" in str(e)
    assert raised
    # complete + active branch only → passes (claude_code.model NOT required)
    ok = {"agent": "codex", "proposer": {"codex": {"model": "gpt-5.5"}}}
    require_schema(ok, schema, resolved_cfg=ok, context="forge")  # no raise

def test_missing_sizing_config_progressive_false_and_true():
    from common.evaluate import missing_sizing_config
    # progressive=false: single_stage must list every locomo leaf (null ok)
    assert missing_sizing_config("locomo", {"single_stage": {"n_conversations": None, "n_qa": None}}, False, "") == []
    assert missing_sizing_config("locomo", {"single_stage": {"n_conversations": None}}, False, "") == [".single_stage.n_qa"]
    # null block → the block itself is missing
    assert missing_sizing_config("locomo", {"single_stage": None}, False, "") == [".single_stage"]
    # progressive=true: all 4 entries, full leaves, +threshold on stage1/2
    full = {"stages": {
        "sanity_check": {"n_conversations": 1, "n_qa": 3},
        "stage1": {"n_conversations": 2, "n_qa": 20, "threshold": 0.3},
        "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
        "stage3": {"n_conversations": 6, "n_qa": 60},
    }}
    assert missing_sizing_config("locomo", full, True, "datasets.locomo") == []
    # missing threshold on stage1
    bad = {"stages": dict(full["stages"], stage1={"n_conversations": 2, "n_qa": 20})}
    assert missing_sizing_config("locomo", bad, True, "datasets.locomo") == ["datasets.locomo.stages.stage1.threshold"]


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
