"""Tests for common/config.py (shared baseline config resolution).
Zero-dep runner — run under BOTH venvs:
    venv/bin/python tests/test_config.py
    baselines/venv/bin/python tests/test_config.py
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


def test_amem_default_config_roundtrips_to_argparse_defaults():
    # Backward-compat anchor: no --config, no CLI → DEFAULT_CONFIG unchanged,
    # and DEFAULT_CONFIG equals the historical argparse defaults.
    import importlib
    m = importlib.import_module("baselines.harness.amem.run")
    from common.config import resolve_config
    d = m.DEFAULT_CONFIG
    assert d["dataset"] == "locomo" and d["split"] == "test"
    assert d["sampling_seed"] == 42 and d["memory_cache"] is True
    assert d["progressive"] is False and d["retrieve_k"] == 10
    assert d["amem_llm_model"] == "gpt-4o-mini"
    # all-None overrides → identical to defaults
    none_over = {k: None for k in d}
    assert resolve_config(d, None, none_over) == d


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
