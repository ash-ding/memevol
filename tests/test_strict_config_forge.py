"""Tests for Task 3: strict-config wiring into forge's nested config resolver
(forge/orchestrator.py::FORGE_REQUIRED_SCHEMA + _forge_strict_validate +
_forge_provided_tree, gated into _resolve_config via common.config.strict_on).

Zero-dep runner — run under the ROOT (forge) venv:
    venv/bin/python tests/test_strict_config_forge.py

Kept SEPARATE from tests/test_strict_config.py (Task 2, baseline run.py
strict-config cases) because that file needs the BASELINES venv (imports
baseline run.py modules -> sentence_transformers for amem); forge cases here
import forge.orchestrator, which needs only the root venv.

Each test loads forge/orchestrator.py as a standalone module (via importlib,
custom module name — never registered under its real dotted name, and
`__name__ != "__main__"` so no code under an `if __name__ == "__main__":`
guard ever fires) so it can call the internal `_forge_strict_validate`
directly without building a full argparse Namespace / going through the
whole CLI.
"""
import sys
import traceback
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Hand-built complete fixture — a full forge YAML dict satisfying
# FORGE_REQUIRED_SCHEMA + the dynamic `datasets` sizing loop. Kept in-file for
# the synthetic missing-key / codex-branch cases below (they need to mutate
# individual leaves in isolation); the real-file anchors
# (test_forge_search_example_passes_strict / test_forge_test_example_passes_strict,
# Task 5) load configs/search_example.yaml + configs/test_example.yaml directly.
# ---------------------------------------------------------------------------

def _forge_min_cfg(agent="claude_code"):
    cfg = {
        "steps": 5, "smoke_test": False,
        "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "progressive": True, "random_sample": False, "sampling_seed": 42,
        "max_sample_concurrent": 3,
        "memory_cache": True, "adopt_orphans": True,
        "agent": agent,
        "llm": {
            "anthropic_transport": "api",
            "vertex": {"project_id": "p", "region": "r", "credentials": None},
        },
        "proposer": {
            "max_turns": 80, "timeout_s": 1500,
            "claude_code": {
                "model": "claude-opus-4-7", "effort": "medium",
                "disallowed_tools": ["mcp__*"], "auth": "subscription",
                "vertex": {"project_id": "p", "region": "r", "credentials": None},
            },
            "codex": {"model": "gpt-5.5", "reasoning_effort": "medium"},
        },
        "propose": {"k_per_step": 2},
        "sanity": {"enabled": True, "max_retries": 2},
        "seed": {"enabled": False},
        "gpu": {"enabled": False},
        "prompts": {"version": "latest"},
        "datasets": {
            "locomo": {
                "stages": {
                    "sanity_check": {"n_conversations": 1, "n_qa": 3},
                    "stage1": {"n_conversations": 2, "n_qa": 20, "threshold": 0.30},
                    "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
                    "stage3": {"n_conversations": 6, "n_qa": 60},
                },
            },
        },
    }
    if agent == "codex":
        # "complete claude_code cfg minus codex.model" — the codex branch
        # becomes active (agent=="codex") but is missing its REQUIRED model.
        del cfg["proposer"]["codex"]["model"]
    return cfg


def test_forge_search_example_passes_strict():
    """configs/search_example.yaml is the real, checked-in exhaustive forge
    search template — it must pass FORGE_REQUIRED_SCHEMA as shipped."""
    orch = _load("_orch1", "forge/orchestrator.py")
    from common.config import load_config_file
    fc = load_config_file(str(PROJECT_ROOT / "configs" / "search_example.yaml"))
    orch._forge_strict_validate(fc, resolved_cfg={**fc, "progressive": True})  # no raise


def test_forge_test_example_passes_strict():
    """configs/test_example.yaml is the real, checked-in exhaustive held-out
    template — it must pass the smaller HELDOUT_REQUIRED_SCHEMA as shipped
    (heldout is progressive=false)."""
    orch = _load("_orch1b", "forge/orchestrator.py")
    from common.config import load_config_file
    fc = load_config_file(str(PROJECT_ROOT / "configs" / "test_example.yaml"))
    orch._forge_strict_validate(fc, resolved_cfg={**fc, "progressive": False},
                                schema=orch.HELDOUT_REQUIRED_SCHEMA)  # no raise


def test_forge_missing_top_key_raises():
    orch = _load("_orch2", "forge/orchestrator.py")
    from common.config import ConfigCompletenessError
    fc = {"steps": 1}  # nearly everything missing
    raised = False
    try:
        orch._forge_strict_validate(fc, resolved_cfg={**fc, "progressive": True})
    except ConfigCompletenessError as e:
        raised = "judge_model" in str(e) and "datasets" in str(e)
    assert raised


def test_forge_codex_branch_conditional():
    orch = _load("_orch3", "forge/orchestrator.py")
    from common.config import ConfigCompletenessError
    # agent=codex → proposer.codex.model required; claude_code branch NOT
    fc = _forge_min_cfg(agent="codex")
    raised = False
    try:
        orch._forge_strict_validate(fc, resolved_cfg=fc)
    except ConfigCompletenessError as e:
        raised = "proposer.codex.model" in str(e)
    assert raised


def test_forge_complete_cfg_claude_code_branch_passes():
    # Sanity companion to the codex case: the default claude_code-agent
    # fixture (codex.model present but IGNORED since agent!=codex) must NOT
    # raise — regression guard against the Cond wiring flipping direction.
    orch = _load("_orch4", "forge/orchestrator.py")
    fc = _forge_min_cfg(agent="claude_code")
    orch._forge_strict_validate(fc, resolved_cfg=fc)  # no raise


def test_forge_strict_off_via_config_flag():
    """--no-strict-config / strict_config: false must bypass the gate inside
    _resolve_config end-to-end (not just the internal validator), even for a
    badly incomplete --config file."""
    import tempfile
    import os
    import yaml
    orch = _load("_orch5", "forge/orchestrator.py")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"datasets": {"locomo": {}}, "strict_config": False}, f)
        cfg = orch._resolve_config(
            orch.build_arg_parser().parse_args(["--config", p]))
        assert cfg["strict_config"] is False


def test_forge_strict_on_via_config_flag_raises_end_to_end():
    """The full _resolve_config path (not just _forge_strict_validate in
    isolation) raises ConfigCompletenessError for an incomplete --config file
    when strict_config is left at its default (True)."""
    import tempfile
    import os
    import yaml
    orch = _load("_orch6", "forge/orchestrator.py")
    from common.config import ConfigCompletenessError
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.yaml")
        with open(p, "w") as f:
            yaml.safe_dump({"datasets": {"locomo": {}}}, f)
        raised = False
        try:
            orch._resolve_config(orch.build_arg_parser().parse_args(["--config", p]))
        except ConfigCompletenessError:
            raised = True
        assert raised


def test_forge_no_config_file_never_trips_strict():
    """Strict mode is gated on args.config being non-None — an in-process
    config (no --config file at all) must never trip it, however incomplete."""
    orch = _load("_orch7", "forge/orchestrator.py")
    from argparse import Namespace
    args = orch.build_arg_parser().parse_args(["--datasets", "locomo"])
    assert args.config is None
    cfg = orch._resolve_config(args)  # must not raise
    assert "locomo" in cfg["datasets"]


# ---------------------------------------------------------------------------
# forge.heldout's smaller real config surface (Task 3 Step 7 — plan gap
# review): forge.heldout also drives _resolve_config, but never reads the
# search-loop-only fields (steps/smoke_test/random_sample/sampling_seed/
# adopt_orphans/agent/proposer.*/propose.*/sanity.*/seed.*/prompts.*), so
# FORGE_REQUIRED_SCHEMA wrongly demanded them for heldout configs.
# HELDOUT_REQUIRED_SCHEMA is the smaller schema forge.heldout passes via
# _resolve_config's `required_schema` kwarg. These two synthetic cases use an
# in-file fixture for isolated leaf-deletion; the real-file anchor
# (test_forge_test_example_passes_strict, Task 5, above) covers
# configs/test_example.yaml directly.
# ---------------------------------------------------------------------------

def _heldout_min_cfg():
    return {
        "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3, "memory_cache": True,
        "gpu": {"enabled": False},
        "llm": {
            "anthropic_transport": "api",
            "vertex": {"project_id": "p", "region": "r", "credentials": None},
        },
        "datasets": {
            "locomo": {
                "single_stage": {"n_conversations": None, "n_qa": None},
            },
        },
    }


def test_heldout_schema_omits_search_fields():
    """A cfg complete for HELDOUT_REQUIRED_SCHEMA but with NO proposer/sanity/
    seed (search-loop-only fields heldout never reads) must NOT raise."""
    orch = _load("_orch8", "forge/orchestrator.py")
    fc = _heldout_min_cfg()
    assert "proposer" not in fc and "sanity" not in fc and "seed" not in fc
    orch._forge_strict_validate(fc, resolved_cfg={**fc, "progressive": False},
                                schema=orch.HELDOUT_REQUIRED_SCHEMA)  # no raise


def test_heldout_schema_missing_model_raises():
    orch = _load("_orch9", "forge/orchestrator.py")
    from common.config import ConfigCompletenessError
    fc = _heldout_min_cfg()
    del fc["model"]
    raised = False
    try:
        orch._forge_strict_validate(fc, resolved_cfg={**fc, "progressive": False},
                                    schema=orch.HELDOUT_REQUIRED_SCHEMA)
    except ConfigCompletenessError as e:
        raised = "model" in str(e)
    assert raised


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
