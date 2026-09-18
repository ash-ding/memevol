"""Packaging a harness baseline as a forge harness.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_package_baseline.py                 # no_memory
    uv run --project baselines/harness/mem0 python tests/test_package_baseline.py

The second spelling also exercises a baseline with vendored `src/` and real
dependencies; in the repo-root venv those tests skip themselves rather than
fail, since importing mem0's memo.py needs mem0's venv.
"""
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from tools.package_baseline import (_SHARED_MODULES, _export_requirements,  # noqa: E402
                                    package)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def test_exported_requirements_are_pinned_and_carry_their_markers():
    """Read straight out of uv.lock, a Windows-only wheel (portalocker pulls
    pywin32) is pinned as if it were needed and the image build dies on it.
    The export writes it with its marker instead, which pip applies."""
    if not _have_uv():
        print("    (skipped: needs uv on PATH)")
        return
    text = _export_requirements(PROJECT_ROOT / "baselines" / "harness" / "mem0")
    pins = [l for l in text.splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith(" ")]
    assert len(pins) > 10, pins[:5]
    assert all("==" in p for p in pins), "every dependency is pinned"
    windows_only = [p for p in pins if p.startswith("pywin32")]
    assert all(";" in p and "win32" in p for p in windows_only), windows_only
    assert not any("memevol-baseline" in p for p in pins), \
        "the project itself is not a dependency"


def test_packaging_produces_a_harness_forge_can_load():
    from forge.contract import load_harness_class

    with tempfile.TemporaryDirectory() as td:
        out = package("no_memory", "faithful", None, Path(td) / "pkg")

        # The shape an evolved candidate has: the interface, then everything
        # else under src/. No nested package path, and nothing named memo.py
        # inside src/ to be mistaken for a second interface.
        assert (out / "memo.py").exists() and (out / "meta.json").exists()
        assert (out / "src" / "adapter.py").exists()
        assert (out / "src" / "model_config.py").exists()
        assert not (out / "baselines").exists(), "the nested package path is gone"
        assert not (out / "src" / "memo.py").exists()
        assert not any(
            "baselines." in f.read_text(encoding="utf-8")
            for f in (out / "src").rglob("*.py")
            if f.name in ("adapter.py",) + _SHARED_MODULES), \
            "an absolute baselines.* import survived the flattening"

        cls = load_harness_class(out)
        assert cls.__name__ == "PackagedHarness"
        memo = cls()
        assert memo.config == {}, "no_memory has no method knobs"

        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["parent_ids"] == [] and meta["arm"] == "faithful"
        assert meta["packaged_from"] == "baselines/harness/no_memory"


def test_packaging_replaces_a_previous_package_rather_than_merging():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "pkg"
        package("no_memory", "faithful", None, out)
        stale = out / "src" / "stale.py"
        stale.write_text("# left over from an older packaging\n", encoding="utf-8")
        package("no_memory", "faithful", None, out)
        assert not stale.exists()


def _mem0_importable() -> bool:
    """True in mem0's own venv. The vendored package lives under `src/` and is
    only reachable through memo.py (which puts it on sys.path), so importing
    `mem0` directly would say nothing."""
    try:
        import baselines.harness.mem0.memo  # noqa: F401
        return True
    except Exception:
        return False


def test_a_vendored_baseline_travels_with_its_src_and_resolved_config():
    """mem0 in its own venv: the vendored package comes along, and the unified
    arm's models are baked into memo.py."""
    if not _mem0_importable():
        print("    (skipped: needs baselines/harness/mem0's venv)")
        return
    from forge.contract import load_harness_class

    with tempfile.TemporaryDirectory() as td:
        out = package("mem0", "unified",
                      {"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"},
                      Path(td) / "pkg")

        assert (out / "src" / "mem0").is_dir(), "the vendored package travels with it"
        reqs = (out / "requirements.txt").read_text(encoding="utf-8")
        assert "qdrant-client==" in reqs and "openai==" in reqs
        assert "pywin32" not in reqs or "sys_platform" in reqs, \
            "a Windows-only pin must carry its marker or not be there at all"

        cls = load_harness_class(out)
        memo = cls()
        assert memo.config["mem0_llm_model"] == "gpt-5-mini"
        assert memo.config["embedding_model"] == "text-embedding-3-small"
        # A caller that supplies a config still wins (eval_harness does).
        assert cls(config={"mem0_llm_model": "x"}).config == {"mem0_llm_model": "x"}


def test_the_unified_arm_is_packaged_on_a_model_these_methods_can_drive():
    """The vendored methods batch large prompts, fan out hard and carry their
    own timeouts. Measured on mem0's real extraction call, a reasoning model
    took 15-23 s against 2-3 s here, emitted ~5x the completion tokens, and
    repeatedly hit windows where it never returned — LightMem and SimpleMem
    never completed a build that way. Whatever the default is, the shim has to
    understand it: a string it cannot parse reaches the SDK verbatim."""
    from tools.package_baseline import build_parser
    from baselines.harness.model_config import normalise_chat_params

    args = build_parser().parse_args(["mem0", "--arm", "unified", "--out", "x"])
    model = args.unified_llm
    assert not model.startswith(("gpt-5", "o1", "o3", "o4")), \
        f"the unified default is a reasoning model again: {model}"

    out = normalise_chat_params({"model": model, "messages": []})
    assert out["model"] == model.split("/")[0]
    assert "reasoning_effort" not in out


def test_an_effort_suffix_still_reaches_the_shim_intact():
    """Opting into a reasoning model stays available."""
    from baselines.harness.model_config import normalise_chat_params

    out = normalise_chat_params({"model": "gpt-5-mini/low", "messages": []})
    assert out["model"] == "gpt-5-mini"
    assert out["reasoning_effort"] == "low"


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
