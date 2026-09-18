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

from tools.package_baseline import _export_requirements, package   # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _have_uv() -> bool:
    return shutil.which("uv") is not None


def test_exported_requirements_are_pinned_and_container_only():
    """Read straight out of uv.lock, a Windows-only wheel (portalocker pulls
    pywin32) would be pinned and fail the image build — so the export asks for
    the container's platform."""
    if not _have_uv():
        print("    (skipped: needs uv on PATH)")
        return
    text = _export_requirements(PROJECT_ROOT / "baselines" / "harness" / "mem0")
    pins = [l for l in text.splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith(" ")]
    assert len(pins) > 10, pins[:5]
    assert all("==" in p for p in pins), "every dependency is pinned"
    assert not any(p.startswith("pywin32") for p in pins), pins
    assert not any("memevol-baseline" in p for p in pins), \
        "the project itself is not a dependency"


def test_packaging_produces_a_harness_forge_can_load():
    from forge.contract import load_harness_class

    with tempfile.TemporaryDirectory() as td:
        out = package("no_memory", "faithful", None, Path(td) / "pkg")

        assert (out / "harness.py").exists() and (out / "meta.json").exists()
        # The method keeps its real package path, so memo.py's own imports work.
        assert (out / "baselines" / "harness" / "no_memory" / "memo.py").exists()
        assert (out / "baselines" / "harness" / "model_config.py").exists()

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
        stale = out / "baselines" / "harness" / "no_memory" / "stale.py"
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
    arm's models are baked into harness.py."""
    if not _mem0_importable():
        print("    (skipped: needs baselines/harness/mem0's venv)")
        return
    from forge.contract import load_harness_class

    with tempfile.TemporaryDirectory() as td:
        out = package("mem0", "unified",
                      {"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"},
                      Path(td) / "pkg")

        pkg = out / "baselines" / "harness" / "mem0"
        assert (pkg / "src" / "mem0").is_dir(), "the vendored package travels with it"
        reqs = (out / "requirements.txt").read_text(encoding="utf-8")
        assert "qdrant-client==" in reqs and "openai==" in reqs
        assert "pywin32" not in reqs, "a Windows-only wheel would fail the build"

        cls = load_harness_class(out)
        memo = cls()
        assert memo.config["mem0_llm_model"] == "gpt-5-mini"
        assert memo.config["embedding_model"] == "text-embedding-3-small"
        # A caller that supplies a config still wins (eval_harness does).
        assert cls(config={"mem0_llm_model": "x"}).config == {"mem0_llm_model": "x"}


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
