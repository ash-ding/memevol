"""Tests for the evolvemem baseline (baselines/evolve/evolvemem/).

This baseline is DELIBERATELY INDEPENDENT of the rest of the repo: it vendors
upstream's whole `EvolveMem/` subtree — package and entry points — and runs it on
upstream's own terms to reproduce arXiv:2605.13941. It implements no `MemoClass`
and never calls `common.evaluate`, so there is no 3-hook contract to test here
(see its README for why, and for what that costs).

What IS worth guarding is the property the whole thing rests on: the vendored
tree is upstream's, unmodified, and its entry points still import. Zero LLM
calls, zero network.

    uv run --project baselines/evolve/evolvemem python tests/test_evolvemem_baseline.py
"""
import ast
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE = PROJECT_ROOT / "baselines/evolve/evolvemem"
UPSTREAM = MODULE / "upstream"
COMMIT = "db80b6a7c591e0ea730a058e9f5fc4eb06572299"


def test_vendored_tree_is_present_and_whole():
    """The layout mirrors upstream's `EvolveMem/` exactly, so `diff -r` against
    the pinned commit covers package AND runners in one shot."""
    assert (UPSTREAM / "evolvemem" / "__init__.py").is_file()
    assert (UPSTREAM / "run_benchmark.py").is_file()
    assert (UPSTREAM / "run_evolution.py").is_file()
    assert (UPSTREAM / "requirements.txt").is_file()
    n = len(list((UPSTREAM / "evolvemem").rglob("*.py")))
    assert n == 32, f"expected 32 vendored modules, found {n}"


def test_no_vendored_file_imports_this_repo():
    """Independence is the point: nothing under upstream/ may reach into
    common/, benchmarks/ or baselines/. If that ever changes, the module has
    silently re-coupled and the README's claim is false."""
    offenders = []
    for path in UPSTREAM.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                # level > 0 is a RELATIVE import inside the vendored package —
                # `from .benchmarks.base import ...` is upstream's own subpackage,
                # not this repo's benchmarks/.
                names = [node.module]
            for n in names:
                if n.split(".")[0] in {"common", "benchmarks", "baselines", "forge", "seeds"}:
                    offenders.append(f"{path.relative_to(UPSTREAM)}: {n}")
    assert not offenders, "vendored code imports this repo: " + "; ".join(offenders)


def test_entry_points_parse_and_expose_their_cli():
    """Cheap structural check that the runners are intact — a truncated or
    partially-copied vendor would pass a file-exists test but not this one."""
    for name in ("run_benchmark.py", "run_evolution.py"):
        src = (UPSTREAM / name).read_text(encoding="utf-8")
        tree = ast.parse(src, name)
        funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "main" in funcs, f"{name} has no main()"
        assert "ArgumentParser" in src, f"{name} exposes no CLI"


def test_vendored_package_imports():
    """The engine and its action space import from the vendored copy, and from
    nowhere else."""
    sys.path.insert(0, str(UPSTREAM))
    import evolvemem
    from evolvemem.evolution import EvolutionEngine, weak_initial_config
    from evolvemem.multi_retriever import RetrievalConfig

    assert str(UPSTREAM) in evolvemem.__file__, evolvemem.__file__
    # the paper's theta_0: BM25-only, k_kw=5, B_ctx=8 (§4.1 Implementation)
    weak = weak_initial_config()
    assert weak.fusion_mode == "keyword_only"
    assert weak.semantic_top_k == 0 and weak.keyword_top_k == 5 and weak.max_context == 8
    assert hasattr(EvolutionEngine, "evolve")
    # the action space the loop optimises
    assert len(getattr(RetrievalConfig, "__dataclass_fields__", {})) >= 40


def test_readme_pins_the_upstream_commit():
    """A reproduction claim is only checkable if the README names the exact
    commit the tree came from."""
    readme = (MODULE / "README.md").read_text(encoding="utf-8")
    assert COMMIT in readme, "README does not pin the vendored commit"
    assert "diff -r" in readme, "README carries no verification command"


def test_byte_identity_if_upstream_clone_is_available():
    """Full verification needs a clone; skip cleanly when there isn't one so the
    suite stays offline-runnable. Set EVOLVEMEM_UPSTREAM_CLONE to enable."""
    import os
    clone = os.environ.get("EVOLVEMEM_UPSTREAM_CLONE")
    if not clone or not Path(clone, ".git").is_dir():
        print("      (skipped: set EVOLVEMEM_UPSTREAM_CLONE to a SimpleMem clone)")
        return
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(f"git -C {clone} archive {COMMIT} EvolveMem | tar -x -C {d}",
                       shell=True, check=True)
        out = subprocess.run(
            # -x __pycache__: bytecode is a build artifact of having RUN the
            # vendored code, not a modification of it.
            ["diff", "-r", "-x", "__pycache__", f"{d}/EvolveMem", str(UPSTREAM)],
            capture_output=True, text=True)
        assert out.returncode == 0, "vendored tree differs from upstream:\n" + out.stdout


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
