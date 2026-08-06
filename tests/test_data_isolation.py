"""Tests for search-mode data isolation (forge/data_isolation.py).

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_data_isolation.py

Covers:
  - stage_search_data: deterministic filtered artifacts (locomo first-6,
    longmemeval search-300 + manifest, oracle stub) and bind targets
  - LongMemEval _compute_split manifest hook (staged env) + no-manifest
    regression (host env)
  - orchestrator._isolation_binds gating (search on / test off / disabled)
  - evaluator run_evaluation argv carries the overlay binds
"""
import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from forge import data_isolation as DI  # noqa: E402
from forge.data_isolation import stage_search_data  # noqa: E402


@contextlib.contextmanager
def _synthetic_lme():
    """Small fake LongMemEval data — the real m variant is ~2.6 GB, so unit
    tests must not stage it. Monkeypatches DATA_PATHS + the split cache."""
    import datasets.longmemeval.env as lme
    with tempfile.TemporaryDirectory() as td:
        samples = [{"question_id": q, "haystack_sessions": []}
                   for q in ("q1", "q2", "q3")]
        paths = {}
        for variant, name in (("s", "longmemeval_s_cleaned.json"),
                              ("m", "longmemeval_m_cleaned.json")):
            fp = Path(td) / name
            fp.write_text(json.dumps(samples))
            paths[variant] = fp
        paths["oracle"] = Path(td) / "longmemeval_oracle.json"
        orig_paths, orig_cache = lme.DATA_PATHS, lme._SPLIT_CACHE
        lme.DATA_PATHS = {**lme.DATA_PATHS, **paths}
        lme._SPLIT_CACHE = (["q1", "q2"], ["q3"])   # search / test
        try:
            yield
        finally:
            lme.DATA_PATHS, lme._SPLIT_CACHE = orig_paths, orig_cache


def _stage(td):
    with _synthetic_lme():
        staging = Path(td) / "staging"
        return stage_search_data(staging), staging


# ---------------- staging artifacts ----------------

def test_locomo_filtered_is_search_prefix():
    with tempfile.TemporaryDirectory() as td:
        DI._stage_locomo(Path(td))
        filtered = json.loads((Path(td) / "locomo10.json").read_text())
        full = json.loads(
            Path(REPO, "datasets", "locomo", "locomo10.json").read_text())
        assert len(filtered) == 6
        assert [s["sample_id"] for s in filtered] == \
               [s["sample_id"] for s in full[:6]]


def test_longmemeval_filtered_matches_env_split():
    with tempfile.TemporaryDirectory() as td:
        with _synthetic_lme():
            DI._stage_longmemeval(Path(td))
        staging = Path(td)
        for name in ("longmemeval_s_cleaned.json", "longmemeval_m_cleaned.json"):
            filtered = json.loads((staging / name).read_text())
            assert sorted(s["question_id"] for s in filtered) == ["q1", "q2"]
        manifest = json.loads((staging / "split_manifest.json").read_text())
        assert manifest["search"] == ["q1", "q2"]
        assert manifest["test"] == []
        assert json.loads((staging / "longmemeval_oracle.json").read_text()) == []


def test_bind_targets_shadow_container_paths():
    with tempfile.TemporaryDirectory() as td:
        binds, _ = _stage(td)
        dsts = [b.split(":")[1] for b in binds]
        for expected in (
            "/app/datasets/locomo/locomo10.json",
            "/app/datasets/longmemeval/longmemeval_s_cleaned.json",
            "/app/datasets/longmemeval/longmemeval_m_cleaned.json",
            "/staging/split_manifest.json",
            "/app/datasets/longmemeval/longmemeval_oracle.json",
        ):
            assert expected in dsts, expected
        assert all(b.endswith(":ro") for b in binds)
        # dynamicmem overlay only when local user_data exists
        if os.path.isdir(os.path.join(REPO, "datasets", "dynamicmem", "user_data")):
            assert "/app/datasets/dynamicmem/user_data" in dsts
            backbound = [d for d in dsts
                         if d.startswith("/app/datasets/dynamicmem/user_data/")]
            assert len(backbound) == 6
            assert all("00" + str(i) in d or f"00{i}" in d
                       for i, d in enumerate(sorted(backbound), start=1))


def test_staging_cache_skips_regeneration():
    with tempfile.TemporaryDirectory() as td:
        b1, staging = _stage(td)
        out = staging / "locomo10.json"
        first_mtime = out.stat().st_mtime_ns
        assert (staging / "locomo10.json.src.json").exists()  # fingerprint sidecar
        b2, _ = _stage(td)
        assert b1 == b2
        assert out.stat().st_mtime_ns == first_mtime  # cache hit — not rewritten


# ---------------- LongMemEval manifest hook ----------------

def test_manifest_hook_used_when_present():
    import datasets.longmemeval.env as lme
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "split_manifest.json").write_text(
            json.dumps({"search": ["q2", "q1"], "test": []}))
        orig_dir, orig_cache = lme._DATA_DIR, lme._SPLIT_CACHE
        lme._DATA_DIR, lme._SPLIT_CACHE = Path(td), None
        try:
            s, t = lme._compute_split()
        finally:
            lme._DATA_DIR, lme._SPLIT_CACHE = orig_dir, orig_cache
        assert s == ["q1", "q2"] and t == []


def test_no_manifest_recomputes_normally():
    import datasets.longmemeval.env as lme
    orig_cache = lme._SPLIT_CACHE
    lme._SPLIT_CACHE = None
    try:
        s, t = lme._compute_split()
    finally:
        lme._SPLIT_CACHE = orig_cache
    assert len(s) == 300 and len(t) == 200


# ---------------- orchestrator gating ----------------

@contextlib.contextmanager
def _test_workspace():
    from forge.paths import paths
    import forge.orchestrator as O
    paths.set_run_id("_test_isolation")
    O._ISOLATION_BINDS_CACHE = None
    try:
        yield paths.workspace
    finally:
        shutil.rmtree(paths.workspace, ignore_errors=True)
        O._ISOLATION_BINDS_CACHE = None
        paths._run_id = None


def test_isolation_binds_gating():
    import forge.orchestrator as O
    calls = []
    orig = DI.stage_search_data
    DI.stage_search_data = lambda *a, **k: calls.append(1) or ["fake:/x:ro"]
    try:
        with _test_workspace():
            assert O._isolation_binds({"data_isolation": False}) is None
            binds = O._isolation_binds({"data_isolation": True})
            assert binds == ["fake:/x:ro"]
            # process-level cache — second call doesn't restage
            assert O._isolation_binds({"data_isolation": True}) is binds
            assert len(calls) == 1
    finally:
        DI.stage_search_data = orig


# ---------------- evaluator argv ----------------

def test_run_evaluation_carries_isolation_binds():
    from forge import evaluator as E
    captured = {}

    async def fake_exec(*cmd, **kw):
        captured["cmd"] = list(cmd)
        class P:
            returncode = 0
            async def communicate(self): return b"", b""
            def kill(self): pass
            async def wait(self): return 0
        return P()

    async def main(binds):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            orig = E.asyncio.create_subprocess_exec
            E.asyncio.create_subprocess_exec = fake_exec
            try:
                await E.run_evaluation(
                    harness_dir=Path(td), image_path=Path("/x.sif"), out_dir=out,
                    dataset="locomo", split="search", stage="stage1",
                    stage_spec={"n_samples": 1, "n_qa": 3},
                    data_isolation_binds=binds,
                )
            finally:
                E.asyncio.create_subprocess_exec = orig
        return " ".join(captured["cmd"])

    cmd = asyncio.run(main(["/tmp/x.json:/app/datasets/locomo/locomo10.json:ro"]))
    assert "/tmp/x.json:/app/datasets/locomo/locomo10.json:ro" in cmd
    cmd = asyncio.run(main(None))
    assert "/tmp/x.json" not in cmd


# ---------------- runner ----------------

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
