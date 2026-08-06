"""Tests for forge host-side pipeline mechanics:

  - env_builder: atomic delta-image builds (no partial SIF can ever occupy
    the content-addressed cache path — audit finding M3)
  - orchestrator._publish_run_artifacts: stale-artifact cleanup so a crashed
    attempt can never inherit the previous attempt's score.json (audit M4)

Zero-dependency runner:

    uv run python tests/test_forge_pipeline.py
"""
import asyncio
import json
import os
import stat
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

import forge.env_builder as eb
from forge.orchestrator import _publish_run_artifacts


# ---------------- helpers ----------------

def _write_shim(bin_dir: Path, body: str) -> None:
    """Install a fake `singularity` executable on the builder's PATH."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "singularity"
    shim.write_text("#!/bin/bash\n" + body + "\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)


class _Patched:
    """Temporarily patch env_builder module globals."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.saved = {}

    def __enter__(self):
        for k, v in self.kwargs.items():
            self.saved[k] = getattr(eb, k)
            setattr(eb, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(eb, k, v)


def _setup(tmp: Path, shim_body: str, timeout_s: int = 60):
    """Common scaffolding: harness dir with requirements, fake base sif,
    image cache dir, shim on PATH."""
    harness = tmp / "harness"
    harness.mkdir()
    (harness / "requirements.txt").write_text("rank_bm25\n")
    base = tmp / "eval-base.sif"
    base.write_text("BASE")
    images = tmp / "images"
    images.mkdir()
    shim_dir = tmp / "bin"
    _write_shim(shim_dir, shim_body)
    return _Patched(
        EVAL_BASE_SIF=base,
        FORGE_IMAGES_DIR=images,
        PROOT_BIN_DIR=shim_dir,       # prepended to PATH by ensure_image
        _BUILD_TIMEOUT_S=timeout_s,
    ), harness, images


def _no_residue(images: Path):
    leftovers = [p.name for p in images.iterdir() if ".building." in p.name]
    assert not leftovers, f"partial build artifacts left behind: {leftovers}"


# ---------------- env_builder: atomic build ----------------

def test_build_success_lands_at_cache_path():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # argv: singularity build --force <dest> <def>  → dest is $3
        patch, harness, images = _setup(tmp, 'echo IMG > "$3"; exit 0')
        with patch:
            out = asyncio.run(eb.ensure_image(harness))
        assert out.parent == images and out.suffix == ".sif"
        assert out.read_text().strip() == "IMG"
        _no_residue(images)


def test_build_failure_leaves_no_cache_entry():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Simulate singularity dying after partially writing the dest file.
        patch, harness, images = _setup(tmp, 'echo PARTIAL > "$3"; exit 1')
        with patch:
            try:
                asyncio.run(eb.ensure_image(harness))
                assert False, "expected EnvBuildError"
            except eb.EnvBuildError:
                pass
        sifs = [p.name for p in images.iterdir() if p.suffix == ".sif"]
        assert not sifs, f"failed build polluted the cache: {sifs}"
        _no_residue(images)


def test_build_timeout_leaves_no_cache_entry():
    """The audited bug: a timed-out build left a truncated SIF at the FINAL
    cache path, permanently poisoning every later cache hit for that sha."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        patch, harness, images = _setup(
            tmp, 'echo PARTIAL > "$3"; sleep 30', timeout_s=1
        )
        with patch:
            try:
                asyncio.run(eb.ensure_image(harness))
                assert False, "expected EnvBuildError (timeout)"
            except eb.EnvBuildError as exc:
                assert "timeout" in str(exc).lower()
        sifs = [p.name for p in images.iterdir() if p.suffix == ".sif"]
        assert not sifs, f"timed-out build polluted the cache: {sifs}"
        _no_residue(images)


def test_cache_hit_skips_build():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        marker = tmp / "shim_invoked"
        patch, harness, images = _setup(tmp, f'touch "{marker}"; echo IMG > "$3"')
        normalized = eb._normalize_requirements("rank_bm25\n")
        sha = eb._hash_requirements(normalized)
        (images / f"{sha}.sif").write_text("CACHED")
        with patch:
            out = asyncio.run(eb.ensure_image(harness))
        assert out.read_text() == "CACHED"
        assert not marker.exists(), "cache hit must not invoke singularity"


# ---------------- orchestrator: publish cleans stale artifacts ----------------

def test_publish_removes_stale_artifacts_on_crash():
    """Attempt N leaves score.json+traces in dst; attempt N+1's container
    dies before writing anything. After publish, dst must NOT contain the
    old attempt's results."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        dst = tmp / "dst"
        (dst / "traces").mkdir(parents=True)
        (dst / "score.json").write_text('{"old": true}')
        (dst / "token_usage.json").write_text("{}")
        (dst / "traces" / "user_old.json").write_text("{}")

        run_dir = tmp / "runs" / "attempt2"
        run_dir.mkdir(parents=True)
        (run_dir / "subprocess.log").write_text("crashed early")

        _publish_run_artifacts(run_dir, dst)

        assert not (dst / "score.json").exists(), "stale score.json survived"
        assert not (dst / "token_usage.json").exists()
        assert not (dst / "traces" / "user_old.json").exists()
        assert (dst / "subprocess.log").read_text() == "crashed early"
        assert not run_dir.exists(), "staging dir must be dropped"


def test_publish_replaces_with_fresh_results():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        dst = tmp / "dst"
        (dst / "traces").mkdir(parents=True)
        (dst / "score.json").write_text('{"old": true}')
        (dst / "traces" / "user_old.json").write_text("{}")

        run_dir = tmp / "runs" / "attempt2"
        (run_dir / "traces").mkdir(parents=True)
        (run_dir / "score.json").write_text('{"fresh": true}')
        (run_dir / "traces" / "user_new.json").write_text('{"n_qa": 1}')

        _publish_run_artifacts(run_dir, dst)

        assert json.loads((dst / "score.json").read_text()) == {"fresh": True}
        names = sorted(p.name for p in (dst / "traces").iterdir())
        assert names == ["user_new.json"], names


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
