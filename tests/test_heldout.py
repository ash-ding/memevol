"""Tests for forge/heldout.py + evaluate_harness's coverage=full path.

Zero-dependency runner (no pytest in the venvs):

    venv/bin/python tests/test_heldout.py
    baselines/venv/bin/python tests/test_heldout.py

Covers:
  - _stage_harness: copy-into-run isolation, missing-harness.py / duplicate-id
    errors
  - heldout._run: ensure_image + evaluate_harness wiring (fakes), coverage &
    dataset passthrough, heldout_results.json shape
  - evaluate_harness coverage=full: single "single" stage plan sized by the
    REQUIRED single_stage config block (fake run_evaluation writing
    score.json), stages.json reached="single", stage metric = FULL_STAGE,
    memcache_dir wired when memory_cache=True, ValueError when single_stage
    is absent
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

from forge import heldout as H  # noqa: E402
from forge import orchestrator as O  # noqa: E402
from forge.paths import paths  # noqa: E402

_RUN_ID = "_test_heldout"


@contextlib.contextmanager
def _test_workspace():
    paths.set_run_id(_RUN_ID)
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield paths.workspace
    finally:
        shutil.rmtree(paths.workspace, ignore_errors=True)
        paths._run_id = None


def _mk_src_harness(td, name="3_abcd1234"):
    src = Path(td) / name
    src.mkdir()
    (src / "harness.py").write_text("# harness body\n")
    (src / "meta.json").write_text("{}")
    return src


# ---------------- _stage_harness ----------------

def test_stage_harness_copies_and_isolates():
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        assert hid == "3_abcd1234"
        dst = paths.harnesses_dir / hid
        assert (dst / "harness.py").exists() and (dst / "meta.json").exists()
        # source untouched, copies independent
        (dst / "harness.py").write_text("mutated")
        assert (src / "harness.py").read_text() == "# harness body\n"


def test_stage_harness_rejects_missing_harness_py():
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        empty = Path(td) / "not_a_harness"
        empty.mkdir()
        try:
            H._stage_harness(empty)
        except SystemExit as exc:
            assert "harness.py" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


def test_stage_harness_rejects_duplicate_id():
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        H._stage_harness(src)
        try:
            H._stage_harness(src)
        except SystemExit as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("expected SystemExit")


# ---------------- heldout._run wiring ----------------

def test_run_wires_coverage_and_writes_results():
    captured = {}

    async def fake_ensure_image(harness_dir):
        return Path("/fake/image.sif")

    async def fake_evaluate_harness(hid, image_path, **kw):
        captured["hid"] = hid
        captured["coverage"] = kw.get("coverage")
        captured["split"] = kw.get("split")
        captured["datasets"] = list(kw.get("datasets_config", {}))
        return {"locomo": {"raw_score": 0.5, "score_max": 1, "stage": 4.0,
                           "tokens": 123, "robustness": 0.1, "eliminated": False}}

    cfg = {
        "coverage": "full", "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
        "max_sample_concurrent": 3,
        "memory_cache": True, "gpu": {"enabled": False}, "llm": {},
        "datasets": {"locomo": {}},
    }
    with tempfile.TemporaryDirectory() as td, _test_workspace() as ws:
        src = _mk_src_harness(td)
        orig_ei, orig_ev = H.ensure_image, H.evaluate_harness
        H.ensure_image, H.evaluate_harness = fake_ensure_image, fake_evaluate_harness
        try:
            asyncio.run(H._run(cfg, [str(src)]))
        finally:
            H.ensure_image, H.evaluate_harness = orig_ei, orig_ev

        assert captured["coverage"] == "full"
        assert captured["split"] == "test"
        assert captured["datasets"] == ["locomo"]
        results = json.loads((ws / "heldout_results.json").read_text())
        r = results["3_abcd1234"]
        assert r["objectives"]["accuracy_locomo"] == 0.5
        assert r["objectives"]["stage_locomo"] == 4.0
        assert r["per_ds"]["locomo"]["tokens"] == 123


# ---------------- config-driven harness resolution ----------------

def test_harnesses_from_yaml():
    import argparse, yaml
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "t.yaml"
        cfg_path.write_text(yaml.safe_dump(
            {"harnesses": ["a/b/1_x", "seeds/no_memory"]}))
        args = argparse.Namespace(config=str(cfg_path), harnesses=None)
        got = H._resolve_harnesses(args, H._yaml_raw(args))
        assert got == ["a/b/1_x", "seeds/no_memory"]


def test_cli_harness_replaces_yaml():
    import argparse
    args = argparse.Namespace(config=None, harnesses=["cli/one"])
    got = H._resolve_harnesses(args, {"harnesses": ["yaml/other"]})
    assert got == ["cli/one"]


def test_no_harnesses_anywhere_errors():
    import argparse
    args = argparse.Namespace(config=None, harnesses=None)
    try:
        H._resolve_harnesses(args, {})
    except SystemExit as exc:
        assert "harnesses" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_test_example_yaml_parses():
    """configs/test_example.yaml must survive _resolve_config + carry the
    heldout keys (placeholder harness paths are NOT existence-checked here)."""
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo, "configs", "test_example.yaml")
    raw = yaml.safe_load(open(cfg_path))
    assert isinstance(raw.get("harnesses"), list) and raw["harnesses"]
    assert raw.get("coverage") == "full"
    cfg = _resolve_config(build_arg_parser().parse_args(["--config", cfg_path]))
    assert cfg["coverage"] == "full"
    assert set(cfg["datasets"]) == {"dynamicmem", "locomo", "longmemeval_s"}


def test_search_example_yaml_parses():
    from forge.orchestrator import build_arg_parser, _resolve_config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo, "configs", "search_example.yaml")
    cfg = _resolve_config(build_arg_parser().parse_args(["--config", cfg_path]))
    assert cfg["coverage"] == "sample"


# ---------------- progressive/coverage consistency (fix-round finding 1) ----------------

def _consistent(cfg):
    """progressive=True must always mean coverage=='sample' and vice versa —
    the persisted config.yaml snapshot must never show a contradictory pair."""
    return cfg["progressive"] == (cfg["coverage"] == "sample")


def test_heldout_omitted_coverage_stays_consistent_with_progressive():
    """Regression for fix-round finding 1: a heldout YAML that omits
    `coverage:` entirely used to leave cfg["progressive"] at _resolve_config's
    default (True) while _apply_heldout_coverage_default forced
    cfg["coverage"] to "full" — a self-contradictory pair persisted into
    config.yaml. Must now resync to progressive=False (matching the "full"
    that heldout defaults to when coverage is omitted everywhere)."""
    import argparse, yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "no_coverage.yaml"
        # No `coverage:` key AND no `progressive:` key — purely defaults.
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"],
            "datasets": {"locomo": {}},
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        # Before the heldout-specific override: _resolve_config's own default
        # (progressive=True <-> coverage="sample") must already be consistent.
        assert _consistent(cfg)
        assert cfg["progressive"] is True and cfg["coverage"] == "sample"

        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        # heldout's "default to full when omitted" kicks in...
        assert cfg["coverage"] == "full"
        # ...and progressive MUST be resynced to match — no contradictory pair.
        assert _consistent(cfg)
        assert cfg["progressive"] is False


def test_heldout_explicit_coverage_cli_stays_consistent():
    """--coverage sample on the CLI is honored as-is (not forced to full),
    and progressive/coverage stay consistent."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
        }))
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--coverage", "sample"])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["coverage"] == "sample"
        assert _consistent(cfg)
        assert cfg["progressive"] is True


def test_heldout_yaml_coverage_full_stays_consistent():
    """An explicit `coverage: full` in the heldout YAML (no CLI flag) is
    honored, and progressive resyncs to False."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "coverage": "full",
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["coverage"] == "full"
        assert _consistent(cfg)
        assert cfg["progressive"] is False


def test_heldout_yaml_progressive_true_no_coverage_not_forced_full():
    """Fix-round finding: a heldout YAML that sets `progressive: true` and
    OMITS `coverage:` used to be silently forced to coverage="full" (and
    progressive resynced to False) by the "default to full" override, which
    only checked for an explicit `coverage:` key. `progressive` is the
    canonical knob, so an explicit `progressive: true` (with no `coverage:`
    anywhere) must resolve to progressive=True / coverage="sample" — NOT
    full — same as it would for the search-loop orchestrator."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "progressive": True,
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["progressive"] is True
        assert cfg["coverage"] == "sample"
        assert _consistent(cfg)


# ---------------- guardrail: reject progressive/coverage=sample on heldout --

def test_reject_progressive_on_heldout_raises():
    """A heldout config resolving to progressive=True (the staged gauntlet)
    must be rejected before any harness runs — held-out numbers require a
    full, uniform pass, not a stage-eliminated subset score."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "progressive": True,
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["progressive"] is True  # sanity: this IS the gauntlet case
        try:
            H._reject_progressive_on_heldout(cfg)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2)")


def test_reject_progressive_on_heldout_via_coverage_sample_alias():
    """`coverage: sample` (the back-compat alias) must also be rejected —
    after `_sync_coverage_progressive` this implies progressive=True too,
    but the guard checks `coverage` directly regardless, defensively."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "coverage": "sample",
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["coverage"] == "sample"
        try:
            H._reject_progressive_on_heldout(cfg)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2)")


def test_reject_progressive_on_heldout_default_passes():
    """The normal/default heldout case (neither `coverage` nor `progressive`
    given anywhere → coverage='full' / progressive=False) must NOT raise —
    the guardrail only fires when the user explicitly opted into the
    gauntlet."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        assert cfg["coverage"] == "full" and cfg["progressive"] is False
        H._reject_progressive_on_heldout(cfg)  # must not raise


def test_reject_progressive_on_heldout_explicit_full_passes():
    """An explicit `coverage: full` (no `progressive` key) also must not
    raise — matches configs/test_example.yaml's documented default."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "coverage": "full",
        }))
        args = H._heldout_arg_parser().parse_args(["--config", str(cfg_path)])
        cfg = _resolve_config(args)
        raw_yaml = H._yaml_raw(args)
        H._apply_heldout_coverage_default(cfg, args, raw_yaml)
        H._reject_progressive_on_heldout(cfg)  # must not raise


# ---------------- evaluate_harness coverage=full integration ----------------

def _fake_run_evaluation_factory(calls):
    async def fake_run_evaluation(*, harness_dir, image_path, out_dir, dataset,
                                  split, stage, stage_spec, **kw):
        calls.append({"stage": stage, "spec": stage_spec, "split": split})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "score.json").write_text(json.dumps({
            "benchmark_eval_score": {
                "benchmark_overall_eval_score": 0.42,
                "benchmark_overall_eval_standard_deviation": 0.0,
                "score_max": 1,
            },
            "per_user": {"u1": {"reward": 0.42, "n_qa": 3, "failure_info": None}},
            "invalid_users": [],
        }))
        (out_dir / "token_usage.json").write_text(json.dumps(
            {"gpt-5-mini": {"total_tokens": 777}}))
        return out_dir
    return fake_run_evaluation


def test_evaluate_harness_full_single_pass():
    """coverage='full' (progressive=false) now resolves its plan via
    resolve_sampling_plan: ONE pass named "single", sized by the REQUIRED
    `single_stage` block (2026-07-26) — replacing the old automatic
    full_wire_spec whole-split. Reached-stage telemetry still maps to
    O.FULL_STAGE (the "single" plan name is the new full-ish tier)."""
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config={"locomo": {"single_stage": {"n_conversations": None, "n_qa": None}}},
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=False, coverage="full",
            ))
        finally:
            O.run_evaluation = orig

        # ONE call, stage "single", uncapped spec (from single_stage), test split
        assert len(calls) == 1, calls
        assert calls[0]["stage"] == "single"
        assert calls[0]["split"] == "test"
        assert calls[0]["spec"] == {"n_samples": None, "n_qa": None}

        m = per_ds["locomo"]
        assert m["stage"] == O.FULL_STAGE
        assert m["eliminated"] is False
        assert abs(m["raw_score"] - 0.42) < 1e-9

        stages = json.loads(
            (paths.harnesses_dir / hid / "locomo" / "stages.json").read_text())
        assert stages["reached"] == "single"
        assert list(stages["stages"]) == ["single"]
        assert stages["stages"]["single"]["threshold"] is None
        # final artifacts copied to the dataset root
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()


def test_evaluate_harness_full_missing_single_stage_raises():
    """coverage='full' with no `single_stage` block configured must raise —
    no silent automatic whole-split anymore (single_stage is required)."""
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        raised = False
        try:
            asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config={"locomo": {"stages": {}}},
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=False, coverage="full",
            ))
        except ValueError as e:
            raised = "single_stage" in str(e)
        assert raised


def test_evaluate_harness_full_wires_memcache_dir():
    """CRITICAL regression (fix round 1, 2026-07-26 review): the HOST-side
    memcache gate in evaluate_harness's `_run_stage_fn` closure must
    recognize the "single" plan name resolve_sampling_plan emits for
    progressive=false — otherwise `memory_cache=True` silently becomes a
    no-op for every `forge.heldout` run (configs/test_example.yaml sets
    `memory_cache: true`) and every progressive=false search config, since
    `memcache_dir` is never passed to `run_evaluation` and `--memcache-dir`
    is never forwarded to the container."""
    calls = []

    async def fake_run_evaluation(*, harness_dir, image_path, out_dir, dataset,
                                  split, stage, stage_spec, memcache_dir=None, **kw):
        calls.append({"stage": stage, "memcache_dir": memcache_dir})
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "score.json").write_text(json.dumps({
            "benchmark_eval_score": {
                "benchmark_overall_eval_score": 0.5,
                "benchmark_overall_eval_standard_deviation": 0.0,
                "score_max": 1,
            },
            "per_user": {"u1": {"reward": 0.5, "n_qa": 1, "failure_info": None}},
            "invalid_users": [],
        }))
        (out_dir / "token_usage.json").write_text(json.dumps({}))
        return out_dir

    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        orig = O.run_evaluation
        O.run_evaluation = fake_run_evaluation
        try:
            asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config={"locomo": {"single_stage": {"n_conversations": None, "n_qa": None}}},
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=True, coverage="full",
            ))
        finally:
            O.run_evaluation = orig

        assert len(calls) == 1, calls
        assert calls[0]["stage"] == "single"
        assert calls[0]["memcache_dir"] is not None, (
            "memory_cache=True must wire --memcache-dir for the 'single' "
            "(progressive=false) stage too — the host-side gate must "
            "include 'single', not just stage1/2/3/full"
        )
        expected_dir = paths.harnesses_dir / hid / "locomo" / "memory_cache"
        assert calls[0]["memcache_dir"] == expected_dir
        assert expected_dir.is_dir()


def test_evaluate_harness_sample_unchanged():
    """Regression lock: coverage omitted → the 3-stage gauntlet as before."""
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])  # fill defaults
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config=ds_cfg,
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=False,
            ))
        finally:
            O.run_evaluation = orig
        # 0.42 clears the default locomo thresholds (0.30/0.35) → all 3 stages
        assert [c["stage"] for c in calls] == ["stage1", "stage2", "stage3"]
        assert per_ds["locomo"]["stage"] == 3.0


def test_evaluate_harness_smoke_single_sanity_pass():
    """smoke=True → one sanity-sized run, artifacts at the dataset root,
    stage recorded as 0.0 (the old mode=dev semantics)."""
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config=ds_cfg,
                split="search", smoke=True,
                model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=False,
            ))
        finally:
            O.run_evaluation = orig
        assert [c["stage"] for c in calls] == ["sanity"]
        assert calls[0]["split"] == "search"
        assert per_ds["locomo"]["stage"] == 0.0
        # artifacts at dataset root (no per-stage subdir)
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()


def test_evaluate_harness_stage3_null_full_final():
    """coverage=sample gauntlet where stage3 uses null sizes → stage1/2 gate
    on concrete sizes, stage3 runs a full-coverage wire spec, reaches 3.0."""
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {"stages": {
            "stage1": {"n_conversations": 2, "n_qa": 20, "threshold": 0.3},
            "stage2": {"n_conversations": 4, "n_qa": 40, "threshold": 0.35},
            "stage3": {"n_conversations": None, "n_qa": None},
        }}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config=ds_cfg,
                split="search", smoke=False,
                model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1,
                memory_cache=False, coverage="sample",
            ))
        finally:
            O.run_evaluation = orig
        # 0.42 clears the 0.3/0.35 thresholds → all three stages run
        assert [c["stage"] for c in calls] == ["stage1", "stage2", "stage3"]
        # stage3's wire spec carries None (full coverage); earlier stages don't
        assert calls[2]["spec"] == {"n_samples": None, "n_qa": None}
        assert calls[0]["spec"] == {"n_samples": 2, "n_qa": 20}
        assert per_ds["locomo"]["stage"] == 3.0  # reached stage3 (not 4.0 = coverage=full)


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
