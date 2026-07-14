"""Tests for forge/heldout.py + evaluate_harness's coverage=full path.

Zero-dependency runner (no pytest in the venvs):

    venv/bin/python tests/test_heldout.py
    baselines/venv/bin/python tests/test_heldout.py

Covers:
  - _stage_harness: copy-into-run isolation, missing-harness.py / duplicate-id
    errors
  - heldout._run: ensure_image + evaluate_harness wiring (fakes), coverage &
    dataset passthrough, heldout_results.json shape
  - evaluate_harness coverage=full: single "full" stage plan (fake
    run_evaluation writing score.json), stages.json reached="full",
    stage metric = FULL_STAGE
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
        "update_type": "all_at_once", "max_sample_concurrent": 3,
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
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"),
                datasets_config={"locomo": {"stages": {}}},
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                update_type="all_at_once", max_sample_concurrent=1,
                memory_cache=False, coverage="full",
            ))
        finally:
            O.run_evaluation = orig

        # ONE call, stage "full", uncapped spec, test split
        assert len(calls) == 1, calls
        assert calls[0]["stage"] == "full"
        assert calls[0]["split"] == "test"
        assert calls[0]["spec"] == {"n_samples": None, "n_qa": None}

        m = per_ds["locomo"]
        assert m["stage"] == O.FULL_STAGE
        assert m["eliminated"] is False
        assert abs(m["raw_score"] - 0.42) < 1e-9

        stages = json.loads(
            (paths.harnesses_dir / hid / "locomo" / "stages.json").read_text())
        assert stages["reached"] == "full"
        assert list(stages["stages"]) == ["full"]
        assert stages["stages"]["full"]["threshold"] is None
        # final artifacts copied to the dataset root
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()


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
                update_type="all_at_once", max_sample_concurrent=1,
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
                update_type="all_at_once", max_sample_concurrent=1,
                memory_cache=False,
            ))
        finally:
            O.run_evaluation = orig
        assert [c["stage"] for c in calls] == ["sanity"]
        assert calls[0]["split"] == "search"
        assert per_ds["locomo"]["stage"] == 0.0
        # artifacts at dataset root (no per-stage subdir)
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()


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
