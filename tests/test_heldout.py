"""Tests for forge/heldout.py + evaluate_harness's progressive=false path.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_heldout.py

Covers:
  - _stage_harness: copy-into-run isolation, missing-harness.py / duplicate-id
    errors
  - heldout._run: ensure_image + evaluate_harness wiring (fakes), progressive & dataset
    dataset passthrough, heldout_results.json shape
  - evaluate_harness progressive=false: single "single" stage plan sized by the
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

def test_run_wires_progressive_and_writes_results():
    captured = {}

    async def fake_ensure_image(harness_dir):
        return Path("/fake/image.sif")

    async def fake_evaluate_harness(hid, image_path, **kw):
        captured["hid"] = hid
        captured["progressive"] = kw.get("progressive")
        captured["split"] = kw.get("split")
        captured["datasets"] = list(kw.get("datasets_config", {}))
        return {"locomo": {"raw_score": 0.5, "score_max": 1, "stage": 4.0,
                           "tokens": 123, "robustness": 0.1, "eliminated": False}}

    cfg = {
        "progressive": False, "model": "gpt-5-mini", "judge_model": "gpt-5-mini",
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

        assert captured["progressive"] is False
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
    heldout keys (placeholder harness paths are NOT existence-checked here).

    --no-strict-config: this file intentionally omits every search-loop-only
    field (steps, propose.*, proposer.*, sanity.*, seed.*, prompts.*, per its
    own header comment) that FORGE_REQUIRED_SCHEMA otherwise requires — this
    test is about progressive/dataset resolution, not strict-config
    completeness of the example file (a follow-up task can decide whether to
    make configs/test_example.yaml itself schema-complete)."""
    import yaml
    from forge.orchestrator import build_arg_parser, _resolve_config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo, "configs", "test_example.yaml")
    raw = yaml.safe_load(open(cfg_path))
    assert isinstance(raw.get("harnesses"), list) and raw["harnesses"]
    assert raw.get("progressive") is False
    cfg = _resolve_config(build_arg_parser().parse_args(
        ["--config", cfg_path, "--no-strict-config"]))
    assert cfg["progressive"] is False
    assert set(cfg["datasets"]) == {"dynamicmem", "locomo", "longmemeval_s"}


def test_search_example_yaml_parses():
    from forge.orchestrator import build_arg_parser, _resolve_config
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo, "configs", "search_example.yaml")
    cfg = _resolve_config(build_arg_parser().parse_args(["--config", cfg_path]))
    assert cfg["progressive"] is True


# ---------------- heldout progressive default ----------------

def test_heldout_omitted_progressive_defaults_to_false():
    """A heldout config that omits `progressive:` gets progressive=False (a
    single-stage pass) — heldout's own default (`_apply_heldout_progressive_default`),
    overriding DEFAULT_CONFIG's search-loop `progressive=True`."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "no_progressive.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
        }))
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--no-strict-config"])
        cfg = _resolve_config(args)
        assert cfg["progressive"] is True   # _resolve_config's search-loop default
        H._apply_heldout_progressive_default(cfg, args, H._yaml_raw(args))
        assert cfg["progressive"] is False  # heldout forces the single-stage pass


def test_heldout_yaml_progressive_true_respected():
    """An explicit `progressive: true` in the heldout YAML is respected (NOT
    silently flipped to False by the default) — it is then rejected downstream by
    `_reject_progressive_on_heldout`, but the default must not swallow it."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "progressive": True,
        }))
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--no-strict-config"])
        cfg = _resolve_config(args)
        H._apply_heldout_progressive_default(cfg, args, H._yaml_raw(args))
        assert cfg["progressive"] is True


# ---------------- guardrail: reject progressive=true on heldout ----------------

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
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--no-strict-config"])
        cfg = _resolve_config(args)
        H._apply_heldout_progressive_default(cfg, args, H._yaml_raw(args))
        assert cfg["progressive"] is True  # sanity: this IS the gauntlet case
        try:
            H._reject_progressive_on_heldout(cfg)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected SystemExit(2)")


def test_reject_progressive_on_heldout_default_passes():
    """The normal/default heldout case (no `progressive:` given anywhere →
    progressive=False) must NOT raise — the guardrail only fires when the user
    explicitly opted into the gauntlet."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
        }))
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--no-strict-config"])
        cfg = _resolve_config(args)
        H._apply_heldout_progressive_default(cfg, args, H._yaml_raw(args))
        assert cfg["progressive"] is False
        H._reject_progressive_on_heldout(cfg)  # must not raise


def test_reject_progressive_on_heldout_explicit_false_passes():
    """An explicit `progressive: false` also must not raise — matches
    configs/test_example.yaml's documented default."""
    import yaml
    from forge.orchestrator import _resolve_config
    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "c.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "harnesses": ["a/b/1_x"], "datasets": {"locomo": {}},
            "progressive": False,
        }))
        args = H._heldout_arg_parser().parse_args(
            ["--config", str(cfg_path), "--no-strict-config"])
        cfg = _resolve_config(args)
        H._apply_heldout_progressive_default(cfg, args, H._yaml_raw(args))
        H._reject_progressive_on_heldout(cfg)  # must not raise


# ---------------- evaluate_harness single-container integration ----------------
# ONE container per (harness, dataset) runs the whole plan in-container
# (evaluate_memo); the host launches it, publishes /out, and reads back
# metrics.json. These tests fake run_evaluation to write what a container
# would, and assert the host-side plan wiring + read-back. The in-container
# promotion/sizing behavior itself is covered by tests/test_run_gauntlet.py.

def _fake_run_evaluation_factory(calls, *, stage="single", stage_num=4.0,
                                 raw=0.42, write_metrics=True):
    async def fake_run_evaluation(*, harness_dir, image_path, out_dir, dataset,
                                  split, plan, memcache_dir=None, **kw):
        calls.append({"plan": plan, "split": split, "memcache_dir": memcache_dir})
        out_dir.mkdir(parents=True, exist_ok=True)
        smoke = bool(plan.get("smoke"))
        (out_dir / "score.json").write_text(json.dumps({
            "benchmark_eval_score": {
                "benchmark_overall_eval_score": raw,
                "benchmark_overall_eval_standard_deviation": 0.0,
            },
            "per_user": {"u1": {"reward": raw, "n_qa": 3, "failure_info": None}},
            "invalid_users": [],
        }))
        (out_dir / "token_usage.json").write_text(json.dumps(
            {"gpt-5-mini": {"total_tokens": 777}}))
        if not smoke:
            (out_dir / "stages.json").write_text(json.dumps({
                "reached": stage, "eliminated": False,
                "stages": {stage: {"raw_score": raw, "score_max": 1,
                                   "normalized": raw, "threshold": None,
                                   "tokens": 777, "spec": {}}},
            }))
        if write_metrics:
            (out_dir / "metrics.json").write_text(json.dumps({
                "raw_score": raw, "score_max": 1, "per_user_stddev": None,
                "tokens": 777,
                "stage": 0.0 if smoke else stage_num,
                "eliminated": False,
            }))
        return out_dir
    return fake_run_evaluation


def test_evaluate_harness_full_single_pass():
    """progressive=false → ONE container launch whose plan carries the REQUIRED
    single_stage block (smoke off); metrics come back from metrics.json with
    stage == FULL_STAGE; artifacts published at the dataset root."""
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
                memory_cache=False, progressive=False,
            ))
        finally:
            O.run_evaluation = orig

        assert len(calls) == 1, calls   # ONE container for the whole plan
        assert calls[0]["split"] == "test"
        plan = calls[0]["plan"]
        assert plan["progressive"] is False and plan["smoke"] is False
        assert plan["single_stage"] == {"n_conversations": None, "n_qa": None}

        m = per_ds["locomo"]
        assert m["stage"] == O.FULL_STAGE
        assert m["eliminated"] is False
        assert abs(m["raw_score"] - 0.42) < 1e-9
        # container artifacts published at the dataset root
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()
        assert (paths.harnesses_dir / hid / "locomo" / "stages.json").exists()


def test_evaluate_harness_full_missing_single_stage_raises():
    """progressive=false with no `single_stage` block must fail fast on the
    HOST (pre-flight resolve_sampling_plan) — before any container launches."""
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
                memory_cache=False, progressive=False,
            ))
        except ValueError as e:
            raised = "single_stage" in str(e)
        assert raised


def test_evaluate_harness_wires_memcache_dir_eval_not_smoke():
    """memory_cache=True must pass the persistent memcache dir to
    run_evaluation for evals (gauntlet AND the progressive=false single pass)
    and NEVER for smoke runs (harness code can change during sanity retries)."""
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {"single_stage": {"n_conversations": None, "n_qa": None}}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])

        calls = []
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(calls)
        try:
            asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"), datasets_config=ds_cfg,
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1, memory_cache=True, progressive=False,
            ))
        finally:
            O.run_evaluation = orig
        assert calls[0]["memcache_dir"] is not None
        expected_dir = paths.harnesses_dir / hid / "locomo" / "memory_cache"
        assert calls[0]["memcache_dir"] == expected_dir and expected_dir.is_dir()

        # smoke: never mounted
        calls2 = []
        O.run_evaluation = _fake_run_evaluation_factory(calls2)
        try:
            asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"), datasets_config=ds_cfg,
                split="search", smoke=True, model="gpt-5-mini",
                judge_model="gpt-5-mini", max_sample_concurrent=1,
                memory_cache=True, progressive=False,
            ))
        finally:
            O.run_evaluation = orig
        assert calls2[0]["memcache_dir"] is None


def test_evaluate_harness_gauntlet_plan():
    """Default progressive=true → ONE container whose plan carries the resolved
    stages block; metrics.json's stage (3.0) flows back to the frontier."""
    calls = []
    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])  # fill defaults
        orig = O.run_evaluation
        O.run_evaluation = _fake_run_evaluation_factory(
            calls, stage="stage3", stage_num=3.0)
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
        assert len(calls) == 1, calls
        plan = calls[0]["plan"]
        assert plan["progressive"] is True and plan["smoke"] is False
        assert plan["stages"]["stage1"]["n_conversations"] == 2  # resolved defaults ride along
        assert per_ds["locomo"]["stage"] == 3.0


def test_evaluate_harness_smoke_single_sanity_pass():
    """smoke=True → plan.smoke on the search split, artifacts at the dataset
    root, stage recorded as 0.0."""
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
        assert len(calls) == 1
        assert calls[0]["plan"]["smoke"] is True
        assert calls[0]["split"] == "search"
        assert per_ds["locomo"]["stage"] == 0.0
        assert (paths.harnesses_dir / hid / "locomo" / "score.json").exists()


def test_evaluate_harness_missing_metrics_degrades():
    """A container that produced no metrics.json (crashed before any stage)
    degrades to a 0-score eliminated entry + a failure-marker score.json —
    never indistinguishable from a genuine 0-score run."""
    calls = []

    async def fake_run_evaluation(*, out_dir, **kw):
        calls.append(kw.get("plan"))
        out_dir.mkdir(parents=True, exist_ok=True)  # container wrote NOTHING
        return out_dir

    with tempfile.TemporaryDirectory() as td, _test_workspace():
        src = _mk_src_harness(td)
        hid = H._stage_harness(src)
        ds_cfg = {"locomo": {}}
        O._resolve_dataset_stages("locomo", ds_cfg["locomo"])
        orig = O.run_evaluation
        O.run_evaluation = fake_run_evaluation
        try:
            per_ds = asyncio.run(O.evaluate_harness(
                hid, Path("/fake.sif"), datasets_config=ds_cfg,
                split="test", model="gpt-5-mini", judge_model="gpt-5-mini",
                max_sample_concurrent=1, memory_cache=False,
            ))
        finally:
            O.run_evaluation = orig
        m = per_ds["locomo"]
        assert m["raw_score"] == 0.0 and m["eliminated"] is True
        # host wrote the failure-marker score.json at the dataset root
        score = json.loads(
            (paths.harnesses_dir / hid / "locomo" / "score.json").read_text())
        assert score["invalid_users"], score


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
