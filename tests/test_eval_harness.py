"""The shared harness entrypoint (baselines/harness/eval_harness.py): the lazy
registry, frame-config resolution, the per-run directory and its records.
Zero-dependency runner (no pytest); needs only the ROOT project env — the
evaluation itself is stubbed at run_baseline, so no baseline deps and no
network are involved:

    uv run python tests/test_eval_harness.py
"""
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.harness import eval_harness as eh  # noqa: E402
from common.memo_class import MemoClass  # noqa: E402

EXAMPLE = PROJECT_ROOT / "baselines" / "harness" / "config.example.yaml"


class _FakeMemo(MemoClass):
    CONFIG_DEFAULTS = {"top_k": 5, "llm": "gpt-4o-mini", "device": None}
    UNIFIED_OVERRIDES = {"llm": "gpt-5-mini"}


def _frame(**changes):
    cfg = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    cfg.update(changes)
    return cfg


def _with_fake_harness(fn):
    """Run `fn(tmp)` with a fake harness `fake` registered and its runs/ under a temp dir."""
    tmp = Path(tempfile.mkdtemp())
    fake_mod = type(sys)("_fake_memo"); fake_mod.FakeMemo = _FakeMemo
    sys.modules["_fake_memo"] = fake_mod
    saved_memos, saved_dir = dict(eh.MEMOS), eh.HARNESS_DIR
    eh.MEMOS["fake"] = "_fake_memo:FakeMemo"
    eh.HARNESS_DIR = tmp
    try:
        fn(tmp)
    finally:
        eh.MEMOS.clear(); eh.MEMOS.update(saved_memos)
        eh.HARNESS_DIR = saved_dir
        sys.modules.pop("_fake_memo", None)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- registry ----------------

def test_load_memo_is_lazy_and_names_the_venv_on_import_failure():
    saved = dict(eh.MEMOS)
    eh.MEMOS["ghost"] = "baselines.harness.ghost.memo:Ghost"
    try:
        eh.load_memo("ghost")
    except ImportError as e:
        assert "uv run --project baselines/harness/ghost" in str(e)
    else:
        raise AssertionError("expected ImportError")
    finally:
        eh.MEMOS.clear(); eh.MEMOS.update(saved)
    try:
        eh.load_memo("nope")
    except KeyError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected KeyError")


# ---------------- config resolution ----------------

def test_arm_selects_the_unified_overrides():
    assert _FakeMemo.resolve_config("faithful")["llm"] == "gpt-4o-mini"
    assert _FakeMemo.resolve_config("unified")["llm"] == "gpt-5-mini"
    assert _FakeMemo.resolve_config("unified", {"top_k": 1}) == {"top_k": 1, "llm": "gpt-5-mini", "device": None}


def test_instances_see_defaults_under_any_partial_config():
    # self.config[k] must be safe even when the framework hands a partial dict.
    assert _FakeMemo().config["top_k"] == 5
    assert _FakeMemo(config={"top_k": 2}).config == {"top_k": 2, "llm": "gpt-4o-mini", "device": None}


def test_frame_config_rejects_bad_arm_and_non_mapping_memo():
    tmp = Path(tempfile.mkdtemp())
    try:
        for bad, needle in (({"arm": "paper"}, "arm"), ({"memo": [1]}, "memo"),
                            ({"memory_cache": True}, "memory cache was removed")):
            p = tmp / "c.yaml"; p.write_text(yaml.safe_dump(_frame(**bad)), encoding="utf-8")
            try:
                eh.load_frame_config(p)
            except ValueError as e:
                assert needle in str(e)
            else:
                raise AssertionError(f"expected ValueError for {bad}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- the run directory ----------------

def _stub_run_baseline(calls):
    async def fake(**kw):
        calls.append(kw)
        (kw["out_dir"] / "score.json").write_text("{}", encoding="utf-8")
        return {kw["dataset"]: {"raw_score": 0.5, "stage": "single", "tokens": 12, "eliminated": False}}
    return fake


def test_main_writes_one_directory_per_run_with_records():
    def body(tmp):
        calls = []
        real = eh.run_baseline
        eh.run_baseline = _stub_run_baseline(calls)
        try:
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(_frame(
                harness="fake", arm="unified", memo={"top_k": 7}, run_name="r1")), encoding="utf-8")
            eh.main(["--config", str(cfg_path)])

            run_dir = tmp / "fake" / "runs" / "r1"
            assert run_dir.is_dir()
            # the evaluator got the run dir as out_dir
            kw = calls[0]
            assert kw["out_dir"] == run_dir
            assert kw["memo_class"] is _FakeMemo
            assert kw["memo_config"] == {"top_k": 7, "llm": "gpt-5-mini", "device": None}

            # config.resolved.yaml carries the FULLY expanded memo config
            resolved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
            assert resolved["memo"] == {"top_k": 7, "llm": "gpt-5-mini", "device": None}
            assert resolved["harness"] == "fake" and resolved["arm"] == "unified"

            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            assert record["status"] == "ok" and record["memo_class"] == "_fake_memo:FakeMemo"
            assert "wall_clock_s" in record and "started" in record and "ended" in record

            lines = (tmp / "fake" / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1 and json.loads(lines[0])["raw_score"] == 0.5
            assert (tmp / "fake" / "runs" / "latest").read_text(encoding="utf-8").strip() == "r1"

            # a second run with the same name must NOT overwrite the first
            try:
                eh.main(["--config", str(cfg_path)])
            except FileExistsError:
                pass
            else:
                raise AssertionError("expected FileExistsError")
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_resolved_config_round_trips():
    # Feeding a run's config.resolved.yaml back as --config must resolve to the
    # identical memo config — it is the audit record, so it has to be a valid input.
    def body(tmp):
        calls = []
        real = eh.run_baseline
        eh.run_baseline = _stub_run_baseline(calls)
        try:
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", arm="faithful",
                                                     memo={"top_k": 3}, run_name="a")), encoding="utf-8")
            eh.main(["--config", str(cfg_path)])
            resolved = tmp / "fake" / "runs" / "a" / "config.resolved.yaml"
            replay = yaml.safe_load(resolved.read_text(encoding="utf-8"))
            replay["run_name"] = "b"
            (tmp / "replay.yaml").write_text(yaml.safe_dump(replay), encoding="utf-8")
            eh.main(["--config", str(tmp / "replay.yaml")])
            assert calls[0]["memo_config"] == calls[1]["memo_config"] == {"top_k": 3, "llm": "gpt-4o-mini", "device": None}
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_failed_run_is_recorded_and_not_indexed():
    def body(tmp):
        real = eh.run_baseline

        async def boom(**kw):
            raise RuntimeError("kaboom")
        eh.run_baseline = boom
        try:
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", run_name="x")), encoding="utf-8")
            try:
                eh.main(["--config", str(cfg_path)])
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected the failure to propagate")
            record = json.loads((tmp / "fake" / "runs" / "x" / "run.json").read_text(encoding="utf-8"))
            assert record["status"].startswith("failed: RuntimeError")
            assert not (tmp / "fake" / "runs" / "index.jsonl").exists()
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_describe_prints_both_arms():
    def body(tmp):
        out = eh.describe("fake")
        assert "top_k: 5" in out and "llm: gpt-4o-mini" in out and "llm: gpt-5-mini" in out
    _with_fake_harness(body)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print("all passed" if not failed else f"{failed} failed")
    sys.exit(1 if failed else 0)
