"""The shared harness entrypoint (baselines/harness/eval_harness.py): the lazy
registry, method-config resolution (arm / unified_models / memo:), the
environment guard, the per-run directory and its records.
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
UNIFIED = {"llm": "gpt-5-mini", "embedding": "text-embedding-3-small"}

# A fake harness module shaped like a real memo.py: module-level data + a class
# that only reads self.config.
FAKE_DEFAULTS = {"top_k": 5, "llm": "gpt-4o-mini", "embedder": "all-MiniLM-L6-v2",
                 "dims": 384, "device": None}
FAKE_MODEL_KEYS = {"llm": ("llm",), "embedding": ("embedder",), "embedding_dims": ("dims",)}


class _FakeMemo(MemoClass):
    pass


def _frame(**changes):
    cfg = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    cfg.update(changes)
    return cfg


def _with_fake_harness(fn):
    """Run `fn(tmp)` with a fake harness `fake` registered and its runs/ under a temp dir."""
    tmp = Path(tempfile.mkdtemp())
    fake_mod = type(sys)("_fake_memo")
    fake_mod.FakeMemo = _FakeMemo
    fake_mod.CONFIG_DEFAULTS = FAKE_DEFAULTS
    fake_mod.UNIFIED_MODEL_KEYS = FAKE_MODEL_KEYS
    _FakeMemo.__module__ = "_fake_memo"
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


def _expect(exc_type, fn, needle):
    try:
        fn()
    except exc_type as e:
        assert needle in str(e), (needle, str(e))
    else:
        raise AssertionError(f"expected {exc_type.__name__} containing {needle!r}")


# ---------------- registry ----------------

def test_load_memo_is_lazy_and_names_the_venv_on_import_failure():
    saved = dict(eh.MEMOS)
    eh.MEMOS["ghost"] = "baselines.harness.ghost.memo:Ghost"
    try:
        _expect(ImportError, lambda: eh.load_memo("ghost"), "uv run --project baselines/harness/ghost")
    finally:
        eh.MEMOS.clear(); eh.MEMOS.update(saved)
    _expect(KeyError, lambda: eh.load_memo("nope"), "nope")


# ---------------- method-config resolution ----------------

def _resolve(arm="faithful", unified_models=None, **overrides):
    return eh.resolve_memo_config(FAKE_DEFAULTS, FAKE_MODEL_KEYS, arm=arm,
                                  unified_models=unified_models, overrides=overrides)


def test_faithful_is_the_defaults():
    assert _resolve() == FAKE_DEFAULTS


def test_unified_writes_the_models_and_the_embedder_width():
    assert _resolve("unified", UNIFIED) == {
        "top_k": 5, "llm": "gpt-5-mini", "embedder": "text-embedding-3-small",
        "dims": 1536, "device": None}


def test_memo_overrides_apply_on_top_of_either_arm():
    assert _resolve("faithful", top_k=9)["top_k"] == 9
    assert _resolve("unified", UNIFIED, top_k=9) == {**_resolve("unified", UNIFIED), "top_k": 9}


def test_memo_cannot_touch_a_key_the_arm_controls():
    for key in ("llm", "embedder", "dims"):
        _expect(ValueError, lambda: _resolve("faithful", **{key: "x"}), "controlled by `arm`")


def test_memo_unknown_key_aborts():
    _expect(KeyError, lambda: _resolve("faithful", topk=9), "topk")


def test_model_keys_must_be_declared_defaults_with_known_roles():
    _expect(ValueError, lambda: eh.resolve_memo_config(
        FAKE_DEFAULTS, {"llm": ("missing_key",)}, arm="faithful"), "missing_key")
    _expect(ValueError, lambda: eh.resolve_memo_config(
        FAKE_DEFAULTS, {"reranker": ("llm",)}, arm="faithful"), "reranker")


def test_arm_and_unified_models_must_agree():
    _expect(ValueError, lambda: eh.validate_arm("paper", None), "arm must be")
    _expect(ValueError, lambda: eh.validate_arm("faithful", UNIFIED), "only read under `arm: unified`")
    _expect(ValueError, lambda: eh.validate_arm("unified", None), "needs `unified_models`")
    _expect(ValueError, lambda: eh.validate_arm("unified", {"llm": "gpt-5-mini"}), "needs `unified_models`")
    _expect(ValueError, lambda: eh.validate_arm(
        "unified", {"llm": "gpt-5-mini", "embedding": "all-MiniLM-L6-v2"}), "API embedding model")
    _expect(ValueError, lambda: eh.validate_arm(
        "unified", {"llm": "gpt-5-mini", "embedding": "text-embedding-9"}), "unknown API embedding model")
    eh.validate_arm("faithful", None)
    eh.validate_arm("unified", UNIFIED)


def test_frame_config_validation():
    tmp = Path(tempfile.mkdtemp())
    try:
        for bad, exc, needle in (
            ({"arm": "paper"}, ValueError, "arm"),
            ({"memo": [1]}, ValueError, "memo"),
            ({"memory_cache": True}, ValueError, "memory cache was removed"),
            ({"arm": "faithful", "unified_models": UNIFIED}, ValueError, "only read under"),
            ({"arm": "unified", "unified_models": None}, ValueError, "needs `unified_models`"),
        ):
            p = tmp / "c.yaml"; p.write_text(yaml.safe_dump(_frame(**bad)), encoding="utf-8")
            _expect(exc, lambda: eh.load_frame_config(p), needle)
        cfg = dict(_frame()); cfg.pop("unified_models")
        p = tmp / "c.yaml"; p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        _expect(ValueError, lambda: eh.load_frame_config(p), "unified_models")   # a frame key: must be listed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- environment guard ----------------

def test_environment_guard_names_what_would_change_the_run():
    eh.check_environment("zep", {"OPENAI_API_KEY": "sk-...", "PATH": "/bin"})   # credentials are fine
    _expect(RuntimeError, lambda: eh.check_environment("amem", {"OPENAI_BASE_URL": "http://x"}),
            "OPENAI_BASE_URL")
    _expect(RuntimeError, lambda: eh.check_environment("mem0", {"OPENROUTER_API_KEY": "k"}),
            "OPENROUTER_API_KEY")
    _expect(RuntimeError, lambda: eh.check_environment("zep", {"SEMAPHORE_LIMIT": "5"}), "SEMAPHORE_LIMIT")
    eh.check_environment("amem", {"SEMAPHORE_LIMIT": "5"})   # graphiti's knob only matters to zep


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
            text = "# my notes\n" + yaml.safe_dump(_frame(
                harness="fake", arm="unified", unified_models=UNIFIED, memo={"top_k": 7}, run_name="r1"))
            cfg_path.write_text(text, encoding="utf-8")
            eh.main(["--config", str(cfg_path)])

            run_dir = tmp / "fake" / "runs" / "r1"
            kw = calls[0]
            assert kw["out_dir"] == run_dir and kw["memo_class"] is _FakeMemo
            expected = {"top_k": 7, "llm": "gpt-5-mini", "embedder": "text-embedding-3-small",
                        "dims": 1536, "device": None}
            assert kw["memo_config"] == expected

            # config.yaml is the input byte-for-byte (comments included)
            assert (run_dir / "config.yaml").read_text(encoding="utf-8") == text
            # memo_config.resolved.yaml is the full record — and NOT a valid --config
            assert yaml.safe_load((run_dir / "memo_config.resolved.yaml").read_text(encoding="utf-8")) == expected
            _expect(Exception, lambda: eh.load_frame_config(run_dir / "memo_config.resolved.yaml"), "")

            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            assert record["status"] == "ok" and record["memo_class"] == "_fake_memo:FakeMemo"
            assert record["unified_models"] == UNIFIED
            assert "wall_clock_s" in record and "started" in record and "ended" in record

            lines = (tmp / "fake" / "runs" / "index.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(lines) == 1 and json.loads(lines[0])["raw_score"] == 0.5
            assert (tmp / "fake" / "runs" / "latest").read_text(encoding="utf-8").strip() == "r1"

            # a second run with the same name must NOT overwrite the first
            _expect(FileExistsError, lambda: eh.main(["--config", str(cfg_path)]), "already exists")
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_rerun_from_config_yaml_switches_arm_cleanly():
    # The regression this layout fixes: the old config.resolved.yaml pinned every
    # model key under `memo:`, so flipping `arm` on a re-run silently kept the
    # old models while the records claimed the new arm.
    def body(tmp):
        calls = []
        real = eh.run_baseline
        eh.run_baseline = _stub_run_baseline(calls)
        try:
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", arm="faithful",
                                                     memo={"top_k": 3}, run_name="a")), encoding="utf-8")
            eh.main(["--config", str(cfg_path)])
            replay = yaml.safe_load((tmp / "fake" / "runs" / "a" / "config.yaml").read_text(encoding="utf-8"))
            replay.update(run_name="b", arm="unified", unified_models=UNIFIED)
            (tmp / "replay.yaml").write_text(yaml.safe_dump(replay), encoding="utf-8")
            eh.main(["--config", str(tmp / "replay.yaml")])
            assert calls[0]["memo_config"] == {**FAKE_DEFAULTS, "top_k": 3}
            assert calls[1]["memo_config"] == {"top_k": 3, "llm": "gpt-5-mini",
                                              "embedder": "text-embedding-3-small", "dims": 1536,
                                              "device": None}
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_a_config_error_leaves_no_run_directory():
    def body(tmp):
        real = eh.run_baseline
        eh.run_baseline = _stub_run_baseline([])
        try:
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", memo={"topk": 1}, run_name="t")),
                                encoding="utf-8")
            _expect(KeyError, lambda: eh.main(["--config", str(cfg_path)]), "topk")
            assert not (tmp / "fake" / "runs" / "t").exists()
            # fixing the typo and re-running under the same name just works
            cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", memo={"top_k": 1}, run_name="t")),
                                encoding="utf-8")
            eh.main(["--config", str(cfg_path)])
            assert (tmp / "fake" / "runs" / "t" / "run.json").exists()
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_main_refuses_to_start_under_a_behaviour_changing_env_var():
    import os

    def body(tmp):
        cfg_path = tmp / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(_frame(harness="fake", run_name="e")), encoding="utf-8")
        os.environ["OPENROUTER_API_KEY"] = "k"
        try:
            _expect(RuntimeError, lambda: eh.main(["--config", str(cfg_path)]), "OPENROUTER_API_KEY")
            assert not (tmp / "fake" / "runs" / "e").exists()
        finally:
            os.environ.pop("OPENROUTER_API_KEY", None)
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
            _expect(RuntimeError, lambda: eh.main(["--config", str(cfg_path)]), "kaboom")
            record = json.loads((tmp / "fake" / "runs" / "x" / "run.json").read_text(encoding="utf-8"))
            assert record["status"].startswith("failed: RuntimeError")
            assert not (tmp / "fake" / "runs" / "index.jsonl").exists()
        finally:
            eh.run_baseline = real
    _with_fake_harness(body)


def test_describe_prints_defaults_and_what_unified_sets():
    def body(tmp):
        out = eh.describe("fake")
        assert "top_k: 5" in out and "llm: gpt-4o-mini" in out
        assert "unified_models.llm -> llm" in out
        assert "unified_models.embedding -> embedder" in out
        assert "width of unified_models.embedding -> dims" in out
    _with_fake_harness(body)


def test_logger_configure_routes_the_file_tape_explicitly():
    import logging
    from common import logger as common_logger
    tmp = Path(tempfile.mkdtemp())
    saved = (common_logger._configured_log_dir, common_logger._configured_log_file)
    try:
        common_logger.configure(tmp, "tape.log")
        log = common_logger.get_logger("_test_configure_routes")
        log.info("hello")
        for h in log.handlers:
            h.flush()
        assert "hello" in (tmp / "tape.log").read_text(encoding="utf-8")
    finally:
        common_logger._configured_log_dir, common_logger._configured_log_file = saved
        lg = logging.getLogger("_test_configure_routes")
        for h in list(lg.handlers):
            h.close(); lg.removeHandler(h)
        common_logger._initialized_loggers.pop("_test_configure_routes", None)
        shutil.rmtree(tmp, ignore_errors=True)


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
