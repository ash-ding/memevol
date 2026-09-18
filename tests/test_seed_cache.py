"""`seeds/` as a cache of evaluated harnesses.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_seed_cache.py

The point of the cache is to not pay twice for the same evaluation — so what
matters is that "the same evaluation" is defined tightly enough. Most of these
tests are about the key: which inputs must change it, and that nothing else
(ordering, formatting) does.
"""
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

from forge import seed_cache as SC          # noqa: E402
from forge import orchestrator as O         # noqa: E402
from forge.paths import paths               # noqa: E402

_CFG = {
    "datasets": {"locomo": {"single_stage": {"n_conversations": 2, "n_qa": 20}}},
    "split": "search",
    "progressive": False,
    "model": "gpt-5-mini",
    "judge_model": "gpt-5-mini",
    "random_sample": False,
    "sampling_seed": 42,
}


@contextlib.contextmanager
def _cache_dir():
    """Point the cache at a temp dir — never touch the repo's real seeds/."""
    tmp = Path(tempfile.mkdtemp())
    real_sc, real_orch = SC.SEEDS_DIR, O.SEEDS_DIR
    SC.SEEDS_DIR = tmp
    O.SEEDS_DIR = tmp
    try:
        yield tmp
    finally:
        SC.SEEDS_DIR, O.SEEDS_DIR = real_sc, real_orch
        shutil.rmtree(tmp, ignore_errors=True)


def _evaluated_harness(root: Path, dataset="locomo", score=0.5):
    """A harness dir as it looks after an evaluation: code + results."""
    d = root / "abc123abc123"
    (d / dataset / "traces").mkdir(parents=True)
    (d / "harness.py").write_text("# code\n", encoding="utf-8")
    (d / "meta.json").write_text('{"parent_ids": []}', encoding="utf-8")
    (d / dataset / "score.json").write_text(json.dumps({"raw_score": score}),
                                            encoding="utf-8")
    (d / dataset / "token_usage.json").write_text("{}", encoding="utf-8")
    (d / dataset / "traces" / "u1.json").write_text('{"steps": []}', encoding="utf-8")
    return d


# ---------------- the key ----------------

def test_the_key_ignores_ordering_but_not_content():
    a = SC.eval_key({"dataset": "locomo", "model": "gpt-5-mini"})
    b = SC.eval_key({"model": "gpt-5-mini", "dataset": "locomo"})
    assert a == b, "key order must not change the key"
    assert a != SC.eval_key({"dataset": "locomo", "model": "gpt-5"})


def test_every_input_that_can_change_a_number_changes_the_key():
    base = SC.eval_inputs(_CFG, "locomo")
    key = SC.eval_key(base)

    def changed(**overrides):
        cfg = json.loads(json.dumps(_CFG))
        cfg.update({k: v for k, v in overrides.items() if k != "sizing"})
        if "sizing" in overrides:
            cfg["datasets"]["locomo"]["single_stage"] = overrides["sizing"]
        return SC.eval_key(SC.eval_inputs(cfg, "locomo"))

    assert changed(model="gpt-5") != key, "a different answerer is a different score"
    assert changed(judge_model="gpt-5") != key, "a different judge is a different score"
    assert changed(split="test") != key
    assert changed(progressive=True) != key
    assert changed(random_sample=True) != key
    assert changed(sizing={"n_conversations": 4, "n_qa": 20}) != key, \
        "per-query cost is amortized over the sampled queries — sizing matters"
    # ... and the evaluation code itself.
    assert SC.eval_key({**base, "repo_version": "deadbeef"}) != key


def test_a_dirty_tree_never_matches_a_clean_one():
    clean = SC.eval_key(SC.eval_inputs(_CFG, "locomo") | {"repo_version": "abc123"})
    dirty = SC.eval_key(SC.eval_inputs(_CFG, "locomo") | {"repo_version": "abc123-dirty"})
    assert clean != dirty


def test_a_per_dataset_judge_override_is_part_of_the_key():
    cfg = json.loads(json.dumps(_CFG))
    cfg["datasets"]["locomo"]["judge_model"] = "gpt-5.4-2026-03-05"
    assert SC.eval_inputs(cfg, "locomo")["judge_model"] == "gpt-5.4-2026-03-05"
    assert SC.eval_key(SC.eval_inputs(cfg, "locomo")) != SC.eval_key(SC.eval_inputs(_CFG, "locomo"))


# ---------------- storing + finding ----------------

def test_store_keeps_the_code_the_results_and_a_readable_manifest():
    with _cache_dir() as cache, tempfile.TemporaryDirectory() as td:
        src = _evaluated_harness(Path(td))
        written = SC.store(src, "abc123abc123", _CFG,
                           {"locomo": {"raw_score": 0.5, "score_max": 1,
                                       "cost_tokens_per_query": 12.5}},
                           sanity_status="passed", run_id="r1")

        assert len(written) == 1
        entry = cache / "abc123abc123"
        assert (entry / "harness.py").read_text() == "# code\n"
        assert not (entry / "locomo").exists(), "results live under evals/, not the code dir"

        result = written[0]
        assert json.loads((result / "score.json").read_text())["raw_score"] == 0.5
        assert (result / "traces" / "u1.json").exists(), "traces are kept too"
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest["inputs"]["dataset"] == "locomo"
        assert manifest["inputs"]["model"] == "gpt-5-mini"
        assert manifest["raw_score"] == 0.5 and manifest["run_id"] == "r1"
        assert manifest["cost_tokens_per_query"] == 12.5

        found = SC.lookup("abc123abc123", manifest["eval_key"])
        assert found == result
        assert SC.lookup("abc123abc123", "0" * 16) is None
        assert SC.lookup("nosuchharness", manifest["eval_key"]) is None


def test_one_harness_keeps_a_result_per_configuration():
    with _cache_dir() as cache, tempfile.TemporaryDirectory() as td:
        src = _evaluated_harness(Path(td))
        SC.store(src, "abc123abc123", _CFG, {"locomo": {"raw_score": 0.5}},
                 sanity_status="passed", run_id="r1")
        bigger = json.loads(json.dumps(_CFG))
        bigger["datasets"]["locomo"]["single_stage"] = {"n_conversations": 4, "n_qa": 40}
        SC.store(src, "abc123abc123", bigger, {"locomo": {"raw_score": 0.4}},
                 sanity_status="passed", run_id="r2")

        evals = sorted((cache / "abc123abc123" / "evals").iterdir())
        assert len(evals) == 2, [p.name for p in evals]
        index = json.loads((cache / "abc123abc123" / "index.json").read_text())
        assert len(index["evals"]) == 2
        assert {e["dataset"] for e in index["evals"]} == {"locomo"}
        assert SC.entries()[0]["harness_id"] == "abc123abc123"


def test_a_dataset_that_produced_nothing_is_not_cached():
    with _cache_dir(), tempfile.TemporaryDirectory() as td:
        src = _evaluated_harness(Path(td))
        written = SC.store(src, "abc123abc123", _CFG,
                           {"locomo": {"raw_score": 0.5},
                            "dynamicmem": {"raw_score": 0.0}},   # no score.json on disk
                           sanity_status="passed", run_id="r1")
        assert len(written) == 1, [str(w) for w in written]
        manifest = json.loads((written[0] / "manifest.json").read_text())
        assert manifest["inputs"]["dataset"] == "locomo", "only the one with results"


def test_a_cache_failure_never_fails_the_run():
    """The results are already computed; losing the cache copy must not lose
    them."""
    with _cache_dir():
        written = SC.store(Path("/definitely/not/here"), "abc123abc123", _CFG,
                           {"locomo": {"raw_score": 0.5}},
                           sanity_status="passed", run_id="r1")
        assert written == []


# ---------------- what the orchestrator caches ----------------

def test_only_harnesses_with_real_results_are_cached():
    with _cache_dir() as cache, tempfile.TemporaryDirectory() as td:
        src = _evaluated_harness(Path(td))
        paths.set_run_id("test_seed_cache")
        try:
            for status in ("failed", "env_build_failed", "proposer_failed", "duplicate"):
                O._cache_evaluated(src, "abc123abc123", _CFG,
                                   {"locomo": {"raw_score": 0.0}}, status)
            assert not list(cache.iterdir()), "nothing worth caching was stored"

            O._cache_evaluated(src, "abc123abc123", _CFG,
                               {"locomo": {"raw_score": 0.5}}, "passed")
            assert (cache / "abc123abc123" / "index.json").exists()
        finally:
            paths._run_id = None


def test_a_seed_source_is_a_cache_entry_or_a_path():
    with _cache_dir() as cache, tempfile.TemporaryDirectory() as td:
        src = _evaluated_harness(Path(td))
        SC.store(src, "abc123abc123", _CFG, {"locomo": {"raw_score": 0.5}},
                 sanity_status="passed", run_id="r1")

        assert O._resolve_seed_source("abc123abc123") == cache / "abc123abc123"
        assert O._resolve_seed_source("baselines/harness/no_memory") is not None, \
            "a method that has never been through forge is reachable by path"
        assert O._resolve_seed_source("no_such_thing") is None


# ---------------- the no_memory baseline ----------------

def test_no_memory_is_a_registered_baseline_that_answers_from_nothing():
    import asyncio
    from baselines.harness.eval_harness import MEMOS, load_memo

    assert "no_memory" in MEMOS
    memo = load_memo("no_memory")(config={})
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(memo.build_memory_from_data(None)) is None
        assert loop.run_until_complete(memo.retrieve_memory_for_query(None)) == {}
    finally:
        loop.close()
    assert not hasattr(memo, "use_memory_to_answer")


def test_no_memory_also_ships_the_forge_harness_shape():
    """The same method written against forge's contract, so a workspace can
    seed from it before it has ever been cached."""
    from forge.contract import load_harness_class
    from forge.paths import PROJECT_ROOT

    cls = load_harness_class(PROJECT_ROOT / "baselines" / "harness" / "no_memory")
    assert cls.__name__ == "NoMemoryHarness"


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
