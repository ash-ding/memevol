"""Workspace identity: hash-named harness dirs, duplicate reuse, history.json.

Zero-dependency runner (no pytest in the venvs):

    uv run python tests/test_forge_workspace.py

A harness directory is named by the hash of its own code, so the same code is
always the same directory — which is what makes a repeated proposal free
instead of a second evaluation. Names then carry no ordering, so the order
steps produced things lives in `history.json`; these tests pin both halves.
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")

from forge import orchestrator as O           # noqa: E402
from forge.paths import paths                 # noqa: E402
from forge.selection import Entry, Frontier    # noqa: E402

_RUN_ID = "test_workspace_ids"


@contextlib.contextmanager
def _workspace():
    paths.set_run_id(_RUN_ID)
    paths.harnesses_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield paths.workspace
    finally:
        shutil.rmtree(paths.workspace, ignore_errors=True)
        paths._run_id = None


def _pending(body="# harness body\n", meta=None):
    """A harness dir as the proposer leaves it: pending name, code inside."""
    d = paths.harnesses_dir / O._new_pending_id()
    d.mkdir(parents=True)
    (d / "harness.py").write_text(body, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta or {"parent_ids": ["abc123def456"]}),
                                 encoding="utf-8")
    return d


# ---------------- naming ----------------

def test_a_settled_harness_is_named_by_its_code_hash():
    with _workspace():
        d = _pending()
        final_id, final_dir, is_dup = O._finalize_harness_dir(d)

        assert not is_dup
        assert O._HARNESS_ID_RX.match(final_id), final_id
        assert len(final_id) == O.HARNESS_ID_LEN
        assert final_dir.name == final_id and final_dir.exists()
        assert not d.exists(), "the pending dir must be gone, not left behind"

        meta = json.loads((final_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["content_hash"].startswith(final_id)
        assert meta["parent_ids"] == ["abc123def456"], "the proposer's meta survives"
        assert meta["created_at"]


def test_the_name_follows_the_code_not_the_arrival_order():
    with _workspace():
        first, _, _ = O._finalize_harness_dir(_pending("# one\n"))
        second, _, _ = O._finalize_harness_dir(_pending("# two\n"))
        assert first != second
        # Same code written later lands on the first one's name.
        again, again_dir, is_dup = O._finalize_harness_dir(_pending("# one\n"))
        assert again == first and is_dup


def test_identical_code_is_a_duplicate_and_keeps_the_original_results():
    """The proposer reproducing an earlier harness must not cost a second
    evaluation — and must not clobber the results already on disk."""
    with _workspace():
        first_id, first_dir, _ = O._finalize_harness_dir(_pending())
        (first_dir / "locomo").mkdir()
        (first_dir / "locomo" / "score.json").write_text('{"raw_score": 0.42}',
                                                         encoding="utf-8")

        again = _pending()
        dup_id, dup_dir, is_dup = O._finalize_harness_dir(again)

        assert is_dup and dup_id == first_id and dup_dir == first_dir
        assert not again.exists(), "the duplicate copy is dropped"
        assert json.loads((first_dir / "locomo" / "score.json").read_text())["raw_score"] == 0.42


def test_finalizing_an_already_named_dir_is_a_no_op():
    with _workspace():
        final_id, final_dir, _ = O._finalize_harness_dir(_pending())
        again_id, again_dir, is_dup = O._finalize_harness_dir(final_dir)
        assert (again_id, again_dir, is_dup) == (final_id, final_dir, False)


# ---------------- history.json ----------------

def test_history_records_each_step_in_order():
    with _workspace():
        O._record_history(0, "seed", O._history_record(
            "aaaaaaaaaaaa", "skipped", {"locomo": {"raw_score": 0.1}}, []))
        O._record_history(1, "propose", O._history_record(
            "bbbbbbbbbbbb", "passed", {"locomo": {"raw_score": 0.3}}, ["aaaaaaaaaaaa"]))
        O._record_history(1, "propose", O._history_record(
            "cccccccccccc", "duplicate", {"locomo": {"raw_score": 0.3}}, ["aaaaaaaaaaaa"]))

        history = json.loads(paths.history_path.read_text(encoding="utf-8"))
        assert history["run_id"] == _RUN_ID
        assert [s["step"] for s in history["steps"]] == [0, 1]
        assert history["steps"][0]["kind"] == "seed"
        step1 = history["steps"][1]
        assert [h["id"] for h in step1["harnesses"]] == ["bbbbbbbbbbbb", "cccccccccccc"], \
            "both candidates of a step are recorded, in the order they settled"
        assert step1["harnesses"][0]["parent_ids"] == ["aaaaaaaaaaaa"]
        assert step1["harnesses"][0]["scores"] == {"locomo": 0.3}
        assert step1["harnesses"][1]["sanity"] == "duplicate"


def test_history_is_written_per_candidate_not_at_the_end():
    """A run that dies mid-search must still explain what it produced."""
    with _workspace():
        O._record_history(1, "propose", O._history_record(
            "aaaaaaaaaaaa", "passed", {}, []))
        assert paths.history_path.exists()
        assert len(json.loads(paths.history_path.read_text())["steps"][0]["harnesses"]) == 1


def test_a_corrupt_history_is_replaced_rather_than_fatal():
    with _workspace():
        paths.history_path.write_text("{not json", encoding="utf-8")
        O._record_history(1, "propose", O._history_record("aaaaaaaaaaaa", "passed", {}, []))
        history = json.loads(paths.history_path.read_text(encoding="utf-8"))
        assert [s["step"] for s in history["steps"]] == [1]


# ---------------- orphan adoption ----------------

def test_orphan_scan_returns_unknown_dirs_oldest_first():
    with _workspace():
        known_id, _, _ = O._finalize_harness_dir(_pending("# known\n"))
        frontier = Frontier()
        frontier.add(Entry(id=known_id, objectives={}, parent_ids=[]))

        orphan_id, orphan_dir, _ = O._finalize_harness_dir(_pending("# orphan\n"))
        time.sleep(0.01)
        unsettled = _pending("# never settled\n")      # pending_*: no identity
        (paths.harnesses_dir / "notes").mkdir()        # not a harness at all

        found = O._scan_orphan_dirs(frontier)
        assert [p.name for p in found] == [orphan_dir.name, unsettled.name], found
        assert known_id not in [p.name for p in found]


def test_classify_discards_a_dir_whose_code_never_settled():
    with _workspace():
        assert O._classify_orphan(_pending(), ["locomo"]) == "incomplete"


def test_classify_accepts_a_settled_dir_with_results():
    with _workspace():
        _, d, _ = O._finalize_harness_dir(_pending())
        for sub in ("locomo/sanity", "locomo"):
            (d / sub).mkdir(parents=True, exist_ok=True)
            (d / sub / "score.json").write_text("{}", encoding="utf-8")
        assert O._classify_orphan(d, ["locomo"]) == "complete"


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
